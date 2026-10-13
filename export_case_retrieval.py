"""Export per-query top-50 retrieval of H²MATS and SPGMT for case studies.

Usage:

    python export_case_retrieval.py --dataset porto --distance-type TP \
        --ours-checkpoint saved_models/porto/TP_xxx/epoch_49.pt \
        --spgmt-checkpoint saved_models/porto/TP_yyy --cuda 0

Checkpoints are found as in eval_spatial_model.py: a relative path is tried
from the launch directory, then from this script's directory, and a run
directory means its last epoch_*.pt. SPGMT's positional encoding and kNN
structure graphs follow usePE/useSI in setting.py unless --spgmt-pe /
--spgmt-knn say otherwise; its printed metrics show whether they match the
checkpoint.

Both models embed the standard test split, then every test trajectory queries
the others, exactly as in Trainer.Spa_eval. The output
case_study/<dataset>_<distance>_retrieval.npz holds each model's top-50 lists,
the ground-truth top-50, the test trajectories and the road network (node
coordinates, edges, L1-L3 regions), so plot_case_study.py needs nothing else.
"""
import argparse
import os
import os.path as osp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import spatial_data_utils
import test_method
from eval_spatial_model import refresh_config_paths, resolve_checkpoint
from hybrid_geo_network import HybridGeoTrajSimEncoder
from model_network import GraphTrajSimEncoder
from setting import SetParameter
from Trainer import move_hierarchy_data_to_device

TOP_K = 50
METRICS = ("HR-10", "HR-50", "R10@50", "R1@1", "R1@10", "R1@50")


def require(paths):
    missing = [path for path in paths if not osp.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"missing input files (relative to {os.getcwd()}):\n  " + "\n  ".join(missing))


def find_checkpoint(path, config, launch_dir):
    """Resolve a checkpoint like eval_spatial_model.py; call after changing to this script's directory."""
    try:
        found = resolve_checkpoint(SimpleNamespace(checkpoint=path.strip(), run_time=None, epoch=None),
                                   config, launch_dir)
    except FileNotFoundError:
        # repr() exposes stray characters such as a trailing \r from a Windows-edited script.
        raise FileNotFoundError(
            f"checkpoint {path!r} not found under {launch_dir} or {os.getcwd()}") from None
    return osp.abspath(found)


def build_ours(config, device):
    g = config.gtraj
    return HybridGeoTrajSimEncoder(
        feature_size=config.feature_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        concat=config.concat,
        device=device,
        useLSTM=g["useLSTM"],
        curvature=g["hybridCurvature"],
        seq_encoder=g["seqEncoder"],
        mamba_layers=g["mambaLayers"],
        mamba_d_state=g["mambaDState"],
        mamba_d_conv=g["mambaDConv"],
        mamba_expand=g["mambaExpand"],
        coord_encoder=g["coordEncoder"],
        hmt_backbone=g["hmtBackbone"],
        hmt_layers=g["hmtLayers"],
        hmt_kernel_size=g["hmtKernelSize"],
        hmt_dropout=g["hmtDropout"],
        hmt_transformer_heads=g["hmtTransformerHeads"],
        hmt_scale_aligned_graph=g["hmtScaleAlignedGraph"],
        coord_fusion=g["coordFusion"],
        coord_transformer_checkpoint=g["coordTransformerCheckpoint"],
        graph_encoder=g["graphEncoder"],
        hierarchy_blocks=g["hierarchyBlocks"],
        hierarchy_dropout=g["hierarchyDropout"],
        graph_scale_dropout=g["graphScaleDropout"],
        use_transition_features=g["useTransitionFeatures"],
        transition_residual_init=g["transitionResidualInit"],
    )


def build_spgmt(config, device, use_pe, use_knn):
    return GraphTrajSimEncoder(
        feature_size=config.feature_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        concat=config.concat,
        device=device,
        usePE=use_pe,
        useSI=use_knn,
        useLSTM=True,
        dataset=config.dataset,
        alpha1=config.alpha1,
        alpha2=config.alpha2,
        seq_encoder="transformer",
        coord_encoder="lstm",
    )


def spgmt_network(config, device, use_pe, use_knn):
    if use_knn:
        graphs = [graph.to(device)
                  for graph in spatial_data_utils.load_neighbor(config.dataset, config.num_knn)]
    else:
        # load_netowrk reads the edge file from the module's own config.
        spatial_data_utils.config = config
        graphs = spatial_data_utils.load_netowrk(config.dataset).to(device)
    # Without positional encoding the anchor distances are never read.
    anchors = (torch.load(anchor_file(config), map_location=device) if use_pe else None)
    return [graphs, anchors]


def anchor_file(config):
    return osp.join("dataset", config.dataset, "distance_to_anchor_node.pt")


def knn_files(config):
    knn_dir = osp.join("dataset", config.dataset, str(config.num_knn))
    return [osp.join(knn_dir, f"k{k}_{kind}.npy")
            for k in (config.num_knn, 2 * config.num_knn, 3 * config.num_knn)
            for kind in ("neighbor", "distance")]


@torch.no_grad()
def embed(name, net, checkpoint, network_input, loader, size, device):
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.to(device)
    net.eval()
    embeddings = None
    for data, coor, _, lengths, idx in tqdm(loader, desc=f"embed {name}"):
        batch = net(network_input, data, coor, lengths).float()
        if embeddings is None:
            embeddings = torch.zeros((size, batch.size(1)), device=device)
        embeddings[idx] = batch
    return embeddings


@torch.no_grad()
def rank(embeddings, gt_top1, batch_size=256):
    """Top-50 of every query (self excluded) and the rank of its true nearest neighbour."""
    norms = (embeddings * embeddings).sum(dim=1)
    gt_top1 = torch.as_tensor(gt_top1, device=embeddings.device)
    top, gt1_rank = [], []
    for start in range(0, embeddings.size(0), batch_size):
        rows = embeddings[start:start + batch_size]
        local = torch.arange(rows.size(0), device=rows.device)
        dist = norms[start:start + rows.size(0), None] + norms[None, :] - 2 * rows @ embeddings.t()
        dist = torch.clamp(dist, min=0.0)
        dist[local, start + local] = float("inf")
        top.append(torch.topk(dist, k=TOP_K, dim=1, largest=False).indices.cpu())
        target = dist[local, gt_top1[start:start + rows.size(0)]]
        gt1_rank.append((dist < target[:, None]).sum(dim=1).cpu() + 1)
    return torch.cat(top).numpy().astype(np.int32), torch.cat(gt1_rank).numpy().astype(np.int32)


def official_metrics(embeddings, labels, device):
    """The paper's six metrics, computed by test_method exactly as Trainer.Spa_eval does."""
    acc = test_method.test_spa_model(embeddings, labels, device).mean(axis=0)
    acc[0] /= 10.0
    acc[1] /= 50.0
    acc[2] /= 10.0
    return acc.numpy().astype(np.float32)


def road_network_arrays(config):
    hierarchy = {
        level: np.load(osp.join(config.hierarchy_dir, f"node_to_{level}.npy")).astype(np.int32)
        for level in ("l1", "l2", "l3")
    }
    nodes = pd.read_csv(config.node_file)
    xy = np.full((len(hierarchy["l1"]), 2), np.nan, dtype=np.float64)
    xy[nodes["node"].to_numpy()] = nodes[["lng", "lat"]].to_numpy()
    edges = pd.read_csv(config.edge_file)[["s_node", "e_node"]].to_numpy(dtype=np.int32)
    return dict(node_xy=xy, edges=edges, **{f"node_to_{k}": v for k, v in hierarchy.items()})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["beijing", "porto", "tdrive"], default="porto")
    parser.add_argument("--distance-type", choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
                        default="TP")
    parser.add_argument("--ours-checkpoint", required=True)
    parser.add_argument("--spgmt-checkpoint", required=True)
    parser.add_argument("--spgmt-pe", action=argparse.BooleanOptionalAction, default=None,
                        help="SPGMT anchor positional encoding (default: usePE in setting.py).")
    parser.add_argument("--spgmt-knn", action=argparse.BooleanOptionalAction, default=None,
                        help="SPGMT kNN structure graphs (default: useSI in setting.py).")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    launch_dir = Path.cwd()
    output = Path(args.output).resolve() if args.output else None
    os.chdir(Path(__file__).resolve().parent)

    config = SetParameter()
    config.dataset = args.dataset
    config.distance_type = args.distance_type
    refresh_config_paths(config)
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    output = output or Path("case_study") / f"{config.dataset}_{config.distance_type}_retrieval.npz"

    ours_checkpoint = find_checkpoint(args.ours_checkpoint, config, launch_dir)
    spgmt_checkpoint = find_checkpoint(args.spgmt_checkpoint, config, launch_dir)
    use_pe = config.gtraj["usePE"] if args.spgmt_pe is None else args.spgmt_pe
    use_knn = config.gtraj["useSI"] if args.spgmt_knn is None else args.spgmt_knn
    print(f"H2MATS checkpoint: {ours_checkpoint}")
    print(f"SPGMT checkpoint:  {spgmt_checkpoint} (positional encoding={use_pe}, kNN graphs={use_knn})")

    require([
        config.spatial_test_set, config.node_file, config.edge_file,
        osp.join("data", config.dataset, "node_features.npy"),
        *[osp.join(config.hierarchy_dir, f"node_to_{level}.npy") for level in ("l1", "l2", "l3")],
        *([anchor_file(config)] if use_pe else []),
        *(knn_files(config) if use_knn else []),
    ])

    loader, size = spatial_data_utils.vali_data_loader(config.spatial_test_set, config.gtraj["test_batch"])
    test_set = np.load(config.spatial_test_set, allow_pickle=True)
    labels = np.asarray(test_set["vali_y"], dtype=np.int64)
    trajectories = [np.asarray(traj, dtype=np.int32) for traj in test_set["vali_list"]]
    print(f"{config.dataset} {config.distance_type}: {size} test trajectories, device {device}")

    results = {}
    runs = (
        ("ours", build_ours, ours_checkpoint, lambda: move_hierarchy_data_to_device(
            spatial_data_utils.load_hierarchy_multilevel(config.dataset), device)),
        ("spgmt", lambda c, d: build_spgmt(c, d, use_pe, use_knn), spgmt_checkpoint,
         lambda: spgmt_network(config, device, use_pe, use_knn)),
    )
    for name, build, checkpoint, load_network in runs:
        network_input = load_network()
        embeddings = embed(name, build(config, device), checkpoint, network_input, loader, size, device)
        results[f"{name}_top"], results[f"{name}_gt1_rank"] = rank(embeddings, labels[:, 0])
        results[f"{name}_metrics"] = official_metrics(embeddings, torch.as_tensor(labels), device)
        print(name, " ".join(f"{m}={100 * v:.2f}" for m, v in zip(METRICS, results[f"{name}_metrics"])))
        del embeddings, network_input
        torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dataset=config.dataset,
        distance_type=config.distance_type,
        metric_names=np.asarray(METRICS),
        gt_top=labels.astype(np.int32),
        traj_nodes=np.concatenate(trajectories),
        traj_offsets=np.cumsum([0] + [len(traj) for traj in trajectories]).astype(np.int64),
        **results,
        **road_network_arrays(config),
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
