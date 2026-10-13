import os
import os.path as osp

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import spatial_data_utils
from hybrid_geo_network import HybridGeoTrajSimEncoder
from model_network import GraphTrajSimEncoder
from Trainer import move_hierarchy_data_to_device


def build_spatial_encoder(config):
    device = "cuda:" + str(config.cuda) if torch.cuda.is_available() else "cpu"
    if config.gtraj.get("useHybridGeo", False):
        net = HybridGeoTrajSimEncoder(
            feature_size=config.feature_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            concat=config.concat,
            device=device,
            useLSTM=config.gtraj["useLSTM"],
            curvature=config.gtraj.get("hybridCurvature", 1.0),
            seq_encoder=config.gtraj.get("seqEncoder", "transformer"),
            mamba_layers=config.gtraj.get("mambaLayers", 2),
            mamba_d_state=config.gtraj.get("mambaDState", 16),
            mamba_d_conv=config.gtraj.get("mambaDConv", 4),
            mamba_expand=config.gtraj.get("mambaExpand", 2),
            coord_encoder=config.gtraj.get("coordEncoder", "lstm"),
            hmt_backbone=config.gtraj.get("hmtBackbone", "conv"),
            hmt_layers=config.gtraj.get("hmtLayers", 2),
            hmt_kernel_size=config.gtraj.get("hmtKernelSize", 5),
            hmt_dropout=config.gtraj.get("hmtDropout", 0.1),
            hmt_transformer_heads=config.gtraj.get("hmtTransformerHeads", 4),
            hmt_scale_aligned_graph=config.gtraj.get("hmtScaleAlignedGraph", False),
            coord_fusion=config.gtraj.get("coordFusion", "late_concat"),
            graph_encoder=config.gtraj.get("graphEncoder", "hybrid_multilevel"),
            hierarchy_blocks=config.gtraj.get("hierarchyBlocks", 2),
            hierarchy_dropout=config.gtraj.get("hierarchyDropout", 0.1),
            graph_scale_dropout=config.gtraj.get("graphScaleDropout", 0.2),
            use_transition_features=config.gtraj.get("useTransitionFeatures", True),
            transition_residual_init=config.gtraj.get("transitionResidualInit", -2.0),
        )
    else:
        net = GraphTrajSimEncoder(
            feature_size=config.feature_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            concat=config.concat,
            device=device,
            usePE=config.gtraj["usePE"],
            useSI=config.gtraj["useSI"],
            useLSTM=config.gtraj["useLSTM"],
            dataset=config.dataset,
            alpha1=config.alpha1,
            alpha2=config.alpha2,
            seq_encoder=config.gtraj.get("seqEncoder", "transformer"),
            mamba_layers=config.gtraj.get("mambaLayers", 2),
            mamba_d_state=config.gtraj.get("mambaDState", 16),
            mamba_d_conv=config.gtraj.get("mambaDConv", 4),
            mamba_expand=config.gtraj.get("mambaExpand", 2),
            coord_encoder=config.gtraj.get("coordEncoder", "lstm"),
        )
    return net.to(device), device


def load_network_input(config, device):
    if config.gtraj.get("useHybridGeo", False):
        return move_hierarchy_data_to_device(
            spatial_data_utils.load_hierarchy_multilevel(config.dataset),
            device,
        )
    if config.gtraj["useSI"]:
        road_network = spatial_data_utils.load_neighbor(config.dataset, config.num_knn)
        road_network = [item.to(device) for item in road_network]
    else:
        road_network = spatial_data_utils.load_netowrk(config.dataset).to(device)
    distance_to_anchor_node = spatial_data_utils.load_region_node(config.dataset).to(device)
    return [road_network, distance_to_anchor_node]


def embedding_dim(config):
    if config.gtraj["useLSTM"]:
        return config.hidden_size * 2
    return config.hidden_size


def load_eval_npz(eval_file):
    data = np.load(eval_file, allow_pickle=True)
    metric_distances = None
    for key in ("metric_distances", "vali_dis", "label_distances"):
        if key in data:
            metric_distances = np.asarray(data[key], dtype=np.float32)
            break
    return data["vali_list"], data["vali_coor"], data["vali_y"], metric_distances


def coordinate_normalization_stats(coordinates):
    count = 0
    total = np.zeros(2, dtype=np.float64)
    total_square = np.zeros(2, dtype=np.float64)
    for coordinate in coordinates:
        coordinate = np.asarray(coordinate, dtype=np.float64)
        if coordinate.ndim != 2 or coordinate.shape[1] != 2:
            raise ValueError(f"Expected coordinates with shape [length, 2], got {coordinate.shape}.")
        count += len(coordinate)
        total += coordinate.sum(axis=0)
        total_square += np.square(coordinate).sum(axis=0)
    if count == 0:
        raise ValueError("Cannot normalize an empty coordinate collection.")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-12)
    return np.asarray([mean[0], mean[1], std[0], std[1]], dtype=np.float64)


def normalize_coordinates(coordinates, stats):
    stats = np.asarray(stats, dtype=np.float64).reshape(-1)
    if stats.shape != (4,):
        raise ValueError(f"Coordinate statistics must contain [mean_x, mean_y, std_x, std_y], got {stats.shape}.")
    mean = stats[:2]
    std = np.maximum(stats[2:], 1e-12)
    normalized = [
        ((np.asarray(item, dtype=np.float64) - mean) / std).astype(np.float32)
        for item in coordinates
    ]
    output = np.empty(len(normalized), dtype=object)
    output[:] = normalized
    return output


def _object_sequence_array(items, dtype):
    arrays = [np.asarray(item, dtype=dtype) for item in items]
    output = np.empty(len(arrays), dtype=object)
    output[:] = arrays
    return output


class _TrajectoryArrayDataset(Dataset):
    def __init__(self, trajectories, coordinates, coordinate_stats):
        self.trajectories = trajectories
        self.coordinates = coordinates
        coordinate_stats = np.asarray(coordinate_stats, dtype=np.float64)
        self.coordinate_mean = coordinate_stats[:2]
        self.coordinate_std = np.maximum(coordinate_stats[2:], 1e-12)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        coordinate = (
            (np.asarray(self.coordinates[idx], dtype=np.float64) - self.coordinate_mean)
            / self.coordinate_std
        ).astype(np.float32)
        return self.trajectories[idx], coordinate, idx


def _collate_trajectory_arrays(items):
    trajectories = [torch.as_tensor(item[0], dtype=torch.long) for item in items]
    coordinates = [torch.as_tensor(item[1], dtype=torch.float32) for item in items]
    lengths = torch.as_tensor([len(item) for item in trajectories], dtype=torch.long)
    indices = np.asarray([item[2] for item in items], dtype=np.int64)
    trajectories = pad_sequence(trajectories, batch_first=True, padding_value=0)
    coordinates = pad_sequence(coordinates, batch_first=True, padding_value=0)
    return trajectories, coordinates, lengths, indices


def export_trajectory_embeddings(
    config,
    load_model,
    trajectories,
    coordinates,
    output_file=None,
    batch_size=None,
    coordinate_stats=None,
    metadata=None,
    include_coordinates=True,
    include_scale_outputs=True,
):
    """Encode independent trajectories with gallery-compatible coordinate statistics."""
    trajectories = _object_sequence_array(trajectories, np.int64)
    coordinates = _object_sequence_array(coordinates, np.float32)
    if len(trajectories) != len(coordinates):
        raise ValueError(
            f"Trajectory count ({len(trajectories)}) does not match coordinate count ({len(coordinates)})."
        )
    for idx, (trajectory, coordinate) in enumerate(zip(trajectories, coordinates)):
        if len(trajectory) != len(coordinate):
            raise ValueError(
                f"Trajectory {idx} has {len(trajectory)} nodes but {len(coordinate)} coordinates."
            )

    coordinate_stats = (
        coordinate_normalization_stats(coordinates)
        if coordinate_stats is None
        else np.asarray(coordinate_stats, dtype=np.float64)
    )
    net, device = build_spatial_encoder(config)
    state = torch.load(load_model, map_location=device)
    net.load_state_dict(state)
    if config.gtraj.get("usePretrainedFrozenGraph", False):
        if not config.gtraj.get("useHybridGeo", False):
            raise ValueError("usePretrainedFrozenGraph currently requires useHybridGeo=True.")
        net.freeze_graph_encoder(cache=config.gtraj.get("cacheFrozenGraph", True))
    net.eval()
    network_input = load_network_input(config, device)
    if config.gtraj.get("usePretrainedFrozenGraph", False) and config.gtraj.get("cacheFrozenGraph", True):
        net.prepare_frozen_graph(network_input)

    batch_size = batch_size or config.gtraj["test_batch"]
    data_loader = DataLoader(
        _TrajectoryArrayDataset(trajectories, coordinates, coordinate_stats),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_trajectory_arrays,
    )
    embeddings = torch.zeros((len(trajectories), embedding_dim(config)), dtype=torch.float32)
    uses_hmt = config.gtraj.get("useHybridGeo", False) and config.gtraj.get("seqEncoder") == "hmt"
    capture_scale_outputs = uses_hmt and include_scale_outputs
    scale_weights = (
        torch.zeros((len(trajectories), 4), dtype=torch.float32)
        if capture_scale_outputs else None
    )
    segment_lengths = (
        torch.zeros((len(trajectories), 4), dtype=torch.long)
        if capture_scale_outputs else None
    )
    scale_embeddings = (
        torch.zeros((len(trajectories), 4, config.embedding_size), dtype=torch.float32)
        if capture_scale_outputs else None
    )

    with torch.no_grad():
        for data, coor, data_length, indices in tqdm(data_loader, desc="encode perturbation queries"):
            out = net(network_input, data, coor, data_length)
            embeddings[indices] = out.detach().to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            if capture_scale_outputs:
                scale_embeddings[indices] = net.last_scale_embeddings.detach().cpu()
                scale_weights[indices] = net.last_scale_weights.detach().cpu()
                segment_lengths[indices] = net.last_segment_lengths.detach().cpu()

    payload = {
        "embeddings": embeddings.numpy().astype(np.float32),
        "trajectories": trajectories,
        "coordinate_stats": coordinate_stats,
    }
    if include_coordinates:
        payload["coordinates"] = coordinates
    if metadata:
        payload.update(metadata)
    if capture_scale_outputs:
        payload["scale_embeddings"] = scale_embeddings.numpy()
        payload["scale_weights"] = scale_weights.numpy()
        payload["segment_lengths"] = segment_lengths.numpy()

    if output_file is not None:
        out_dir = osp.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(output_file, **payload)
    return payload


def export_spatial_embeddings(config, load_model, eval_file, output_file=None, batch_size=None):
    net, device = build_spatial_encoder(config)
    state = torch.load(load_model, map_location=device)
    net.load_state_dict(state)
    if config.gtraj.get("usePretrainedFrozenGraph", False):
        if not config.gtraj.get("useHybridGeo", False):
            raise ValueError("usePretrainedFrozenGraph currently requires useHybridGeo=True.")
        net.freeze_graph_encoder(cache=config.gtraj.get("cacheFrozenGraph", True))
    net.eval()

    network_input = load_network_input(config, device)
    if config.gtraj.get("usePretrainedFrozenGraph", False) and config.gtraj.get("cacheFrozenGraph", True):
        net.prepare_frozen_graph(network_input)
    batch_size = batch_size or config.gtraj["test_batch"]
    data_loader, data_len = spatial_data_utils.vali_data_loader(eval_file, batch_size)

    embeddings = torch.zeros((data_len, embedding_dim(config)), device=device, requires_grad=False)
    labels = torch.zeros((data_len, 50), requires_grad=False, dtype=torch.long)
    uses_hmt = config.gtraj.get("useHybridGeo", False) and config.gtraj.get("seqEncoder") == "hmt"
    scale_weights = torch.zeros((data_len, 4), dtype=torch.float32) if uses_hmt else None
    segment_lengths = torch.zeros((data_len, 4), dtype=torch.long) if uses_hmt else None
    scale_embeddings = (
        torch.zeros((data_len, 4, config.embedding_size), dtype=torch.float32)
        if uses_hmt else None
    )

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="encode trajectories"):
            data, coor, label, data_length, idx = batch
            out = net(network_input, data, coor, data_length)
            embeddings[idx] = out.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            labels[idx] = label
            if uses_hmt:
                scale_embeddings[idx] = net.last_scale_embeddings.detach().cpu()
                scale_weights[idx] = net.last_scale_weights.detach().cpu()
                segment_lengths[idx] = net.last_segment_lengths.detach().cpu()

    trajectories, coordinates, _, metric_distances = load_eval_npz(eval_file)
    payload = {
        "embeddings": embeddings.cpu().numpy().astype(np.float32),
        "labels": labels.cpu().numpy().astype(np.int64),
        "trajectories": trajectories,
        "coordinates": coordinates,
        "coordinate_stats": coordinate_normalization_stats(coordinates),
    }
    if metric_distances is not None:
        if metric_distances.shape != labels.shape:
            raise ValueError(
                "Metric distances must be row-aligned with validation labels; "
                f"got distances={metric_distances.shape}, labels={labels.shape}."
            )
        payload["metric_distances"] = metric_distances
    if uses_hmt:
        payload["scale_embeddings"] = scale_embeddings.numpy()
        payload["scale_weights"] = scale_weights.numpy()
        payload["segment_lengths"] = segment_lengths.numpy()

    if output_file is not None:
        out_dir = osp.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(output_file, **payload)
    return payload
