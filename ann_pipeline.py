import argparse
import json
import os
import os.path as osp
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ann_index import ANNConfig, HierarchicalTrajectoryANN, LocalVectorIndex
from paired_retrieval_experiment import (
    DEFAULT_PERTURBATIONS,
    SUPPORTED_PERTURBATIONS,
    evaluate_paired_queries,
    generate_perturbed_queries,
    load_road_neighbor_data,
    remap_perturbed_source_ids,
)


def new_default_config():
    from setting import SetParameter

    return SetParameter()


def setup_logging():
    try:
        from setting import setup_logger

        setup_logger()
    except Exception:
        import logging

        logging.basicConfig(level=logging.INFO)


def apply_dataset(config, dataset):
    if dataset is None or dataset == config.dataset:
        return config
    config.dataset = dataset
    config.node_file = osp.join("dataset", dataset, "node.csv")
    config.edge_file = osp.join("dataset", dataset, "edge_weight.csv")
    config.traj_file = osp.join("dataset", dataset, "matching_result.pt")
    config.tdrive_traj_file = osp.join("dataset", dataset, "matching_result.csv")
    config.tdrive_time_file = osp.join("dataset", dataset, "time_drop_list.csv")
    config.set_dataset_mode(config.dataset_mode)
    config.spatial_long_test_set = osp.join("ground_truth", dataset, config.distance_type, "spatial_long_test_set.npz")
    config.spatial_num_test_set = osp.join("ground_truth", dataset, config.distance_type, "spatial_num_test_set.npz")
    config.hierarchy_dir = osp.join("dataset", dataset, "hierarchy_louvain")
    return config


def refresh_distance_paths(config):
    config.set_dataset_mode(config.dataset_mode)
    config.spatial_long_test_set = osp.join("ground_truth", config.dataset, config.distance_type, "spatial_long_test_set.npz")
    config.spatial_num_test_set = osp.join("ground_truth", config.dataset, config.distance_type, "spatial_num_test_set.npz")


def apply_model_overrides(config, args):
    if getattr(args, "cuda", None) is not None:
        config.cuda = str(args.cuda)
    if getattr(args, "test_batch", None) is not None:
        config.gtraj["test_batch"] = args.test_batch
    if getattr(args, "use_hybrid_geo", None) is not None:
        config.gtraj["useHybridGeo"] = args.use_hybrid_geo
    if getattr(args, "use_si", None) is not None:
        config.gtraj["useSI"] = args.use_si
    if getattr(args, "graph_encoder", None) is not None:
        config.gtraj["graphEncoder"] = args.graph_encoder
        if args.graph_encoder == "simple_gcn":
            config.gtraj["hmtScaleAlignedGraph"] = False
            config.gtraj["hierarchyLossWeight"] = 0.0
    if getattr(args, "seq_encoder", None) is not None:
        config.gtraj["seqEncoder"] = args.seq_encoder
    if getattr(args, "coord_encoder", None) is not None:
        config.gtraj["coordEncoder"] = args.coord_encoder
    if getattr(args, "mamba_layers", None) is not None:
        config.gtraj["mambaLayers"] = args.mamba_layers
    if getattr(args, "mamba_d_state", None) is not None:
        config.gtraj["mambaDState"] = args.mamba_d_state
    if getattr(args, "mamba_d_conv", None) is not None:
        config.gtraj["mambaDConv"] = args.mamba_d_conv
    if getattr(args, "mamba_expand", None) is not None:
        config.gtraj["mambaExpand"] = args.mamba_expand
    if getattr(args, "hmt_backbone", None) is not None:
        config.gtraj["hmtBackbone"] = args.hmt_backbone
    if getattr(args, "hmt_layers", None) is not None:
        config.gtraj["hmtLayers"] = args.hmt_layers
    if getattr(args, "hmt_kernel_size", None) is not None:
        config.gtraj["hmtKernelSize"] = args.hmt_kernel_size
    if getattr(args, "hmt_dropout", None) is not None:
        config.gtraj["hmtDropout"] = args.hmt_dropout
    if getattr(args, "hmt_transformer_heads", None) is not None:
        config.gtraj["hmtTransformerHeads"] = args.hmt_transformer_heads
    if getattr(args, "hmt_scale_aligned_graph", None) is not None:
        config.gtraj["hmtScaleAlignedGraph"] = args.hmt_scale_aligned_graph
    if getattr(args, "coord_fusion", None) is not None:
        config.gtraj["coordFusion"] = args.coord_fusion
    return config


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        checkpoints = sorted(
            path.glob("epoch_*.pt"),
            key=lambda p: int(re.search(r"epoch_(\d+)\.pt$", p.name).group(1))
            if re.search(r"epoch_(\d+)\.pt$", p.name)
            else -1,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No epoch_*.pt checkpoint found in {path}")
        path = checkpoints[-1]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)


def load_embedding_payload(path):
    data = np.load(path, allow_pickle=True)
    payload = {
        "embeddings": data["embeddings"].astype(np.float32),
        "labels": data["labels"].astype(np.int64) if "labels" in data else None,
        "trajectories": data["trajectories"],
        "coordinates": data["coordinates"] if "coordinates" in data else None,
        "source_ids": data["source_ids"].astype(np.int64) if "source_ids" in data else None,
        "scale_embeddings": (
            data["scale_embeddings"].astype(np.float32)
            if "scale_embeddings" in data else None
        ),
    }
    if "coordinate_stats" in data:
        payload["coordinate_stats"] = data["coordinate_stats"].astype(np.float64)
    if "scale_weights" in data:
        payload["scale_weights"] = data["scale_weights"].astype(np.float32)
    if "segment_lengths" in data:
        payload["segment_lengths"] = data["segment_lengths"].astype(np.int64)
    return payload


def load_paired_query_payload(path):
    data = np.load(path, allow_pickle=True)
    required = {
        "embeddings",
        "trajectories",
        "source_ids",
        "perturbation_types",
        "severities",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"Paired query file {path} is missing arrays: {missing}.")
    payload = {
        "embeddings": data["embeddings"].astype(np.float32),
        "trajectories": data["trajectories"],
        "source_ids": data["source_ids"].astype(np.int64),
        "perturbation_types": data["perturbation_types"].astype(str),
        "severities": data["severities"].astype(np.float32),
        "gallery_size": int(data["gallery_size"].item()) if "gallery_size" in data else None,
    }
    if "variant_ids" in data:
        payload["variant_ids"] = data["variant_ids"].astype(np.int64)
    return payload


def slice_embedding_payload(payload, max_vectors):
    if not max_vectors or max_vectors <= 0:
        return payload
    max_vectors = min(int(max_vectors), len(payload["embeddings"]))
    row_aligned_keys = {
        "embeddings",
        "labels",
        "trajectories",
        "coordinates",
        "scale_embeddings",
        "scale_weights",
        "segment_lengths",
        "source_ids",
    }
    sliced = {}
    for key, value in payload.items():
        if key in row_aligned_keys and value is not None:
            sliced[key] = value[:max_vectors]
        else:
            sliced[key] = value
    return sliced


def compute_retrieval_metrics(predictions, labels):
    metrics = np.zeros((len(labels), 6), dtype=np.float32)
    for idx, truth in enumerate(labels):
        pred = predictions[idx]
        truth = np.asarray(truth, dtype=np.int64)
        pred10 = set(pred[:10])
        pred50 = set(pred[:50])
        true10 = set(truth[:10])
        true50 = set(truth[:50])
        true1 = set(truth[:1])
        metrics[idx, 0] = len(pred10 & true10) / 10.0
        metrics[idx, 1] = len(pred50 & true50) / 50.0
        metrics[idx, 2] = len(pred50 & true10) / 10.0
        metrics[idx, 3] = len(set(pred[:1]) & true1)
        metrics[idx, 4] = len(pred10 & true1)
        metrics[idx, 5] = len(pred50 & true1)
    return metrics.mean(axis=0)


def evaluate_ann(ann, embeddings, trajectories, labels, top_k=50, max_queries=None):
    total = len(embeddings) if max_queries is None or max_queries <= 0 else min(max_queries, len(embeddings))
    predictions = []
    start = perf_counter()
    for idx in range(total):
        ids, _ = ann.query(
            embeddings[idx],
            trajectories[idx],
            top_k=top_k,
            exclude_ids={idx},
        )
        predictions.append(ids)
    query_seconds = perf_counter() - start
    metrics = compute_retrieval_metrics(predictions, labels[:total])
    return {
        "metrics": metrics,
        "avg_query_ms": query_seconds / max(total, 1) * 1000.0,
        "queries": total,
    }


def make_ann_config(args):
    return ANNConfig(
        metric=args.metric,
        backend=args.backend,
        normalize=args.normalize,
        pattern_level=args.pattern_level,
        path_max_len=args.path_max_len,
        l3_top_r=args.l3_top_r,
        l3_expand_hops=args.l3_expand_hops,
        path_top_p=args.path_top_p,
        l1_top_buckets=args.l1_top_buckets,
        candidates_per_bucket=args.candidates_per_bucket,
        global_fallback=args.global_fallback,
        exact_bucket_threshold=args.exact_bucket_threshold,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )


def build_command(args):
    config = apply_dataset(new_default_config(), args.dataset)
    if args.distance_type is not None:
        config.distance_type = args.distance_type
        refresh_distance_paths(config)
    if args.dataset_mode is not None:
        config.set_dataset_mode(args.dataset_mode)
    apply_model_overrides(config, args)

    eval_file = args.eval_file or config.spatial_test_set
    embedding_file = args.embedding_file or osp.join(
        "ann_artifacts",
        config.dataset,
        config.distance_type,
        "test_embeddings.npz",
    )
    if args.model is not None:
        try:
            from trajectory_embedding import export_spatial_embeddings, export_trajectory_embeddings
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Cannot import trajectory_embedding.py. Make sure trajectory_embedding.py is in the same "
                f"directory as ann_pipeline.py: {SCRIPT_DIR}"
            ) from exc

        model_path = resolve_checkpoint(args.model)
        uses_raw_arrays = args.trajectory_array is not None or args.coordinate_array is not None
        if uses_raw_arrays:
            if args.trajectory_array is None or args.coordinate_array is None:
                raise ValueError("--trajectory-array and --coordinate-array must be provided together.")
            trajectories = np.load(args.trajectory_array, allow_pickle=True)
            coordinates = np.load(args.coordinate_array, allow_pickle=True)
            if args.max_vectors and args.max_vectors > 0:
                trajectories = trajectories[: args.max_vectors]
                coordinates = coordinates[: args.max_vectors]
            coordinate_stats = None
            if args.coordinate_stats_file:
                coordinate_stats = np.asarray(
                    np.load(args.coordinate_stats_file, allow_pickle=False),
                    dtype=np.float64,
                ).reshape(-1)
                if coordinate_stats.shape != (4,):
                    raise ValueError(
                        "--coordinate-stats-file must contain "
                        "[mean_x, mean_y, std_x, std_y]."
                    )
            payload = export_trajectory_embeddings(
                config=config,
                load_model=model_path,
                trajectories=trajectories,
                coordinates=coordinates,
                output_file=embedding_file,
                batch_size=args.batch_size,
                coordinate_stats=coordinate_stats,
                include_coordinates=not args.compact_output,
                include_scale_outputs=not args.compact_output,
            )
        else:
            payload = export_spatial_embeddings(
                config=config,
                load_model=model_path,
                eval_file=eval_file,
                output_file=embedding_file,
                batch_size=args.batch_size,
            )
    else:
        payload = load_embedding_payload(embedding_file)
    payload.setdefault("labels", None)
    payload = slice_embedding_payload(payload, args.max_vectors)

    if args.export_only:
        print(f"Trajectory embeddings exported to: {embedding_file}")
        return

    index_dir = args.index_dir or osp.join("ann_artifacts", config.dataset, config.distance_type, "hier_ann")
    hierarchy_dir = args.hierarchy_dir or config.hierarchy_dir
    ann = HierarchicalTrajectoryANN(make_ann_config(args))
    ann.build(payload["embeddings"], payload["trajectories"], hierarchy_dir)
    ann.save(index_dir)
    print(f"ANN index saved to: {index_dir}")
    print(
        f"vectors={len(payload['embeddings'])}, pattern_level={ann.config.pattern_level}, "
        f"patterns={len(ann.pattern_to_ids)}, build_seconds={ann.build_seconds:.3f}"
    )

    if args.evaluate:
        if payload["labels"] is None:
            raise ValueError("The original ANN evaluation requires a labels array in the embedding file.")
        result = evaluate_ann(
            ann,
            payload["embeddings"],
            payload["trajectories"],
            payload["labels"],
            top_k=args.top_k,
            max_queries=args.max_queries,
        )
        print_metrics(result)


def eval_command(args):
    payload = load_embedding_payload(args.embedding_file)
    if payload["labels"] is None:
        raise ValueError("The original ANN evaluation requires a labels array in the embedding file.")
    ann = HierarchicalTrajectoryANN.load(args.index_dir)
    result = evaluate_ann(
        ann,
        payload["embeddings"],
        payload["trajectories"],
        payload["labels"],
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    print_metrics(result)


def _configure_model(args):
    config = apply_dataset(new_default_config(), args.dataset)
    if args.distance_type is not None:
        config.distance_type = args.distance_type
        refresh_distance_paths(config)
    if args.dataset_mode is not None:
        config.set_dataset_mode(args.dataset_mode)
    return apply_model_overrides(config, args)


def perturb_build_command(args):
    config = _configure_model(args)
    gallery = load_embedding_payload(args.gallery_embedding_file)
    if gallery["coordinates"] is None:
        raise ValueError(
            "The gallery embedding file must contain raw coordinates to create aligned perturbations."
        )

    perturbations = tuple(item.strip() for item in args.perturbations.split(",") if item.strip())
    severities = tuple(float(item.strip()) for item in args.severities.split(",") if item.strip())
    variants_per_setting = int(getattr(args, "variants_per_setting", 1))
    compact_output = bool(getattr(args, "compact_output", False))
    neighbor_adjacency = None
    node_coordinates = None
    if "neighbor_replace" in perturbations:
        neighbor_adjacency, node_coordinates = load_road_neighbor_data(
            getattr(args, "edge_file", None) or config.edge_file,
            getattr(args, "node_file", None) or config.node_file,
        )
    generated = generate_perturbed_queries(
        gallery["trajectories"],
        gallery["coordinates"],
        source_count=args.source_count,
        perturbations=perturbations,
        severities=severities,
        variants_per_setting=variants_per_setting,
        seed=args.seed,
        min_length=args.min_length,
        gallery_size=args.gallery_size,
        neighbor_adjacency=neighbor_adjacency,
        node_coordinates=node_coordinates,
        neighbor_connectivity=getattr(args, "neighbor_connectivity", "prefer"),
    )
    generated = remap_perturbed_source_ids(generated, gallery.get("source_ids"))

    try:
        from trajectory_embedding import coordinate_normalization_stats, export_trajectory_embeddings
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import trajectory_embedding.py. Make sure it is in the same directory as ann_pipeline.py."
        ) from exc

    coordinate_stats = gallery.get("coordinate_stats")
    if coordinate_stats is None:
        coordinate_stats = coordinate_normalization_stats(gallery["coordinates"])
    metadata = {
        key: value
        for key, value in generated.items()
        if key not in {"trajectories", "coordinates"}
    }
    payload = export_trajectory_embeddings(
        config=config,
        load_model=resolve_checkpoint(args.model),
        trajectories=generated["trajectories"],
        coordinates=generated["coordinates"],
        output_file=args.query_file,
        batch_size=args.batch_size,
        coordinate_stats=coordinate_stats,
        metadata=metadata,
        include_coordinates=not compact_output,
        include_scale_outputs=not compact_output,
    )
    print(f"Perturbation query file saved to: {args.query_file}")
    print(
        f"gallery_size={int(generated['gallery_size'])}, "
        f"sources={len(np.unique(generated['source_ids']))}, queries={len(payload['embeddings'])}, "
        f"perturbations={','.join(perturbations)}, severities={','.join(map(str, severities))}, "
        f"variants_per_setting={variants_per_setting}, compact={compact_output}"
    )
    if "neighbor_replace" in perturbations:
        requested = generated["requested_replacement_counts"]
        actual = generated["actual_replacement_counts"]
        valid = generated["path_valid_ratios"]
        effective = generated["effective_severities"]
        print(
            f"neighbor replacements: requested_mean={requested.mean():.2f}, "
            f"actual_mean={actual.mean():.2f}, "
            f"effective_severity_mean={effective.mean():.4f}, "
            f"path_valid_mean={np.nanmean(valid):.4f}, "
            f"connectivity={getattr(args, 'neighbor_connectivity', 'prefer')}"
        )


def _prepare_query_embedding(ann, embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    if ann.config.normalize or ann.config.metric == "cosine":
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
    return embedding


def _index_disk_size(index_dir):
    total = 0
    for root, _, files in os.walk(index_dir):
        for filename in files:
            total += osp.getsize(osp.join(root, filename))
    return total


def _print_paired_result(result):
    metric_keys = sorted(
        (key for key in result if key.startswith("id_recall_at_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    metrics = [f"{key.upper()}={result[key]:.4f}" for key in metric_keys]
    mrr_key = next(key for key in result if key.startswith("mrr_at_"))
    metrics.append(f"{mrr_key.upper()}={result[mrr_key]:.4f}")
    print(
        f"method={result['method']}, backend={result['backend']}, queries={result['queries']}, "
        f"avg_ms={result['avg_query_ms']:.3f}, p95_ms={result['p95_query_ms']:.3f}, "
        f"qps={result['qps']:.2f}"
    )
    print(", ".join(metrics))
    for group in result["groups"]:
        group_metrics = ", ".join(f"{key.upper()}={group[key]:.4f}" for key in metric_keys)
        print(
            f"  type={group['perturbation_type']}, severity={group['severity']:.2f}, "
            f"queries={group['queries']}, {group_metrics}"
        )


def perturb_eval_command(args):
    queries = load_paired_query_payload(args.query_file)
    ann = HierarchicalTrajectoryANN.load(args.index_dir)
    if queries["embeddings"].shape[1] != ann.embeddings.shape[1]:
        raise ValueError(
            f"Query dimension {queries['embeddings'].shape[1]} does not match index dimension "
            f"{ann.embeddings.shape[1]}."
        )
    if len(queries["source_ids"]) and int(queries["source_ids"].max()) >= len(ann.embeddings):
        raise ValueError(
            f"Query source ID {int(queries['source_ids'].max())} is outside the "
            f"{len(ann.embeddings)}-trajectory index. Rebuild queries with the matching gallery size."
        )

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods).difference({"hierarchical", "global", "exact"}))
    if unknown:
        raise ValueError(f"Unknown paired retrieval methods: {unknown}.")

    results = []
    for method in methods:
        method_max_queries = args.max_queries
        if method == "hierarchical":
            query_function = lambda embedding, trajectory, top_k: ann.query(
                embedding, trajectory, top_k=top_k, exclude_ids=None
            )[0]
            local_backends = sorted({index.backend for index in ann.bucket_indexes.values()})
            backend = "hierarchical-" + "+".join(local_backends)
        elif method == "global":
            query_function = lambda embedding, trajectory, top_k: ann.global_index.query(
                _prepare_query_embedding(ann, embedding), top_k
            )[0]
            backend = ann.global_index.backend
        else:
            exact_index = LocalVectorIndex(
                np.arange(len(ann.embeddings), dtype=np.int64),
                ann.embeddings,
                ANNConfig(metric=ann.config.metric, backend="exact"),
            )
            query_function = lambda embedding, trajectory, top_k: exact_index.query(
                _prepare_query_embedding(ann, embedding), top_k
            )[0]
            backend = "exact"
            if args.exact_max_queries > 0:
                method_max_queries = (
                    args.exact_max_queries
                    if not method_max_queries or method_max_queries <= 0
                    else min(method_max_queries, args.exact_max_queries)
                )

        result = evaluate_paired_queries(
            query_function=query_function,
            embeddings=queries["embeddings"],
            trajectories=queries["trajectories"],
            source_ids=queries["source_ids"],
            perturbation_types=queries["perturbation_types"],
            severities=queries["severities"],
            top_k=args.top_k,
            max_queries=method_max_queries,
        )
        result.update(
            {
                "method": method,
                "backend": backend,
                "gallery_size": int(len(ann.embeddings)),
                "index_disk_mb": _index_disk_size(args.index_dir) / (1024.0 ** 2),
                "index_load_build_seconds": float(ann.build_seconds),
            }
        )
        results.append(result)
        _print_paired_result(result)

    output = {
        "protocol": "paired_trajectory_identification",
        "query_file": str(args.query_file),
        "index_dir": str(args.index_dir),
        "results": results,
    }
    if args.result_file:
        out_dir = osp.dirname(args.result_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Paired retrieval results saved to: {args.result_file}")


def print_metrics(result):
    names = ["HR@10", "HR@50", "R10@50", "R1@1", "R1@10", "R1@50"]
    metric_text = ", ".join(f"{name}={value:.4f}" for name, value in zip(names, result["metrics"]))
    print(f"queries={result['queries']}, avg_query_ms={result['avg_query_ms']:.3f}")
    print(metric_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Build and evaluate hierarchical ANN for trajectory embeddings.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, require_artifacts=False):
        p.add_argument("--embedding-file", required=require_artifacts, default=None)
        p.add_argument("--index-dir", required=require_artifacts, default=None)
        p.add_argument("--top-k", type=int, default=50)
        p.add_argument("--max-queries", type=int, default=0)
        p.add_argument("--backend", default="auto", choices=["auto", "hnsw", "exact"])
        p.add_argument("--metric", default="l2", choices=["l2", "cosine"])
        p.add_argument("--normalize", action="store_true")
        p.add_argument("--pattern-level", default="l3", choices=["l2", "l3"])
        p.add_argument("--path-max-len", type=int, default=3)
        p.add_argument("--l3-top-r", type=int, default=4)
        p.add_argument("--l3-expand-hops", type=int, default=1)
        p.add_argument("--path-top-p", type=int, default=16)
        p.add_argument("--l1-top-buckets", type=int, default=4)
        p.add_argument("--candidates-per-bucket", type=int, default=128)
        p.add_argument("--global-fallback", type=int, default=1000)
        p.add_argument("--exact-bucket-threshold", type=int, default=256)
        p.add_argument("--hnsw-m", type=int, default=16)
        p.add_argument("--hnsw-ef-construction", type=int, default=200)
        p.add_argument("--hnsw-ef-search", type=int, default=80)

    build = sub.add_parser("build")
    add_common(build)
    build.add_argument("--dataset", default=None, choices=["beijing", "porto", "tdrive"])
    build.add_argument("--dataset-mode", default=None, choices=["standard", "50w", "full"])
    build.add_argument("--distance-type", default=None, choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"])
    build.add_argument("--model", default=None, help="Checkpoint path. If omitted, --embedding-file must already exist.")
    build.add_argument("--eval-file", default=None)
    build.add_argument(
        "--trajectory-array",
        default=None,
        help="Raw node-sequence .npy input for gallery export without similarity ground truth.",
    )
    build.add_argument(
        "--coordinate-array",
        default=None,
        help="Raw coordinate-sequence .npy paired with --trajectory-array.",
    )
    build.add_argument("--hierarchy-dir", default=None)
    build.add_argument("--batch-size", type=int, default=None)
    build.add_argument("--test-batch", type=int, default=None)
    build.add_argument(
        "--compact-output",
        action="store_true",
        help=(
            "For raw-array export, omit coordinates and HMT scale outputs. "
            "Recommended for million-row Gallery embedding files."
        ),
    )
    build.add_argument(
        "--export-only",
        action="store_true",
        help="Export embeddings and stop without building the auxiliary ANN index.",
    )
    build.add_argument(
        "--coordinate-stats-file",
        default=None,
        help=(
            "Optional .npy file containing Gallery coordinate statistics "
            "[mean_x, mean_y, std_x, std_y] to reuse for Query export."
        ),
    )
    build.add_argument("--cuda", type=str, default=None)
    build.add_argument("--seq-encoder", choices=["transformer", "mamba", "bimamba", "hmt"], default=None)
    build.add_argument(
        "--graph-encoder",
        choices=["hybrid_multilevel", "simple_gcn"],
        default=None,
    )
    build.add_argument("--coord-encoder", choices=["lstm", "mamba", "transformer"], default=None)
    build.add_argument("--mamba-layers", type=int, default=None)
    build.add_argument("--mamba-d-state", type=int, default=None)
    build.add_argument("--mamba-d-conv", type=int, default=None)
    build.add_argument("--mamba-expand", type=int, default=None)
    build.add_argument(
        "--hmt-backbone",
        choices=["conv", "mamba", "bimamba", "transformer", "hybrid"],
        default=None,
    )
    build.add_argument("--hmt-layers", type=int, default=None)
    build.add_argument("--hmt-kernel-size", type=int, default=None)
    build.add_argument("--hmt-dropout", type=float, default=None)
    build.add_argument("--hmt-transformer-heads", type=int, default=None)
    build.add_argument(
        "--hmt-scale-aligned-graph",
        dest="hmt_scale_aligned_graph",
        action="store_true",
        default=None,
    )
    build.add_argument(
        "--no-hmt-scale-aligned-graph",
        dest="hmt_scale_aligned_graph",
        action="store_false",
    )
    build.add_argument("--coord-fusion", choices=["gated_attention", "late_concat"], default=None)
    build.add_argument("--use-hybrid-geo", dest="use_hybrid_geo", action="store_true", default=None)
    build.add_argument("--no-use-hybrid-geo", dest="use_hybrid_geo", action="store_false")
    build.add_argument("--use-si", dest="use_si", action="store_true", default=None)
    build.add_argument("--no-use-si", dest="use_si", action="store_false")
    build.add_argument(
        "--max-vectors",
        type=int,
        default=0,
        help="Optionally build a prefix index for 50k/100k/200k/500k scale experiments.",
    )
    build.add_argument("--evaluate", action="store_true")

    evaluate = sub.add_parser("eval")
    add_common(evaluate, require_artifacts=True)

    perturb_build = sub.add_parser(
        "perturb-build",
        help="Generate and encode paired perturbation queries without trajectory-distance ground truth.",
    )
    perturb_build.add_argument("--gallery-embedding-file", required=True)
    perturb_build.add_argument("--query-file", required=True)
    perturb_build.add_argument("--model", required=True)
    perturb_build.add_argument("--dataset", default=None, choices=["beijing", "porto", "tdrive"])
    perturb_build.add_argument("--dataset-mode", default=None, choices=["standard", "50w", "full"])
    perturb_build.add_argument(
        "--distance-type",
        default=None,
        choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
    )
    perturb_build.add_argument("--gallery-size", type=int, default=0)
    perturb_build.add_argument("--source-count", type=int, default=5000)
    perturb_build.add_argument(
        "--perturbations",
        default=",".join(DEFAULT_PERTURBATIONS),
        help="Comma-separated perturbations: drop,crop,local_gap,neighbor_replace.",
    )
    perturb_build.add_argument("--severities", default="0.1,0.2,0.3")
    perturb_build.add_argument("--edge-file", default=None)
    perturb_build.add_argument("--node-file", default=None)
    perturb_build.add_argument(
        "--neighbor-connectivity",
        choices=["prefer", "strict"],
        default="prefer",
        help=(
            "prefer chooses the first-order neighbor best connected to the local "
            "context; strict only replaces nodes when both adjacent edges remain valid."
        ),
    )
    perturb_build.add_argument(
        "--variants-per-setting",
        type=int,
        default=1,
        help=(
            "Independent random replicas for every perturbation/severity pair. "
            "Total variants equal types x severities x replicas; use 4 for the "
            "default 3x3 grid or 12 for neighbor_replace-only with 3 severities "
            "to generate 36 variants per source."
        ),
    )
    perturb_build.add_argument(
        "--compact-output",
        action="store_true",
        help=(
            "Omit coordinates and HMT scale outputs from the encoded file. This is "
            "recommended for 500k/1M Gallery generation and keeps all arrays needed "
            "by hier_hash_large_scale.py and build_initial_only_gallery.py."
        ),
    )
    perturb_build.add_argument("--min-length", type=int, default=4)
    perturb_build.add_argument("--seed", type=int, default=2026)
    perturb_build.add_argument("--batch-size", type=int, default=None)
    perturb_build.add_argument("--test-batch", type=int, default=None)
    perturb_build.add_argument("--cuda", type=str, default=None)
    perturb_build.add_argument("--seq-encoder", choices=["transformer", "mamba", "bimamba", "hmt"], default=None)
    perturb_build.add_argument(
        "--graph-encoder",
        choices=["hybrid_multilevel", "simple_gcn"],
        default=None,
    )
    perturb_build.add_argument("--coord-encoder", choices=["lstm", "mamba", "transformer"], default=None)
    perturb_build.add_argument("--mamba-layers", type=int, default=None)
    perturb_build.add_argument("--mamba-d-state", type=int, default=None)
    perturb_build.add_argument("--mamba-d-conv", type=int, default=None)
    perturb_build.add_argument("--mamba-expand", type=int, default=None)
    perturb_build.add_argument(
        "--hmt-backbone",
        choices=["conv", "mamba", "bimamba", "transformer", "hybrid"],
        default=None,
    )
    perturb_build.add_argument("--hmt-layers", type=int, default=None)
    perturb_build.add_argument("--hmt-kernel-size", type=int, default=None)
    perturb_build.add_argument("--hmt-dropout", type=float, default=None)
    perturb_build.add_argument("--hmt-transformer-heads", type=int, default=None)
    perturb_build.add_argument(
        "--hmt-scale-aligned-graph",
        dest="hmt_scale_aligned_graph",
        action="store_true",
        default=None,
    )
    perturb_build.add_argument(
        "--no-hmt-scale-aligned-graph",
        dest="hmt_scale_aligned_graph",
        action="store_false",
    )
    perturb_build.add_argument("--coord-fusion", choices=["gated_attention", "late_concat"], default=None)
    perturb_build.add_argument("--use-hybrid-geo", dest="use_hybrid_geo", action="store_true", default=None)
    perturb_build.add_argument("--no-use-hybrid-geo", dest="use_hybrid_geo", action="store_false")
    perturb_build.add_argument("--use-si", dest="use_si", action="store_true", default=None)
    perturb_build.add_argument("--no-use-si", dest="use_si", action="store_false")

    perturb_eval = sub.add_parser(
        "perturb-eval",
        help="Evaluate paired source-trajectory identification on an existing ANN index.",
    )
    perturb_eval.add_argument("--query-file", required=True)
    perturb_eval.add_argument("--index-dir", required=True)
    perturb_eval.add_argument("--methods", default="hierarchical,global,exact")
    perturb_eval.add_argument("--top-k", type=int, default=50)
    perturb_eval.add_argument("--max-queries", type=int, default=0)
    perturb_eval.add_argument(
        "--exact-max-queries",
        type=int,
        default=200,
        help="Cap exact-vector queries; use 0 to evaluate all queries.",
    )
    perturb_eval.add_argument("--result-file", default=None)
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    if args.command == "build":
        build_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "perturb-build":
        perturb_build_command(args)
    elif args.command == "perturb-eval":
        perturb_eval_command(args)


if __name__ == "__main__":
    main()
