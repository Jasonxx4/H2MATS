"""Build a leakage-controlled large-scale trajectory benchmark.

The benchmark keeps all valid trajectories outside the original training
selection, adds perturbation-based distractors to reach a requested Gallery
size, and creates independent perturbed Queries whose clean source remains in
the Gallery.  It deliberately stores raw trajectories as object ``.npy``
arrays because the existing embedding exporter accepts that format directly.
"""

import argparse
import ast
import csv
import json
import os
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np

from tp_retrieval_evaluation import SUPPORTED_METRICS

from paired_retrieval_experiment import (
    SUPPORTED_PERTURBATIONS,
    _kept_indices,
    _neighbor_replacement,
    load_road_neighbor_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DEFAULTS = {
    "beijing": {
        "trajectory_file": "dataset/beijing/matching_result.pt",
        "train_index_file": "data/beijing/st_traj/shuffle_index_list.npy",
        "edge_file": "dataset/beijing/edge_weight.csv",
        "node_file": "dataset/beijing/node.csv",
        "point_distance_file": "ground_truth/beijing/Point_dis_matrix.npy",
        "node_count": 112557,
        "neterp_anchor_node": 72526,
    },
    "porto": {
        "trajectory_file": "dataset/porto/matching_result.pt",
        "train_index_file": "data/porto/st_traj/shuffle_index_list.npy",
        "edge_file": "dataset/porto/edge_weight.csv",
        "node_file": "dataset/porto/node.csv",
        "point_distance_file": "ground_truth/porto/Point_dis_matrix.npy",
        "node_count": 128466,
        "neterp_anchor_node": None,
    },
    "tdrive": {
        "trajectory_file": "dataset/tdrive/matching_result.csv",
        "train_index_file": "data/tdrive/st_traj/shuffle_index_list.npy",
        "edge_file": "dataset/tdrive/edge_weight.csv",
        "node_file": "dataset/tdrive/node.csv",
        "point_distance_file": "ground_truth/tdrive/Point_dis_matrix.npy",
        "node_count": 74671,
        "neterp_anchor_node": None,
    },
}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _object_array(items, dtype):
    result = np.empty(len(items), dtype=object)
    result[:] = [np.asarray(item, dtype=dtype) for item in items]
    return result


def _load_matching_result(path):
    """Load the pure-Python payload from a torch zip without importing torch."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        records = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Node_list", "Coor_list"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Trajectory CSV {path} must contain {sorted(required)}."
                )
            for row in reader:
                try:
                    records.append([
                        ast.literal_eval(row["Node_list"]),
                        ast.literal_eval(row["Coor_list"]),
                    ])
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Invalid trajectory row in {path}.") from exc
        return records
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if not candidates:
                raise ValueError(f"No data.pkl member found in {path}.")
            with archive.open(candidates[0]) as handle:
                payload = pickle.load(handle)
    except (zipfile.BadZipFile, pickle.PickleError, EOFError, ValueError) as exc:
        raise RuntimeError(
            f"Could not read {path} as the expected torch serialization archive."
        ) from exc
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"Expected a trajectory list in {path}, got {type(payload)!r}.")
    return payload


def _dataset_value(args, key, override=None):
    if override:
        return override
    return DATASET_DEFAULTS[args.dataset][key]


def _resource_value(args, key, override=None):
    dataset = getattr(args, "dataset", "beijing")
    if override:
        return override
    return DATASET_DEFAULTS[dataset][key]


def _load_training_ids(path, valid_raw_ids, train_count, shuffle_seed):
    """Return raw IDs used by the original shuffled training selection."""
    valid_raw_ids = np.asarray(valid_raw_ids, dtype=np.int64)
    path = Path(path) if path else None
    if path is not None and path.exists():
        shuffled = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64).reshape(-1)
        # spatial_preprocess stores positions in the length-filtered array.
        if len(shuffled) and int(shuffled.max()) < len(valid_raw_ids):
            positions = shuffled[: int(train_count)]
            return valid_raw_ids[positions]
        # Be permissive for a manually supplied file containing raw IDs.
        if np.all(np.isin(shuffled[: int(train_count)], valid_raw_ids)):
            return shuffled[: int(train_count)]
        raise ValueError(
            f"Training index file {path} is neither filtered-array positions nor raw IDs."
        )

    order = list(range(len(valid_raw_ids)))
    random.Random(int(shuffle_seed)).shuffle(order)
    return valid_raw_ids[np.asarray(order[: int(train_count)], dtype=np.int64)]


def _validate_trajectory_record(record, raw_id):
    if not isinstance(record, (list, tuple)) or len(record) != 2:
        raise ValueError(f"Trajectory {raw_id} must be [node_sequence, coordinate_sequence].")
    nodes = np.asarray(record[0], dtype=np.int64).reshape(-1)
    coordinates = np.asarray(record[1], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(nodes) != len(coordinates):
        raise ValueError(
            f"Trajectory {raw_id} has incompatible nodes/coordinates: "
            f"{nodes.shape} and {coordinates.shape}."
        )
    return nodes, coordinates


def _make_perturbation(nodes, coordinates, perturbation, severity, rng, context):
    if perturbation == "neighbor_replace":
        if context["adjacency"] is None or context["node_coordinates"] is None:
            raise ValueError(
                "neighbor_replace requires --edge-file and --node-file (or dataset defaults)."
            )
        output_nodes, output_coordinates, details = _neighbor_replacement(
            nodes,
            coordinates,
            severity,
            rng,
            context["adjacency"],
            context["node_coordinates"],
            connectivity=context["neighbor_connectivity"],
        )
        return output_nodes, output_coordinates, {
            "actual_count": int(details["actual_count"]),
            "path_valid_ratio": float(details["path_valid_ratio"]),
        }

    kept = _kept_indices(len(nodes), perturbation, severity, context["min_length"], rng)
    return nodes[kept], coordinates[kept], {
        "actual_count": int(len(nodes) - len(kept)),
        "path_valid_ratio": np.nan,
    }


def _append_metadata(metadata, source_id, perturbation, severity, original_length, output_length, details):
    metadata["source_ids"].append(int(source_id))
    metadata["perturbation_types"].append(str(perturbation))
    metadata["severities"].append(float(severity))
    metadata["original_lengths"].append(int(original_length))
    metadata["trajectory_lengths"].append(int(output_length))
    metadata["actual_replacement_counts"].append(int(details["actual_count"]))
    metadata["path_valid_ratios"].append(float(details["path_valid_ratio"]))


def _empty_metadata():
    return {
        "source_ids": [],
        "perturbation_types": [],
        "severities": [],
        "original_lengths": [],
        "trajectory_lengths": [],
        "actual_replacement_counts": [],
        "path_valid_ratios": [],
    }


def _metadata_arrays(metadata):
    return {
        "source_ids": np.asarray(metadata["source_ids"], dtype=np.int64),
        "perturbation_types": np.asarray(metadata["perturbation_types"]),
        "severities": np.asarray(metadata["severities"], dtype=np.float32),
        "original_lengths": np.asarray(metadata["original_lengths"], dtype=np.int64),
        "trajectory_lengths": np.asarray(metadata["trajectory_lengths"], dtype=np.int64),
        "actual_replacement_counts": np.asarray(
            metadata["actual_replacement_counts"], dtype=np.int64
        ),
        "path_valid_ratios": np.asarray(metadata["path_valid_ratios"], dtype=np.float32),
    }


def _parse_float_list(value):
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise ValueError("Perturbation severities must be comma-separated values in (0, 1).")
    return values


def build_benchmark(args):
    traj_path = _resolve(_dataset_value(args, "trajectory_file", args.trajectory_file))
    records = _load_matching_result(traj_path)
    valid = []
    for raw_id, record in enumerate(records):
        nodes, coordinates = _validate_trajectory_record(record, raw_id)
        if len(nodes) >= int(args.min_source_length):
            valid.append((raw_id, nodes, coordinates))
    if not valid:
        raise ValueError("No valid trajectories remain after length filtering.")

    valid_raw_ids = np.asarray([item[0] for item in valid], dtype=np.int64)
    train_ids = _load_training_ids(
        _resolve(_dataset_value(args, "train_index_file", args.train_index_file)),
        valid_raw_ids,
        args.train_count,
        args.shuffle_seed,
    )
    train_set = set(int(item) for item in train_ids)
    gallery_sources = [item for item in valid if int(item[0]) not in train_set]
    available_gallery_sources = len(gallery_sources)
    if available_gallery_sources < int(args.query_count):
        raise ValueError("The non-training trajectory pool is smaller than query_count.")
    target_size = int(args.target_size)
    if target_size < int(args.query_count):
        raise ValueError("target_size must be at least query_count.")

    perturbations = tuple(item.strip() for item in args.perturbations.split(",") if item.strip())
    unknown = sorted(set(perturbations).difference(SUPPORTED_PERTURBATIONS))
    if unknown or not perturbations:
        raise ValueError(f"Unknown or empty perturbations: {unknown}; expected {SUPPORTED_PERTURBATIONS}.")
    severities = _parse_float_list(args.severities)
    rng = np.random.default_rng(int(args.seed))
    query_indices = rng.choice(len(gallery_sources), size=int(args.query_count), replace=False)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    query_index_set = set(int(item) for item in query_indices)
    remaining_indices = np.asarray(
        [idx for idx in range(len(gallery_sources)) if idx not in query_index_set],
        dtype=np.int64,
    )
    if available_gallery_sources > target_size:
        remaining_indices = rng.choice(
            remaining_indices,
            size=target_size - len(query_indices),
            replace=False,
        )
    # Put Query positives first so every nested Gallery prefix retains them.
    selected_indices = np.concatenate((query_indices, remaining_indices))
    gallery_sources = [gallery_sources[int(idx)] for idx in selected_indices]
    query_indices = np.arange(int(args.query_count), dtype=np.int64)
    query_index_set = set(int(item) for item in query_indices)
    augment_count = target_size - len(gallery_sources)
    augment_pool = np.asarray(
        [idx for idx in range(len(gallery_sources)) if idx not in query_index_set],
        dtype=np.int64,
    )
    if augment_count > len(augment_pool):
        raise ValueError(
            f"Need {augment_count} perturbations but only {len(augment_pool)} non-query sources "
            "are available. Increase the source pool or lower target_size."
        )
    augment_indices = rng.choice(augment_pool, size=augment_count, replace=False)

    context = {
        "min_length": int(args.min_length),
        "adjacency": None,
        "node_coordinates": None,
        "neighbor_connectivity": args.neighbor_connectivity,
    }
    if "neighbor_replace" in perturbations:
        adjacency, node_coordinates = load_road_neighbor_data(
            str(_resolve(_dataset_value(args, "edge_file", args.edge_file))),
            str(_resolve(_dataset_value(args, "node_file", args.node_file))),
        )
        context["adjacency"] = adjacency
        context["node_coordinates"] = node_coordinates

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_nodes = [item[1] for item in gallery_sources]
    gallery_coordinates = [item[2] for item in gallery_sources]
    gallery_metadata = _empty_metadata()
    for item in gallery_sources:
        _append_metadata(
            gallery_metadata, item[0], "clean", 0.0, len(item[1]), len(item[1]),
            {"actual_count": 0, "path_valid_ratio": 1.0},
        )

    query_nodes, query_coordinates = [], []
    query_metadata = _empty_metadata()
    for query_number, source_index in enumerate(query_indices):
        source_id, nodes, coordinates = gallery_sources[int(source_index)]
        perturbation = perturbations[query_number % len(perturbations)]
        severity = severities[query_number % len(severities)]
        output_nodes, output_coordinates, details = _make_perturbation(
            nodes, coordinates, perturbation, severity, rng, context
        )
        query_nodes.append(output_nodes)
        query_coordinates.append(output_coordinates)
        _append_metadata(
            query_metadata, source_id, perturbation, severity,
            len(nodes), len(output_nodes), details,
        )

    for variant_number, source_index in enumerate(augment_indices):
        source_id, nodes, coordinates = gallery_sources[int(source_index)]
        perturbation = perturbations[variant_number % len(perturbations)]
        severity = severities[variant_number % len(severities)]
        output_nodes, output_coordinates, details = _make_perturbation(
            nodes, coordinates, perturbation, severity, rng, context
        )
        gallery_nodes.append(output_nodes)
        gallery_coordinates.append(output_coordinates)
        _append_metadata(
            gallery_metadata, source_id, perturbation, severity,
            len(nodes), len(output_nodes), details,
        )

    gallery_metadata_arrays = _metadata_arrays(gallery_metadata)
    gallery_metadata_arrays["is_initial"] = np.asarray(
        [item == "clean" for item in gallery_metadata["perturbation_types"]], dtype=bool
    )
    query_metadata_arrays = _metadata_arrays(query_metadata)
    query_metadata_arrays["is_initial"] = np.zeros(int(args.query_count), dtype=bool)

    np.save(output_dir / "gallery_trajectories.npy", _object_array(gallery_nodes, np.int64), allow_pickle=True)
    np.save(output_dir / "gallery_coordinates.npy", _object_array(gallery_coordinates, np.float32), allow_pickle=True)
    np.save(output_dir / "query_trajectories.npy", _object_array(query_nodes, np.int64), allow_pickle=True)
    np.save(output_dir / "query_coordinates.npy", _object_array(query_coordinates, np.float32), allow_pickle=True)
    np.savez_compressed(output_dir / "gallery_metadata.npz", **gallery_metadata_arrays)
    np.savez_compressed(output_dir / "query_metadata.npz", **query_metadata_arrays)

    manifest = {
        "protocol": "non_training_real_plus_perturbation_gallery",
        "dataset": args.dataset,
        "trajectory_file": str(traj_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "raw_trajectory_count": int(len(records)),
        "valid_trajectory_count": int(len(valid)),
        "min_source_length": int(args.min_source_length),
        "training_count_requested": int(args.train_count),
        "training_count_used": int(len(train_ids)),
        "training_ids": sorted(train_set) if args.save_training_ids else None,
        "real_non_training_pool_count": int(available_gallery_sources),
        "real_non_training_gallery_count": int(len(gallery_sources)),
        "perturbation_gallery_count": int(augment_count),
        "gallery_count": int(len(gallery_nodes)),
        "query_count": int(len(query_nodes)),
        "target_size": int(target_size),
        "query_sources_have_clean_gallery_positive": True,
        "query_perturbations_are_excluded_from_gallery": True,
        "perturbations": list(perturbations),
        "severities": list(severities),
        "seed": int(args.seed),
        "shuffle_seed": int(args.shuffle_seed),
        "train_index_file": str(
            _resolve(_dataset_value(args, "train_index_file", args.train_index_file)).resolve()
        ),
        "files": {
            "gallery_trajectories": "gallery_trajectories.npy",
            "gallery_coordinates": "gallery_coordinates.npy",
            "gallery_metadata": "gallery_metadata.npz",
            "query_trajectories": "query_trajectories.npy",
            "query_coordinates": "query_coordinates.npy",
            "query_metadata": "query_metadata.npz",
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({key: manifest[key] for key in (
        "raw_trajectory_count", "valid_trajectory_count", "training_count_used",
        "real_non_training_pool_count",
        "real_non_training_gallery_count", "perturbation_gallery_count",
        "gallery_count", "query_count",
    )}, indent=2))


def compute_tp(args):
    from tp_retrieval_evaluation import compute_tp_ground_truth

    root = _resolve(args.benchmark_dir)
    gallery = np.load(root / "gallery_trajectories.npy", allow_pickle=True)
    queries = np.load(root / "query_trajectories.npy", allow_pickle=True)
    output = _resolve(args.output_file) if args.output_file else root / "tp_ground_truth.npz"
    ids, distances, report = compute_tp_ground_truth(
        queries,
        gallery,
        str(_resolve(_resource_value(args, "point_distance_file", args.point_distance_file))),
        top_k=int(args.top_k),
        max_queries=int(args.query_count),
        num_threads=int(args.num_threads),
        device=args.device,
        gpu_batch_size=int(args.gpu_batch_size),
        cache_file=str(output),
        force_recompute=bool(args.force_recompute),
    )
    report_path = Path(str(output) + ".json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return ids, distances


def compute_tp_table(args):
    """Write a Query x Gallery float32 TP table without retaining it in RAM."""
    from tp_retrieval_evaluation import close_mmap_array, compute_tp_distance_table

    root = _resolve(args.benchmark_dir)
    gallery = np.load(root / "gallery_trajectories.npy", allow_pickle=True)
    queries = np.load(root / "query_trajectories.npy", allow_pickle=True)
    output = _resolve(args.output_file) if args.output_file else root / "tp_distance_table.npy"
    expected_bytes = int(min(len(queries), int(args.query_count)) * len(gallery) * 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = os.statvfs(str(output.parent)).f_bavail * os.statvfs(str(output.parent)).f_frsize if hasattr(os, "statvfs") else None
    if free_bytes is not None and free_bytes < expected_bytes:
        raise OSError(
            f"Insufficient free space for {output}: need at least "
            f"{expected_bytes / (1024 ** 3):.2f} GiB, have {free_bytes / (1024 ** 3):.2f} GiB."
        )
    table, report = compute_tp_distance_table(
        queries,
        gallery,
        str(_resolve(_resource_value(args, "point_distance_file", args.point_distance_file))),
        max_queries=int(args.query_count),
        num_threads=int(args.num_threads),
        device=args.device,
        gpu_batch_size=int(args.gpu_batch_size),
        cache_file=str(output),
        force_recompute=bool(args.force_recompute),
    )
    report.update(
        {
            "table_file": str(output),
            "dtype": "float32",
            "shape": [int(table.shape[0]), int(table.shape[1])],
            "storage_gib": float(table.nbytes / (1024 ** 3)),
            "gallery_order": "gallery_trajectories.npy row order",
            "query_order": "query_trajectories.npy row order",
        }
    )
    close_mmap_array(table)
    report_path = Path(str(output) + ".json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


def _metric_resource_kwargs(args):
    metric = str(args.metric)
    point_distance_file = _resource_value(
        args, "point_distance_file", getattr(args, "point_distance_file", None)
    )
    edge_file = _resource_value(
        args, "edge_file", getattr(args, "edge_file", None)
    )
    node_count = int(getattr(args, "node_count", 0) or 0)
    if node_count <= 0:
        node_count = int(DATASET_DEFAULTS[getattr(args, "dataset", "beijing")]["node_count"])
    anchor = getattr(args, "neterp_anchor_node", None)
    if anchor is None:
        anchor = DATASET_DEFAULTS[getattr(args, "dataset", "beijing")]["neterp_anchor_node"]
    return {
        "point_distance_file": str(_resolve(point_distance_file)),
        "edge_file": str(_resolve(edge_file)),
        "node_count": node_count,
        "max_trajectory_length": int(getattr(args, "max_trajectory_length", 0) or 0),
        "neterp_anchor_node": anchor,
    }


def _load_benchmark_trajectories(args):
    root = _resolve(args.benchmark_dir)
    gallery_file = root / "gallery_trajectories.npy"
    query_file = root / "query_trajectories.npy"
    if not gallery_file.is_file() or not query_file.is_file():
        raise FileNotFoundError(
            f"Benchmark directory must contain gallery_trajectories.npy and "
            f"query_trajectories.npy: {root}"
        )
    return (
        np.load(query_file, allow_pickle=True),
        np.load(gallery_file, allow_pickle=True),
    )


def _write_metric_report(output, report, *, shape=None, table=False):
    report = dict(report)
    report["metric"] = str(report.get("metric", ""))
    report["output_file"] = str(output)
    if shape is not None:
        report["shape"] = [int(item) for item in shape]
    if table:
        report["dtype"] = "float32"
        report["storage_gib"] = float(np.prod(shape) * 4 / (1024 ** 3))
        report["gallery_order"] = "gallery_trajectories.npy row order"
        report["query_order"] = "query_trajectories.npy row order"
    with open(Path(str(output) + ".json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def compute_metric(args):
    from tp_retrieval_evaluation import compute_metric_ground_truth

    queries, gallery = _load_benchmark_trajectories(args)
    output = _resolve(args.output_file) if args.output_file else (
        _resolve(args.benchmark_dir) / f"{str(args.metric).lower()}_ground_truth.npz"
    )
    resource_kwargs = _metric_resource_kwargs(args)
    _, _, report = compute_metric_ground_truth(
        queries,
        gallery,
        args.metric,
        **resource_kwargs,
        top_k=int(args.top_k),
        max_queries=int(args.query_count),
        gallery_batch_size=int(args.gallery_batch_size),
        num_threads=int(args.num_threads),
        cache_file=str(output),
        force_recompute=bool(args.force_recompute),
    )
    return _write_metric_report(output, report)


def compute_metric_table(args):
    from tp_retrieval_evaluation import close_mmap_array, compute_metric_distance_table

    queries, gallery = _load_benchmark_trajectories(args)
    output = _resolve(args.output_file) if args.output_file else (
        _resolve(args.benchmark_dir) / f"{str(args.metric).lower()}_distance_table.npy"
    )
    query_count = min(len(queries), int(args.query_count)) if int(args.query_count) > 0 else len(queries)
    expected_bytes = int(query_count * len(gallery) * 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "statvfs"):
        stat = os.statvfs(str(output.parent))
        free_bytes = stat.f_bavail * stat.f_frsize
        if free_bytes < expected_bytes:
            raise OSError(
                f"Insufficient free space for {output}: need at least "
                f"{expected_bytes / (1024 ** 3):.2f} GiB, "
                f"have {free_bytes / (1024 ** 3):.2f} GiB."
            )
    table, report = compute_metric_distance_table(
        queries,
        gallery,
        args.metric,
        **_metric_resource_kwargs(args),
        max_queries=int(args.query_count),
        gallery_batch_size=int(args.gallery_batch_size),
        num_threads=int(args.num_threads),
        cache_file=str(output),
        force_recompute=bool(args.force_recompute),
    )
    shape = table.shape
    _write_metric_report(output, report, shape=shape, table=True)
    close_mmap_array(table)


def attach_metadata(args):
    embedding_file = _resolve(args.embedding_file)
    metadata_file = _resolve(args.metadata_file)
    output_file = _resolve(args.output_file)
    with np.load(embedding_file, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    with np.load(metadata_file, allow_pickle=True) as data:
        for key in data.files:
            if key in payload and not args.overwrite:
                raise ValueError(f"Embedding already contains metadata key {key!r}; use --overwrite.")
            payload[key] = data[key]
    row_count = len(payload["embeddings"])
    for key, value in payload.items():
        if key in {"coordinate_stats"}:
            continue
        if hasattr(value, "shape") and len(value.shape) and len(value) != row_count:
            raise ValueError(f"Metadata key {key!r} has {len(value)} rows; expected {row_count}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **payload)
    print(f"Attached metadata: {output_file}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="beijing")
    build.add_argument("--trajectory-file", default=None)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--target-size", type=int, default=1_000_000)
    build.add_argument("--query-count", type=int, default=2_000)
    build.add_argument("--train-count", type=int, default=50_000)
    build.add_argument("--train-index-file", default=None)
    build.add_argument("--shuffle-seed", type=int, default=1953)
    build.add_argument("--min-source-length", type=int, default=10)
    build.add_argument("--min-length", type=int, default=4)
    build.add_argument("--perturbations", default="drop,crop,local_gap")
    build.add_argument("--severities", default="0.1,0.2,0.3")
    build.add_argument("--seed", type=int, default=2026)
    build.add_argument("--edge-file", default=None)
    build.add_argument("--node-file", default=None)
    build.add_argument("--neighbor-connectivity", choices=["prefer", "strict"], default="prefer")
    build.add_argument("--save-training-ids", action="store_true")
    build.set_defaults(func=build_benchmark)

    tp = subparsers.add_parser("tp")
    tp.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="beijing")
    tp.add_argument("--benchmark-dir", required=True)
    tp.add_argument("--point-distance-file", default=None)
    tp.add_argument("--output-file", default=None)
    tp.add_argument("--query-count", type=int, default=2_000)
    tp.add_argument("--top-k", type=int, default=50)
    tp.add_argument("--num-threads", type=int, default=0)
    tp.add_argument("--device", default="cpu")
    tp.add_argument("--gpu-batch-size", type=int, default=512)
    tp.add_argument("--force-recompute", action="store_true")
    tp.set_defaults(func=compute_tp)

    table = subparsers.add_parser("tp-table")
    table.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="beijing")
    table.add_argument("--benchmark-dir", required=True)
    table.add_argument("--point-distance-file", default=None)
    table.add_argument("--output-file", default=None)
    table.add_argument("--query-count", type=int, default=2_000)
    table.add_argument("--num-threads", type=int, default=0)
    table.add_argument("--device", default="cpu")
    table.add_argument("--gpu-batch-size", type=int, default=512)
    table.add_argument("--force-recompute", action="store_true")
    table.set_defaults(func=compute_tp_table)

    def add_metric_options(command):
        command.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="beijing")
        command.add_argument("--benchmark-dir", required=True)
        command.add_argument("--metric", choices=SUPPORTED_METRICS, required=True)
        command.add_argument("--point-distance-file", default=None)
        command.add_argument("--edge-file", default=None)
        command.add_argument("--node-count", type=int, default=0)
        command.add_argument("--max-trajectory-length", type=int, default=0)
        command.add_argument("--neterp-anchor-node", type=int, default=None)
        command.add_argument("--output-file", default=None)
        command.add_argument("--query-count", type=int, default=2_000)
        command.add_argument("--top-k", type=int, default=50)
        command.add_argument("--gallery-batch-size", type=int, default=1024)
        command.add_argument("--num-threads", type=int, default=0)
        command.add_argument(
            "--force-recompute",
            action="store_true",
            help="Ignore a matching metric cache and recompute it.",
        )

    metric = subparsers.add_parser(
        "metric",
        help="Compute exact Top-K truth for any supported trajectory metric.",
    )
    add_metric_options(metric)
    metric.set_defaults(func=compute_metric)

    metric_table = subparsers.add_parser(
        "metric-table",
        help="Write a dense Query x Gallery table for any supported metric.",
    )
    add_metric_options(metric_table)
    metric_table.set_defaults(func=compute_metric_table)

    attach = subparsers.add_parser("attach-metadata")
    attach.add_argument("--embedding-file", required=True)
    attach.add_argument("--metadata-file", required=True)
    attach.add_argument("--output-file", required=True)
    attach.add_argument("--overwrite", action="store_true")
    attach.set_defaults(func=attach_metadata)
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    arguments.func(arguments)
