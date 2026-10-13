import csv
from time import perf_counter

import numpy as np


DEFAULT_PERTURBATIONS = ("drop", "crop", "local_gap")
SUPPORTED_PERTURBATIONS = DEFAULT_PERTURBATIONS + ("neighbor_replace",)


def _object_sequence_array(items, dtype):
    arrays = [np.asarray(item, dtype=dtype) for item in items]
    output = np.empty(len(arrays), dtype=object)
    output[:] = arrays
    return output


def _validate_inputs(trajectories, coordinates):
    if coordinates is None:
        raise ValueError("Coordinates are required to create perturbation queries.")
    if len(trajectories) != len(coordinates):
        raise ValueError(
            f"Trajectory count ({len(trajectories)}) does not match coordinate count "
            f"({len(coordinates)})."
        )
    for idx, (trajectory, coordinate) in enumerate(zip(trajectories, coordinates)):
        if len(trajectory) != len(coordinate):
            raise ValueError(
                f"Trajectory {idx} has {len(trajectory)} nodes but {len(coordinate)} coordinates."
            )


def _kept_indices(length, perturbation, severity, min_length, rng):
    if perturbation not in SUPPORTED_PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation '{perturbation}'. Expected one of {SUPPORTED_PERTURBATIONS}."
        )
    if not 0.0 < severity < 1.0:
        raise ValueError(f"Perturbation severity must be in (0, 1), got {severity}.")
    if length <= min_length:
        raise ValueError(
            f"A trajectory of length {length} cannot be perturbed while preserving "
            f"min_length={min_length}."
        )

    remove_count = max(1, int(round(length * severity)))
    remove_count = min(remove_count, length - min_length)

    if perturbation == "drop":
        internal = np.arange(1, length - 1, dtype=np.int64)
        remove_count = min(remove_count, len(internal))
        removed = rng.choice(internal, size=remove_count, replace=False)
        keep = np.ones(length, dtype=bool)
        keep[removed] = False
        return np.flatnonzero(keep)

    if perturbation == "crop":
        left_removed = int(rng.integers(0, remove_count + 1))
        right_removed = remove_count - left_removed
        stop = length - right_removed if right_removed else length
        return np.arange(left_removed, stop, dtype=np.int64)

    start = int(rng.integers(1, length - remove_count))
    keep = np.ones(length, dtype=bool)
    keep[start : start + remove_count] = False
    return np.flatnonzero(keep)


def load_road_neighbor_data(edge_file, node_file):
    """Load undirected first-order neighbors and raw node coordinates."""
    adjacency = {}
    with open(edge_file, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"s_node", "e_node"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{edge_file} must contain columns {sorted(required)}.")
        for row in reader:
            source = int(row["s_node"])
            target = int(row["e_node"])
            if source == target:
                continue
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

    node_coordinates = {}
    with open(node_file, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"node", "lng", "lat"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{node_file} must contain columns {sorted(required)}.")
        for row in reader:
            node_coordinates[int(row["node"])] = np.asarray(
                [float(row["lng"]), float(row["lat"])], dtype=np.float32
            )
    return {
        node: np.asarray(sorted(neighbors), dtype=np.int64)
        for node, neighbors in adjacency.items()
    }, node_coordinates


def _path_valid_ratio(trajectory, adjacency):
    trajectory = np.asarray(trajectory, dtype=np.int64)
    if len(trajectory) < 2:
        return 1.0
    valid = 0
    for left, right in zip(trajectory[:-1], trajectory[1:]):
        neighbors = adjacency.get(int(left), ())
        valid += int(int(right) in neighbors)
    return float(valid / (len(trajectory) - 1))


def _neighbor_replacement(
    trajectory,
    coordinates,
    severity,
    rng,
    adjacency,
    node_coordinates,
    connectivity="prefer",
):
    if connectivity not in {"prefer", "strict"}:
        raise ValueError("neighbor_connectivity must be 'prefer' or 'strict'.")
    trajectory = np.asarray(trajectory, dtype=np.int64)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    target_count = min(
        max(1, int(round(len(trajectory) * severity))), max(len(trajectory) - 2, 0)
    )
    eligible = []
    candidates_by_position = {}
    for position in range(1, len(trajectory) - 1):
        node = int(trajectory[position])
        previous = int(trajectory[position - 1])
        following = int(trajectory[position + 1])
        candidates = [
            int(candidate)
            for candidate in adjacency.get(node, ())
            if int(candidate) != node and int(candidate) in node_coordinates
        ]
        if not candidates:
            continue
        context_scores = {
            candidate: int(previous in adjacency.get(candidate, ()))
            + int(following in adjacency.get(candidate, ()))
            for candidate in candidates
        }
        if connectivity == "strict":
            candidates = [
                candidate for candidate in candidates if context_scores[candidate] == 2
            ]
            if not candidates:
                continue
        candidates_by_position[position] = (candidates, context_scores)
        eligible.append(position)

    actual_target = min(target_count, len(eligible))
    if connectivity == "strict" and actual_target:
        selected_list = []
        for position in rng.permutation(eligible):
            if all(abs(int(position) - chosen) > 1 for chosen in selected_list):
                selected_list.append(int(position))
                if len(selected_list) >= actual_target:
                    break
        selected = np.asarray(sorted(selected_list), dtype=np.int64)
    else:
        selected = (
            np.sort(rng.choice(eligible, size=actual_target, replace=False))
            if actual_target
            else np.empty((0,), dtype=np.int64)
        )
    output_trajectory = trajectory.copy()
    output_coordinates = coordinates.copy()
    replacement_nodes = []
    for position in selected:
        candidates, context_scores = candidates_by_position[int(position)]
        best_score = max(context_scores[candidate] for candidate in candidates)
        best = [
            candidate for candidate in candidates if context_scores[candidate] == best_score
        ]
        replacement = int(rng.choice(best))
        output_trajectory[position] = replacement
        output_coordinates[position] = node_coordinates[replacement]
        replacement_nodes.append(replacement)
    return output_trajectory, output_coordinates, {
        "target_count": int(target_count),
        "actual_count": int(len(selected)),
        "positions": np.asarray(selected, dtype=np.int64),
        "replacement_nodes": np.asarray(replacement_nodes, dtype=np.int64),
        "path_valid_ratio": _path_valid_ratio(output_trajectory, adjacency),
    }


def generate_perturbed_queries(
    trajectories,
    coordinates,
    source_count=5000,
    perturbations=DEFAULT_PERTURBATIONS,
    severities=(0.1, 0.2, 0.3),
    variants_per_setting=1,
    seed=2026,
    min_length=4,
    gallery_size=None,
    neighbor_adjacency=None,
    node_coordinates=None,
    neighbor_connectivity="prefer",
):
    """Create paired queries whose known positive is the unmodified gallery item."""
    trajectories = _object_sequence_array(trajectories, np.int64)
    coordinates = _object_sequence_array(coordinates, np.float32)
    _validate_inputs(trajectories, coordinates)

    gallery_size = len(trajectories) if not gallery_size or gallery_size <= 0 else int(gallery_size)
    if gallery_size > len(trajectories):
        raise ValueError(
            f"gallery_size={gallery_size} exceeds available trajectories ({len(trajectories)})."
        )
    if min_length < 2:
        raise ValueError("min_length must be at least 2.")

    perturbations = tuple(str(item) for item in perturbations)
    severities = tuple(float(item) for item in severities)
    if not perturbations:
        raise ValueError("At least one perturbation type is required.")
    if not severities:
        raise ValueError("At least one perturbation severity is required.")
    variants_per_setting = int(variants_per_setting)
    if variants_per_setting <= 0:
        raise ValueError("variants_per_setting must be a positive integer.")

    eligible = np.asarray(
        [idx for idx in range(gallery_size) if len(trajectories[idx]) > min_length],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise ValueError(
            f"No trajectory in the first {gallery_size} gallery items is longer than {min_length}."
        )

    rng = np.random.default_rng(seed)
    if source_count and source_count > 0:
        source_count = min(int(source_count), len(eligible))
        source_ids = rng.choice(eligible, size=source_count, replace=False)
    else:
        source_ids = eligible

    query_trajectories = []
    query_coordinates = []
    query_source_ids = []
    query_types = []
    query_severities = []
    query_variant_ids = []
    requested_replacement_counts = []
    actual_replacement_counts = []
    replaced_positions = []
    replacement_node_ids = []
    path_valid_ratios = []
    effective_severities = []
    original_lengths = []
    query_lengths = []

    for source_id in source_ids:
        trajectory = np.asarray(trajectories[source_id], dtype=np.int64)
        coordinate = np.asarray(coordinates[source_id], dtype=np.float32)
        for perturbation in perturbations:
            for severity in severities:
                for variant_id in range(variants_per_setting):
                    if perturbation == "neighbor_replace":
                        if neighbor_adjacency is None or node_coordinates is None:
                            raise ValueError(
                                "neighbor_replace requires road adjacency and node coordinates."
                            )
                        query_trajectory, query_coordinate, replacement = (
                            _neighbor_replacement(
                                trajectory,
                                coordinate,
                                severity,
                                rng,
                                neighbor_adjacency,
                                node_coordinates,
                                connectivity=neighbor_connectivity,
                            )
                        )
                    else:
                        kept = _kept_indices(
                            len(trajectory),
                            perturbation,
                            severity,
                            min_length,
                            rng,
                        )
                        query_trajectory = trajectory[kept]
                        query_coordinate = coordinate[kept]
                        replacement = {
                            "target_count": 0,
                            "actual_count": 0,
                            "positions": np.empty((0,), dtype=np.int64),
                            "replacement_nodes": np.empty((0,), dtype=np.int64),
                            "path_valid_ratio": np.nan,
                        }
                    query_trajectories.append(query_trajectory)
                    query_coordinates.append(query_coordinate)
                    query_source_ids.append(int(source_id))
                    query_types.append(perturbation)
                    query_severities.append(float(severity))
                    query_variant_ids.append(variant_id)
                    requested_replacement_counts.append(replacement["target_count"])
                    actual_replacement_counts.append(replacement["actual_count"])
                    replaced_positions.append(replacement["positions"])
                    replacement_node_ids.append(replacement["replacement_nodes"])
                    path_valid_ratios.append(replacement["path_valid_ratio"])
                    effective_severities.append(
                        (
                            replacement["actual_count"] / max(len(trajectory), 1)
                            if perturbation == "neighbor_replace"
                            else (len(trajectory) - len(query_trajectory))
                            / max(len(trajectory), 1)
                        )
                    )
                    original_lengths.append(len(trajectory))
                    query_lengths.append(len(query_trajectory))

    return {
        "trajectories": _object_sequence_array(query_trajectories, np.int64),
        "coordinates": _object_sequence_array(query_coordinates, np.float32),
        "source_ids": np.asarray(query_source_ids, dtype=np.int64),
        "perturbation_types": np.asarray(query_types),
        "severities": np.asarray(query_severities, dtype=np.float32),
        "variant_ids": np.asarray(query_variant_ids, dtype=np.int64),
        "requested_replacement_counts": np.asarray(
            requested_replacement_counts, dtype=np.int64
        ),
        "actual_replacement_counts": np.asarray(
            actual_replacement_counts, dtype=np.int64
        ),
        "replaced_positions": _object_sequence_array(replaced_positions, np.int64),
        "replacement_node_ids": _object_sequence_array(
            replacement_node_ids, np.int64
        ),
        "path_valid_ratios": np.asarray(path_valid_ratios, dtype=np.float32),
        "effective_severities": np.asarray(
            effective_severities, dtype=np.float32
        ),
        "original_lengths": np.asarray(original_lengths, dtype=np.int64),
        "query_lengths": np.asarray(query_lengths, dtype=np.int64),
        "variants_per_setting": np.asarray(variants_per_setting, dtype=np.int64),
        "generation_seed": np.asarray(seed, dtype=np.int64),
        "gallery_size": np.asarray(gallery_size, dtype=np.int64),
    }


def remap_perturbed_source_ids(generated, gallery_source_ids):
    """Map perturbation source row IDs into the Gallery's source-ID space."""
    if gallery_source_ids is None:
        return generated
    gallery_source_ids = np.asarray(gallery_source_ids, dtype=np.int64).reshape(-1)
    local_source_ids = np.asarray(generated["source_ids"], dtype=np.int64).reshape(-1)
    if np.any((local_source_ids < 0) | (local_source_ids >= len(gallery_source_ids))):
        raise ValueError(
            "Perturbation source IDs contain a row outside the source Gallery."
        )
    remapped = dict(generated)
    remapped["local_source_ids"] = local_source_ids.copy()
    remapped["source_ids"] = gallery_source_ids[local_source_ids]
    return remapped


def paired_identification_metrics(predictions, source_ids, ks=(1, 10, 50)):
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if len(predictions) != len(source_ids):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match source count ({len(source_ids)})."
        )
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or ks[0] <= 0:
        raise ValueError("All recall cutoffs must be positive integers.")

    ranks = np.zeros(len(source_ids), dtype=np.int64)
    max_k = ks[-1]
    for idx, (prediction, source_id) in enumerate(zip(predictions, source_ids)):
        prediction = np.asarray(prediction, dtype=np.int64)[:max_k]
        positions = np.flatnonzero(prediction == source_id)
        if positions.size:
            ranks[idx] = int(positions[0]) + 1

    result = {
        f"id_recall_at_{k}": float(np.mean((ranks > 0) & (ranks <= k)))
        for k in ks
    }
    reciprocal = np.zeros(len(ranks), dtype=np.float64)
    hit_mask = ranks > 0
    reciprocal[hit_mask] = 1.0 / ranks[hit_mask]
    result[f"mrr_at_{max_k}"] = float(reciprocal.mean())
    result["mean_rank_if_hit"] = float(ranks[hit_mask].mean()) if np.any(hit_mask) else None
    result["misses_at_max_k"] = int(np.count_nonzero(~hit_mask))
    return result


def _latency_metrics(latencies):
    latencies = np.asarray(latencies, dtype=np.float64)
    total_seconds = float(latencies.sum())
    return {
        "avg_query_ms": float(latencies.mean() * 1000.0) if len(latencies) else 0.0,
        "p50_query_ms": float(np.percentile(latencies, 50) * 1000.0) if len(latencies) else 0.0,
        "p95_query_ms": float(np.percentile(latencies, 95) * 1000.0) if len(latencies) else 0.0,
        "qps": float(len(latencies) / total_seconds) if total_seconds > 0 else 0.0,
    }


def evaluate_paired_queries(
    query_function,
    embeddings,
    trajectories,
    source_ids,
    perturbation_types,
    severities,
    top_k=50,
    max_queries=None,
):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    total = len(embeddings) if not max_queries or max_queries <= 0 else min(int(max_queries), len(embeddings))
    predictions = []
    latencies = []
    for idx in range(total):
        start = perf_counter()
        ids = query_function(embeddings[idx], trajectories[idx], top_k)
        latencies.append(perf_counter() - start)
        predictions.append(np.asarray(ids, dtype=np.int64))

    cutoffs = tuple(k for k in (1, 10, 50) if k <= top_k)
    if top_k not in cutoffs:
        cutoffs = tuple(sorted(set(cutoffs + (top_k,))))
    result = {
        "queries": total,
        "top_k": int(top_k),
        **paired_identification_metrics(predictions, source_ids[:total], cutoffs),
        **_latency_metrics(latencies),
    }

    perturbation_types = np.asarray(perturbation_types)[:total]
    severities = np.asarray(severities, dtype=np.float32)[:total]
    groups = []
    group_keys = sorted(
        {(str(kind), round(float(level), 6)) for kind, level in zip(perturbation_types, severities)}
    )
    for kind, level in group_keys:
        mask = (perturbation_types == kind) & np.isclose(severities, level)
        indices = np.flatnonzero(mask)
        group_predictions = [predictions[idx] for idx in indices]
        group = {
            "perturbation_type": kind,
            "severity": level,
            "queries": int(len(indices)),
            **paired_identification_metrics(group_predictions, np.asarray(source_ids)[indices], cutoffs),
        }
        groups.append(group)
    result["groups"] = groups
    return result
