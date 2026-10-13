"""Restore per-row coordinates for a compact trajectory embedding NPZ.

Large benchmark embedding files intentionally omit ``coordinates`` to keep
the NPZ small.  This utility restores them from the original trajectory
records without changing the trajectory row order.  It is deliberately
strict: it stops when a source ID, perturbation, or node sequence cannot be
matched exactly enough to prove that the restored coordinates are the ones
used to create the benchmark.

Supported benchmark transformations are the ones used by
``build_augmented_tp_benchmark.py``:

* clean rows use the original coordinate sequence;
* drop/crop/local_gap rows use the coordinates at the matching subsequence;
* neighbor_replace rows keep original coordinates at unchanged positions and
  use ``node.csv`` coordinates at replaced positions.

The output is an object ``.npy`` array with one float32 [length, 2] array per
row.  No input file is modified.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_augmented_tp_benchmark import _load_matching_result


PERTURBATION_SUBSEQUENCES = {"drop", "crop", "local_gap"}
CLEAN_PERTURBATIONS = {"clean", "initial", "original"}


def _load_node_coordinates(path):
    import csv

    coordinates = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"node", "lng", "lat"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}; "
                f"got {reader.fieldnames}."
            )
        for row in reader:
            node = int(row["node"])
            coordinates[node] = np.asarray(
                [float(row["lng"]), float(row["lat"])], dtype=np.float32
            )
    if not coordinates:
        raise ValueError(f"No node coordinates were read from {path}.")
    return coordinates


def _as_nodes(value, label):
    nodes = np.asarray(value, dtype=np.int64)
    if nodes.ndim != 1 or not len(nodes):
        raise ValueError(f"{label} must be a non-empty one-dimensional node array.")
    return nodes


def _as_coordinates(value, label):
    coordinates = np.asarray(value, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"{label} must have shape [length, 2], got {coordinates.shape}."
        )
    return coordinates


def _subsequence_positions(source, source_coordinates, target):
    """Return exact coordinates or fail on ambiguous repeated-node matches."""
    positions = []
    target_index = 0
    for source_index, node in enumerate(source):
        if target_index < len(target) and int(node) == int(target[target_index]):
            positions.append(source_index)
            target_index += 1
            if target_index == len(target):
                break
    if target_index != len(target):
        raise ValueError("the target node sequence is not a source subsequence")
    reverse_positions = np.empty(len(target), dtype=np.int64)
    target_index = len(target) - 1
    for source_index in range(len(source) - 1, -1, -1):
        if target_index >= 0 and int(source[source_index]) == int(target[target_index]):
            reverse_positions[target_index] = source_index
            target_index -= 1
            if target_index < 0:
                break

    # Any alignment for target[i] lies between the greedy forward and
    # backward matches. If all occurrences there have the same coordinates,
    # the restored coordinate is independent of which alignment was used.
    result = source_coordinates[np.asarray(positions, dtype=np.int64)].copy()
    for target_index, (first, last) in enumerate(zip(positions, reverse_positions)):
        if first == last:
            continue
        candidates = source_coordinates[first : last + 1][
            source[first : last + 1] == target[target_index]
        ]
        if not np.all(candidates == result[target_index]):
            raise ValueError(
                f"ambiguous matching positions for target node {target_index}; "
                "the repeated source node has different coordinates"
            )
    return result


def _load_payload(path):
    with np.load(path, allow_pickle=True) as data:
        required = {"trajectories", "source_ids"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}.")
        payload = {key: data[key] for key in (
            "trajectories", "source_ids", "perturbation_types", "coordinate_stats",
            "original_lengths", "trajectory_lengths", "actual_replacement_counts",
        ) if key in data.files}
    trajectories = np.asarray(payload["trajectories"], dtype=object).reshape(-1)
    source_ids = np.asarray(payload["source_ids"], dtype=np.int64).reshape(-1)
    if len(trajectories) != len(source_ids):
        raise ValueError(
            f"{path}: trajectories has {len(trajectories)} rows but source_ids has "
            f"{len(source_ids)} rows."
        )
    perturbations = payload.get("perturbation_types")
    if perturbations is None:
        perturbations = np.asarray(["unknown"] * len(trajectories), dtype=str)
    else:
        perturbations = np.asarray(perturbations).reshape(-1).astype(str)
        if len(perturbations) != len(trajectories):
            raise ValueError(
                f"{path}: perturbation_types has {len(perturbations)} rows but "
                f"trajectories has {len(trajectories)} rows."
            )
    return payload, trajectories, source_ids, perturbations


def _restore_rows(
    trajectories,
    source_ids,
    perturbations,
    records,
    node_coordinates,
):
    output = np.empty(len(trajectories), dtype=object)
    counts = Counter()
    subsequence_rows = 0
    replacement_rows = 0
    exact_clean_rows = 0
    max_replacements = 0

    for row_id, (trajectory, source_id, perturbation) in enumerate(
        zip(trajectories, source_ids, perturbations)
    ):
        source_id = int(source_id)
        if source_id < 0 or source_id >= len(records):
            raise ValueError(
                f"row {row_id}: source_id={source_id} is outside the original "
                f"trajectory range [0, {len(records)}). This usually means the "
                "NPZ stores row IDs rather than raw matching_result IDs."
            )
        record = records[source_id]
        if not isinstance(record, (list, tuple)) or len(record) != 2:
            raise ValueError(f"source record {source_id} is not [nodes, coordinates].")
        source_nodes = _as_nodes(record[0], f"source record {source_id} nodes")
        source_coordinates = _as_coordinates(
            record[1], f"source record {source_id} coordinates"
        )
        if len(source_nodes) != len(source_coordinates):
            raise ValueError(
                f"source record {source_id}: nodes/coordinates lengths differ: "
                f"{len(source_nodes)} != {len(source_coordinates)}."
            )
        row_nodes = _as_nodes(trajectory, f"NPZ row {row_id} trajectory")
        kind = str(perturbation).strip().lower()
        counts[kind] += 1

        if kind in CLEAN_PERTURBATIONS or (
            kind == "unknown" and np.array_equal(row_nodes, source_nodes)
        ):
            if not np.array_equal(row_nodes, source_nodes):
                raise ValueError(
                    f"row {row_id}, source {source_id}: marked {kind!r} but its "
                    "node sequence is not the original sequence."
                )
            output[row_id] = source_coordinates.copy()
            exact_clean_rows += 1
            continue

        if kind in PERTURBATION_SUBSEQUENCES:
            try:
                output[row_id] = _subsequence_positions(
                    source_nodes, source_coordinates, row_nodes
                )
            except ValueError as exc:
                raise ValueError(
                    f"row {row_id}, source {source_id}, perturbation {kind!r}: {exc}"
                ) from exc
            subsequence_rows += 1
            continue

        if kind == "neighbor_replace":
            if node_coordinates is None:
                raise ValueError(
                    "neighbor_replace rows require --node-file so replacement "
                    "coordinates can be recovered exactly."
                )
            if len(row_nodes) != len(source_nodes):
                raise ValueError(
                    f"row {row_id}, source {source_id}: neighbor_replace must "
                    f"preserve length, got {len(row_nodes)} vs {len(source_nodes)}."
                )
            restored = source_coordinates.copy()
            replacement_count = 0
            for position, (old_node, new_node) in enumerate(
                zip(source_nodes, row_nodes)
            ):
                if int(old_node) == int(new_node):
                    continue
                if int(new_node) not in node_coordinates:
                    raise ValueError(
                        f"row {row_id}, position {position}: replacement node "
                        f"{int(new_node)} is absent from the node coordinate file."
                    )
                restored[position] = node_coordinates[int(new_node)]
                replacement_count += 1
            output[row_id] = restored
            replacement_rows += 1
            max_replacements = max(max_replacements, replacement_count)
            continue

        raise ValueError(
            f"row {row_id}, source {source_id}: unsupported perturbation type "
            f"{perturbation!r}. Refusing to guess coordinates."
        )

    return output, {
        "rows": int(len(output)),
        "perturbation_counts": dict(sorted(counts.items())),
        "exact_clean_rows": int(exact_clean_rows),
        "subsequence_rows": int(subsequence_rows),
        "neighbor_replace_rows": int(replacement_rows),
        "max_replacements_in_one_row": int(max_replacements),
    }


def _coordinate_stats(coordinates):
    count = 0
    total = np.zeros(2, dtype=np.float64)
    total_square = np.zeros(2, dtype=np.float64)
    for item in coordinates:
        item = np.asarray(item, dtype=np.float64)
        count += len(item)
        total += item.sum(axis=0)
        total_square += np.square(item).sum(axis=0)
    if count == 0:
        raise ValueError("Cannot compute coordinate statistics for empty output.")
    mean = total / count
    std = np.sqrt(np.maximum(total_square / count - np.square(mean), 0.0))
    return np.asarray([mean[0], mean[1], std[0], std[1]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-file", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--node-file",
        default=None,
        help="Required only when the NPZ contains neighbor_replace rows.",
    )
    parser.add_argument("--report-file", default=None)
    parser.add_argument(
        "--check-gallery-stats",
        action="store_true",
        help=(
            "Compare restored-coordinate stats with the NPZ coordinate_stats. "
            "Use this for a Gallery file; Query stats are usually inherited "
            "from Gallery and should be checked against the Gallery separately."
        ),
    )
    args = parser.parse_args()

    embedding_file = Path(args.embedding_file)
    trajectory_file = Path(args.trajectory_file)
    output_file = Path(args.output_file)
    payload, trajectories, source_ids, perturbations = _load_payload(embedding_file)
    records = _load_matching_result(trajectory_file)
    node_coordinates = None
    if np.any(np.char.lower(perturbations.astype(str)) == "neighbor_replace"):
        if not args.node_file:
            raise ValueError(
                "The NPZ contains neighbor_replace rows; pass --node-file dataset/porto/node.csv."
            )
        node_coordinates = _load_node_coordinates(args.node_file)

    coordinates, report = _restore_rows(
        trajectories,
        source_ids,
        perturbations,
        records,
        node_coordinates,
    )
    for row_id, (nodes, coordinate) in enumerate(zip(trajectories, coordinates)):
        if len(nodes) != len(coordinate):
            raise ValueError(
                f"Restored row {row_id} has {len(nodes)} nodes but "
                f"{len(coordinate)} coordinates."
            )
    stats = _coordinate_stats(coordinates)
    report["embedding_file"] = str(embedding_file.resolve())
    report["trajectory_file"] = str(trajectory_file.resolve())
    report["output_file"] = str(output_file.resolve())
    report["restored_coordinate_stats"] = stats.tolist()
    if "coordinate_stats" in payload:
        expected_stats = np.asarray(payload["coordinate_stats"], dtype=np.float64).reshape(-1)
        if expected_stats.shape != (4,):
            raise ValueError(
                f"NPZ coordinate_stats must have shape (4,), got {expected_stats.shape}."
            )
        report["npz_coordinate_stats"] = expected_stats.tolist()
        report["coordinate_stats_max_abs_error"] = float(
            np.max(np.abs(stats - expected_stats))
        )
        if args.check_gallery_stats and not np.allclose(
            stats, expected_stats, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(
                "Restored Gallery coordinate statistics do not match the NPZ "
                f"coordinate_stats: restored={stats.tolist()}, "
                f"npz={expected_stats.tolist()}."
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, coordinates, allow_pickle=True)
    report_file = Path(args.report_file) if args.report_file else Path(str(output_file) + ".json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
