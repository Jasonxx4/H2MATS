"""Build a row-aligned train+validation set for post-hoc code encoders.

The nominal Porto/Beijing split has up to 15k encoder-training rows and 5k
validation rows. Preprocessing can remove invalid trajectories, so the actual
row counts may be lower. This utility combines the valid trajectories and
recomputes Top-K labels in the combined local index space. Recomputing labels
is necessary because the original train_y and vali_y arrays use different
local index spaces and cannot be concatenated safely.
"""

import argparse
import os
import os.path as osp

import numpy as np

from tp_retrieval_evaluation import exact_tp_distances, flatten_trajectories


def _load_rows(path, trajectory_key, coordinate_key):
    with np.load(path, allow_pickle=True) as data:
        missing = [
            key for key in (trajectory_key, coordinate_key) if key not in data.files
        ]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}.")
        trajectories = _as_row_sequence(data[trajectory_key])
        coordinates = _as_row_sequence(data[coordinate_key])

    if len(trajectories) != len(coordinates):
        raise ValueError(
            f"{path} has {len(trajectories)} trajectories but "
            f"{len(coordinates)} coordinate rows."
        )
    if not len(trajectories):
        raise ValueError(f"{path} contains no trajectories.")
    for row_id, trajectory in enumerate(trajectories):
        if len(np.asarray(trajectory).reshape(-1)) == 0:
            raise ValueError(f"{path} contains an empty trajectory at row {row_id}.")
    return trajectories, coordinates


def _as_row_sequence(values):
    """Preserve the first dimension for both object and regular arrays."""
    values = np.asarray(values, dtype=object)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim == 1:
        return values
    rows = np.empty(values.shape[0], dtype=object)
    for row_id in range(values.shape[0]):
        rows[row_id] = np.asarray(values[row_id])
    return rows


def _validate_split_count(split_name, actual_count, expected_count, path):
    """Treat configured split counts as nominal upper bounds."""
    if expected_count is None:
        return
    expected_count = int(expected_count)
    if expected_count < 0:
        raise ValueError(f"Expected {split_name} row limit must be non-negative.")
    if actual_count > expected_count:
        raise ValueError(
            f"Expected at most {expected_count} {split_name} rows, "
            f"found {actual_count} in {path}."
        )
    if actual_count < expected_count:
        print(
            f"[head-data] {split_name}: using {actual_count}/{expected_count} "
            f"valid rows from {path} after preprocessing."
        )


def _stable_topk_ids(distances, candidate_ids, top_k):
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if len(distances) != len(candidate_ids):
        raise ValueError("Distances and candidate IDs must have equal lengths.")
    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    distances = distances[finite]
    candidate_ids = candidate_ids[finite]
    order = np.lexsort((candidate_ids, distances))[: int(top_k)]
    return candidate_ids[order], distances[order]


def _merge_topk(best_ids, best_distances, candidate_ids, candidate_distances, top_k):
    if len(best_ids) == 0:
        return _stable_topk_ids(candidate_distances, candidate_ids, top_k)
    ids = np.concatenate((best_ids, candidate_ids))
    distances = np.concatenate((best_distances, candidate_distances))
    return _stable_topk_ids(distances, ids, top_k)


def build_head_training_set(
    train_file,
    validation_file,
    output_file,
    point_distance_file,
    *,
    top_k=50,
    gallery_batch_size=1024,
    num_threads=0,
    overwrite=False,
    expected_train_count=None,
    expected_validation_count=None,
):
    """Create a combined head-training NPZ with recomputed TP labels."""
    if osp.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}; pass --overwrite to replace it."
        )
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")
    if int(gallery_batch_size) <= 0:
        raise ValueError("gallery_batch_size must be positive.")
    if not osp.isfile(point_distance_file):
        raise FileNotFoundError(
            f"Point-distance matrix not found: {point_distance_file}"
        )

    train_trajectories, train_coordinates = _load_rows(
        train_file, "train_list", "train_coor"
    )
    validation_trajectories, validation_coordinates = _load_rows(
        validation_file, "vali_list", "vali_coor"
    )
    _validate_split_count(
        "training", len(train_trajectories), expected_train_count, train_file
    )
    _validate_split_count(
        "validation",
        len(validation_trajectories),
        expected_validation_count,
        validation_file,
    )
    trajectories = np.concatenate((train_trajectories, validation_trajectories))
    coordinates = np.concatenate((train_coordinates, validation_coordinates))
    total = len(trajectories)

    point_distances = np.load(point_distance_file, mmap_mode="r")
    if point_distances.ndim != 2 or point_distances.shape[0] != point_distances.shape[1]:
        raise ValueError("Point-distance matrix must be square.")
    nodes, offsets = flatten_trajectories(trajectories)
    if len(nodes) and (
        int(nodes.min()) < 0 or int(nodes.max()) >= point_distances.shape[0]
    ):
        raise ValueError("A trajectory node is outside the point-distance matrix.")
    if int(num_threads) > 0:
        try:
            import numba

            numba.set_num_threads(int(num_threads))
        except ImportError:
            pass

    width = min(int(top_k), total - 1)
    labels = np.full((total, int(top_k)), -1, dtype=np.int64)
    label_distances = np.full((total, int(top_k)), np.nan, dtype=np.float32)
    candidate_ids = np.arange(total, dtype=np.int64)
    gallery_batches = []
    for start in range(0, total, int(gallery_batch_size)):
        stop = min(start + int(gallery_batch_size), total)
        batch_nodes, batch_offsets = flatten_trajectories(
            trajectories[start:stop]
        )
        gallery_batches.append((start, stop, batch_nodes, batch_offsets))

    try:
        from tqdm import tqdm

        iterator = tqdm(range(total), desc="build head TP labels")
    except ImportError:
        iterator = range(total)

    for query_id in iterator:
        query = nodes[offsets[query_id] : offsets[query_id + 1]]
        best_ids = np.empty(0, dtype=np.int64)
        best_distances = np.empty(0, dtype=np.float32)
        for start, stop, batch_nodes, batch_offsets in gallery_batches:
            distances = exact_tp_distances(
                query,
                batch_nodes,
                batch_offsets,
                point_distances,
            )
            local_ids = candidate_ids[start:stop]
            if start <= query_id < stop:
                distances = distances.copy()
                distances[query_id - start] = np.inf
            best_ids, best_distances = _merge_topk(
                best_ids,
                best_distances,
                local_ids,
                distances,
                width,
            )
        labels[query_id, : len(best_ids)] = best_ids
        label_distances[query_id, : len(best_distances)] = best_distances

    output_dir = osp.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        output_file,
        vali_list=trajectories,
        vali_coor=coordinates,
        vali_y=labels,
        vali_dis=label_distances,
        source_ids=np.arange(total, dtype=np.int64),
        train_count=np.asarray(len(train_trajectories), dtype=np.int64),
        validation_count=np.asarray(len(validation_trajectories), dtype=np.int64),
        expected_train_count=np.asarray(
            -1 if expected_train_count is None else int(expected_train_count),
            dtype=np.int64,
        ),
        expected_validation_count=np.asarray(
            -1
            if expected_validation_count is None
            else int(expected_validation_count),
            dtype=np.int64,
        ),
        training_scope=np.asarray("train+validation"),
        metric=np.asarray("TP"),
    )
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge encoder train/validation rows and build head labels."
    )
    parser.add_argument(
        "--train-file",
        default="ground_truth/porto/TP/spatial_train_set.npz",
    )
    parser.add_argument(
        "--validation-file",
        default="ground_truth/porto/TP/spatial_vali_set.npz",
    )
    parser.add_argument(
        "--output-file",
        default="ground_truth/porto/TP/spatial_head_train_20k.npz",
    )
    parser.add_argument(
        "--point-distance-file",
        default="ground_truth/porto/Point_dis_matrix.npy",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gallery-batch-size", type=int, default=1024)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--expected-train-count", type=int, default=None)
    parser.add_argument("--expected-validation-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = build_head_training_set(
        args.train_file,
        args.validation_file,
        args.output_file,
        args.point_distance_file,
        top_k=args.top_k,
        gallery_batch_size=args.gallery_batch_size,
        num_threads=args.num_threads,
        overwrite=args.overwrite,
        expected_train_count=args.expected_train_count,
        expected_validation_count=args.expected_validation_count,
    )
    print(f"saved head training set: {output}")
