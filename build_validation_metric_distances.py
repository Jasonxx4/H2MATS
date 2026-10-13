"""Build label-aligned metric distances for hash-head distillation.

The output row ``i`` corresponds to validation_embeddings[i] and the output
column ``j`` corresponds to validation_embeddings[i]'s labels[j].  Computing
only the labelled candidates avoids materializing an N x N TP distance table.
"""

import argparse
import os
import os.path as osp

import numpy as np

from hier_hash_large_scale import load_embedding_payload
from trajectory_metric_backend import TrajectoryMetricBackend


def build_distances(
    embedding_file,
    output_file,
    point_distance_file,
    num_threads=0,
    overwrite=False,
):
    if osp.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}; pass --overwrite to replace it."
        )
    payload = load_embedding_payload(embedding_file, role="query")
    with np.load(embedding_file, allow_pickle=True) as data:
        label_key = next(
            (key for key in ("labels", "vali_y", "train_y") if key in data.files),
            None,
        )
        if label_key is None:
            raise ValueError(
                f"Embedding file {embedding_file} has no labels/vali_y/train_y array."
            )
        labels = np.asarray(data[label_key], dtype=np.int64)

    trajectories = np.asarray(payload["trajectories"], dtype=object).reshape(-1)
    if labels.ndim != 2 or len(labels) != len(trajectories):
        raise ValueError(
            "Labels must have shape [N, K] aligned with validation trajectories; "
            f"got labels={labels.shape}, trajectories={trajectories.shape}."
        )
    if not osp.isfile(point_distance_file):
        raise FileNotFoundError(
            f"TP point-distance matrix not found: {point_distance_file}"
        )

    out_dir = osp.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    distances = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype=np.float32,
        shape=labels.shape,
    )
    distances[:] = np.nan
    backend = TrajectoryMetricBackend(
        "TP",
        point_distance_file=point_distance_file,
        num_threads=num_threads,
    )
    try:
        try:
            from tqdm import tqdm

            iterator = tqdm(range(len(trajectories)), desc="validation TP distances")
        except ImportError:
            iterator = range(len(trajectories))
        for row_id in iterator:
            row_labels = labels[row_id]
            valid = (row_labels >= 0) & (row_labels < len(trajectories))
            if not np.any(valid):
                continue
            candidate_ids = row_labels[valid]
            distances[row_id, valid] = backend.distances(
                trajectories[row_id], trajectories[candidate_ids]
            )
            if row_id % 256 == 0:
                distances.flush()
    finally:
        backend.close()
        distances.flush()
        del distances
    print(
        f"saved TP distances: {output_file}, shape={labels.shape}, "
        f"finite={int(np.isfinite(np.load(output_file, mmap_mode='r')).sum())}"
    )
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build validation label-aligned TP distances for hash training."
    )
    parser.add_argument("--embedding-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--point-distance-file", required=True)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_distances(
        args.embedding_file,
        args.output_file,
        args.point_distance_file,
        num_threads=args.num_threads,
        overwrite=args.overwrite,
    )
