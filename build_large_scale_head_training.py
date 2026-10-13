"""Build large-scale query/Gallery supervision for post-hoc hash heads.

The workflow has two stages:

1. ``truth`` computes exact Top-K TP truth for a new Query set against an
   existing large Gallery, while storing only Top-K IDs and distances.
2. ``assemble`` combines pre-exported Query and Gallery embeddings into the
   row-aligned format consumed by ``hash_global_pipeline.py``.  Only Query
   rows have labels and act as training anchors; Gallery rows are candidates
   and negative-sampling sources.
"""

import argparse
import os
import os.path as osp

import numpy as np

from tp_retrieval_evaluation import compute_tp_ground_truth


def _find_array(data, keys, path):
    for key in keys:
        if key in data.files:
            return data[key]
    raise ValueError(f"{path} is missing trajectory array; expected one of {keys}.")


def _as_row_sequence(values):
    values = np.asarray(values, dtype=object)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    rows = np.empty(values.shape[0], dtype=object)
    for row_id in range(values.shape[0]):
        rows[row_id] = np.asarray(values[row_id])
    return rows


def _load_trajectories(path, role):
    path = str(path)
    if osp.isdir(path):
        filename = "query_trajectories.npy" if role == "query" else "gallery_trajectories.npy"
        path = osp.join(path, filename)
    if not osp.isfile(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    if path.lower().endswith(".npy"):
        return _as_row_sequence(np.load(path, allow_pickle=True))
    with np.load(path, allow_pickle=True) as data:
        if role == "query":
            keys = ("trajectories", "query_trajectories", "vali_list")
        else:
            keys = ("trajectories", "gallery_trajectories", "vali_list")
        return _as_row_sequence(_find_array(data, keys, path))


def _load_embedding_payload(path, role):
    with np.load(path, allow_pickle=True) as data:
        if "embeddings" not in data.files:
            raise ValueError(f"{path} does not contain embeddings.")
        payload = {
            "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
        }
        for key in ("trajectories", "vali_list", "query_trajectories", "gallery_trajectories"):
            if key in data.files:
                payload["trajectories"] = _as_row_sequence(data[key])
                break
        for key in ("coordinates", "vali_coor", "query_coordinates", "gallery_coordinates"):
            if key in data.files:
                payload["coordinates"] = _as_row_sequence(data[key])
                break
        if "source_ids" in data.files:
            payload["source_ids"] = np.asarray(data["source_ids"], dtype=np.int64).reshape(-1)
    if payload["embeddings"].ndim != 2:
        raise ValueError(f"{path} embeddings must be a 2-D matrix.")
    for key in ("trajectories", "coordinates", "source_ids"):
        if key in payload and len(payload[key]) != len(payload["embeddings"]):
            raise ValueError(
                f"{path} has {len(payload[key])} {key} rows but "
                f"{len(payload['embeddings'])} embeddings."
            )
    return payload


def build_truth(args):
    queries = _load_trajectories(args.query_file, "query")
    gallery = _load_trajectories(args.gallery_file, "gallery")
    ids, distances, report = compute_tp_ground_truth(
        queries,
        gallery,
        args.point_distance_file,
        top_k=args.top_k,
        max_queries=args.max_queries,
        num_threads=args.num_threads,
        device=args.device,
        gpu_batch_size=args.gpu_batch_size,
        cache_file=args.cache_file,
        force_recompute=args.force_recompute,
    )
    output_dir = osp.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        args.output_file,
        gallery_ids=ids,
        ids=ids,
        metric_distances=distances,
        distances=distances,
        query_count=np.asarray(len(ids), dtype=np.int64),
        gallery_count=np.asarray(len(gallery), dtype=np.int64),
        top_k=np.asarray(args.top_k, dtype=np.int64),
        query_file=np.asarray(osp.abspath(args.query_file)),
        gallery_file=np.asarray(osp.abspath(args.gallery_file)),
    )
    print(
        f"saved large-scale TP truth: {args.output_file}, "
        f"queries={report['queries']}, gallery={report['gallery_rows']}, "
        f"top_k={args.top_k}"
    )
    return args.output_file


def _load_truth(path):
    with np.load(path, allow_pickle=False) as data:
        id_key = "gallery_ids" if "gallery_ids" in data.files else "ids"
        distance_key = (
            "metric_distances" if "metric_distances" in data.files else "distances"
        )
        if id_key not in data.files or distance_key not in data.files:
            raise ValueError(
                f"{path} must contain gallery_ids/ids and metric_distances/distances."
            )
        ids = np.asarray(data[id_key], dtype=np.int64)
        distances = np.asarray(data[distance_key], dtype=np.float32)
    if ids.ndim != 2 or distances.shape != ids.shape:
        raise ValueError(
            f"Truth IDs and distances must have equal 2-D shapes; "
            f"got ids={ids.shape}, distances={distances.shape}."
        )
    return ids, distances


def assemble(args):
    query = _load_embedding_payload(args.query_embedding_file, "query")
    gallery = _load_embedding_payload(args.gallery_embedding_file, "gallery")
    gallery_ids, metric_distances = _load_truth(args.truth_file)
    query_count = len(query["embeddings"])
    gallery_count = len(gallery["embeddings"])
    if len(gallery_ids) != query_count:
        raise ValueError(
            f"Truth has {len(gallery_ids)} queries but query embeddings have {query_count}."
        )
    valid_gallery_ids = gallery_ids >= 0
    if np.any(gallery_ids[valid_gallery_ids] >= gallery_count):
        raise ValueError("Truth contains Gallery IDs outside the Gallery embedding rows.")
    if query["embeddings"].shape[1] != gallery["embeddings"].shape[1]:
        raise ValueError("Query and Gallery embedding dimensions must match.")

    embeddings = np.concatenate((query["embeddings"], gallery["embeddings"]), axis=0)
    # -1 marks a Query with no valid TP neighbor or unused Top-K padding. Keep
    # it invalid while remapping only real Gallery row IDs into union rows.
    labels = np.full(gallery_ids.shape, -1, dtype=np.int64)
    labels[valid_gallery_ids] = gallery_ids[valid_gallery_ids] + query_count
    distances = metric_distances

    payload = {
        "embeddings": embeddings,
        "labels": labels,
        "metric_distances": distances,
        "query_count": np.asarray(query_count, dtype=np.int64),
        "gallery_count": np.asarray(gallery_count, dtype=np.int64),
        "anchor_count": np.asarray(query_count, dtype=np.int64),
        "candidate_offset": np.asarray(query_count, dtype=np.int64),
    }
    if args.include_trajectories and "trajectories" in query and "trajectories" in gallery:
        payload["trajectories"] = np.concatenate(
            (query["trajectories"], gallery["trajectories"])
        )
    if args.include_trajectories and "coordinates" in query and "coordinates" in gallery:
        payload["coordinates"] = np.concatenate(
            (query["coordinates"], gallery["coordinates"])
        )
    if "source_ids" in query and "source_ids" in gallery:
        payload["source_ids"] = np.concatenate(
            (query["source_ids"], gallery["source_ids"])
        )
    else:
        payload["source_ids"] = np.arange(query_count + gallery_count, dtype=np.int64)

    output_dir = osp.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(args.output_file, **payload)
    print(
        f"saved large-scale head embeddings: {args.output_file}, "
        f"anchors={query_count}, candidates={gallery_count}, "
        f"labels={labels.shape}"
    )
    return args.output_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    truth = subparsers.add_parser("truth")
    truth.add_argument("--query-file", required=True)
    truth.add_argument("--gallery-file", required=True)
    truth.add_argument("--point-distance-file", required=True)
    truth.add_argument("--output-file", required=True)
    truth.add_argument("--cache-file", default=None)
    truth.add_argument("--top-k", type=int, default=256)
    truth.add_argument("--max-queries", type=int, default=0)
    truth.add_argument("--num-threads", type=int, default=64)
    truth.add_argument("--device", default="cpu")
    truth.add_argument("--gpu-batch-size", type=int, default=512)
    truth.add_argument("--force-recompute", action="store_true")
    truth.set_defaults(func=build_truth)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--query-embedding-file", required=True)
    assemble_parser.add_argument("--gallery-embedding-file", required=True)
    assemble_parser.add_argument("--truth-file", required=True)
    assemble_parser.add_argument("--output-file", required=True)
    assemble_parser.add_argument(
        "--include-trajectories",
        action="store_true",
        help="Include Query+Gallery trajectories/coordinates for adaptive weight fitting.",
    )
    assemble_parser.set_defaults(func=assemble)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
