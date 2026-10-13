"""Exact trajectory ground truth and retrieval metrics for large-scale experiments."""

import hashlib
import os
import os.path as osp
from time import perf_counter

import numpy as np


TP_CACHE_VERSION = 1
METRIC_CACHE_VERSION = 1
SUPPORTED_METRICS = ("TP", "DITA", "discret_frechet", "LCRS", "NetERP")


def flatten_trajectories(trajectories):
    # Preserve the first dimension as the trajectory-row dimension. Calling
    # reshape(-1) here turns a regular [rows, length] candidate batch into
    # scalar rows, which changes TP semantics for equal-length trajectories.
    trajectories = np.asarray(trajectories, dtype=object)
    if trajectories.ndim == 0:
        trajectories = trajectories.reshape(1)
    offsets = np.zeros(len(trajectories) + 1, dtype=np.int64)
    rows = []
    for row_id, trajectory in enumerate(trajectories):
        row = np.asarray(trajectory, dtype=np.int64).reshape(-1)
        if not len(row):
            raise ValueError(f"Trajectory row {row_id} is empty.")
        rows.append(row)
        offsets[row_id + 1] = offsets[row_id] + len(row)
    nodes = np.concatenate(rows) if rows else np.empty((0,), dtype=np.int64)
    return nodes, offsets


def _tp_distances_python(query, gallery_nodes, gallery_offsets, point_distances):
    distances = np.full(len(gallery_offsets) - 1, np.inf, dtype=np.float32)
    for gallery_id in range(len(distances)):
        gallery = gallery_nodes[
            gallery_offsets[gallery_id] : gallery_offsets[gallery_id + 1]
        ]
        pairwise = np.asarray(point_distances[np.ix_(query, gallery)])
        if np.any(pairwise < 0):
            continue
        distances[gallery_id] = max(
            float(pairwise.min(axis=1).max()),
            float(pairwise.min(axis=0).max()),
        )
    return distances


try:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _tp_distances_numba(query, gallery_nodes, gallery_offsets, point_distances):
        gallery_count = len(gallery_offsets) - 1
        result = np.empty(gallery_count, dtype=np.float32)
        for gallery_id in numba.prange(gallery_count):
            start = gallery_offsets[gallery_id]
            end = gallery_offsets[gallery_id + 1]
            invalid = False
            directed_query = 0.0
            for query_pos in range(len(query)):
                minimum = np.inf
                for gallery_pos in range(start, end):
                    distance = point_distances[
                        query[query_pos], gallery_nodes[gallery_pos]
                    ]
                    if distance < 0:
                        invalid = True
                        break
                    if distance < minimum:
                        minimum = distance
                if invalid:
                    break
                if minimum > directed_query:
                    directed_query = minimum

            directed_gallery = 0.0
            if not invalid:
                for gallery_pos in range(start, end):
                    minimum = np.inf
                    for query_pos in range(len(query)):
                        distance = point_distances[
                            gallery_nodes[gallery_pos], query[query_pos]
                        ]
                        if distance < 0:
                            invalid = True
                            break
                        if distance < minimum:
                            minimum = distance
                    if invalid:
                        break
                    if minimum > directed_gallery:
                        directed_gallery = minimum
            result[gallery_id] = (
                np.inf
                if invalid
                else max(directed_query, directed_gallery)
            )
        return result

except ImportError:
    numba = None
    _tp_distances_numba = None


def exact_tp_distances(query, gallery_nodes, gallery_offsets, point_distances):
    query = np.asarray(query, dtype=np.int64).reshape(-1)
    if not len(query):
        raise ValueError("A TP Query trajectory cannot be empty.")
    if _tp_distances_numba is not None:
        return _tp_distances_numba(
            query, gallery_nodes, gallery_offsets, point_distances
        )
    return _tp_distances_python(
        query, gallery_nodes, gallery_offsets, point_distances
    )


def _tp_distances_cuda(
    query,
    gallery_nodes,
    gallery_offsets,
    point_distances,
    device,
    gallery_batch_size=512,
):
    """Compute TP distances in bounded GPU batches.

    The point-distance matrix remains a CPU mmap.  Only the selected
    ``query x gallery_batch`` blocks are copied to CUDA, so this works even
    when the full matrix is much larger than GPU memory.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--tp-device cuda requires PyTorch.") from exc
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA device {device} is not available.")
    target = torch.device(device)
    query = np.asarray(query, dtype=np.int64).reshape(-1)
    gallery_nodes = np.asarray(gallery_nodes, dtype=np.int64).reshape(-1)
    gallery_offsets = np.asarray(gallery_offsets, dtype=np.int64).reshape(-1)
    gallery_count = len(gallery_offsets) - 1
    result = np.full(gallery_count, np.inf, dtype=np.float32)
    batch_size = max(int(gallery_batch_size), 1)
    query_len = len(query)
    if query_len == 0:
        return result

    for batch_start in range(0, gallery_count, batch_size):
        batch_stop = min(batch_start + batch_size, gallery_count)
        starts = gallery_offsets[batch_start:batch_stop]
        stops = gallery_offsets[batch_start + 1:batch_stop + 1]
        max_len = int(np.max(stops - starts)) if len(starts) else 0
        if max_len <= 0:
            continue
        node_block = np.zeros((batch_stop - batch_start, max_len), dtype=np.int64)
        valid_mask = np.zeros(node_block.shape, dtype=bool)
        for row, (start, stop) in enumerate(zip(starts, stops)):
            length = int(stop - start)
            node_block[row, :length] = gallery_nodes[start:stop]
            valid_mask[row, :length] = True

        # NumPy performs the mmap-backed gather on CPU; the compact block is
        # then transferred once to CUDA for both directed Hausdorff terms.
        block = np.asarray(point_distances[np.ix_(query, node_block.reshape(-1))])
        block = block.reshape(query_len, batch_stop - batch_start, max_len)
        invalid = (~valid_mask[None, :, :]) | (block < 0)
        values = torch.as_tensor(block, dtype=torch.float32, device=target)
        mask = torch.as_tensor(invalid, dtype=torch.bool, device=target)
        values = values.masked_fill(mask, float("inf"))
        valid_tensor = torch.as_tensor(valid_mask[None, :, :], dtype=torch.bool, device=target)
        invalid_rows = torch.any(mask & valid_tensor, dim=(0, 2))
        directed_query = torch.min(values, dim=2).values.max(dim=0).values
        gallery_min = torch.min(values, dim=0).values
        gallery_min = gallery_min.masked_fill(~valid_tensor[0], float("-inf"))
        directed_gallery = gallery_min.max(dim=1).values
        distances = torch.maximum(directed_query, directed_gallery)
        distances = torch.where(
            invalid_rows,
            torch.full_like(distances, float("inf")),
            distances,
        )
        result[batch_start:batch_stop] = distances.detach().cpu().numpy()
        del values, mask, block
    return result


def _stable_topk_finite(distances, top_k):
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    finite_ids = np.flatnonzero(np.isfinite(distances))
    top_k = min(int(top_k), len(finite_ids))
    if top_k <= 0:
        return np.empty((0,), dtype=np.int64)
    finite_distances = distances[finite_ids]
    if top_k < len(finite_ids):
        threshold = np.partition(finite_distances, top_k - 1)[top_k - 1]
        below = finite_ids[finite_distances < threshold]
        equal = finite_ids[finite_distances == threshold]
        chosen = np.concatenate([below, equal[: top_k - len(below)]])
    else:
        chosen = finite_ids
    order = np.lexsort((chosen, distances[chosen]))
    return chosen[order].astype(np.int64, copy=False)


def compute_tp_distance_table(
    query_trajectories,
    gallery_trajectories,
    point_distance_file,
    *,
    max_queries=200,
    num_threads=0,
    device="cpu",
    gpu_batch_size=512,
    cache_file=None,
    force_recompute=False,
):
    device = str(device or "cpu")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("TP device must be 'cpu' or 'cuda[:index]'.")
    query_trajectories = np.asarray(query_trajectories, dtype=object).reshape(-1)
    gallery_trajectories = np.asarray(gallery_trajectories, dtype=object).reshape(-1)
    query_count = len(query_trajectories)
    if max_queries and int(max_queries) > 0:
        query_count = min(query_count, int(max_queries))
    if query_count <= 0 or len(gallery_trajectories) <= 0:
        raise ValueError("TP distance table requires non-empty Query and Gallery.")
    query_trajectories = query_trajectories[:query_count]

    query_nodes, query_offsets = flatten_trajectories(query_trajectories)
    gallery_nodes, gallery_offsets = flatten_trajectories(gallery_trajectories)
    fingerprint = _trajectory_fingerprint(
        query_nodes, query_offsets, gallery_nodes, gallery_offsets
    )
    meta_file = f"{cache_file}.meta.npz" if cache_file else None
    if cache_file and not force_recompute and osp.isfile(cache_file) and osp.isfile(meta_file):
        try:
            with np.load(meta_file, allow_pickle=False) as meta:
                valid = (
                    int(meta["version"].item()) == TP_CACHE_VERSION
                    and str(meta["fingerprint"].item()) == fingerprint
                    and int(meta["query_count"].item()) == query_count
                    and int(meta["gallery_count"].item()) == len(gallery_trajectories)
                )
            if valid:
                distances = np.load(cache_file, mmap_mode="r")
                return distances, {
                    "enabled": True,
                    "queries": int(query_count),
                    "gallery_rows": int(len(gallery_trajectories)),
                    "compute_seconds": 0.0,
                    "cache_reused": True,
                    "cache_file": str(cache_file),
                    "device": device,
                    "gpu_batch_size": int(gpu_batch_size),
                    "fingerprint": fingerprint,
                }
        except (KeyError, OSError, ValueError):
            pass

    if not osp.isfile(point_distance_file):
        raise FileNotFoundError(
            f"TP point-distance matrix not found: {point_distance_file}"
        )
    point_distances = np.load(point_distance_file, mmap_mode="r")
    if point_distances.ndim != 2 or point_distances.shape[0] != point_distances.shape[1]:
        raise ValueError("TP point-distance matrix must be square.")
    if len(query_nodes) and (
        int(query_nodes.min()) < 0 or int(query_nodes.max()) >= point_distances.shape[0]
    ):
        raise ValueError("A Query trajectory node is outside the point-distance matrix.")
    if len(gallery_nodes) and (
        int(gallery_nodes.min()) < 0
        or int(gallery_nodes.max()) >= point_distances.shape[0]
    ):
        raise ValueError("A Gallery trajectory node is outside the point-distance matrix.")
    if numba is not None and int(num_threads) > 0 and device == "cpu":
        numba.set_num_threads(int(num_threads))

    if cache_file:
        out_dir = osp.dirname(cache_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        table = np.lib.format.open_memmap(
            cache_file,
            mode="w+",
            dtype=np.float32,
            shape=(query_count, len(gallery_trajectories)),
        )
    else:
        table = np.empty((query_count, len(gallery_trajectories)), dtype=np.float32)

    try:
        from tqdm import tqdm

        iterator = tqdm(range(query_count), desc="exact TP distance table")
    except ImportError:
        iterator = range(query_count)

    start_time = perf_counter()
    for query_id in iterator:
        query = query_nodes[query_offsets[query_id] : query_offsets[query_id + 1]]
        if device.startswith("cuda"):
            distances = _tp_distances_cuda(
                query,
                gallery_nodes,
                gallery_offsets,
                point_distances,
                device=device,
                gallery_batch_size=gpu_batch_size,
            )
        else:
            distances = exact_tp_distances(
                query, gallery_nodes, gallery_offsets, point_distances
            )
        table[query_id, :] = distances
    if hasattr(table, "flush"):
        table.flush()
    compute_seconds = perf_counter() - start_time

    if cache_file:
        np.savez_compressed(
            meta_file,
            version=np.asarray(TP_CACHE_VERSION, dtype=np.int64),
            fingerprint=np.asarray(fingerprint),
            query_count=np.asarray(query_count, dtype=np.int64),
            gallery_count=np.asarray(len(gallery_trajectories), dtype=np.int64),
        )
        table = np.load(cache_file, mmap_mode="r")
    return table, {
        "enabled": True,
        "queries": int(query_count),
        "gallery_rows": int(len(gallery_trajectories)),
        "compute_seconds": float(compute_seconds),
        "cache_reused": False,
        "cache_file": str(cache_file) if cache_file else None,
        "point_distance_file": str(point_distance_file),
        "device": device,
        "gpu_batch_size": int(gpu_batch_size),
        "fingerprint": fingerprint,
    }


def tp_ground_truth_from_distance_table(distance_table, selected_columns, *, top_k=50):
    distance_table = np.asarray(distance_table)
    selected_columns = np.asarray(selected_columns, dtype=np.int64).reshape(-1)
    query_count = int(distance_table.shape[0])
    width = min(int(top_k), len(selected_columns))
    truth_ids = np.full((query_count, width), -1, dtype=np.int64)
    truth_distances = np.full((query_count, width), np.inf, dtype=np.float32)
    invalid_pair_count = 0
    queries_without_valid_neighbor = 0
    for query_id in range(query_count):
        distances = np.asarray(
            distance_table[query_id, selected_columns], dtype=np.float32
        )
        invalid_pair_count += int(np.count_nonzero(~np.isfinite(distances)))
        local_ids = _stable_topk_finite(distances, width)
        if not len(local_ids):
            queries_without_valid_neighbor += 1
            continue
        truth_ids[query_id, : len(local_ids)] = local_ids
        truth_distances[query_id, : len(local_ids)] = distances[local_ids]
    return truth_ids, truth_distances, {
        "invalid_pair_count": int(invalid_pair_count),
        "queries_without_valid_neighbor": int(queries_without_valid_neighbor),
    }


def close_mmap_array(array):
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _trajectory_fingerprint(query_nodes, query_offsets, gallery_nodes, gallery_offsets):
    digest = hashlib.sha256()
    for array in (query_nodes, query_offsets, gallery_nodes, gallery_offsets):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _metric_fingerprint(
    metric,
    query_nodes,
    query_offsets,
    gallery_nodes,
    gallery_offsets,
    *,
    point_distance_file=None,
    edge_file=None,
    node_count=0,
    max_trajectory_length=0,
    neterp_anchor_node=None,
):
    digest = hashlib.sha256()
    digest.update(str(metric).encode("utf-8"))
    digest.update(
        _trajectory_fingerprint(
            query_nodes, query_offsets, gallery_nodes, gallery_offsets
        ).encode("ascii")
    )
    for value in (
        point_distance_file,
        edge_file,
        int(node_count),
        int(max_trajectory_length),
        neterp_anchor_node,
    ):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_cached_ground_truth(path, fingerprint, query_count, gallery_count, top_k):
    if not path or not osp.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            valid = (
                int(data["version"].item()) == TP_CACHE_VERSION
                and str(data["fingerprint"].item()) == fingerprint
                and int(data["query_count"].item()) == query_count
                and int(data["gallery_count"].item()) == gallery_count
                and int(data["top_k"].item()) == top_k
            )
            if not valid:
                return None
            return {
                "ids": np.asarray(data["ids"], dtype=np.int64),
                "distances": np.asarray(data["distances"], dtype=np.float32),
                "invalid_pair_count": int(data["invalid_pair_count"].item()),
                "queries_without_valid_neighbor": int(
                    data["queries_without_valid_neighbor"].item()
                ),
            }
    except (KeyError, OSError, ValueError):
        return None


def compute_tp_ground_truth(
    query_trajectories,
    gallery_trajectories,
    point_distance_file,
    *,
    top_k=50,
    max_queries=200,
    num_threads=0,
    device="cpu",
    gpu_batch_size=512,
    cache_file=None,
    force_recompute=False,
):
    top_k = int(top_k)
    device = str(device or "cpu")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("TP device must be 'cpu' or 'cuda[:index]'.")
    if top_k <= 0:
        raise ValueError("TP top_k must be positive.")
    query_trajectories = np.asarray(query_trajectories, dtype=object).reshape(-1)
    gallery_trajectories = np.asarray(gallery_trajectories, dtype=object).reshape(-1)
    query_count = len(query_trajectories)
    if max_queries and int(max_queries) > 0:
        query_count = min(query_count, int(max_queries))
    if query_count <= 0 or len(gallery_trajectories) <= 0:
        raise ValueError("TP evaluation requires non-empty Query and Gallery trajectories.")
    query_trajectories = query_trajectories[:query_count]

    query_nodes, query_offsets = flatten_trajectories(query_trajectories)
    gallery_nodes, gallery_offsets = flatten_trajectories(gallery_trajectories)
    fingerprint = _trajectory_fingerprint(
        query_nodes, query_offsets, gallery_nodes, gallery_offsets
    )
    cached = None if force_recompute else _load_cached_ground_truth(
        cache_file, fingerprint, query_count, len(gallery_trajectories), top_k
    )
    if cached is not None:
        report = {
            "enabled": True,
            "queries": int(query_count),
            "gallery_rows": int(len(gallery_trajectories)),
            "top_k": int(top_k),
            "compute_seconds": 0.0,
            "cache_reused": True,
            "cache_file": str(cache_file),
            "device": str(device or "cpu"),
            "gpu_batch_size": int(gpu_batch_size),
            "fingerprint": fingerprint,
            "invalid_pair_count": cached["invalid_pair_count"],
            "queries_without_valid_neighbor": cached[
                "queries_without_valid_neighbor"
            ],
        }
        return cached["ids"], cached["distances"], report

    if not osp.isfile(point_distance_file):
        raise FileNotFoundError(
            f"TP point-distance matrix not found: {point_distance_file}"
        )
    point_distances = np.load(point_distance_file, mmap_mode="r")
    if point_distances.ndim != 2 or point_distances.shape[0] != point_distances.shape[1]:
        raise ValueError("TP point-distance matrix must be square.")
    if len(query_nodes) and (
        int(query_nodes.min()) < 0 or int(query_nodes.max()) >= point_distances.shape[0]
    ):
        raise ValueError("A Query trajectory node is outside the point-distance matrix.")
    if len(gallery_nodes) and (
        int(gallery_nodes.min()) < 0
        or int(gallery_nodes.max()) >= point_distances.shape[0]
    ):
        raise ValueError("A Gallery trajectory node is outside the point-distance matrix.")
    if numba is not None and int(num_threads) > 0 and device == "cpu":
        numba.set_num_threads(int(num_threads))

    width = min(top_k, len(gallery_trajectories))
    truth_ids = np.full((query_count, width), -1, dtype=np.int64)
    truth_distances = np.full((query_count, width), np.inf, dtype=np.float32)
    invalid_pair_count = 0
    queries_without_valid_neighbor = 0
    try:
        from tqdm import tqdm

        iterator = tqdm(range(query_count), desc="exact TP ground truth")
    except ImportError:
        iterator = range(query_count)

    start_time = perf_counter()
    for query_id in iterator:
        query = query_nodes[query_offsets[query_id] : query_offsets[query_id + 1]]
        if device.startswith("cuda"):
            distances = _tp_distances_cuda(
                query,
                gallery_nodes,
                gallery_offsets,
                point_distances,
                device=device,
                gallery_batch_size=gpu_batch_size,
            )
        else:
            distances = exact_tp_distances(
                query, gallery_nodes, gallery_offsets, point_distances
            )
        invalid_pair_count += int(np.count_nonzero(~np.isfinite(distances)))
        ids = _stable_topk_finite(distances, width)
        if not len(ids):
            queries_without_valid_neighbor += 1
            continue
        truth_ids[query_id, : len(ids)] = ids
        truth_distances[query_id, : len(ids)] = distances[ids]
    compute_seconds = perf_counter() - start_time

    if cache_file:
        out_dir = osp.dirname(cache_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            cache_file,
            version=np.asarray(TP_CACHE_VERSION, dtype=np.int64),
            fingerprint=np.asarray(fingerprint),
            query_count=np.asarray(query_count, dtype=np.int64),
            gallery_count=np.asarray(len(gallery_trajectories), dtype=np.int64),
            top_k=np.asarray(top_k, dtype=np.int64),
            ids=truth_ids,
            distances=truth_distances,
            invalid_pair_count=np.asarray(invalid_pair_count, dtype=np.int64),
            queries_without_valid_neighbor=np.asarray(
                queries_without_valid_neighbor, dtype=np.int64
            ),
        )
    report = {
        "enabled": True,
        "queries": int(query_count),
        "gallery_rows": int(len(gallery_trajectories)),
        "top_k": int(top_k),
        "compute_seconds": float(compute_seconds),
        "cache_reused": False,
        "cache_file": str(cache_file) if cache_file else None,
        "point_distance_file": str(point_distance_file),
        "device": device,
        "gpu_batch_size": int(gpu_batch_size),
        "fingerprint": fingerprint,
        "invalid_pair_count": int(invalid_pair_count),
        "queries_without_valid_neighbor": int(queries_without_valid_neighbor),
    }
    return truth_ids, truth_distances, report


def _make_metric_backend(
    metric,
    trajectories,
    *,
    point_distance_file=None,
    edge_file=None,
    node_count=0,
    max_trajectory_length=0,
    neterp_anchor_node=None,
    num_threads=0,
):
    from trajectory_metric_backend import TrajectoryMetricBackend

    trajectories = np.asarray(trajectories, dtype=object).reshape(-1)
    if not len(trajectories):
        raise ValueError("Metric evaluation requires non-empty trajectories.")
    if int(node_count) <= 0:
        node_count = max(
            int(np.max(np.concatenate([np.asarray(item, dtype=np.int64) for item in trajectories]))) + 1,
            1,
        )
    if int(max_trajectory_length) <= 0:
        max_trajectory_length = max(len(item) for item in trajectories)
    return TrajectoryMetricBackend(
        metric,
        point_distance_file=point_distance_file,
        edge_file=edge_file,
        node_count=node_count,
        max_trajectory_length=max_trajectory_length,
        neterp_anchor_node=neterp_anchor_node,
        num_threads=num_threads,
    )


def _metric_acceleration_name():
    try:
        import trajectory_metric_backend

        if trajectory_metric_backend.numba is not None:
            return "numba-parallel-gallery"
    except ImportError:
        pass
    return "python-fallback"


def compute_metric_distance_table(
    query_trajectories,
    gallery_trajectories,
    metric,
    *,
    point_distance_file=None,
    edge_file=None,
    node_count=0,
    max_trajectory_length=0,
    neterp_anchor_node=None,
    max_queries=2000,
    gallery_batch_size=1024,
    num_threads=0,
    cache_file=None,
    force_recompute=False,
):
    """Write a Query x Gallery distance table for any supported metric.

    Non-TP metrics are evaluated in Gallery chunks through the same backend
    used by online reranking. The output is a float32 NumPy memmap, so the
    dense table does not need to fit in RAM.
    """
    metric = str(metric)
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected {SUPPORTED_METRICS}.")
    if metric == "TP":
        from tp_retrieval_evaluation import compute_tp_distance_table

        table, report = compute_tp_distance_table(
            query_trajectories,
            gallery_trajectories,
            point_distance_file,
            max_queries=max_queries,
            num_threads=num_threads,
            device="cpu",
            cache_file=cache_file,
            force_recompute=force_recompute,
        )
        report["metric"] = metric
        return table, report

    if int(gallery_batch_size) <= 0:
        raise ValueError("gallery_batch_size must be positive.")
    query_trajectories = np.asarray(query_trajectories, dtype=object).reshape(-1)
    gallery_trajectories = np.asarray(gallery_trajectories, dtype=object).reshape(-1)
    query_count = len(query_trajectories)
    if int(max_queries) > 0:
        query_count = min(query_count, int(max_queries))
    if query_count <= 0 or len(gallery_trajectories) <= 0:
        raise ValueError("Metric distance table requires non-empty Query and Gallery.")
    query_trajectories = query_trajectories[:query_count]
    query_nodes, query_offsets = flatten_trajectories(query_trajectories)
    gallery_nodes, gallery_offsets = flatten_trajectories(gallery_trajectories)
    fingerprint = _metric_fingerprint(
        metric,
        query_nodes,
        query_offsets,
        gallery_nodes,
        gallery_offsets,
        point_distance_file=point_distance_file,
        edge_file=edge_file,
        node_count=node_count,
        max_trajectory_length=max_trajectory_length,
        neterp_anchor_node=neterp_anchor_node,
    )
    meta_file = f"{cache_file}.meta.npz" if cache_file else None
    if cache_file and not force_recompute and osp.isfile(cache_file) and osp.isfile(meta_file):
        try:
            with np.load(meta_file, allow_pickle=False) as meta:
                valid = (
                    int(meta["version"].item()) == METRIC_CACHE_VERSION
                    and str(meta["metric"].item()) == metric
                    and str(meta["fingerprint"].item()) == fingerprint
                    and int(meta["query_count"].item()) == query_count
                    and int(meta["gallery_count"].item()) == len(gallery_trajectories)
                )
            if valid:
                table = np.load(cache_file, mmap_mode="r")
                return table, {
                    "enabled": True,
                    "metric": metric,
                    "queries": int(query_count),
                    "gallery_rows": int(len(gallery_trajectories)),
                    "cache_reused": True,
                    "cache_file": str(cache_file),
                    "gallery_batch_size": int(gallery_batch_size),
                    "acceleration": _metric_acceleration_name(),
                    "fingerprint": fingerprint,
                }
        except (KeyError, OSError, ValueError):
            pass

    if cache_file:
        out_dir = osp.dirname(cache_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        table = np.lib.format.open_memmap(
            cache_file,
            mode="w+",
            dtype=np.float32,
            shape=(query_count, len(gallery_trajectories)),
        )
    else:
        table = np.empty((query_count, len(gallery_trajectories)), dtype=np.float32)

    backend = _make_metric_backend(
        metric,
        np.concatenate((query_trajectories, gallery_trajectories)),
        point_distance_file=point_distance_file,
        edge_file=edge_file,
        node_count=node_count,
        max_trajectory_length=max_trajectory_length,
        neterp_anchor_node=neterp_anchor_node,
        num_threads=num_threads,
    )
    start_time = perf_counter()
    try:
        from tqdm import tqdm

        iterator = tqdm(range(query_count), desc=f"exact {metric} distance table")
    except ImportError:
        iterator = range(query_count)
    try:
        for query_id in iterator:
            query = query_trajectories[query_id]
            for start in range(0, len(gallery_trajectories), int(gallery_batch_size)):
                stop = min(start + int(gallery_batch_size), len(gallery_trajectories))
                table[query_id, start:stop] = backend.distances(
                    query, gallery_trajectories[start:stop]
                )
            if hasattr(table, "flush"):
                table.flush()
    finally:
        backend.close()
    compute_seconds = perf_counter() - start_time
    if hasattr(table, "flush"):
        table.flush()
    if cache_file:
        np.savez_compressed(
            meta_file,
            version=np.asarray(METRIC_CACHE_VERSION, dtype=np.int64),
            metric=np.asarray(metric),
            fingerprint=np.asarray(fingerprint),
            query_count=np.asarray(query_count, dtype=np.int64),
            gallery_count=np.asarray(len(gallery_trajectories), dtype=np.int64),
        )
        table = np.load(cache_file, mmap_mode="r")
    return table, {
        "enabled": True,
        "metric": metric,
        "queries": int(query_count),
        "gallery_rows": int(len(gallery_trajectories)),
        "compute_seconds": float(compute_seconds),
        "cache_reused": False,
        "cache_file": str(cache_file) if cache_file else None,
        "gallery_batch_size": int(gallery_batch_size),
        "acceleration": _metric_acceleration_name(),
        "fingerprint": fingerprint,
    }


def compute_metric_ground_truth(
    query_trajectories,
    gallery_trajectories,
    metric,
    *,
    point_distance_file=None,
    edge_file=None,
    node_count=0,
    max_trajectory_length=0,
    neterp_anchor_node=None,
    top_k=50,
    max_queries=2000,
    gallery_batch_size=1024,
    num_threads=0,
    cache_file=None,
    force_recompute=False,
):
    """Compute exact Top-K truth without materializing a dense table."""
    metric = str(metric)
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected {SUPPORTED_METRICS}.")
    if metric == "TP":
        return compute_tp_ground_truth(
            query_trajectories,
            gallery_trajectories,
            point_distance_file,
            top_k=top_k,
            max_queries=max_queries,
            num_threads=num_threads,
            cache_file=cache_file,
            force_recompute=force_recompute,
        )
    if int(top_k) <= 0 or int(gallery_batch_size) <= 0:
        raise ValueError("top_k and gallery_batch_size must be positive.")
    query_trajectories = np.asarray(query_trajectories, dtype=object).reshape(-1)
    gallery_trajectories = np.asarray(gallery_trajectories, dtype=object).reshape(-1)
    query_count = len(query_trajectories)
    if int(max_queries) > 0:
        query_count = min(query_count, int(max_queries))
    if query_count <= 0 or len(gallery_trajectories) <= 0:
        raise ValueError("Metric evaluation requires non-empty Query and Gallery.")
    query_trajectories = query_trajectories[:query_count]
    width = min(int(top_k), len(gallery_trajectories))
    query_nodes, query_offsets = flatten_trajectories(query_trajectories)
    gallery_nodes, gallery_offsets = flatten_trajectories(gallery_trajectories)
    fingerprint = _metric_fingerprint(
        metric,
        query_nodes,
        query_offsets,
        gallery_nodes,
        gallery_offsets,
        point_distance_file=point_distance_file,
        edge_file=edge_file,
        node_count=node_count,
        max_trajectory_length=max_trajectory_length,
        neterp_anchor_node=neterp_anchor_node,
    )
    if cache_file and not force_recompute and osp.isfile(cache_file):
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                valid = (
                    int(data["version"].item()) == METRIC_CACHE_VERSION
                    and str(data["metric"].item()) == metric
                    and str(data["fingerprint"].item()) == fingerprint
                    and int(data["query_count"].item()) == query_count
                    and int(data["gallery_count"].item()) == len(gallery_trajectories)
                    and int(data["top_k"].item()) == int(top_k)
                )
                if valid:
                    return (
                        np.asarray(data["ids"], dtype=np.int64),
                        np.asarray(data["distances"], dtype=np.float32),
                        {
                            "enabled": True,
                            "metric": metric,
                            "queries": int(query_count),
                            "gallery_rows": int(len(gallery_trajectories)),
                            "top_k": int(top_k),
                            "cache_reused": True,
                            "cache_file": str(cache_file),
                            "gallery_batch_size": int(gallery_batch_size),
                            "acceleration": _metric_acceleration_name(),
                            "fingerprint": fingerprint,
                            "invalid_pair_count": int(data["invalid_pair_count"].item()),
                            "queries_without_valid_neighbor": int(
                                data["queries_without_valid_neighbor"].item()
                            ),
                        },
                    )
        except (KeyError, OSError, ValueError):
            pass

    backend = _make_metric_backend(
        metric,
        np.concatenate((query_trajectories, gallery_trajectories)),
        point_distance_file=point_distance_file,
        edge_file=edge_file,
        node_count=node_count,
        max_trajectory_length=max_trajectory_length,
        neterp_anchor_node=neterp_anchor_node,
        num_threads=num_threads,
    )
    truth_ids = np.full((query_count, width), -1, dtype=np.int64)
    truth_distances = np.full((query_count, width), np.inf, dtype=np.float32)
    invalid_pair_count = 0
    queries_without_valid_neighbor = 0
    start_time = perf_counter()
    try:
        from tqdm import tqdm

        query_iterator = tqdm(
            range(query_count), desc=f"exact {metric} ground truth"
        )
    except ImportError:
        query_iterator = range(query_count)
    try:
        for query_id in query_iterator:
            query = query_trajectories[query_id]
            best_ids = np.empty((0,), dtype=np.int64)
            best_distances = np.empty((0,), dtype=np.float32)
            for start in range(0, len(gallery_trajectories), int(gallery_batch_size)):
                stop = min(start + int(gallery_batch_size), len(gallery_trajectories))
                distances = np.asarray(
                    backend.distances(query, gallery_trajectories[start:stop]),
                    dtype=np.float32,
                )
                invalid_pair_count += int(np.count_nonzero(~np.isfinite(distances)))
                finite = np.flatnonzero(np.isfinite(distances))
                if not len(finite):
                    continue
                candidate_ids = finite.astype(np.int64) + int(start)
                candidate_distances = distances[finite]
                all_ids = np.concatenate((best_ids, candidate_ids))
                all_distances = np.concatenate((best_distances, candidate_distances))
                order = np.lexsort((all_ids, all_distances))[:width]
                best_ids = all_ids[order]
                best_distances = all_distances[order]
            if not len(best_ids):
                queries_without_valid_neighbor += 1
                continue
            truth_ids[query_id, : len(best_ids)] = best_ids
            truth_distances[query_id, : len(best_distances)] = best_distances
    finally:
        backend.close()
    compute_seconds = perf_counter() - start_time
    if cache_file:
        out_dir = osp.dirname(cache_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            cache_file,
            version=np.asarray(METRIC_CACHE_VERSION, dtype=np.int64),
            metric=np.asarray(metric),
            fingerprint=np.asarray(fingerprint),
            query_count=np.asarray(query_count, dtype=np.int64),
            gallery_count=np.asarray(len(gallery_trajectories), dtype=np.int64),
            top_k=np.asarray(top_k, dtype=np.int64),
            ids=truth_ids,
            distances=truth_distances,
            invalid_pair_count=np.asarray(invalid_pair_count, dtype=np.int64),
            queries_without_valid_neighbor=np.asarray(
                queries_without_valid_neighbor, dtype=np.int64
            ),
        )
    return truth_ids, truth_distances, {
        "enabled": True,
        "metric": metric,
        "queries": int(query_count),
        "gallery_rows": int(len(gallery_trajectories)),
        "top_k": int(top_k),
        "compute_seconds": float(compute_seconds),
        "cache_reused": False,
        "cache_file": str(cache_file) if cache_file else None,
        "gallery_batch_size": int(gallery_batch_size),
        "acceleration": _metric_acceleration_name(),
        "fingerprint": fingerprint,
        "invalid_pair_count": int(invalid_pair_count),
        "queries_without_valid_neighbor": int(queries_without_valid_neighbor),
    }


def tp_retrieval_metrics(predictions, truth_ids):
    total = min(len(predictions), len(truth_ids))
    if total <= 0:
        return {"queries": 0}
    hr10 = []
    hr50 = []
    r10_at_50 = []
    r1_at_1 = []
    r1_at_10 = []
    r1_at_50 = []
    valid_queries = 0
    for prediction, truth in zip(predictions[:total], truth_ids[:total]):
        prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
        truth = np.asarray(truth, dtype=np.int64).reshape(-1)
        truth = truth[truth >= 0]
        if not len(truth):
            continue
        valid_queries += 1
        truth10 = set(truth[:10].tolist())
        truth50 = set(truth[:50].tolist())
        pred10 = set(prediction[:10].tolist())
        pred50 = set(prediction[:50].tolist())
        hr10.append(len(pred10 & truth10) / max(len(truth10), 1))
        hr50.append(len(pred50 & truth50) / max(len(truth50), 1))
        r10_at_50.append(len(pred50 & truth10) / max(len(truth10), 1))
        top1 = int(truth[0])
        r1_at_1.append(float(top1 in prediction[:1]))
        r1_at_10.append(float(top1 in prediction[:10]))
        r1_at_50.append(float(top1 in prediction[:50]))
    values = {
        "hr_at_10": hr10,
        "hr_at_50": hr50,
        "r10_at_50": r10_at_50,
        "r1_at_1": r1_at_1,
        "r1_at_10": r1_at_10,
        "r1_at_50": r1_at_50,
    }
    return {
        "queries": int(valid_queries),
        **{
            key: float(np.mean(items)) if items else 0.0
            for key, items in values.items()
        },
    }


def resolve_tp_cache_file(configured_path, scale_dir, gallery_size, multi_scale=False):
    if not configured_path:
        return osp.join(scale_dir, "tp_ground_truth.npz")
    if "{gallery_size}" in configured_path:
        return configured_path.format(gallery_size=int(gallery_size))
    if multi_scale:
        root, extension = osp.splitext(configured_path)
        extension = extension or ".npz"
        return f"{root}_gallery_{int(gallery_size)}{extension}"
    return configured_path


def add_tp_arguments(parser):
    parser.add_argument(
        "--tp-eval",
        action="store_true",
        help="Compute/load exact TP Top-K truth and add TP retrieval metrics.",
    )
    parser.add_argument(
        "--tp-point-distance-file",
        default="ground_truth/beijing/Point_dis_matrix.npy",
    )
    parser.add_argument("--tp-max-queries", type=int, default=200)
    parser.add_argument("--tp-top-k", type=int, default=50)
    parser.add_argument("--tp-num-threads", type=int, default=0)
    parser.add_argument(
        "--tp-device",
        default="cpu",
        help="Device for exact TP ground-truth computation: cpu or cuda:N.",
    )
    parser.add_argument(
        "--tp-gpu-batch-size",
        type=int,
        default=512,
        help="Gallery trajectories processed per GPU TP batch.",
    )
    parser.add_argument(
        "--tp-ground-truth-file",
        default=None,
        help=(
            "Optional TP cache path. Use {gallery_size} for a scale sweep; without "
            "it, a scale suffix is added automatically when multiple sizes run."
        ),
    )
    parser.add_argument("--tp-force-recompute", action="store_true")
