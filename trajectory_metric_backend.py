"""Candidate-set trajectory metric backends for adaptive retrieval."""

from pathlib import Path

import numpy as np

from tp_retrieval_evaluation import (
    exact_tp_distances,
    flatten_trajectories,
)

try:
    import numba
except ImportError:  # pragma: no cover - optional acceleration dependency
    numba = None


METRIC_NAMES = (
    "embedding",
    "TP",
    "DITA",
    "discret_frechet",
    "LCRS",
    "NetERP",
)


def _valid_point_distance(point_distances, left, right):
    value = float(point_distances[int(left), int(right)])
    return value if value >= 0.0 else None


def _tp_distance(left, right, point_distances):
    directed_left = 0.0
    for left_node in left:
        minimum = np.inf
        for right_node in right:
            value = _valid_point_distance(point_distances, left_node, right_node)
            if value is None:
                return np.inf
            minimum = min(minimum, value)
        directed_left = max(directed_left, minimum)
    directed_right = 0.0
    for right_node in right:
        minimum = np.inf
        for left_node in left:
            value = _valid_point_distance(point_distances, right_node, left_node)
            if value is None:
                return np.inf
            minimum = min(minimum, value)
        directed_right = max(directed_right, minimum)
    return max(directed_left, directed_right)


def _dita_distance(left, right, point_distances):
    rows, columns = len(left), len(right)
    cost = np.empty((rows, columns), dtype=np.float64)
    value = _valid_point_distance(point_distances, left[0], right[0])
    if value is None:
        return np.inf
    cost[0, 0] = value
    for row in range(1, rows):
        value = _valid_point_distance(point_distances, left[row], right[0])
        if value is None:
            return np.inf
        cost[row, 0] = cost[row - 1, 0] + value
    for column in range(1, columns):
        value = _valid_point_distance(point_distances, left[0], right[column])
        if value is None:
            return np.inf
        cost[0, column] = cost[0, column - 1] + value
    for row in range(1, rows):
        for column in range(1, columns):
            value = _valid_point_distance(
                point_distances, left[row], right[column]
            )
            if value is None:
                return np.inf
            cost[row, column] = value + min(
                cost[row - 1, column - 1],
                cost[row, column - 1],
                cost[row - 1, column],
            )
    return cost[-1, -1]


def _frechet_distance(left, right, point_distances):
    rows, columns = len(left), len(right)
    cost = np.empty((rows, columns), dtype=np.float64)
    value = _valid_point_distance(point_distances, left[0], right[0])
    if value is None:
        return np.inf
    cost[0, 0] = value
    for row in range(1, rows):
        value = _valid_point_distance(point_distances, left[row], right[0])
        if value is None:
            return np.inf
        cost[row, 0] = max(value, cost[row - 1, 0])
    for column in range(1, columns):
        value = _valid_point_distance(point_distances, left[0], right[column])
        if value is None:
            return np.inf
        cost[0, column] = max(value, cost[0, column - 1])
    for row in range(1, rows):
        for column in range(1, columns):
            value = _valid_point_distance(
                point_distances, left[row], right[column]
            )
            if value is None:
                return np.inf
            cost[row, column] = max(
                value,
                min(
                    cost[row - 1, column - 1],
                    cost[row - 1, column],
                    cost[row, column - 1],
                ),
            )
    return cost[-1, -1]


def _lcrs_distance(left, right, edge_keys, node_count, max_trajectory_length):
    previous = np.zeros(len(right) + 1, dtype=np.int32)
    for left_node in left:
        current = np.zeros(len(right) + 1, dtype=np.int32)
        for column, right_node in enumerate(right, start=1):
            edge_key = int(left_node) * int(node_count) + int(right_node)
            position = int(np.searchsorted(edge_keys, edge_key))
            adjacent = position < len(edge_keys) and edge_keys[position] == edge_key
            if adjacent:
                current[column] = previous[column - 1] + 1
            else:
                current[column] = max(current[column - 1], previous[column])
        previous = current
    matched = int(previous[-1])
    if matched <= 0:
        return float(max_trajectory_length * 2)
    return (len(left) + len(right) - matched) / float(matched)


def _neterp_distance(left, right, point_distances, anchor_node):
    rows, columns = len(left), len(right)
    edit = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    for row in range(1, rows + 1):
        value = _valid_point_distance(point_distances, anchor_node, left[row - 1])
        if value is None:
            return np.inf
        edit[row, 0] = edit[row - 1, 0] + value
    for column in range(1, columns + 1):
        value = _valid_point_distance(
            point_distances, anchor_node, right[column - 1]
        )
        if value is None:
            return np.inf
        edit[0, column] = edit[0, column - 1] + value
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            delete = _valid_point_distance(
                point_distances, anchor_node, left[row - 1]
            )
            insert = _valid_point_distance(
                point_distances, anchor_node, right[column - 1]
            )
            replace = _valid_point_distance(
                point_distances, left[row - 1], right[column - 1]
            )
            if delete is None or insert is None or replace is None:
                return np.inf
            edit[row, column] = min(
                edit[row - 1, column] + delete,
                edit[row, column - 1] + insert,
                edit[row - 1, column - 1] + replace,
            )
    return edit[-1, -1]


if numba is not None:

    @numba.njit(fastmath=True, cache=False)
    def _dita_distance_numba(query, gallery, point_distances):
        rows, columns = len(query), len(gallery)
        cost = np.empty((rows, columns), dtype=np.float64)
        value = point_distances[query[0], gallery[0]]
        if value < 0.0:
            return np.inf
        cost[0, 0] = value
        for row in range(1, rows):
            value = point_distances[query[row], gallery[0]]
            if value < 0.0:
                return np.inf
            cost[row, 0] = cost[row - 1, 0] + value
        for column in range(1, columns):
            value = point_distances[query[0], gallery[column]]
            if value < 0.0:
                return np.inf
            cost[0, column] = cost[0, column - 1] + value
        for row in range(1, rows):
            for column in range(1, columns):
                value = point_distances[query[row], gallery[column]]
                if value < 0.0:
                    return np.inf
                cost[row, column] = value + min(
                    cost[row - 1, column - 1],
                    cost[row, column - 1],
                    cost[row - 1, column],
                )
        return cost[rows - 1, columns - 1]


    @numba.njit(fastmath=True, cache=False)
    def _frechet_distance_numba(query, gallery, point_distances):
        rows, columns = len(query), len(gallery)
        cost = np.empty((rows, columns), dtype=np.float64)
        value = point_distances[query[0], gallery[0]]
        if value < 0.0:
            return np.inf
        cost[0, 0] = value
        for row in range(1, rows):
            value = point_distances[query[row], gallery[0]]
            if value < 0.0:
                return np.inf
            cost[row, 0] = max(value, cost[row - 1, 0])
        for column in range(1, columns):
            value = point_distances[query[0], gallery[column]]
            if value < 0.0:
                return np.inf
            cost[0, column] = max(value, cost[0, column - 1])
        for row in range(1, rows):
            for column in range(1, columns):
                value = point_distances[query[row], gallery[column]]
                if value < 0.0:
                    return np.inf
                cost[row, column] = max(
                    value,
                    min(
                        cost[row - 1, column - 1],
                        cost[row - 1, column],
                        cost[row, column - 1],
                    ),
                )
        return cost[rows - 1, columns - 1]


    @numba.njit(fastmath=True, cache=False)
    def _lcrs_distance_numba(
        query, gallery, edge_keys, node_count, max_trajectory_length
    ):
        previous = np.zeros(len(gallery) + 1, dtype=np.int32)
        for query_node in query:
            current = np.zeros(len(gallery) + 1, dtype=np.int32)
            for column in range(1, len(gallery) + 1):
                edge_key = int(query_node) * int(node_count) + int(gallery[column - 1])
                position = np.searchsorted(edge_keys, edge_key)
                if position < len(edge_keys) and edge_keys[position] == edge_key:
                    current[column] = previous[column - 1] + 1
                else:
                    current[column] = max(current[column - 1], previous[column])
            previous = current
        matched = int(previous[-1])
        if matched <= 0:
            return float(max_trajectory_length * 2)
        return (len(query) + len(gallery) - matched) / float(matched)


    @numba.njit(fastmath=True, cache=False)
    def _neterp_distance_numba(query, gallery, point_distances, anchor_node):
        rows, columns = len(query), len(gallery)
        edit = np.zeros((rows + 1, columns + 1), dtype=np.float64)
        for row in range(1, rows + 1):
            value = point_distances[anchor_node, query[row - 1]]
            if value < 0.0:
                return np.inf
            edit[row, 0] = edit[row - 1, 0] + value
        for column in range(1, columns + 1):
            value = point_distances[anchor_node, gallery[column - 1]]
            if value < 0.0:
                return np.inf
            edit[0, column] = edit[0, column - 1] + value
        for row in range(1, rows + 1):
            delete = point_distances[anchor_node, query[row - 1]]
            for column in range(1, columns + 1):
                insert = point_distances[anchor_node, gallery[column - 1]]
                replace = point_distances[query[row - 1], gallery[column - 1]]
                if delete < 0.0 or insert < 0.0 or replace < 0.0:
                    return np.inf
                edit[row, column] = min(
                    edit[row - 1, column] + delete,
                    edit[row, column - 1] + insert,
                    edit[row - 1, column - 1] + replace,
                )
        return edit[rows, columns]


    @numba.njit(parallel=True, fastmath=True, cache=False)
    def _metric_distances_numba(
        metric_code,
        query,
        gallery_nodes,
        gallery_offsets,
        point_distances,
        edge_keys,
        node_count,
        max_trajectory_length,
        anchor_node,
    ):
        result = np.full(len(gallery_offsets) - 1, np.inf, dtype=np.float32)
        for gallery_id in numba.prange(len(result)):
            start = gallery_offsets[gallery_id]
            end = gallery_offsets[gallery_id + 1]
            gallery = gallery_nodes[start:end]
            if metric_code == 1:
                value = _dita_distance_numba(query, gallery, point_distances)
            elif metric_code == 2:
                value = _frechet_distance_numba(query, gallery, point_distances)
            elif metric_code == 3:
                value = _lcrs_distance_numba(
                    query, gallery, edge_keys, node_count, max_trajectory_length
                )
            else:
                value = _neterp_distance_numba(
                    query, gallery, point_distances, anchor_node
                )
            result[gallery_id] = np.float32(value)
        return result


def load_lcrs_edge_keys(edge_file, node_count):
    import pandas as pd

    edge_file = Path(edge_file)
    if not edge_file.is_file():
        raise FileNotFoundError(f"LCRS edge file not found: {edge_file}")
    frame = pd.read_csv(edge_file)
    required = {"s_node", "e_node"}
    if not required.issubset(frame.columns):
        raise ValueError(f"LCRS edge file must contain {sorted(required)}.")
    source = frame["s_node"].to_numpy(dtype=np.int64)
    target = frame["e_node"].to_numpy(dtype=np.int64)
    diagonal = np.arange(int(node_count), dtype=np.int64)
    return np.unique(
        np.concatenate(
            (
                diagonal * node_count + diagonal,
                source * node_count + target,
                target * node_count + source,
            )
        )
    )


class TrajectoryMetricBackend:
    def __init__(
        self,
        name,
        *,
        point_distance_file=None,
        edge_file=None,
        node_count=None,
        max_trajectory_length=None,
        neterp_anchor_node=None,
        num_threads=0,
    ):
        canonical = str(name)
        lookup = {item.lower(): item for item in METRIC_NAMES}
        if canonical.lower() not in lookup:
            raise ValueError(f"Unsupported rerank metric: {name}")
        self.name = lookup[canonical.lower()]
        self.point_distances = None
        self.edge_keys = None
        self.node_count = int(node_count or 0)
        self.max_trajectory_length = int(max_trajectory_length or 0)
        self.neterp_anchor_node = neterp_anchor_node
        self.tp_num_threads = int(num_threads or 0)
        if self.name in {"TP", "DITA", "discret_frechet", "NetERP"}:
            if not point_distance_file:
                raise ValueError(f"{self.name} reranking requires point_distance_file.")
            self.point_distances = np.load(point_distance_file, mmap_mode="r")
            if self.point_distances.ndim != 2:
                raise ValueError("Point-distance matrix must be two-dimensional.")
            self.node_count = int(self.point_distances.shape[0])
            if self.tp_num_threads > 0 and numba is not None:
                try:
                    numba.set_num_threads(self.tp_num_threads)
                except ImportError:
                    pass
        if self.name == "LCRS":
            if self.node_count <= 0 or self.max_trajectory_length <= 0:
                raise ValueError(
                    "LCRS reranking requires node_count and max_trajectory_length."
                )
            self.edge_keys = load_lcrs_edge_keys(edge_file, self.node_count)
        if self.name == "NetERP" and self.neterp_anchor_node is None:
            raise ValueError("NetERP reranking requires neterp_anchor_node.")

    def close(self):
        point_distances = self.point_distances
        if isinstance(point_distances, np.memmap):
            mmap = getattr(point_distances, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.point_distances = None

    def distances(self, query_trajectory, gallery_trajectories):
        query = np.asarray(query_trajectory, dtype=np.int64).reshape(-1)
        if not len(query):
            raise ValueError("Query trajectory cannot be empty.")
        gallery_trajectories = np.asarray(gallery_trajectories, dtype=object)
        if gallery_trajectories.ndim == 0:
            gallery_trajectories = gallery_trajectories.reshape(1)
        if self.name == "TP":
            # Keep the online TP definition identical to the large-scale TP
            # table path, but run the candidate batch through its Numba
            # parallel kernel instead of Python nested loops.
            gallery_nodes, gallery_offsets = flatten_trajectories(
                gallery_trajectories
            )
            return exact_tp_distances(
                query,
                gallery_nodes,
                gallery_offsets,
                self.point_distances,
            )
        if numba is not None and self.name in {
            "DITA", "discret_frechet", "LCRS", "NetERP"
        }:
            gallery_nodes, gallery_offsets = flatten_trajectories(
                gallery_trajectories
            )
            metric_code = {
                "DITA": 1,
                "discret_frechet": 2,
                "LCRS": 3,
                "NetERP": 4,
            }[self.name]
            return _metric_distances_numba(
                metric_code,
                query,
                gallery_nodes,
                gallery_offsets,
                self.point_distances
                if self.point_distances is not None
                else np.empty((1, 1), dtype=np.float32),
                self.edge_keys
                if self.edge_keys is not None
                else np.empty((0,), dtype=np.int64),
                int(self.node_count),
                int(self.max_trajectory_length),
                int(self.neterp_anchor_node or 0),
            )
        distances = np.full(len(gallery_trajectories), np.inf, dtype=np.float32)
        for row_id, gallery_trajectory in enumerate(gallery_trajectories):
            gallery = np.asarray(gallery_trajectory, dtype=np.int64).reshape(-1)
            if not len(gallery):
                continue
            if self.name == "TP":
                value = _tp_distance(query, gallery, self.point_distances)
            elif self.name == "DITA":
                value = _dita_distance(query, gallery, self.point_distances)
            elif self.name == "discret_frechet":
                value = _frechet_distance(query, gallery, self.point_distances)
            elif self.name == "LCRS":
                value = _lcrs_distance(
                    query,
                    gallery,
                    self.edge_keys,
                    self.node_count,
                    self.max_trajectory_length,
                )
            elif self.name == "NetERP":
                value = _neterp_distance(
                    query,
                    gallery,
                    self.point_distances,
                    int(self.neterp_anchor_node),
                )
            else:
                raise RuntimeError("Embedding distances are computed by the code index.")
            distances[row_id] = np.float32(value)
        return distances
