import json
import math
import os
import os.path as osp
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np


def _try_import_hnswlib():
    try:
        import hnswlib

        return hnswlib
    except ImportError:
        return None


@dataclass
class ANNConfig:
    metric: str = "l2"
    backend: str = "auto"
    normalize: bool = False
    pattern_level: str = "l3"
    path_max_len: int = 3
    l3_top_r: int = 4
    l3_expand_hops: int = 1
    path_top_p: int = 16
    l1_top_buckets: int = 4
    candidates_per_bucket: int = 128
    global_fallback: int = 1000
    exact_bucket_threshold: int = 256
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 80


def _as_float32(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {vectors.shape}.")
    return vectors


def _normalize(vectors):
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(denom, 1e-12)


def _l2_distances(query, matrix):
    diff = matrix - query.reshape(1, -1)
    return np.einsum("ij,ij->i", diff, diff)


def _cosine_distances(query, matrix):
    query = query.reshape(1, -1)
    return 1.0 - np.sum(matrix * query, axis=1)


def _topk_exact(query, matrix, ids, k, metric):
    if len(ids) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    k = min(k, len(ids))
    distances = _cosine_distances(query, matrix) if metric == "cosine" else _l2_distances(query, matrix)
    if k == len(ids):
        order = np.argsort(distances)
    else:
        order = np.argpartition(distances, k - 1)[:k]
        order = order[np.argsort(distances[order])]
    return ids[order], distances[order].astype(np.float32)


def _compress_regions(regions):
    out = []
    prev = None
    for item in regions:
        item = int(item)
        if item < 0:
            continue
        if item != prev:
            out.append(item)
            prev = item
    return out


def _fixed_length_signature(sequence, max_len):
    sequence = list(sequence)
    if len(sequence) <= max_len:
        return tuple(sequence)
    positions = np.linspace(0, len(sequence) - 1, max_len)
    return tuple(sequence[int(round(pos))] for pos in positions)


def _dominant_regions(regions, top_r):
    if len(regions) == 0:
        return tuple()
    counts = Counter(int(x) for x in regions if int(x) >= 0)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return tuple(region for region, _ in ranked[:top_r])


def _sequence_lcs_len(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for item_long in long:
        cur = [0]
        for j, item_short in enumerate(short, start=1):
            if item_long == item_short:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _path_similarity(a, b):
    if not a or not b:
        return 0.0
    lcs = _sequence_lcs_len(a, b)
    return lcs / max(len(a), len(b), 1)


def _pattern_key(pattern):
    return ",".join(str(int(x)) for x in pattern)


def _parse_pattern_key(key):
    if key == "":
        return tuple()
    return tuple(int(x) for x in key.split(","))


def _bucket_key(pattern_key, l1_bucket):
    return f"{pattern_key}|{int(l1_bucket)}"


def _split_bucket_key(bucket_key):
    pattern_key, l1_bucket = bucket_key.rsplit("|", 1)
    return pattern_key, int(l1_bucket)


def load_hierarchy_arrays(hierarchy_dir):
    required = ["node_to_l1.npy", "node_to_l2.npy", "node_to_l3.npy"]
    missing = [name for name in required if not osp.exists(osp.join(hierarchy_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing hierarchy files in {hierarchy_dir}: {missing}. "
            "Run generate_louvain_hierarchy.py before building ANN indexes."
        )
    hierarchy = {
        "node_to_l1": np.load(osp.join(hierarchy_dir, "node_to_l1.npy")),
        "node_to_l2": np.load(osp.join(hierarchy_dir, "node_to_l2.npy")),
        "node_to_l3": np.load(osp.join(hierarchy_dir, "node_to_l3.npy")),
    }
    edge_path = osp.join(hierarchy_dir, "l3_edge_index.npy")
    hierarchy["l3_edges"] = np.load(edge_path) if osp.exists(edge_path) else np.zeros((0, 2), dtype=np.int64)
    return hierarchy


def _safe_map_nodes(nodes, assignment):
    nodes = np.asarray(nodes, dtype=np.int64)
    valid = (nodes >= 0) & (nodes < len(assignment))
    mapped = np.full(nodes.shape, -1, dtype=np.int64)
    mapped[valid] = assignment[nodes[valid]]
    return mapped


def build_trajectory_metadata(trajectories, hierarchy, config):
    l3_regions = []
    path_patterns = []
    l2_buckets = []
    l1_buckets = []

    for traj in trajectories:
        l1 = _safe_map_nodes(traj, hierarchy["node_to_l1"])
        l2 = _safe_map_nodes(traj, hierarchy["node_to_l2"])
        l3 = _safe_map_nodes(traj, hierarchy["node_to_l3"])

        l3_regions.append(_dominant_regions(l3, config.l3_top_r))
        if config.pattern_level == "l2":
            pattern_regions = l2
        elif config.pattern_level == "l3":
            pattern_regions = l3
        else:
            raise ValueError(f"Unknown pattern_level: {config.pattern_level}")
        compressed_pattern = _compress_regions(pattern_regions)
        path_patterns.append(_fixed_length_signature(compressed_pattern, config.path_max_len))
        dominant_l2 = _dominant_regions(l2, 1)
        dominant_l1 = _dominant_regions(l1, 1)
        l2_buckets.append(dominant_l2[0] if dominant_l2 else -1)
        l1_buckets.append(dominant_l1[0] if dominant_l1 else -1)

    return {
        "l3_regions": np.asarray(l3_regions, dtype=object),
        "l2_paths": np.asarray(path_patterns, dtype=object),
        "l2_buckets": np.asarray(l2_buckets, dtype=np.int64),
        "l1_buckets": np.asarray(l1_buckets, dtype=np.int64),
    }


class LocalVectorIndex:
    def __init__(self, ids, vectors, config):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.vectors = _as_float32(vectors)
        self.config = config
        self.backend = "exact"
        self.index = None
        self._build()

    def _build(self):
        if len(self.ids) <= self.config.exact_bucket_threshold:
            return
        requested = self.config.backend.lower()
        if requested not in {"auto", "hnsw", "exact"}:
            raise ValueError(f"Unknown ANN backend: {self.config.backend}")
        if requested == "exact":
            return

        hnswlib = _try_import_hnswlib()
        if hnswlib is None:
            return

        space = "cosine" if self.config.metric == "cosine" else "l2"
        index = hnswlib.Index(space=space, dim=self.vectors.shape[1])
        index.init_index(
            max_elements=len(self.ids),
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_m,
        )
        index.add_items(self.vectors, np.arange(len(self.ids)))
        index.set_ef(self.config.hnsw_ef_search)
        self.index = index
        self.backend = "hnsw"

    def query(self, query_vector, k):
        if len(self.ids) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        k = min(k, len(self.ids))
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if self.index is not None:
            labels, distances = self.index.knn_query(query_vector.reshape(1, -1), k=k)
            local_ids = labels[0].astype(np.int64)
            return self.ids[local_ids], distances[0].astype(np.float32)
        return _topk_exact(query_vector, self.vectors, self.ids, k, self.config.metric)


class HierarchicalTrajectoryANN:
    def __init__(self, config=None):
        self.config = config or ANNConfig()
        self.embeddings = None
        self.trajectories = None
        self.metadata = None
        self.hierarchy_dir = None
        self.hierarchy = None
        self.l3_neighbors = defaultdict(set)
        self.pattern_to_ids = {}
        self.pattern_to_l3 = {}
        self.pattern_to_buckets = {}
        self.bucket_to_ids = {}
        self.bucket_indexes = {}
        self.l3_to_patterns = defaultdict(set)
        self.global_index = None

    def build(self, embeddings, trajectories, hierarchy_dir, metadata=None):
        embeddings = _as_float32(embeddings)
        if self.config.normalize or self.config.metric == "cosine":
            embeddings = _normalize(embeddings).astype(np.float32)
        if len(embeddings) != len(trajectories):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) does not match trajectory count ({len(trajectories)})."
            )

        self.embeddings = embeddings
        self.trajectories = np.asarray(trajectories, dtype=object)
        self.hierarchy_dir = hierarchy_dir
        self.hierarchy = load_hierarchy_arrays(hierarchy_dir)
        self.metadata = metadata or build_trajectory_metadata(self.trajectories, self.hierarchy, self.config)
        self._build_l3_neighbors()
        self._build_pattern_tables()
        self._build_vector_indexes()
        return self

    def _build_l3_neighbors(self):
        self.l3_neighbors = defaultdict(set)
        edges = self.hierarchy.get("l3_edges", np.zeros((0, 2), dtype=np.int64))
        for src, dst in edges:
            src, dst = int(src), int(dst)
            self.l3_neighbors[src].add(dst)
            self.l3_neighbors[dst].add(src)

    def _build_pattern_tables(self):
        pattern_to_ids = defaultdict(list)
        bucket_to_ids = defaultdict(list)
        pattern_to_l3 = defaultdict(set)
        pattern_to_buckets = defaultdict(set)
        l3_to_patterns = defaultdict(set)

        for traj_id, path in enumerate(self.metadata["l2_paths"]):
            key = _pattern_key(path)
            bucket = _bucket_key(key, int(self.metadata["l1_buckets"][traj_id]))
            pattern_to_ids[key].append(traj_id)
            bucket_to_ids[bucket].append(traj_id)
            pattern_to_buckets[key].add(bucket)
            for region in self.metadata["l3_regions"][traj_id]:
                region = int(region)
                pattern_to_l3[key].add(region)
                l3_to_patterns[region].add(key)

        self.pattern_to_ids = {key: np.asarray(ids, dtype=np.int64) for key, ids in pattern_to_ids.items()}
        self.bucket_to_ids = {key: np.asarray(ids, dtype=np.int64) for key, ids in bucket_to_ids.items()}
        self.pattern_to_l3 = {key: regions for key, regions in pattern_to_l3.items()}
        self.pattern_to_buckets = {
            key: sorted(buckets, key=lambda item: (-len(self.bucket_to_ids[item]), item))
            for key, buckets in pattern_to_buckets.items()
        }
        self.l3_to_patterns = l3_to_patterns

    def _build_vector_indexes(self):
        self.bucket_indexes = {}
        start = perf_counter()
        for key, ids in self.bucket_to_ids.items():
            self.bucket_indexes[key] = LocalVectorIndex(ids, self.embeddings[ids], self.config)
        all_ids = np.arange(len(self.embeddings), dtype=np.int64)
        self.global_index = LocalVectorIndex(all_ids, self.embeddings, self.config)
        self.build_seconds = perf_counter() - start

    def _expand_l3_regions(self, regions):
        frontier = set(int(x) for x in regions)
        expanded = set(frontier)
        for _ in range(max(0, self.config.l3_expand_hops)):
            next_frontier = set()
            for region in frontier:
                next_frontier.update(self.l3_neighbors.get(region, set()))
            next_frontier -= expanded
            expanded.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return expanded

    def _metadata_for_query(self, trajectory):
        return build_trajectory_metadata([trajectory], self.hierarchy, self.config)

    def _route_patterns(self, query_meta):
        query_regions = self._expand_l3_regions(query_meta["l3_regions"][0])
        query_path = tuple(query_meta["l2_paths"][0])

        candidate_keys = set()
        for region in query_regions:
            candidate_keys.update(self.l3_to_patterns.get(int(region), set()))

        if not candidate_keys:
            candidate_keys = set(self.pattern_to_ids.keys())

        scored = []
        for key in candidate_keys:
            pattern = _parse_pattern_key(key)
            pattern_regions = self.pattern_to_l3.get(key, set())
            region_overlap = len(query_regions & pattern_regions) / max(len(query_regions), 1)
            path_score = _path_similarity(query_path, pattern)
            bucket_bonus = math.log1p(len(self.pattern_to_ids[key])) * 1e-3
            scored.append((region_overlap + path_score + bucket_bonus, key))

        scored.sort(reverse=True)
        return [key for _, key in scored[: self.config.path_top_p]]

    def _route_buckets(self, pattern_keys, query_meta):
        query_l1_bucket = int(query_meta["l1_buckets"][0])
        scored = []
        for rank, pattern_key in enumerate(pattern_keys):
            pattern_priority = len(pattern_keys) - rank
            for key in self.pattern_to_buckets.get(pattern_key, []):
                _, l1_bucket = _split_bucket_key(key)
                l1_bonus = 1.0 if l1_bucket == query_l1_bucket else 0.0
                size_bonus = math.log1p(len(self.bucket_to_ids[key])) * 1e-3
                scored.append((pattern_priority + l1_bonus + size_bonus, key))

        scored.sort(reverse=True)
        limit = max(self.config.path_top_p, self.config.path_top_p * max(1, self.config.l1_top_buckets))
        return [key for _, key in scored[:limit]]

    def query(self, query_embedding, query_trajectory, top_k=50, exclude_ids=None):
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if self.config.normalize or self.config.metric == "cosine":
            query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)

        query_meta = self._metadata_for_query(query_trajectory)
        pattern_keys = self._route_patterns(query_meta)
        bucket_keys = self._route_buckets(pattern_keys, query_meta)
        exclude_ids = set(exclude_ids or [])

        merged = {}
        per_bucket_k = max(top_k, self.config.candidates_per_bucket)
        for key in bucket_keys:
            ids, distances = self.bucket_indexes[key].query(query_embedding, per_bucket_k)
            for traj_id, distance in zip(ids, distances):
                traj_id = int(traj_id)
                if traj_id in exclude_ids:
                    continue
                distance = float(distance)
                if traj_id not in merged or distance < merged[traj_id]:
                    merged[traj_id] = distance

        if self.config.global_fallback > 0:
            ids, distances = self.global_index.query(query_embedding, max(top_k, self.config.global_fallback))
            for traj_id, distance in zip(ids, distances):
                traj_id = int(traj_id)
                if traj_id in exclude_ids:
                    continue
                distance = float(distance)
                if traj_id not in merged or distance < merged[traj_id]:
                    merged[traj_id] = distance

        ranked = sorted(merged.items(), key=lambda x: x[1])[:top_k]
        return np.asarray([x[0] for x in ranked], dtype=np.int64), np.asarray([x[1] for x in ranked], dtype=np.float32)

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        np.save(osp.join(index_dir, "embeddings.npy"), self.embeddings)
        np.save(osp.join(index_dir, "trajectories.npy"), self.trajectories, allow_pickle=True)
        np.savez_compressed(
            osp.join(index_dir, "metadata.npz"),
            l3_regions=self.metadata["l3_regions"],
            l2_paths=self.metadata["l2_paths"],
            l1_buckets=self.metadata["l1_buckets"],
        )
        payload = {
            "config": asdict(self.config),
            "hierarchy_dir": self.hierarchy_dir,
            "num_vectors": int(len(self.embeddings)),
            "dim": int(self.embeddings.shape[1]),
            "num_patterns": int(len(self.pattern_to_ids)),
            "num_buckets": int(len(self.bucket_to_ids)),
            "build_seconds": float(getattr(self, "build_seconds", 0.0)),
        }
        with open(osp.join(index_dir, "ann_config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, index_dir):
        with open(osp.join(index_dir, "ann_config.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = ANNConfig(**payload["config"])
        ann = cls(config)
        embeddings = np.load(osp.join(index_dir, "embeddings.npy"))
        trajectories = np.load(osp.join(index_dir, "trajectories.npy"), allow_pickle=True)
        meta = np.load(osp.join(index_dir, "metadata.npz"), allow_pickle=True)
        metadata = {
            "l3_regions": meta["l3_regions"],
            "l2_paths": meta["l2_paths"],
            "l1_buckets": meta["l1_buckets"],
        }
        return ann.build(embeddings, trajectories, payload["hierarchy_dir"], metadata=metadata)
