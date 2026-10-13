import json
import os
import os.path as osp
from dataclasses import asdict, dataclass

import numpy as np

from ann_index import (
    ANNConfig,
    HierarchicalTrajectoryANN,
    build_trajectory_metadata,
    load_hierarchy_arrays,
)
from hash_global import HashGlobalIndex, hamming_distances


@dataclass
class HierHashFusionConfig:
    routing_mode: str = "legacy"
    pattern_level: str = "l3"
    path_max_len: int = 3
    l3_top_r: int = 4
    l3_expand_hops: int = 1
    path_top_p: int = 16
    l1_top_buckets: int = 4
    l2_top_regions: int = 16
    min_route_candidates: int = 2048
    max_route_candidates: int = 4096
    bucket_max_size: int = 512
    struct_hamming_candidates: int = 512
    global_hash_candidates: int = 1024
    max_rerank_candidates: int = 1536

    def validate(self):
        if self.routing_mode not in {"legacy", "hierarchical"}:
            raise ValueError(f"Unsupported routing_mode: {self.routing_mode}")
        if self.pattern_level not in {"l2", "l3"}:
            raise ValueError(f"Unsupported pattern_level: {self.pattern_level}")
        positive_fields = {
            "path_max_len": self.path_max_len,
            "l3_top_r": self.l3_top_r,
            "path_top_p": self.path_top_p,
            "l1_top_buckets": self.l1_top_buckets,
            "l2_top_regions": self.l2_top_regions,
            "min_route_candidates": self.min_route_candidates,
            "max_route_candidates": self.max_route_candidates,
            "bucket_max_size": self.bucket_max_size,
            "struct_hamming_candidates": self.struct_hamming_candidates,
            "global_hash_candidates": self.global_hash_candidates,
            "max_rerank_candidates": self.max_rerank_candidates,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"Fusion parameters must be positive: {invalid}")
        if self.l3_expand_hops < 0:
            raise ValueError("l3_expand_hops must be non-negative.")
        if self.max_route_candidates < self.min_route_candidates:
            raise ValueError("max_route_candidates must be >= min_route_candidates.")


def _topk_indices(distances, k):
    k = min(int(k), len(distances))
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k == len(distances):
        return np.argsort(distances, kind="stable")
    order = np.argpartition(distances, k - 1)[:k]
    return order[np.argsort(distances[order], kind="stable")]


def _relative_path(path, base_dir):
    return osp.relpath(osp.abspath(path), osp.abspath(base_dir))


class HierHashFusionIndex:
    """Parallel hierarchy and global-hash retrieval with continuous reranking."""

    def __init__(self, hash_index, config=None):
        self.hash_index = hash_index
        self.config = config or HierHashFusionConfig()
        self.config.validate()
        self.router = None
        self.hierarchy_dir = None
        self.metadata = None
        self.metadata_info = {}
        self.routing_bucket_to_ids = {}
        self.routing_bucket_l2 = {}
        self.routing_bucket_l1 = {}
        self.routing_pattern_count = 0
        self.l2_to_ids = {}
        self.l2_to_l1_buckets = {}
        self.l2_to_l3 = {}
        self.l3_to_l2 = {}
        self.l2_to_routing_buckets = {}
        self.l3_region_sizes = {}
        self._l3_rank_cache = {}

    def _router_config(self):
        return ANNConfig(
            metric=self.hash_index.config.metric,
            backend="exact",
            normalize=self.hash_index.config.normalize,
            pattern_level=self.config.pattern_level,
            path_max_len=self.config.path_max_len,
            l3_top_r=self.config.l3_top_r,
            l3_expand_hops=self.config.l3_expand_hops,
            path_top_p=self.config.path_top_p,
            l1_top_buckets=self.config.l1_top_buckets,
            candidates_per_bucket=1,
            global_fallback=0,
        )

    def build(self, trajectories, hierarchy_dir, metadata=None, metadata_info=None):
        trajectories = np.asarray(trajectories, dtype=object)
        if len(trajectories) != len(self.hash_index.embeddings):
            raise ValueError(
                f"Trajectory count ({len(trajectories)}) does not match hash index size "
                f"({len(self.hash_index.embeddings)})."
            )
        router = HierarchicalTrajectoryANN(self._router_config())
        router.trajectories = trajectories
        router.hierarchy_dir = osp.abspath(hierarchy_dir)
        router.hierarchy = load_hierarchy_arrays(hierarchy_dir)
        router.metadata = metadata or build_trajectory_metadata(
            trajectories, router.hierarchy, router.config
        )
        router._build_l3_neighbors()
        self.metadata = router.metadata
        if self.config.routing_mode == "hierarchical":
            self.routing_pattern_count = len({
                tuple(int(value) for value in path)
                for path in router.metadata["l2_paths"]
            })
            self._build_hierarchical_tables(router)
        else:
            router._build_pattern_tables()
            self.routing_pattern_count = len(router.pattern_to_ids)
        self.router = router
        self.hierarchy_dir = router.hierarchy_dir
        self.metadata_info = dict(metadata_info or {})
        return self

    def _build_hierarchical_tables(self, router):
        """Build non-cross-product L2 buckets with optional L1 descent."""
        l2_to_ids = {}
        l2_to_l1 = {}
        for traj_id, l2 in enumerate(self.metadata["l2_buckets"]):
            l2 = int(l2)
            l1 = int(self.metadata["l1_buckets"][traj_id])
            l2_to_ids.setdefault(l2, []).append(traj_id)
            l2_to_l1.setdefault(l2, {}).setdefault(l1, []).append(traj_id)

        parent_sets = {}
        node_to_l2 = np.asarray(router.hierarchy["node_to_l2"], dtype=np.int64)
        node_to_l3 = np.asarray(router.hierarchy["node_to_l3"], dtype=np.int64)
        for l2, l3 in zip(node_to_l2, node_to_l3):
            l2, l3 = int(l2), int(l3)
            if l2 >= 0 and l3 >= 0:
                parent_sets.setdefault(l2, set()).add(l3)

        self.l2_to_ids = {
            l2: np.asarray(ids, dtype=np.int64) for l2, ids in l2_to_ids.items()
        }
        self.l2_to_l1_buckets = {
            l2: {
                l1: np.asarray(ids, dtype=np.int64) for l1, ids in children.items()
            }
            for l2, children in l2_to_l1.items()
        }
        self.l2_to_l3 = {
            l2: set(parents) for l2, parents in parent_sets.items()
        }

        routing_bucket_to_ids = {}
        routing_bucket_l2 = {}
        routing_bucket_l1 = {}
        for l2, ids in self.l2_to_ids.items():
            if len(ids) <= self.config.bucket_max_size:
                key = f"l2:{l2}"
                routing_bucket_to_ids[key] = ids
                routing_bucket_l2[key] = l2
                routing_bucket_l1[key] = -1
            else:
                for l1, child_ids in self.l2_to_l1_buckets.get(l2, {}).items():
                    key = f"l1:{l2}:{l1}"
                    routing_bucket_to_ids[key] = child_ids
                    routing_bucket_l2[key] = l2
                    routing_bucket_l1[key] = l1
        self.routing_bucket_to_ids = routing_bucket_to_ids
        self.routing_bucket_l2 = routing_bucket_l2
        self.routing_bucket_l1 = routing_bucket_l1

        l3_to_l2 = {}
        for l2, parents in self.l2_to_l3.items():
            if l2 not in self.l2_to_ids:
                continue
            for l3 in parents:
                l3_to_l2.setdefault(l3, set()).add(l2)
        self.l3_to_l2 = {
            l3: sorted(children, key=lambda l2: (-len(self.l2_to_ids[l2]), l2))
            for l3, children in l3_to_l2.items()
        }
        self.l3_region_sizes = {
            l3: sum(len(self.l2_to_ids[l2]) for l2 in children)
            for l3, children in self.l3_to_l2.items()
        }
        self._l3_rank_cache = {}
        l2_to_routing_buckets = {}
        for key, l2 in self.routing_bucket_l2.items():
            l2_to_routing_buckets.setdefault(l2, []).append(key)
        self.l2_to_routing_buckets = {
            l2: sorted(
                keys,
                key=lambda key: (-len(self.routing_bucket_to_ids[key]), key),
            )
            for l2, keys in l2_to_routing_buckets.items()
        }

    def _rank_l3_regions(self, query_regions):
        known_l3 = set(self.l3_to_l2)
        seeds = {int(region) for region in query_regions if int(region) in known_l3}
        cache_key = tuple(sorted(seeds))
        if cache_key in self._l3_rank_cache:
            return self._l3_rank_cache[cache_key]
        distances = {region: 0 for region in seeds}
        frontier = set(seeds)
        hop = 0
        while frontier:
            next_frontier = set()
            for region in frontier:
                next_frontier.update(self.router.l3_neighbors.get(region, set()))
            next_frontier &= known_l3
            next_frontier -= set(distances)
            hop += 1
            for region in next_frontier:
                distances[region] = hop
            frontier = next_frontier

        disconnected_hop = max(distances.values(), default=-1) + 1
        for region in known_l3 - set(distances):
            distances[region] = disconnected_hop

        ranked = sorted(
            known_l3,
            key=lambda l3: (distances[l3], -self.l3_region_sizes[l3], l3),
        )
        result = (ranked, distances)
        self._l3_rank_cache[cache_key] = result
        return result

    def _hierarchical_route(self, query_meta, query_code=None):
        query_l3 = self.router._expand_l3_regions(query_meta["l3_regions"][0])
        query_l2 = int(query_meta["l2_buckets"][0])
        query_l1 = int(query_meta["l1_buckets"][0])
        ranked_l3, l3_distances = self._rank_l3_regions(query_l3)

        ranked_l2 = []
        l2_hops = {}
        seen_l2 = set()
        hops = sorted({l3_distances[l3] for l3 in ranked_l3})
        for hop in hops:
            layer_l3 = [l3 for l3 in ranked_l3 if l3_distances[l3] == hop]
            layer_children = []
            layer_overflow = []
            for l3 in layer_l3:
                children = sorted(
                    self.l3_to_l2.get(l3, []),
                    key=lambda l2: (
                        -int(l2 == query_l2),
                        -len(self.l2_to_ids[l2]),
                        l2,
                    ),
                )
                layer_children.extend(children[: self.config.l2_top_regions])
                layer_overflow.extend(children[self.config.l2_top_regions:])
            for l2 in layer_children + layer_overflow:
                if l2 not in seen_l2:
                    seen_l2.add(l2)
                    ranked_l2.append(l2)
                    l2_hops[l2] = hop
        for l2 in self.l2_to_ids:
            if l2 not in seen_l2:
                ranked_l2.append(l2)
                l2_hops[l2] = max(hops, default=0) + 1

        selected_buckets = []
        selected_ids = []
        selected_set = set()
        selected_l2 = []
        max_l3_hop = 0
        for l2 in ranked_l2:
            selected_l2.append(l2)
            max_l3_hop = max(max_l3_hop, l2_hops[l2])
            bucket_keys = list(self.l2_to_routing_buckets.get(l2, []))
            bucket_keys.sort(
                key=lambda key: (
                    -int(self.routing_bucket_l1[key] == query_l1),
                    -len(self.routing_bucket_to_ids[key]),
                    key,
                ),
            )
            for key in bucket_keys:
                selected_buckets.append(key)
                for traj_id in self.routing_bucket_to_ids[key]:
                    if int(traj_id) not in selected_set:
                        selected_set.add(int(traj_id))
                        selected_ids.append(int(traj_id))
                if len(selected_ids) >= self.config.min_route_candidates:
                    break
            if len(selected_ids) >= self.config.min_route_candidates:
                break

        route_ids = np.asarray(selected_ids, dtype=np.int64)
        route_pool_before_cap = int(len(route_ids))
        if query_code is not None and len(route_ids) > self.config.max_route_candidates:
            distances = hamming_distances(query_code, self.hash_index.codes[route_ids])
            route_ids = route_ids[_topk_indices(distances, self.config.max_route_candidates)]
        return route_ids, {
            "l2_count": int(len(selected_l2)),
            "l3_count": int(len({
                parent for l2 in selected_l2 for parent in self.l2_to_l3.get(l2, set())
            })),
            "max_l3_hop": int(max_l3_hop),
            "bucket_count": int(len(selected_buckets)),
            "route_pool_before_cap": route_pool_before_cap,
        }

    def route_candidate_ids(self, query_trajectory, query_code=None):
        if self.router is None:
            raise RuntimeError("The fusion index has not been built or loaded.")
        query_meta = self.router._metadata_for_query(query_trajectory)
        if self.config.routing_mode == "hierarchical":
            route_ids, details = self._hierarchical_route(query_meta, query_code=query_code)
            return route_ids, {
                "pattern_count": int(details["l2_count"]),
                **details,
            }
        pattern_keys = self.router._route_patterns(query_meta)
        bucket_keys = self.router._route_buckets(pattern_keys, query_meta)
        chunks = [
            self.router.bucket_to_ids[key]
            for key in bucket_keys
            if key in self.router.bucket_to_ids
        ]
        if not chunks:
            route_ids = np.empty((0,), dtype=np.int64)
        else:
            route_ids = np.unique(np.concatenate(chunks)).astype(np.int64, copy=False)
        return route_ids, {
            "pattern_count": int(len(pattern_keys)),
            "bucket_count": int(len(bucket_keys)),
        }

    def _structural_hamming_search(self, query_code, route_ids, exclude_ids=None,
                                   novelty_ids=None):
        route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
        exclude_ids = np.asarray(sorted(set(exclude_ids or [])), dtype=np.int64)
        if len(exclude_ids) and len(route_ids):
            route_ids = route_ids[~np.isin(route_ids, exclude_ids)]
        distances = hamming_distances(query_code, self.hash_index.codes[route_ids])
        raw_order = _topk_indices(distances, self.config.struct_hamming_candidates)
        raw_ids = route_ids[raw_order].astype(np.int64, copy=False)

        novelty_ids = np.asarray(sorted(set(novelty_ids or [])), dtype=np.int64)
        if len(novelty_ids) and len(route_ids):
            keep = ~np.isin(route_ids, novelty_ids)
            route_ids = route_ids[keep]
            distances = distances[keep]
        order = _topk_indices(distances, self.config.struct_hamming_candidates)
        return (
            route_ids[order].astype(np.int64, copy=False),
            distances[order].astype(np.int32, copy=False),
            raw_ids,
        )

    def query(self, query_embedding, query_trajectory, top_k=50, exclude_ids=None,
              return_details=False, query_code=None):
        if self.router is None:
            raise RuntimeError("The fusion index has not been built or loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if top_k > self.config.global_hash_candidates:
            raise ValueError(
                f"top_k={top_k} exceeds global_hash_candidates="
                f"{self.config.global_hash_candidates}."
            )
        if top_k > self.config.max_rerank_candidates:
            raise ValueError(
                f"top_k={top_k} exceeds max_rerank_candidates="
                f"{self.config.max_rerank_candidates}."
            )
        if query_code is None:
            query_code = self.hash_index.encode_query_code(query_embedding)
        query_code = np.asarray(query_code, dtype=np.uint8).reshape(-1)

        global_ids, _, global_details = self.hash_index.hamming_search(
            query_code,
            candidate_limit=self.config.global_hash_candidates,
            minimum_candidates=top_k,
            exclude_ids=exclude_ids,
        )
        route_ids, route_details = self.route_candidate_ids(
            query_trajectory, query_code=query_code
        )
        struct_ids, _, raw_struct_ids = self._structural_hamming_search(
            query_code,
            route_ids,
            exclude_ids=exclude_ids,
            novelty_ids=set(int(item) for item in global_ids),
        )
        overlap_count = int(
            len(np.intersect1d(raw_struct_ids, global_ids, assume_unique=True))
        )
        candidate_ids = np.concatenate([global_ids, struct_ids]).astype(
            np.int64, copy=False
        )
        union_count_before_cap = int(len(candidate_ids))
        candidate_ids_before_cap = candidate_ids
        if len(candidate_ids) > self.config.max_rerank_candidates:
            union_hamming = hamming_distances(query_code, self.hash_index.codes[candidate_ids])
            order = _topk_indices(union_hamming, self.config.max_rerank_candidates)
            candidate_ids = candidate_ids[order]

        ids, distances = self.hash_index.rerank_continuous(
            query_embedding, candidate_ids, top_k
        )
        if not return_details:
            return ids, distances
        return ids, distances, {
            "candidate_ids": candidate_ids,
            "union_before_cap_candidate_ids": candidate_ids_before_cap,
            "struct_candidate_ids": raw_struct_ids,
            "struct_unique_candidate_ids": struct_ids,
            "global_candidate_ids": global_ids,
            "route_pool_count": int(len(route_ids)),
            "struct_candidate_count": int(len(raw_struct_ids)),
            "struct_unique_candidate_count": int(len(struct_ids)),
            "global_candidate_count": int(len(global_ids)),
            "overlap_count": overlap_count,
            "union_count_before_cap": union_count_before_cap,
            "rerank_candidate_count": int(len(candidate_ids)),
            "global_mih_candidate_count": global_details["mih_candidate_count"],
            "global_used_full_scan": global_details["used_full_scan"],
            **route_details,
        }

    def save(self, index_dir, hash_index_dir):
        if self.router is None or self.metadata is None:
            raise RuntimeError("Cannot save an unbuilt fusion index.")
        os.makedirs(index_dir, exist_ok=True)
        np.savez_compressed(
            osp.join(index_dir, "routing_metadata.npz"),
            l3_regions=self.metadata["l3_regions"],
            l2_paths=self.metadata["l2_paths"],
            l2_buckets=self.metadata["l2_buckets"],
            l1_buckets=self.metadata["l1_buckets"],
        )
        payload = {
            "format_version": 1,
            "config": asdict(self.config),
            "hash_index_dir": _relative_path(hash_index_dir, index_dir),
            "hierarchy_dir": _relative_path(self.hierarchy_dir, index_dir),
            "num_vectors": int(len(self.hash_index.embeddings)),
            "metadata": self.metadata_info,
        }
        with open(osp.join(index_dir, "fusion_config.json"), "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def load(cls, index_dir, mmap=True, hash_index_dir=None, hierarchy_dir=None):
        with open(osp.join(index_dir, "fusion_config.json"), "r", encoding="utf-8") as file:
            payload = json.load(file)
        hash_index_dir = hash_index_dir or osp.join(index_dir, payload["hash_index_dir"])
        hierarchy_dir = hierarchy_dir or osp.join(index_dir, payload["hierarchy_dir"])
        hash_index = HashGlobalIndex.load(osp.abspath(hash_index_dir), mmap=mmap)
        metadata_file = np.load(osp.join(index_dir, "routing_metadata.npz"), allow_pickle=True)
        metadata = {
            "l3_regions": metadata_file["l3_regions"],
            "l2_paths": metadata_file["l2_paths"],
            "l2_buckets": metadata_file["l2_buckets"] if "l2_buckets" in metadata_file else None,
            "l1_buckets": metadata_file["l1_buckets"],
        }
        if payload["config"].get("routing_mode", "legacy") == "hierarchical":
            if metadata["l2_buckets"] is None:
                raise ValueError("Hierarchical fusion indexes require l2_buckets metadata; rebuild the index.")
        index = cls(hash_index, HierHashFusionConfig(**payload["config"]))
        empty_trajectories = np.empty(len(hash_index.embeddings), dtype=object)
        index.build(
            empty_trajectories,
            osp.abspath(hierarchy_dir),
            metadata=metadata,
            metadata_info=payload.get("metadata", {}),
        )
        return index
