"""Large-scale TopoRouteHash evaluation with multi-resolution path postings."""

import argparse
import gc
import json
import os
import os.path as osp
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ann_index import load_hierarchy_arrays
from dual_metric_fusion import load_weight_payload
from hier_hash_large_scale import (
    _candidate_source_stats,
    _directory_size,
    _distribution_stats,
    _json_dump,
    _topk_indices,
    gallery_composition,
    load_embedding_payload,
    parse_gallery_sizes,
    remap_gallery_source_ids,
    select_gallery_rows,
    source_retrieval_metrics,
)
from tp_retrieval_evaluation import (
    add_tp_arguments,
    close_mmap_array,
    compute_tp_distance_table,
    compute_tp_ground_truth,
    resolve_tp_cache_file,
    tp_ground_truth_from_distance_table,
    tp_retrieval_metrics,
)
from trajectory_metric_backend import TrajectoryMetricBackend


METHODS = (
    "topo-route-hash",
    "adaptive-topo-hash",
    "adaptive-dual-metric",
    "adaptive-dual-metric-no-rerank",
    "global-hash",
    "exact",
)
_MASK64 = (1 << 64) - 1


def _packed_hamming_distances(query_code, codes):
    """Return packed-code Hamming distances without importing the torch stack."""
    query_code = np.asarray(query_code, dtype=np.uint8).reshape(1, -1)
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != query_code.shape[1]:
        raise ValueError(
            f"Packed code shapes do not match: query={query_code.shape}, "
            f"gallery={codes.shape}."
        )
    try:
        from hash_global import hamming_distances
    except ImportError:
        # Keep lightweight tests usable when torch is not installed.
        return np.unpackbits(
            np.bitwise_xor(codes, query_code), axis=1
        ).sum(axis=1, dtype=np.int32)
    return hamming_distances(query_code, codes)


def _same_file(left, right):
    if not left or not right:
        return False


def _row_retrieval_metrics(predictions, labels):
    labels = np.asarray(labels, dtype=np.int64)
    total = min(len(predictions), len(labels))
    metrics = np.zeros((total, 6), dtype=np.float32)
    for idx in range(total):
        prediction = np.asarray(predictions[idx], dtype=np.int64)
        truth = np.asarray(labels[idx], dtype=np.int64)
        pred10 = set(prediction[:10])
        pred50 = set(prediction[:50])
        true10 = set(truth[:10])
        true50 = set(truth[:50])
        true1 = set(truth[:1])
        metrics[idx, 0] = len(pred10 & true10) / 10.0
        metrics[idx, 1] = len(pred50 & true50) / 50.0
        metrics[idx, 2] = len(pred50 & true10) / 10.0
        metrics[idx, 3] = len(set(prediction[:1]) & true1)
        metrics[idx, 4] = len(pred10 & true1)
        metrics[idx, 5] = len(pred50 & true1)
    names = ["HR@10", "HR@50", "R10@50", "R1@1", "R1@10", "R1@50"]
    values = metrics.mean(axis=0) if total else np.zeros(6, dtype=np.float32)
    return {name: float(value) for name, value in zip(names, values)}


def _select_low_confidence_queries(margins, threshold, max_fraction):
    """Select the lowest-margin queries under a deterministic batch cap."""
    margins = np.asarray(margins, dtype=np.float64).reshape(-1)
    if not 0.0 <= float(max_fraction) <= 1.0:
        raise ValueError("max_fraction must be in [0, 1].")
    low_confidence = margins < float(threshold)
    eligible = np.flatnonzero(low_confidence)
    budget = min(
        int(np.floor(len(margins) * float(max_fraction) + 1e-12)),
        len(eligible),
    )
    selected = np.zeros(len(margins), dtype=bool)
    if budget:
        order = np.argsort(margins[eligible], kind="stable")
        selected[eligible[order[:budget]]] = True
    return low_confidence, selected


def _ordered_unique_ids(*parts):
    """Concatenate candidate arrays while preserving source/rank order."""
    values = []
    seen = set()
    for part in parts:
        if part is None:
            continue
        for value in np.asarray(part, dtype=np.int64).reshape(-1):
            value = int(value)
            if value not in seen:
                seen.add(value)
                values.append(value)
    return np.asarray(values, dtype=np.int64)


def _candidate_label_stats(candidate_lists, labels, query_ids=None):
    if query_ids is None:
        query_ids = np.arange(len(candidate_lists), dtype=np.int64)
    else:
        query_ids = np.asarray(query_ids, dtype=np.int64).reshape(-1)
        if len(query_ids) != len(candidate_lists):
            raise ValueError(
                "query_ids must have one entry for every candidate list."
            )
        if np.any((query_ids < 0) | (query_ids >= len(labels))):
            raise ValueError("query_ids contains an index outside labels.")
    total = len(candidate_lists)
    if total <= 0:
        return {
            "truth_top1_hit_rate": 0.0,
            "truth_top10_coverage": 0.0,
            "truth_top50_coverage": 0.0,
        }
    top1_hits = []
    top10_coverage = []
    top50_coverage = []
    for candidates, query_id in zip(candidate_lists, query_ids):
        truth = labels[int(query_id)]
        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
        truth = np.asarray(truth, dtype=np.int64).reshape(-1)
        candidate_set = set(int(item) for item in candidates)
        truth1 = truth[:1]
        truth10 = truth[:10]
        truth50 = truth[:50]
        top1_hits.append(bool(len(truth1) and int(truth1[0]) in candidate_set))
        top10_coverage.append(
            sum(int(item) in candidate_set for item in truth10)
            / max(len(truth10), 1)
        )
        top50_coverage.append(
            sum(int(item) in candidate_set for item in truth50)
            / max(len(truth50), 1)
        )
    return {
        "truth_top1_hit_rate": float(np.mean(top1_hits)),
        "truth_top10_coverage": float(np.mean(top10_coverage)),
        "truth_top50_coverage": float(np.mean(top50_coverage)),
    }


def _candidate_retention_stats(input_lists, selected_lists, labels, query_ids):
    """Measure truth retained by a prefilter on the exact same query rows."""
    query_ids = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if len(input_lists) != len(selected_lists) or len(input_lists) != len(query_ids):
        raise ValueError(
            "input_lists, selected_lists, and query_ids must have equal lengths."
        )
    input_hits = {1: 0, 10: 0, 50: 0}
    selected_hits = {1: 0, 10: 0, 50: 0}
    for input_ids, selected_ids, query_id in zip(
        input_lists, selected_lists, query_ids
    ):
        truth = np.asarray(labels[int(query_id)], dtype=np.int64).reshape(-1)
        input_set = set(int(item) for item in np.asarray(input_ids).reshape(-1))
        selected_set = set(
            int(item) for item in np.asarray(selected_ids).reshape(-1)
        )
        for cutoff in input_hits:
            truth_ids = truth[:cutoff]
            input_hits[cutoff] += sum(
                int(item) in input_set for item in truth_ids
            )
            selected_hits[cutoff] += sum(
                int(item) in selected_set for item in truth_ids
            )
    return {
        f"truth_top{cutoff}_retention": float(
            selected_hits[cutoff] / max(input_hits[cutoff], 1)
        )
        for cutoff in (1, 10, 50)
    }


def _payload_labels(payload):
    path = payload.get("path")
    if not path:
        return None
    with np.load(path, allow_pickle=True) as data:
        for key in ("labels", "vali_y", "train_y"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.int64)
    return None
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def _mix64(value):
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _mix64_vector(values):
    values = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        values ^= values >> np.uint64(31)
    return values


def _token_hash(level, token):
    value = _mix64(0xA24BAED4963EE407 ^ int(level))
    for item in token:
        value = _mix64(value ^ _mix64(int(item) + 2))
    return value


def _compressed_level_path(trajectory, assignment):
    trajectory = np.asarray(trajectory, dtype=np.int64).reshape(-1)
    valid = (trajectory >= 0) & (trajectory < len(assignment))
    mapped = np.asarray(assignment[trajectory[valid]], dtype=np.int64)
    mapped = mapped[mapped >= 0]
    if not len(mapped):
        return np.empty((0,), dtype=np.int64)
    keep = np.concatenate(([True], mapped[1:] != mapped[:-1]))
    return mapped[keep]


def _path_tokens(path, qgram):
    path = np.asarray(path, dtype=np.int64).reshape(-1)
    if not len(path):
        return [(-1,)]
    if len(path) < qgram:
        return [tuple(int(item) for item in path)]
    return [
        tuple(int(item) for item in path[start : start + qgram])
        for start in range(len(path) - qgram + 1)
    ]


def _minhash_signature(token_keys, seeds):
    token_keys = np.unique(np.asarray(token_keys, dtype=np.uint64).reshape(-1))
    signature = np.full(len(seeds), np.uint64(_MASK64), dtype=np.uint64)
    for token in token_keys:
        signature = np.minimum(signature, _mix64_vector(seeds ^ token))
    return signature


def _signature_distance(query_signature, gallery_signatures):
    gallery_signatures = np.asarray(gallery_signatures, dtype=np.uint64)
    if not len(gallery_signatures):
        return np.empty((0,), dtype=np.float32)
    return 1.0 - np.mean(
        gallery_signatures == np.asarray(query_signature, dtype=np.uint64), axis=1
    ).astype(np.float32)


class CompactPostingIndex:
    def __init__(self, keys, offsets, rows):
        self.keys = np.asarray(keys, dtype=np.uint64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.rows = np.asarray(rows)

    @classmethod
    def from_pairs(cls, pair_keys, pair_rows):
        pair_keys = np.asarray(pair_keys, dtype=np.uint64)
        pair_rows = np.asarray(pair_rows)
        if not len(pair_keys):
            return cls(
                np.empty((0,), dtype=np.uint64),
                np.zeros((1,), dtype=np.int64),
                pair_rows,
            )
        order = np.argsort(pair_keys, kind="stable")
        pair_keys = pair_keys[order]
        pair_rows = pair_rows[order]
        keys, starts = np.unique(pair_keys, return_index=True)
        offsets = np.empty(len(keys) + 1, dtype=np.int64)
        offsets[:-1] = starts
        offsets[-1] = len(pair_rows)
        return cls(keys, offsets, pair_rows)

    @classmethod
    def build(cls, selected_keys, row_count, max_keys_per_row):
        capacity = int(row_count) * int(max_keys_per_row)
        pair_keys = np.empty(capacity, dtype=np.uint64)
        pair_rows = np.empty(
            capacity,
            dtype=np.int32 if row_count <= np.iinfo(np.int32).max else np.int64,
        )
        cursor = 0
        for row_id, keys in enumerate(selected_keys):
            keys = np.asarray(keys, dtype=np.uint64)[:max_keys_per_row]
            stop = cursor + len(keys)
            pair_keys[cursor:stop] = keys
            pair_rows[cursor:stop] = row_id
            cursor = stop
        return cls.from_pairs(pair_keys[:cursor], pair_rows[:cursor])

    def lookup(self, key):
        position = int(np.searchsorted(self.keys, np.uint64(key)))
        if position >= len(self.keys) or self.keys[position] != np.uint64(key):
            return np.empty((0,), dtype=self.rows.dtype)
        return self.rows[self.offsets[position] : self.offsets[position + 1]]

    def lookup_many(self, keys, candidate_cap=0):
        chunks = [self.lookup(key) for key in np.unique(keys)]
        chunks = sorted(
            (chunk for chunk in chunks if len(chunk)), key=lambda chunk: len(chunk)
        )
        if not chunks:
            return np.empty((0,), dtype=np.int64)
        if candidate_cap and candidate_cap > 0:
            scan_cap = max(int(candidate_cap) * 2, int(candidate_cap))
            limited = []
            accumulated = 0
            for chunk in chunks:
                remaining = scan_cap - accumulated
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    positions = np.linspace(
                        0, len(chunk) - 1, num=remaining, dtype=np.int64
                    )
                    chunk = chunk[positions]
                limited.append(chunk)
                accumulated += len(chunk)
            chunks = limited
        result = np.unique(np.concatenate(chunks)).astype(np.int64, copy=False)
        if candidate_cap and len(result) > candidate_cap:
            positions = np.linspace(
                0, len(result) - 1, num=int(candidate_cap), dtype=np.int64
            )
            result = result[positions]
        return result

    def save(self, path):
        np.savez(
            path,
            keys=self.keys,
            offsets=self.offsets,
            rows=self.rows,
        )


@dataclass
class TopoRouteHashConfig:
    # ``hybrid`` uses L3/L2 postings plus Global Hash candidates.  The
    # ``global-only`` mode keeps only Global Hash candidates and applies the
    # topology score as a reranker, which is cheaper when Global Hash already
    # has near-complete candidate recall.
    candidate_source: str = "hybrid"
    qgram: int = 2
    minhash_size: int = 8
    postings_per_level: int = 4
    l3_expand_hops: int = 1
    min_route_candidates: int = 1024
    max_route_candidates: int = 4096
    structural_candidates: int = 512
    global_candidates: int = 1024
    max_rerank_candidates: int = 1536
    l3_weight: float = 0.45
    l2_weight: float = 0.35
    l1_weight: float = 0.20
    embedding_weight: float = 0.80
    # Optional continuous multi-scale reranking.  A value of zero preserves
    # the original embedding/topology fusion.  When enabled, the remaining
    # score mass is split between embedding and topology explicitly.
    hierarchy_embedding_weight: float = 0.0
    topology_weight: float = -1.0
    # Adaptive retrieval keeps a Global Hash core and adds topology rescue
    # candidates only for queries whose two routes disagree.
    adaptive_initial_global: int = 512
    adaptive_initial_structural: int = 256
    adaptive_max_global: int = 2048
    adaptive_max_structural: int = 1024
    adaptive_max_rerank: int = 2048
    adaptive_overlap_threshold: float = 0.35
    adaptive_min_route_recall: int = 128
    adaptive_gate: str = "overlap"
    adaptive_global_margin: float = 0.0
    adaptive_metric_fast_path: str = "skip"
    # Upper bound on the fraction of queries allowed to enter the expensive
    # low-confidence metric-reranking branch.  The batch evaluator selects
    # the lowest-margin queries up to this budget; 1.0 disables the cap.
    adaptive_max_tp_fraction: float = 0.15
    # A low-confidence query can still be a high-quality rescue when the
    # Global/structural pools agree strongly.  Such queries use a smaller
    # metric pool; uncertain or expanded rescues use the full budget.
    adaptive_rescue_quality_overlap: float = 0.75
    adaptive_rescue_near_margin_fraction: float = 0.80
    # Optional candidate-set guarantee: derive the Global Hash embedding Top-50
    # from the full Hamming pool and keep all of it eligible for metric reranking.
    adaptive_preserve_global_top50: bool = False
    # Expensive trajectory metrics are evaluated only after this cheap
    # embedding prefilter. Protected candidates keep Global/topology rescue
    # results eligible even when their embedding rank is not high enough.
    metric_prefilter_candidates: int = 256
    metric_prefilter_high_quality_candidates: int = 50
    metric_prefilter_global_protect: int = 64
    metric_prefilter_structural_protect: int = 32
    metric_prefilter_embedding_weight: float = 1.0
    metric_prefilter_hash_weight: float = 0.0
    metric_prefilter_hash_score: str = "hamming"
    metric_prefilter_soft_temperature: float = 1.0
    # Independent Metric-Head retrieval used by adaptive-dual-metric.  The
    # first budget controls its MIH search, the second controls how many of
    # those candidates are merged with Global and topology candidates.
    dual_metric_search_candidates: int = 256
    dual_metric_candidates: int = 96
    dual_metric_tp_candidates: int = 128
    dual_metric_high_quality_tp_candidates: int = 64
    dual_metric_hard_tp_candidates: int = 160
    dual_metric_hard_margin_fraction: float = 0.50
    dual_metric_head_protect: int = 48
    dual_metric_structural_protect: int = 16
    dual_metric_embedding_weight: float = 0.40
    dual_metric_hash_weight: float = 0.40
    dual_metric_topology_weight: float = 0.20
    seed: int = 2026

    def validate(self):
        if self.candidate_source not in {"hybrid", "global-only"}:
            raise ValueError(
                "candidate_source must be 'hybrid' or 'global-only'."
            )
        integer_positive = {
            "qgram": self.qgram,
            "minhash_size": self.minhash_size,
            "postings_per_level": self.postings_per_level,
            "min_route_candidates": self.min_route_candidates,
            "max_route_candidates": self.max_route_candidates,
            "structural_candidates": self.structural_candidates,
            "global_candidates": self.global_candidates,
            "max_rerank_candidates": self.max_rerank_candidates,
            "adaptive_initial_global": self.adaptive_initial_global,
            "adaptive_initial_structural": self.adaptive_initial_structural,
            "adaptive_max_global": self.adaptive_max_global,
            "adaptive_max_structural": self.adaptive_max_structural,
            "adaptive_max_rerank": self.adaptive_max_rerank,
            "adaptive_min_route_recall": self.adaptive_min_route_recall,
            "metric_prefilter_candidates": self.metric_prefilter_candidates,
            "metric_prefilter_high_quality_candidates": self.metric_prefilter_high_quality_candidates,
            "dual_metric_search_candidates": self.dual_metric_search_candidates,
            "dual_metric_candidates": self.dual_metric_candidates,
            "dual_metric_tp_candidates": self.dual_metric_tp_candidates,
            "dual_metric_high_quality_tp_candidates": self.dual_metric_high_quality_tp_candidates,
            "dual_metric_hard_tp_candidates": self.dual_metric_hard_tp_candidates,
        }
        invalid = [name for name, value in integer_positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"TopoRouteHash parameters must be positive: {invalid}")
        if self.l3_expand_hops < 0:
            raise ValueError("l3_expand_hops cannot be negative.")
        if self.max_route_candidates < self.min_route_candidates:
            raise ValueError("max_route_candidates must be >= min_route_candidates.")
        if self.adaptive_max_global < self.adaptive_initial_global:
            raise ValueError("adaptive_max_global must cover adaptive_initial_global.")
        if self.adaptive_max_structural < self.adaptive_initial_structural:
            raise ValueError(
                "adaptive_max_structural must cover adaptive_initial_structural."
            )
        if self.adaptive_max_rerank < self.max_rerank_candidates:
            raise ValueError("adaptive_max_rerank must cover max_rerank_candidates.")
        if not 0.0 <= self.adaptive_overlap_threshold <= 1.0:
            raise ValueError("adaptive_overlap_threshold must be in [0, 1].")
        if self.adaptive_gate not in {"overlap", "global", "hybrid"}:
            raise ValueError("adaptive_gate must be overlap, global, or hybrid.")
        if self.adaptive_global_margin < 0.0:
            raise ValueError("adaptive_global_margin cannot be negative.")
        if not 0.0 <= self.adaptive_rescue_quality_overlap <= 1.0:
            raise ValueError("adaptive_rescue_quality_overlap must be in [0, 1].")
        if not 0.0 <= self.adaptive_rescue_near_margin_fraction <= 1.0:
            raise ValueError(
                "adaptive_rescue_near_margin_fraction must be in [0, 1]."
            )
        if self.adaptive_metric_fast_path not in {"skip", "metric"}:
            raise ValueError("adaptive_metric_fast_path must be skip or metric.")
        if not 0.0 <= self.adaptive_max_tp_fraction <= 1.0:
            raise ValueError("adaptive_max_tp_fraction must be in [0, 1].")
        if self.metric_prefilter_global_protect < 0:
            raise ValueError("metric_prefilter_global_protect cannot be negative.")
        if self.metric_prefilter_structural_protect < 0:
            raise ValueError("metric_prefilter_structural_protect cannot be negative.")
        if self.metric_prefilter_embedding_weight < 0.0:
            raise ValueError("metric_prefilter_embedding_weight cannot be negative.")
        if self.metric_prefilter_hash_weight < 0.0:
            raise ValueError("metric_prefilter_hash_weight cannot be negative.")
        if self.metric_prefilter_hash_score not in {"hamming", "soft"}:
            raise ValueError("metric_prefilter_hash_score must be hamming or soft.")
        if self.metric_prefilter_soft_temperature <= 0.0:
            raise ValueError("metric_prefilter_soft_temperature must be positive.")
        if (
            self.metric_prefilter_embedding_weight
            + self.metric_prefilter_hash_weight <= 0.0
        ):
            raise ValueError("At least one metric prefilter score weight must be positive.")
        if self.metric_prefilter_candidates <= 0:
            raise ValueError("metric_prefilter_candidates must be positive.")
        if self.metric_prefilter_high_quality_candidates <= 0:
            raise ValueError(
                "metric_prefilter_high_quality_candidates must be positive."
            )
        if (
            self.metric_prefilter_high_quality_candidates
            > self.metric_prefilter_candidates
        ):
            raise ValueError(
                "High-quality metric budget cannot exceed the regular metric budget."
            )
        if (
            self.metric_prefilter_global_protect
            + self.metric_prefilter_structural_protect
            > self.metric_prefilter_candidates
        ):
            raise ValueError(
                "Metric protection budgets cannot exceed metric_prefilter_candidates."
            )
        if self.dual_metric_search_candidates < self.dual_metric_candidates:
            raise ValueError(
                "dual_metric_search_candidates must cover dual_metric_candidates."
            )
        if not (
            self.dual_metric_high_quality_tp_candidates
            <= self.dual_metric_tp_candidates
            <= self.dual_metric_hard_tp_candidates
        ):
            raise ValueError(
                "Dual-metric TP budgets must satisfy high-quality <= regular <= hard."
            )
        if not 0.0 <= self.dual_metric_hard_margin_fraction <= 1.0:
            raise ValueError("dual_metric_hard_margin_fraction must be in [0, 1].")
        if self.dual_metric_head_protect < 0 or self.dual_metric_structural_protect < 0:
            raise ValueError("Dual-metric source protection budgets cannot be negative.")
        dual_weights = (
            self.dual_metric_embedding_weight,
            self.dual_metric_hash_weight,
            self.dual_metric_topology_weight,
        )
        if min(dual_weights) < 0.0 or sum(dual_weights) <= 0.0:
            raise ValueError("Dual-metric fusion weights must be non-negative and non-zero.")
        required_rerank = (
            self.global_candidates
            if self.candidate_source == "global-only"
            else self.structural_candidates + self.global_candidates
        )
        if required_rerank > self.max_rerank_candidates:
            raise ValueError(
                "max_rerank_candidates must cover the active candidate budget."
            )
        level_weight = self.l3_weight + self.l2_weight + self.l1_weight
        if min(self.l3_weight, self.l2_weight, self.l1_weight) < 0:
            raise ValueError("Structural level weights cannot be negative.")
        if level_weight <= 0:
            raise ValueError("At least one structural level weight must be positive.")
        if not 0.0 <= self.embedding_weight <= 1.0:
            raise ValueError("embedding_weight must be in [0, 1].")
        if self.hierarchy_embedding_weight < 0.0 or self.hierarchy_embedding_weight > 1.0:
            raise ValueError("hierarchy_embedding_weight must be in [0, 1].")
        if self.topology_weight < -1.0 or self.topology_weight > 1.0:
            raise ValueError("topology_weight must be -1 or in [0, 1].")
        if self.hierarchy_embedding_weight > 0.0:
            topology_weight = (
                1.0 - self.embedding_weight - self.hierarchy_embedding_weight
                if self.topology_weight < 0.0 else self.topology_weight
            )
            if topology_weight < 0.0 or self.embedding_weight + self.hierarchy_embedding_weight + topology_weight > 1.0 + 1e-6:
                raise ValueError(
                    "embedding_weight + hierarchy_embedding_weight + topology_weight "
                    "must be at most 1."
                )


class TopoRouteHashIndex:
    """L3 posting routing, L2/L1 path refinement, and Global Hash fallback."""

    def __init__(self, hash_index, hierarchy, config=None):
        self.hash_index = hash_index
        # Optional second code space used only by the expensive-metric
        # prefilter. The packed global hash codes remain in hash_index.codes.
        self.metric_prefilter_codes = None
        self.metric_prefilter_soft_codes = None
        self.metric_hash_index = None
        self.dual_metric_weight_payload = None
        self.hierarchy = hierarchy
        self.config = config or TopoRouteHashConfig()
        self.config.validate()
        self.seeds = np.asarray(
            [_mix64(self.config.seed + offset) for offset in range(self.config.minhash_size)],
            dtype=np.uint64,
        )
        self.signatures = {}
        self.trajectories = None
        self.scale_embeddings = None
        self.postings = {}
        self.l3_neighbors = {}
        for source, target in np.asarray(
            hierarchy.get("l3_edges", np.zeros((0, 2), dtype=np.int64))
        ):
            source, target = int(source), int(target)
            self.l3_neighbors.setdefault(source, set()).add(target)
            self.l3_neighbors.setdefault(target, set()).add(source)
        self.build_seconds = 0.0

    def _level_features(self, trajectory, level, include_tokens=True):
        assignment = self.hierarchy[f"node_to_l{level}"]
        trajectory = np.asarray(trajectory, dtype=np.int64).reshape(-1)
        path = _compressed_level_path(trajectory, assignment)
        tokens_for_hash = _path_tokens(path, self.config.qgram)
        keys = np.asarray(
            [_token_hash(level, token) for token in tokens_for_hash],
            dtype=np.uint64,
        )
        keys = np.unique(keys)
        signature = _minhash_signature(keys, self.seeds)
        tokens = tokens_for_hash if include_tokens else ()
        return {
            "path": path,
            "tokens": tokens,
            "keys": keys,
            "signature": signature,
        }

    def _features(self, trajectory):
        result = {}
        for level in (1, 2, 3):
            result[level] = self._level_features(
                trajectory, level, include_tokens=(level == 3)
            )
        return result

    def build(self, trajectories, scale_embeddings=None):
        build_start = perf_counter()
        trajectories = np.asarray(trajectories, dtype=object)
        row_count = len(trajectories)
        self.trajectories = trajectories
        if scale_embeddings is not None:
            scale_embeddings = np.asarray(scale_embeddings, dtype=np.float32)
            if scale_embeddings.ndim != 3 or len(scale_embeddings) != row_count:
                raise ValueError(
                    "scale_embeddings must have shape [gallery_rows, scales, dim]."
                )
        self.scale_embeddings = scale_embeddings
        self.signatures = {
            level: np.empty(
                (row_count, self.config.minhash_size), dtype=np.uint64
            )
            for level in (1, 2, 3)
        }
        document_frequency = {2: {}, 3: {}}
        try:
            from tqdm import tqdm

            iterator = tqdm(range(row_count), desc="topo signatures")
        except ImportError:
            iterator = range(row_count)
        for row_id in iterator:
            features = self._features(trajectories[row_id])
            for level in (1, 2, 3):
                self.signatures[level][row_id] = features[level]["signature"]
            if self.config.candidate_source == "hybrid":
                for level in (2, 3):
                    for key in features[level]["keys"]:
                        key = int(key)
                        document_frequency[level][key] = (
                            document_frequency[level].get(key, 0) + 1
                        )

        if self.config.candidate_source == "hybrid":
            capacity = row_count * self.config.postings_per_level
            pair_keys = {
                level: np.empty(capacity, dtype=np.uint64) for level in (2, 3)
            }
            pair_rows = {
                level: np.empty(
                    capacity,
                    dtype=np.int32 if row_count <= np.iinfo(np.int32).max else np.int64,
                )
                for level in (2, 3)
            }
            cursors = {2: 0, 3: 0}
            try:
                from tqdm import tqdm

                iterator = tqdm(range(row_count), desc="topo postings")
            except ImportError:
                iterator = range(row_count)
            for row_id in iterator:
                features = self._features(trajectories[row_id])
                for level in (2, 3):
                    keys = sorted(
                        (int(key) for key in features[level]["keys"]),
                        key=lambda key: (
                            document_frequency[level].get(key, row_count), key
                        ),
                    )[: self.config.postings_per_level]
                    pair_start = cursors[level]
                    stop = pair_start + len(keys)
                    pair_keys[level][pair_start:stop] = keys
                    pair_rows[level][pair_start:stop] = row_id
                    cursors[level] = stop
            self.postings = {
                level: CompactPostingIndex.from_pairs(
                    pair_keys[level][: cursors[level]],
                    pair_rows[level][: cursors[level]],
                )
                for level in (2, 3)
            }
        else:
            self.postings = {}
        self.build_seconds = perf_counter() - build_start
        return self

    def _expanded_l3_keys(self, tokens):
        expanded = set()
        for token in tokens:
            expanded.add(_token_hash(3, token))
            if self.config.l3_expand_hops <= 0:
                continue
            for position, region in enumerate(token):
                frontier = {int(region)}
                visited = set(frontier)
                for _ in range(self.config.l3_expand_hops):
                    next_frontier = set()
                    for item in frontier:
                        next_frontier.update(self.l3_neighbors.get(item, set()))
                    next_frontier -= visited
                    visited.update(next_frontier)
                    frontier = next_frontier
                for neighbor in visited - {int(region)}:
                    variant = list(token)
                    variant[position] = int(neighbor)
                    expanded.add(_token_hash(3, tuple(variant)))
        return np.asarray(sorted(expanded), dtype=np.uint64)

    def _structural_distances(self, features, candidate_ids):
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        distances = np.zeros(len(candidate_ids), dtype=np.float32)
        weights = {
            3: self.config.l3_weight,
            2: self.config.l2_weight,
            1: self.config.l1_weight,
        }
        total_weight = sum(weights.values())
        for level, weight in weights.items():
            distances += weight * _signature_distance(
                features[level]["signature"], self.signatures[level][candidate_ids]
            )
        return distances / total_weight

    def _route_candidates(self, features):
        l3_keys = self._expanded_l3_keys(features[3]["tokens"])
        posting_cap = self.config.max_route_candidates * 4
        l3_ids = self.postings[3].lookup_many(
            l3_keys, candidate_cap=posting_cap
        )
        l2_ids = self.postings[2].lookup_many(
            features[2]["keys"], candidate_cap=posting_cap
        )
        intersection = np.intersect1d(l3_ids, l2_ids, assume_unique=True)
        if len(intersection) >= self.config.min_route_candidates:
            route_ids = intersection
            l2_added = 0
        else:
            route_ids = np.union1d(intersection, l3_ids).astype(
                np.int64, copy=False
            )
            before = len(route_ids)
            if len(route_ids) < self.config.min_route_candidates:
                route_ids = np.union1d(route_ids, l2_ids).astype(
                    np.int64, copy=False
                )
            l2_added = int(len(route_ids) - before)
        route_before_cap = int(len(route_ids))
        if len(route_ids) > self.config.max_route_candidates:
            distances = self._structural_distances(features, route_ids)
            route_ids = route_ids[
                _topk_indices(distances, self.config.max_route_candidates)
            ]
        return route_ids, {
            "l3_posting_candidates": int(len(l3_ids)),
            "l2_posting_candidates": int(len(l2_ids)),
            "l2_intersection_candidates": int(len(intersection)),
            "l2_fallback_added": int(l2_added),
            "route_before_cap": route_before_cap,
            "route_after_cap": int(len(route_ids)),
            "expanded_l3_tokens": int(len(l3_keys)),
        }

    def _continuous_distances(self, query_embedding, candidate_ids):
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if self.hash_index.config.normalize or self.hash_index.config.metric == "cosine":
            query_embedding = query_embedding / max(
                float(np.linalg.norm(query_embedding)), 1e-12
            )
        gallery = self.hash_index.embeddings[candidate_ids].astype(np.float32)
        if self.hash_index.config.metric == "cosine":
            return (1.0 - gallery @ query_embedding).astype(np.float32)
        difference = gallery - query_embedding.reshape(1, -1)
        return np.einsum("ij,ij->i", difference, difference).astype(np.float32)

    def _hierarchy_distances(self, query_scale_embeddings, query_scale_weights, candidate_ids):
        """Compute cheap continuous distances on the precomputed scale features."""
        if self.scale_embeddings is None or query_scale_embeddings is None:
            return None
        query = np.asarray(query_scale_embeddings, dtype=np.float32)
        gallery = np.asarray(self.scale_embeddings[candidate_ids], dtype=np.float32)
        if query.ndim != 2 or gallery.ndim != 3 or query.shape[0] != gallery.shape[1]:
            raise ValueError("Query/Gallery scale embeddings have incompatible shapes.")
        if self.hash_index.config.metric == "cosine" or self.hash_index.config.normalize:
            query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
            gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=2, keepdims=True), 1e-12)
            distances = 1.0 - np.einsum("sd,nsd->ns", query, gallery)
        else:
            difference = gallery - query[None, :, :]
            distances = np.einsum("nsd,nsd->ns", difference, difference)
        if query_scale_weights is None:
            weights = np.full(query.shape[0], 1.0 / query.shape[0], dtype=np.float32)
        else:
            weights = np.asarray(query_scale_weights, dtype=np.float32).reshape(-1)
            if len(weights) != query.shape[0]:
                raise ValueError("Query scale weights do not match scale embeddings.")
            weights = np.maximum(weights, 0.0)
            weights /= max(float(weights.sum()), 1e-12)
        return np.asarray(distances @ weights, dtype=np.float32)

    @staticmethod
    def _rank_fraction(values):
        values = np.asarray(values)
        ranks = np.empty(len(values), dtype=np.float32)
        ranks[np.argsort(values, kind="stable")] = np.arange(len(values), dtype=np.float32)
        return ranks / max(len(values) - 1, 1)

    def retrieve_candidate_sets(
        self, query_trajectory, query_code, top_k=50, exclude_ids=None,
        *, global_candidates=None, structural_candidates=None,
        source_cache=None,
    ):
        """Return route, structural, and Global Hash candidates without reranking.

        This is shared by adaptive metric-aware wrappers.  The legacy ``query``
        path keeps its original defaults, while callers can request a larger
        temporary candidate budget for difficult queries.
        """
        if exclude_ids is None:
            exclude_ids = np.empty((0,), dtype=np.int64)
        else:
            exclude_ids = np.asarray(exclude_ids, dtype=np.int64).reshape(-1)
        exclude_ids = np.unique(exclude_ids)
        active_global = (
            self.config.global_candidates
            if global_candidates is None else int(global_candidates)
        )
        active_structural = (
            self.config.structural_candidates
            if structural_candidates is None else int(structural_candidates)
        )
        if active_global <= 0 or active_structural <= 0:
            raise ValueError("Candidate budgets must be positive.")
        if source_cache is None:
            features = self._features(query_trajectory)
            if self.config.candidate_source == "hybrid":
                route_ids, route_details = self._route_candidates(features)
                if len(exclude_ids) and len(route_ids):
                    route_ids = route_ids[~np.isin(route_ids, exclude_ids)]
                if len(route_ids):
                    route_distances = self._structural_distances(features, route_ids)
                    route_order = _topk_indices(route_distances, len(route_ids))
                    ranked_route_ids = route_ids[route_order]
                else:
                    ranked_route_ids = np.empty((0,), dtype=np.int64)
            else:
                route_ids = np.empty((0,), dtype=np.int64)
                ranked_route_ids = route_ids
                route_details = {
                    "l3_posting_candidates": 0,
                    "l2_posting_candidates": 0,
                    "l2_intersection_candidates": 0,
                    "l2_fallback_added": 0,
                    "route_before_cap": 0,
                    "route_after_cap": 0,
                    "expanded_l3_tokens": 0,
                }
            global_ids, _, global_details = self.hash_index.hamming_search(
                query_code,
                candidate_limit=active_global,
                minimum_candidates=top_k,
                exclude_ids=exclude_ids,
            )
        else:
            features = source_cache["features"]
            route_ids = source_cache["route_ids"]
            ranked_route_ids = source_cache["ranked_route_ids"]
            route_details = source_cache["route_details"]
            global_ids = source_cache["global_ids"][:active_global]
            global_details = source_cache["global_details"]
        if self.config.candidate_source == "hybrid":
            structural_ids = ranked_route_ids[:active_structural]
        else:
            structural_ids = np.empty((0,), dtype=np.int64)
        if self.config.candidate_source == "global-only":
            union_ids = np.asarray(global_ids, dtype=np.int64)
        else:
            union_ids = np.union1d(structural_ids, global_ids).astype(
                np.int64, copy=False
            )
        if len(exclude_ids) and len(union_ids):
            union_ids = union_ids[~np.isin(union_ids, exclude_ids)]
        return {
            "features": features,
            "route_ids": np.asarray(route_ids, dtype=np.int64),
            "structural_candidate_ids": np.asarray(structural_ids, dtype=np.int64),
            "global_candidate_ids": np.asarray(global_ids, dtype=np.int64),
            "union_candidate_ids": np.asarray(union_ids, dtype=np.int64),
            "global_used_full_scan": bool(global_details["used_full_scan"]),
            **route_details,
        }

    def _build_source_cache(self, query_trajectory, query_code, exclude_ids,
                            *, global_limit, top_k, global_cache=None):
        """Build reusable route/Global sources for adaptive two-stage queries."""
        if exclude_ids is None:
            exclude_ids = np.empty((0,), dtype=np.int64)
        else:
            exclude_ids = np.unique(np.asarray(exclude_ids, dtype=np.int64).reshape(-1))
        features = self._features(query_trajectory)
        if self.config.candidate_source == "hybrid":
            route_ids, route_details = self._route_candidates(features)
            if len(exclude_ids) and len(route_ids):
                route_ids = route_ids[~np.isin(route_ids, exclude_ids)]
            if len(route_ids):
                route_distances = self._structural_distances(features, route_ids)
                route_order = _topk_indices(route_distances, len(route_ids))
                ranked_route_ids = route_ids[route_order]
            else:
                ranked_route_ids = np.empty((0,), dtype=np.int64)
        else:
            route_ids = np.empty((0,), dtype=np.int64)
            ranked_route_ids = route_ids
            route_details = {
                "l3_posting_candidates": 0,
                "l2_posting_candidates": 0,
                "l2_intersection_candidates": 0,
                "l2_fallback_added": 0,
                "route_before_cap": 0,
                "route_after_cap": 0,
                "expanded_l3_tokens": 0,
            }
        if global_cache is None:
            global_ids, global_distances, global_details = (
                self.hash_index.hamming_search(
                    query_code,
                    candidate_limit=global_limit,
                    minimum_candidates=top_k,
                    exclude_ids=exclude_ids,
                )
            )
        else:
            global_ids = global_cache["global_ids"]
            global_distances = global_cache["global_distances"]
            global_details = global_cache["global_details"]
        return {
            "features": features,
            "route_ids": np.asarray(route_ids, dtype=np.int64),
            "ranked_route_ids": np.asarray(ranked_route_ids, dtype=np.int64),
            "route_details": route_details,
            "global_ids": np.asarray(global_ids, dtype=np.int64),
            "global_distances": np.asarray(global_distances, dtype=np.int32),
            "global_details": global_details,
        }

    def query(
        self, query_embedding, query_trajectory, query_code, top_k=50,
        query_scale_embeddings=None, query_scale_weights=None, exclude_ids=None,
    ):
        if exclude_ids is None:
            exclude_ids = np.empty((0,), dtype=np.int64)
        else:
            exclude_ids = np.asarray(exclude_ids, dtype=np.int64).reshape(-1)
        exclude_ids = np.unique(exclude_ids)
        candidate_data = self.retrieve_candidate_sets(
            query_trajectory, query_code, top_k=top_k, exclude_ids=exclude_ids
        )
        features = candidate_data.pop("features")
        route_ids = candidate_data["route_ids"]
        structural_ids = candidate_data["structural_candidate_ids"]
        global_ids = candidate_data["global_candidate_ids"]
        candidate_ids = candidate_data["union_candidate_ids"]
        if len(candidate_ids) > self.config.max_rerank_candidates:
            candidate_ids = candidate_ids[: self.config.max_rerank_candidates]
        hierarchy_weight = self.config.hierarchy_embedding_weight
        topology_weight = (
            1.0 - self.config.embedding_weight - hierarchy_weight
            if self.config.topology_weight < 0.0 else self.config.topology_weight
        )
        structural_distance = (
            self._structural_distances(features, candidate_ids)
            if topology_weight > 0.0 else None
        )
        structural_rank = (
            self._rank_fraction(structural_distance)
            if structural_distance is not None else 0.0
        )
        embedding_distance = self._continuous_distances(query_embedding, candidate_ids)
        embedding_rank = self._rank_fraction(embedding_distance)
        hierarchy_distance = self._hierarchy_distances(
            query_scale_embeddings, query_scale_weights, candidate_ids
        )
        if hierarchy_weight > 0.0 and hierarchy_distance is None:
            raise ValueError(
                "Hierarchy reranking is enabled but scale_embeddings are missing "
                "from the Query/Gallery payload."
            )
        if hierarchy_weight > 0.0:
            fused_score = (
                self.config.embedding_weight * embedding_rank
                + hierarchy_weight * self._rank_fraction(hierarchy_distance)
                + topology_weight * structural_rank
            )
        else:
            fused_score = (
                self.config.embedding_weight * embedding_rank
                + (1.0 - self.config.embedding_weight) * structural_rank
            )
        order = _topk_indices(fused_score, top_k)
        final_ids = candidate_ids[order]
        final_global = np.isin(final_ids, global_ids)
        final_struct = np.isin(final_ids, structural_ids)
        candidate_overlap = int(
            len(np.intersect1d(structural_ids, global_ids, assume_unique=True))
        )
        return final_ids, fused_score[order], {
            "route_ids": route_ids,
            "structural_candidate_ids": structural_ids,
            "global_candidate_ids": global_ids,
            "union_candidate_ids": candidate_ids,
            "structural_candidate_count": int(len(structural_ids)),
            "global_candidate_count": int(len(global_ids)),
            "union_candidate_count": int(len(candidate_ids)),
            "struct_global_overlap": candidate_overlap,
            "final_global_only": int(np.count_nonzero(final_global & ~final_struct)),
            "final_struct_only": int(np.count_nonzero(final_struct & ~final_global)),
            "final_overlap": int(np.count_nonzero(final_global & final_struct)),
            "top1_from_struct_only": bool(final_struct[0] and not final_global[0]),
            "global_used_full_scan": bool(candidate_data["global_used_full_scan"]),
            "hierarchy_rerank_enabled": bool(hierarchy_weight > 0.0),
            "hierarchy_distance_available": bool(hierarchy_distance is not None),
            **{key: value for key, value in candidate_data.items()
               if key not in {"route_ids", "structural_candidate_ids",
                              "global_candidate_ids", "union_candidate_ids",
                              "global_used_full_scan"}},
        }

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        for level in (1, 2, 3):
            np.save(osp.join(index_dir, f"minhash_l{level}.npy"), self.signatures[level])
        for level in (2, 3):
            if level in self.postings:
                self.postings[level].save(
                    osp.join(index_dir, f"postings_l{level}.npz")
                )
        with open(osp.join(index_dir, "topo_config.json"), "w", encoding="utf-8") as file:
            json.dump(asdict(self.config), file, indent=2)


class AdaptiveTopoHashIndex(TopoRouteHashIndex):
    """Adaptive Global-core/topology-rescue retrieval with metric reranking."""

    def set_dual_metric_weight_payload(self, payload):
        """Install validated metric/bucket-specific fusion weights."""
        from dual_metric_fusion import validate_weight_payload

        self.dual_metric_weight_payload = validate_weight_payload(payload)

    def _dual_metric_weights(self, metric_name, bucket, variant):
        if self.dual_metric_weight_payload is not None:
            metric_config = self.dual_metric_weight_payload["metrics"].get(metric_name)
            if metric_config is None:
                raise ValueError(
                    f"No learned dual-metric weights for rerank metric {metric_name}."
                )
            weights = metric_config[variant][bucket]
        else:
            weights = (
                self.config.dual_metric_embedding_weight,
                self.config.dual_metric_hash_weight,
                self.config.dual_metric_topology_weight,
            )
        weights = np.asarray(weights, dtype=np.float32)
        return weights / max(float(weights.sum()), 1e-12)

    def _global_confidence(self, distances, initial_count):
        distances = np.asarray(distances, dtype=np.float32).reshape(-1)
        if len(distances) < initial_count:
            return 0.0, False
        probe = min(len(distances), max(2, min(initial_count, 32)))
        margin = float(distances[probe - 1] - distances[0]) / max(
            float(getattr(self.hash_index.config, "bits", 1)), 1.0
        )
        return margin, margin >= self.config.adaptive_global_margin

    def _metric_prefilter_candidates(
        self, query_embedding, candidate_ids, global_ids=None, structural_ids=None,
        budget_override=None, required_ids=None, query_code=None,
    ):
        """Keep a small embedding-ranked pool before expensive metric reranking."""
        candidate_ids = np.unique(np.asarray(candidate_ids, dtype=np.int64).reshape(-1))
        configured_budget = (
            self.config.metric_prefilter_candidates
            if budget_override is None else int(budget_override)
        )
        budget = min(configured_budget, len(candidate_ids))
        if len(candidate_ids) <= budget:
            return candidate_ids, {
                "metric_prefilter_applied": False,
                "metric_prefilter_input_count": int(len(candidate_ids)),
                "metric_prefilter_candidate_count": int(len(candidate_ids)),
                "metric_prefilter_protected_global": 0,
                "metric_prefilter_protected_structural": 0,
            }

        required = (
            np.empty((0,), dtype=np.int64)
            if required_ids is None
            else np.asarray(required_ids, dtype=np.int64).reshape(-1)
        )
        required = required[np.isin(required, candidate_ids)]
        if len(required):
            _, positions = np.unique(required, return_index=True)
            required = required[np.sort(positions)]
        if len(required) > budget:
            raise ValueError(
                "The metric prefilter budget cannot cover all required candidates: "
                f"required={len(required)}, budget={budget}."
            )

        protected_parts = [required] if len(required) else []
        protected_global = 0
        protected_structural = 0
        if global_ids is not None and self.config.metric_prefilter_global_protect:
            values = np.asarray(global_ids, dtype=np.int64).reshape(-1)
            values = values[
                np.isin(values, candidate_ids) & ~np.isin(values, required)
            ]
            global_budget = max(budget - len(required), 0)
            values = values[: min(self.config.metric_prefilter_global_protect, global_budget)]
            protected_parts.append(values)
            protected_global = len(required) + len(values)
        else:
            protected_global = len(required)
        if structural_ids is not None and self.config.metric_prefilter_structural_protect:
            values = np.asarray(structural_ids, dtype=np.int64).reshape(-1)
            already_protected = (
                np.concatenate(protected_parts)
                if protected_parts else np.empty((0,), dtype=np.int64)
            )
            values = values[
                np.isin(values, candidate_ids) & ~np.isin(values, already_protected)
            ]
            values = values[: min(
                self.config.metric_prefilter_structural_protect,
                max(budget - len(already_protected), 0),
            )]
            protected_parts.append(values)
            protected_structural = len(values)
        protected = (
            np.unique(np.concatenate(protected_parts)).astype(np.int64, copy=False)
            if protected_parts else np.empty((0,), dtype=np.int64)
        )
        if len(protected) > budget:
            protected = protected[:budget]
        remaining = candidate_ids[~np.isin(candidate_ids, protected)]
        fill_count = budget - len(protected)
        if fill_count > 0 and len(remaining):
            embedding_distance = self._continuous_distances(query_embedding, remaining)
            embedding_rank = self._rank_fraction(embedding_distance)
            hash_weight = float(self.config.metric_prefilter_hash_weight)
            embedding_weight = float(self.config.metric_prefilter_embedding_weight)
            if hash_weight > 0.0:
                if query_code is None:
                    raise ValueError(
                        "Hash-aware metric prefilter requires the packed query code."
                    )
                if (
                    self.config.metric_prefilter_hash_score == "soft"
                    and self.metric_prefilter_soft_codes is not None
                    and np.asarray(query_code).ndim == 1
                    and np.asarray(query_code).shape[0] == self.metric_prefilter_soft_codes.shape[1]
                ):
                    gallery_codes = self.metric_prefilter_soft_codes[
                        remaining
                    ].astype(np.float32, copy=False)
                    query_soft_code = np.asarray(query_code, dtype=np.float32)
                    bits = float(gallery_codes.shape[1])
                    hash_distance = (
                        bits - np.einsum("ij,j->i", gallery_codes, query_soft_code)
                    ) / (2.0 * bits)
                else:
                    if self.config.metric_prefilter_hash_score == "soft":
                        raise ValueError(
                            "Soft metric prefilter requires metric_prefilter_soft_codes "
                            "and a continuous query metric code."
                        )
                    gallery_codes = (
                        self.metric_prefilter_codes
                        if self.metric_prefilter_codes is not None
                        else self.hash_index.codes
                    )
                    hash_distance = _packed_hamming_distances(
                        query_code, gallery_codes[remaining]
                    )
                hash_rank = self._rank_fraction(hash_distance)
            else:
                hash_rank = 0.0
            prefilter_score = (
                embedding_weight * embedding_rank + hash_weight * hash_rank
            ) / max(embedding_weight + hash_weight, 1e-12)
            fill = remaining[_topk_indices(prefilter_score, fill_count)]
        else:
            fill = np.empty((0,), dtype=np.int64)
        selected = np.unique(np.concatenate((protected, fill))).astype(
            np.int64, copy=False
        )
        return selected, {
            "metric_prefilter_applied": True,
            "metric_prefilter_input_count": int(len(candidate_ids)),
            "metric_prefilter_candidate_count": int(len(selected)),
            "metric_prefilter_protected_global": int(protected_global),
            "metric_prefilter_protected_structural": int(protected_structural),
        }

    def _dual_metric_candidates(
        self,
        query_embedding,
        features,
        global_ids,
        structural_ids,
        metric_query_code,
        metric_score_query_code,
        global_top50_ids,
        tp_budget,
        metric_name,
        difficulty_bucket,
        fusion_variant,
        exclude_ids=None,
        collect_fusion_features=False,
    ):
        """Merge independent Metric-Head retrieval with Global/Topology rescue.

        This method is deliberately separate from ``_metric_prefilter_candidates``:
        the latter can only reorder an existing pool, while this branch adds
        candidates that Global Hash did not retrieve.
        """
        if self.metric_hash_index is None:
            raise RuntimeError(
                "adaptive-dual-metric requires an independent Metric Hash index."
            )
        if metric_query_code is None:
            raise ValueError(
                "adaptive-dual-metric requires packed Metric-Head query codes."
            )
        metric_ids, metric_distances, metric_details = (
            self.metric_hash_index.hamming_search(
                metric_query_code,
                candidate_limit=self.config.dual_metric_search_candidates,
                minimum_candidates=min(50, int(tp_budget)),
                exclude_ids=exclude_ids,
            )
        )
        metric_ids = np.asarray(metric_ids, dtype=np.int64)
        if (
            self.config.metric_prefilter_hash_score == "soft"
            and self.metric_prefilter_soft_codes is not None
            and metric_score_query_code is not None
        ):
            query_soft_code = np.asarray(metric_score_query_code, dtype=np.float32)
            gallery_soft_codes = self.metric_prefilter_soft_codes[
                metric_ids
            ].astype(np.float32, copy=False)
            bits = float(gallery_soft_codes.shape[1])
            metric_distances = (
                bits - np.einsum("ij,j->i", gallery_soft_codes, query_soft_code)
            ) / (2.0 * bits)
            metric_ids = metric_ids[
                _topk_indices(metric_distances, self.config.dual_metric_candidates)
            ]
        else:
            metric_ids = metric_ids[: self.config.dual_metric_candidates]
        global_ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        structural_ids = np.asarray(structural_ids, dtype=np.int64).reshape(-1)
        global_top50_ids = (
            np.empty((0,), dtype=np.int64)
            if global_top50_ids is None
            else np.asarray(global_top50_ids, dtype=np.int64).reshape(-1)
        )
        # Add the protected Global Top-50 before truncating the universe.  This
        # matters when the adaptive active pool is capped below the baseline
        # Global-Hamming pool: protection must be able to reinsert those IDs.
        universe = _ordered_unique_ids(
            global_ids, metric_ids, structural_ids, global_top50_ids
        )
        if exclude_ids is not None and len(universe):
            universe = universe[
                ~np.isin(universe, np.asarray(exclude_ids, dtype=np.int64))
            ]
        if not len(universe):
            return universe, {
                "dual_metric_search_count": int(len(metric_ids)),
                "dual_metric_union_count": 0,
                "dual_metric_tp_budget": 0,
                "dual_metric_protected_global": 0,
                "dual_metric_protected_metric": 0,
                "dual_metric_protected_structural": 0,
                "metric_prefilter_applied": False,
                "metric_prefilter_input_count": 0,
                "metric_prefilter_candidate_count": 0,
                "metric_prefilter_protected_global": 0,
                "metric_prefilter_protected_structural": 0,
            }

        embedding_rank = self._rank_fraction(
            self._continuous_distances(query_embedding, universe)
        )
        structural_rank = self._rank_fraction(
            self._structural_distances(features, universe)
        )
        if (
            self.config.metric_prefilter_hash_score == "soft"
            and self.metric_prefilter_soft_codes is not None
            and metric_score_query_code is not None
        ):
            query_soft_code = np.asarray(metric_score_query_code, dtype=np.float32)
            gallery_soft_codes = self.metric_prefilter_soft_codes[
                universe
            ].astype(np.float32, copy=False)
            bits = float(gallery_soft_codes.shape[1])
            metric_distance = (
                bits - np.einsum("ij,j->i", gallery_soft_codes, query_soft_code)
            ) / (2.0 * bits)
        else:
            metric_distance = _packed_hamming_distances(
                metric_query_code, self.metric_hash_index.codes[universe]
            )
        metric_rank = self._rank_fraction(metric_distance)
        weights = self._dual_metric_weights(
            metric_name, difficulty_bucket, fusion_variant
        )
        fused_score = (
            weights[0] * embedding_rank
            + weights[1] * metric_rank
            + weights[2] * structural_rank
        ) / max(float(weights.sum()), 1e-12)

        # Global baseline Top-50 is a candidate-set safety net.  Additional
        # Metric/Topology quotas preserve the rescue branches before score fill.
        protected_global = global_top50_ids[
            np.isin(global_top50_ids, universe)
        ]
        if not len(protected_global):
            protected_global = global_ids[: self.config.dual_metric_head_protect]
        rescue_slots = max(int(tp_budget) - len(protected_global), 0)
        requested_metric = int(self.config.dual_metric_head_protect)
        requested_structural = int(self.config.dual_metric_structural_protect)
        requested_total = requested_metric + requested_structural
        if requested_total > 0:
            metric_slots = min(
                requested_metric,
                int(round(rescue_slots * requested_metric / requested_total)),
            )
        else:
            metric_slots = 0
        structural_slots = min(
            requested_structural, max(rescue_slots - metric_slots, 0)
        )
        protected_metric = metric_ids[:metric_slots]
        protected_structural = structural_ids[:structural_slots]
        protected = _ordered_unique_ids(
            protected_global, protected_metric, protected_structural
        )
        protected = protected[np.isin(protected, universe)][: int(tp_budget)]
        remaining_mask = ~np.isin(universe, protected)
        remaining = universe[remaining_mask]
        fill = remaining[
            _topk_indices(
                fused_score[remaining_mask],
                max(int(tp_budget) - len(protected), 0),
            )
        ]
        selected = _ordered_unique_ids(protected, fill)[: int(tp_budget)]
        dual_metric_no_rerank_ids = universe[
            _topk_indices(fused_score, len(universe))
        ]
        details = {
            "dual_metric_search_count": int(len(metric_ids)),
            "dual_metric_union_count": int(len(universe)),
            "dual_metric_tp_budget": int(len(selected)),
            "dual_metric_protected_global": int(len(protected_global)),
            "dual_metric_protected_metric": int(len(protected_metric)),
            "dual_metric_protected_structural": int(len(protected_structural)),
            "dual_metric_metric_full_scan": bool(metric_details["used_full_scan"]),
            "dual_metric_no_rerank_ids": dual_metric_no_rerank_ids,
            "metric_prefilter_applied": True,
            "metric_prefilter_input_count": int(len(universe)),
            "metric_prefilter_candidate_count": int(len(selected)),
            "metric_prefilter_protected_global": int(len(protected_global)),
            "metric_prefilter_protected_structural": int(len(protected_structural)),
            "dual_metric_difficulty_bucket": difficulty_bucket,
            "dual_metric_fusion_variant": fusion_variant,
            "dual_metric_fusion_weights": weights.astype(float).tolist(),
        }
        if collect_fusion_features:
            details["dual_metric_fusion_features"] = {
                "universe_ids": universe.copy(),
                "rank_features": np.column_stack(
                    (embedding_rank, metric_rank, structural_rank)
                ).astype(np.float32, copy=False),
                "protected_mask": np.isin(universe, protected),
                "fill_budget": int(max(int(tp_budget) - len(protected), 0)),
                "bucket": difficulty_bucket,
            }
        return selected, details

    def query_adaptive(
        self,
        query_embedding,
        query_trajectory,
        query_code,
        top_k=50,
        *,
        metric_backend=None,
        query_scale_embeddings=None,
        query_scale_weights=None,
        exclude_ids=None,
        global_cache_override=None,
        force_global_fast_path=False,
        force_metric_rerank=False,
        metric_query_code=None,
        metric_hash_query_code=None,
        dual_metric_mode=False,
        dual_metric_no_rerank=False,
        collect_dual_metric_features=False,
    ):
        cfg = self.config
        preserve_global_top50 = bool(
            cfg.adaptive_preserve_global_top50 or dual_metric_mode
        )
        if preserve_global_top50:
            if (
                metric_backend is None
                or getattr(metric_backend, "name", "embedding") == "embedding"
            ):
                raise ValueError(
                    "adaptive_preserve_global_top50 requires a non-embedding "
                    "metric backend."
                )
            if not dual_metric_mode and cfg.metric_prefilter_candidates < 50:
                raise ValueError(
                    "adaptive_preserve_global_top50 requires at least 50 regular "
                    "metric prefilter candidates."
                )
            if not dual_metric_mode and cfg.metric_prefilter_high_quality_candidates < 50:
                raise ValueError(
                    "adaptive_preserve_global_top50 requires at least 50 high-quality "
                    "metric prefilter candidates."
                )
        if (
            dual_metric_mode
            and min(
                cfg.dual_metric_high_quality_tp_candidates,
                cfg.dual_metric_tp_candidates,
                cfg.dual_metric_hard_tp_candidates,
            ) < top_k
        ):
            raise ValueError(
                "All adaptive-dual-metric TP budgets must be >= top_k."
            )
        if (
            metric_backend is not None
            and getattr(metric_backend, "name", "embedding") != "embedding"
            and not dual_metric_mode
            and cfg.metric_prefilter_candidates < top_k
        ):
            raise ValueError(
                "metric_prefilter_candidates must be >= top_k so metric "
                "reranking can return the requested result count."
            )
        global_cache = None
        global_fast_path = False
        global_margin = 0.0
        global_top50_ids = None
        global_baseline_limit = int(
            getattr(self.hash_index.config, "hamming_candidates", 0)
        )
        if preserve_global_top50 and global_baseline_limit < 50:
            raise ValueError(
                "adaptive_preserve_global_top50 requires at least 50 Global Hash "
                "Hamming candidates."
            )
        global_search_limit = cfg.adaptive_max_global
        if preserve_global_top50:
            global_search_limit = max(
                global_search_limit,
                global_baseline_limit,
            )
        initial_global_limit = (
            cfg.adaptive_initial_global
            if preserve_global_top50
            and cfg.adaptive_metric_fast_path == "skip"
            else global_search_limit
        )
        if global_cache_override is not None:
            global_cache = {
                "global_ids": np.asarray(
                    global_cache_override["global_ids"], dtype=np.int64
                ),
                "global_distances": np.asarray(
                    global_cache_override["global_distances"], dtype=np.int32
                ),
                "global_details": dict(global_cache_override.get("global_details", {})),
            }
            global_margin, confident = self._global_confidence(
                global_cache["global_distances"], cfg.adaptive_initial_global
            )
            global_fast_path = cfg.adaptive_gate == "global" or confident
        elif cfg.adaptive_gate in {"global", "hybrid"}:
            global_ids, global_distances, global_details = (
                self.hash_index.hamming_search(
                    query_code,
                    candidate_limit=initial_global_limit,
                    minimum_candidates=top_k,
                    exclude_ids=exclude_ids,
                )
            )
            global_cache = {
                "global_ids": np.asarray(global_ids, dtype=np.int64),
                "global_distances": np.asarray(global_distances, dtype=np.int32),
                "global_details": global_details,
            }
            global_margin, confident = self._global_confidence(
                global_distances, cfg.adaptive_initial_global
            )
            # ``global`` is an explicit Global-Hash-only fast mode.  The
            # ``hybrid`` mode uses the margin only as a gate and falls back to
            # topology rescue for uncertain queries.
            global_fast_path = cfg.adaptive_gate == "global" or confident

        if force_global_fast_path:
            global_fast_path = True
        elif force_metric_rerank:
            # The batch scheduler explicitly selected this query for the
            # expensive branch.  Do not let a stale/high margin override it.
            global_fast_path = False
        elif dual_metric_no_rerank:
            global_fast_path = False

        if global_fast_path:
            if preserve_global_top50:
                if cfg.adaptive_metric_fast_path == "metric":
                    candidate_ids = global_cache["global_ids"]
                    global_reference_ids = candidate_ids[:global_baseline_limit]
                    global_top50_ids, _ = self.hash_index.rerank_continuous(
                        query_embedding,
                        global_reference_ids,
                        min(50, len(global_reference_ids)),
                    )
                else:
                    # High-confidence queries keep the cheap Global embedding
                    # path.  The protected Global Top-50 is only materialized
                    # when a query falls through to the difficult TP path.
                    candidate_ids = global_cache["global_ids"][:cfg.adaptive_initial_global]
            else:
                candidate_ids = global_cache["global_ids"][:cfg.adaptive_initial_global]
            use_metric_fast_path = (
                metric_backend is not None
                and getattr(metric_backend, "name", "embedding") != "embedding"
                and cfg.adaptive_metric_fast_path == "metric"
            )
            if use_metric_fast_path:
                metric_candidate_ids, prefilter_details = self._metric_prefilter_candidates(
                    query_embedding,
                    candidate_ids,
                    global_ids=(
                        global_top50_ids
                        if global_top50_ids is not None else candidate_ids
                    ),
                    budget_override=max(
                        top_k, cfg.metric_prefilter_candidates
                    ),
                    required_ids=global_top50_ids,
                    query_code=(
                        metric_query_code
                        if metric_query_code is not None else query_code
                    ),
                )
                metric_distance = metric_backend.distances(
                    query_trajectory, self.trajectories[metric_candidate_ids]
                )
                order = _topk_indices(metric_distance, top_k)
                final_scores = metric_distance[order]
                rerank_name = metric_backend.name
                metric_rerank_skipped = False
            else:
                metric_candidate_ids = candidate_ids
                prefilter_details = {
                    "metric_prefilter_applied": False,
                    "metric_prefilter_input_count": int(len(candidate_ids)),
                    "metric_prefilter_candidate_count": int(len(candidate_ids)),
                    "metric_prefilter_protected_global": 0,
                    "metric_prefilter_protected_structural": 0,
                }
                metric_rerank_skipped = bool(
                    metric_backend is not None
                    and getattr(metric_backend, "name", "embedding") != "embedding"
                )
                embedding_distance = self._continuous_distances(
                    query_embedding, candidate_ids
                )
                order = _topk_indices(embedding_distance, top_k)
                final_scores = embedding_distance[order]
                rerank_name = "embedding"
            final_ids = metric_candidate_ids[order]
            empty = np.empty((0,), dtype=np.int64)
            route_details = {
                "l3_posting_candidates": 0,
                "l2_posting_candidates": 0,
                "l2_intersection_candidates": 0,
                "l2_fallback_added": 0,
                "route_before_cap": 0,
                "route_after_cap": 0,
                "expanded_l3_tokens": 0,
            }
            return final_ids, final_scores, {
                "route_ids": empty,
                "structural_candidate_ids": empty,
                "global_candidate_ids": candidate_ids,
                "union_candidate_ids": candidate_ids,
                "structural_candidate_count": 0,
                "global_candidate_count": int(len(candidate_ids)),
                "union_candidate_count": int(len(candidate_ids)),
                "global_core_count": int(len(candidate_ids)),
                "adaptive_stage": 0,
                "adaptive_difficult": False,
                "adaptive_overlap_ratio": 1.0,
                "adaptive_metric": rerank_name,
                "adaptive_gate": cfg.adaptive_gate,
                "adaptive_global_margin": global_margin,
                "adaptive_global_fast_path": True,
                "metric_rerank_skipped": metric_rerank_skipped,
                "adaptive_metric_fast_path": cfg.adaptive_metric_fast_path,
                "adaptive_rescue_quality": False,
                "adaptive_rescue_near_margin": False,
                "struct_global_overlap": 0,
                "final_global_only": int(len(final_ids)),
                "final_struct_only": 0,
                "final_overlap": 0,
                "top1_from_struct_only": False,
                "global_used_full_scan": bool(global_cache["global_details"]["used_full_scan"]),
                "hierarchy_rerank_enabled": False,
                "hierarchy_distance_available": False,
                "metric_candidate_ids": (
                    metric_candidate_ids if rerank_name != "embedding" else None
                ),
                "global_top50_ids": global_top50_ids,
                "tp_rerank_applied": bool(rerank_name != "embedding"),
                "adaptive_tp_budget_forced_fast_path": bool(force_global_fast_path),
                "adaptive_tp_budget_selected": bool(force_metric_rerank),
                "adaptive_dual_metric": bool(dual_metric_mode),
                **prefilter_details,
                **route_details,
            }

        source_cache = self._build_source_cache(
            query_trajectory,
            query_code,
            exclude_ids,
            global_limit=(
                initial_global_limit
                if preserve_global_top50
                and cfg.adaptive_metric_fast_path == "skip"
                and global_cache is None
                else (
                    global_search_limit
                    if preserve_global_top50 or global_cache is not None
                    else cfg.adaptive_initial_global
                )
            ),
            top_k=top_k,
            global_cache=global_cache,
        )
        first = self.retrieve_candidate_sets(
            query_trajectory,
            query_code,
            top_k=top_k,
            exclude_ids=exclude_ids,
            global_candidates=cfg.adaptive_initial_global,
            structural_candidates=cfg.adaptive_initial_structural,
            source_cache=source_cache,
        )
        global_core = first["global_candidate_ids"]
        structural_first = first["structural_candidate_ids"]
        overlap = len(np.intersect1d(global_core, structural_first, assume_unique=True))
        denominator = max(min(len(global_core), len(structural_first)), 1)
        overlap_ratio = float(overlap) / float(denominator)
        difficult = (
            len(global_core) < cfg.adaptive_initial_global
            or overlap_ratio < cfg.adaptive_overlap_threshold
            or first["route_after_cap"] < cfg.adaptive_min_route_recall
        )
        if force_metric_rerank:
            difficult = True
        if dual_metric_no_rerank:
            difficult = True
        active = first
        stage = 0
        if difficult:
            stage = 1
            if len(source_cache["global_ids"]) < cfg.adaptive_max_global:
                expanded_ids, expanded_distances, expanded_details = (
                    self.hash_index.hamming_search(
                        query_code,
                        candidate_limit=global_search_limit,
                        minimum_candidates=top_k,
                        exclude_ids=exclude_ids,
                    )
                )
                source_cache["global_ids"] = np.asarray(
                    expanded_ids, dtype=np.int64
                )
                source_cache["global_distances"] = np.asarray(
                    expanded_distances, dtype=np.int32
                )
                source_cache["global_details"] = expanded_details
            active = self.retrieve_candidate_sets(
                query_trajectory,
                query_code,
                top_k=top_k,
                exclude_ids=exclude_ids,
                global_candidates=cfg.adaptive_max_global,
                structural_candidates=cfg.adaptive_max_structural,
                source_cache=source_cache,
            )
        global_ids = np.asarray(active["global_candidate_ids"], dtype=np.int64)
        structural_ids = np.asarray(active["structural_candidate_ids"], dtype=np.int64)
        union_ids = np.asarray(active["union_candidate_ids"], dtype=np.int64)
        protect_global_for_tp = bool(
            preserve_global_top50
            and (difficult or cfg.adaptive_metric_fast_path == "metric")
        )
        if protect_global_for_tp:
            global_reference_ids = source_cache["global_ids"][:global_baseline_limit]
            global_top50_ids, _ = self.hash_index.rerank_continuous(
                query_embedding,
                global_reference_ids,
                min(50, len(global_reference_ids)),
            )
        # A low-confidence query is a high-quality rescue when the initial
        # Global and structural routes already agree strongly and no second
        # candidate expansion was needed.  It still goes through metric
        # reranking, but can use the smaller Top-50-safe budget.
        near_margin_rescue = bool(
            cfg.adaptive_global_margin > 0.0
            and global_margin
            >= cfg.adaptive_global_margin
            * cfg.adaptive_rescue_near_margin_fraction
        )
        high_quality_rescue = bool(
            not difficult
            and not near_margin_rescue
            and overlap_ratio >= cfg.adaptive_rescue_quality_overlap
            and len(global_core) >= cfg.adaptive_initial_global
            and first["route_after_cap"] >= cfg.adaptive_min_route_recall
        )
        # Always protect the Global core. Topology is a rescue set, never an
        # uncontrolled replacement of the stronger global candidates.
        protected_global = global_ids[: cfg.adaptive_max_rerank]
        if len(union_ids) > cfg.adaptive_max_rerank:
            rescue = structural_ids[~np.isin(structural_ids, protected_global)]
            rescue_limit = max(cfg.adaptive_max_rerank - len(protected_global), 0)
            if len(rescue) > rescue_limit:
                structural_distance = self._structural_distances(
                    active["features"], rescue
                )
                rescue = rescue[_topk_indices(structural_distance, rescue_limit)]
            candidate_ids = np.unique(
                np.concatenate((protected_global, rescue))
            ).astype(np.int64, copy=False)
        else:
            candidate_ids = np.unique(
                np.concatenate((protected_global, structural_ids))
            ).astype(np.int64, copy=False)
        if len(candidate_ids) > cfg.adaptive_max_rerank:
            candidate_ids = candidate_ids[: cfg.adaptive_max_rerank]
        if global_top50_ids is not None:
            remaining = candidate_ids[~np.isin(candidate_ids, global_top50_ids)]
            candidate_ids = np.concatenate(
                (
                    global_top50_ids,
                    remaining[: max(cfg.adaptive_max_rerank - len(global_top50_ids), 0)],
                )
            ).astype(np.int64, copy=False)

        features = active["features"]
        hierarchy_weight = cfg.hierarchy_embedding_weight
        topology_weight = (
            1.0 - cfg.embedding_weight - hierarchy_weight
            if cfg.topology_weight < 0.0 else cfg.topology_weight
        )
        # With embedding_weight=1, topology is only used to rescue candidates;
        # computing structural distances for the final ranking is redundant.
        structural_distance = None
        if topology_weight > 0.0:
            structural_distance = self._structural_distances(features, candidate_ids)
        structural_rank = (
            self._rank_fraction(structural_distance)
            if structural_distance is not None else 0.0
        )
        hierarchy_distance = self._hierarchy_distances(
            query_scale_embeddings, query_scale_weights, candidate_ids
        )
        prefilter_details = {
            "metric_prefilter_applied": False,
            "metric_prefilter_input_count": int(len(candidate_ids)),
            "metric_prefilter_candidate_count": int(len(candidate_ids)),
            "metric_prefilter_protected_global": 0,
            "metric_prefilter_protected_structural": 0,
        }
        metric_backend_enabled = bool(
            metric_backend is not None
            and getattr(metric_backend, "name", "embedding") != "embedding"
        )
        metric_only_difficult = bool(
            preserve_global_top50 and cfg.adaptive_metric_fast_path == "skip"
        )
        use_metric_rerank = bool(
            metric_backend_enabled
            and not dual_metric_no_rerank
            and (not metric_only_difficult or difficult)
        )
        metric_budget = 0
        dual_metric_no_rerank_ids = None
        if dual_metric_mode:
            hard_margin = bool(
                cfg.adaptive_global_margin > 0.0
                and global_margin
                < cfg.adaptive_global_margin
                * cfg.dual_metric_hard_margin_fraction
            )
            hard_rescue = bool(
                hard_margin
                or (
                    overlap_ratio < cfg.adaptive_overlap_threshold
                    and first["route_after_cap"] < cfg.adaptive_min_route_recall
                )
            )
            dual_high_quality = bool(
                not hard_rescue
                and overlap_ratio >= cfg.adaptive_rescue_quality_overlap
                and first["route_after_cap"] >= cfg.adaptive_min_route_recall
            )
            if hard_rescue:
                metric_budget = cfg.dual_metric_hard_tp_candidates
                difficulty_bucket = "hard"
            elif dual_high_quality:
                metric_budget = cfg.dual_metric_high_quality_tp_candidates
                difficulty_bucket = "high_quality"
            else:
                metric_budget = cfg.dual_metric_tp_candidates
                difficulty_bucket = "ordinary"
            if use_metric_rerank or dual_metric_no_rerank:
                fusion_variant = (
                    "no_rerank" if dual_metric_no_rerank else "rerank"
                )
                metric_candidate_ids, prefilter_details = self._dual_metric_candidates(
                    query_embedding,
                    features,
                    global_ids,
                    structural_ids,
                    metric_hash_query_code,
                    metric_query_code,
                    global_top50_ids,
                    metric_budget,
                    getattr(metric_backend, "name", "embedding"),
                    difficulty_bucket,
                    fusion_variant,
                    exclude_ids=exclude_ids,
                    collect_fusion_features=collect_dual_metric_features,
                )
                dual_metric_no_rerank_ids = prefilter_details.pop(
                    "dual_metric_no_rerank_ids", None
                )
            if use_metric_rerank:
                high_quality_rescue = dual_high_quality
        if use_metric_rerank and not dual_metric_mode:
            metric_budget = max(
                top_k,
                (
                    cfg.metric_prefilter_high_quality_candidates
                    if high_quality_rescue
                    else cfg.metric_prefilter_candidates
                ),
            )
            metric_candidate_ids, prefilter_details = self._metric_prefilter_candidates(
                query_embedding,
                candidate_ids,
                global_ids=(
                    global_top50_ids if global_top50_ids is not None else global_ids
                ),
                structural_ids=structural_ids,
                budget_override=metric_budget,
                required_ids=global_top50_ids,
                query_code=(
                    metric_query_code
                    if metric_query_code is not None else query_code
                ),
            )
        if use_metric_rerank:
            metric_distance = metric_backend.distances(
                query_trajectory, self.trajectories[metric_candidate_ids]
            )
            order = _topk_indices(metric_distance, top_k)
            final_scores = metric_distance[order]
            rerank_name = metric_backend.name
            ranked_candidate_ids = metric_candidate_ids
        elif dual_metric_no_rerank and dual_metric_no_rerank_ids is not None:
            ranked_candidate_ids = dual_metric_no_rerank_ids
            order = np.arange(min(top_k, len(ranked_candidate_ids)), dtype=np.int64)
            final_scores = np.arange(len(order), dtype=np.float32)
            rerank_name = "dual-hash-fused"
        else:
            ranked_candidate_ids = (
                dual_metric_no_rerank_ids
                if dual_metric_no_rerank_ids is not None
                else candidate_ids
            )
            embedding_distance = self._continuous_distances(
                query_embedding, ranked_candidate_ids
            )
            embedding_rank = self._rank_fraction(embedding_distance)
            if hierarchy_weight > 0.0 and hierarchy_distance is None:
                raise ValueError(
                    "Hierarchy reranking is enabled but scale_embeddings are missing."
                )
            if hierarchy_weight > 0.0:
                fused_score = (
                    cfg.embedding_weight * embedding_rank
                    + hierarchy_weight * self._rank_fraction(hierarchy_distance)
                    + topology_weight * structural_rank
                )
            else:
                fused_score = (
                    cfg.embedding_weight * embedding_rank
                    + (1.0 - cfg.embedding_weight) * structural_rank
                )
            order = _topk_indices(fused_score, top_k)
            final_scores = fused_score[order]
            rerank_name = "embedding"
        final_ids = ranked_candidate_ids[order]
        final_global = np.isin(final_ids, global_ids)
        final_struct = np.isin(final_ids, structural_ids)
        details = {
            "route_ids": active["route_ids"],
            "structural_candidate_ids": structural_ids,
            "global_candidate_ids": global_ids,
            "union_candidate_ids": candidate_ids,
            "structural_candidate_count": int(len(structural_ids)),
            "global_candidate_count": int(len(global_ids)),
            "union_candidate_count": int(len(candidate_ids)),
            "global_core_count": int(len(global_core)),
            "adaptive_stage": int(stage),
            "adaptive_difficult": bool(difficult),
            "adaptive_overlap_ratio": overlap_ratio,
            "adaptive_metric": rerank_name,
            "adaptive_gate": cfg.adaptive_gate,
            "adaptive_global_margin": global_margin,
            "adaptive_global_fast_path": False,
            "metric_rerank_skipped": bool(
                metric_backend_enabled and not use_metric_rerank
            ),
            "adaptive_metric_fast_path": cfg.adaptive_metric_fast_path,
            "adaptive_rescue_quality": high_quality_rescue,
            "adaptive_rescue_near_margin": near_margin_rescue,
            "metric_prefilter_budget": int(metric_budget)
            if use_metric_rerank
            else 0,
            "struct_global_overlap": int(
                len(np.intersect1d(structural_ids, global_ids, assume_unique=True))
            ),
            "final_global_only": int(np.count_nonzero(final_global & ~final_struct)),
            "final_struct_only": int(np.count_nonzero(final_struct & ~final_global)),
            "final_overlap": int(np.count_nonzero(final_global & final_struct)),
            "top1_from_struct_only": bool(
                len(final_ids) and final_struct[0] and not final_global[0]
            ),
            "global_used_full_scan": bool(active["global_used_full_scan"]),
            "hierarchy_rerank_enabled": bool(cfg.hierarchy_embedding_weight > 0.0),
            "hierarchy_distance_available": bool(hierarchy_distance is not None),
            "metric_candidate_ids": (
                metric_candidate_ids if use_metric_rerank else None
            ),
            "dual_metric_no_rerank_ids": (
                np.asarray(dual_metric_no_rerank_ids, dtype=np.int64)
                if dual_metric_no_rerank_ids is not None else None
            ),
            "global_top50_ids": global_top50_ids,
            "tp_rerank_applied": bool(use_metric_rerank),
            "adaptive_tp_budget_forced_fast_path": bool(force_global_fast_path),
            "adaptive_tp_budget_selected": bool(force_metric_rerank),
            "adaptive_dual_metric": bool(dual_metric_mode),
            "adaptive_dual_metric_no_rerank": bool(dual_metric_no_rerank),
            **prefilter_details,
        }
        details.update(
            {
                key: value
                for key, value in active.items()
                if key not in {
                    "features", "route_ids", "structural_candidate_ids",
                    "global_candidate_ids", "union_candidate_ids",
                    "global_used_full_scan",
                }
            }
        )
        return final_ids, final_scores, details

    def query_adaptive_dual_metric(
        self,
        query_embedding,
        query_trajectory,
        query_code,
        metric_hash_query_code,
        top_k=50,
        **kwargs,
    ):
        """Global fast path plus independent Metric-Head rescue for hard queries."""
        return self.query_adaptive(
            query_embedding,
            query_trajectory,
            query_code,
            top_k=top_k,
            metric_hash_query_code=metric_hash_query_code,
            dual_metric_mode=True,
            **kwargs,
        )


def _hash_config(args, head):
    from hash_global import HashGlobalConfig

    return HashGlobalConfig(
        bits=head.config.bits,
        subcode_bits=16,
        probe_radius=args.probe_radius,
        hamming_candidates=args.hamming_candidates,
        metric=args.metric,
        normalize=args.normalize,
        embedding_storage=args.embedding_storage,
        encode_batch_size=args.encode_batch_size,
        backend=getattr(args, "hash_backend", "mih"),
        flat_block_size=getattr(args, "hash_flat_block_size", 65536),
        hnsw_m=getattr(args, "hash_hnsw_m", 16),
        hnsw_ef_construction=getattr(args, "hash_hnsw_ef_construction", 200),
        hnsw_ef_search=getattr(args, "hash_hnsw_ef_search", 256),
    )


def _topo_config(args):
    return TopoRouteHashConfig(
        candidate_source=args.candidate_source,
        qgram=args.qgram,
        minhash_size=args.minhash_size,
        postings_per_level=args.postings_per_level,
        l3_expand_hops=args.l3_expand_hops,
        min_route_candidates=args.min_route_candidates,
        max_route_candidates=args.max_route_candidates,
        structural_candidates=args.structural_candidates,
        global_candidates=args.global_candidates,
        max_rerank_candidates=args.max_rerank_candidates,
        l3_weight=args.l3_weight,
        l2_weight=args.l2_weight,
        l1_weight=args.l1_weight,
        embedding_weight=args.embedding_weight,
        hierarchy_embedding_weight=args.hierarchy_embedding_weight,
        topology_weight=args.topology_weight,
        adaptive_initial_global=getattr(args, "adaptive_initial_global", 512),
        adaptive_initial_structural=getattr(args, "adaptive_initial_structural", 256),
        adaptive_max_global=getattr(args, "adaptive_max_global", 2048),
        adaptive_max_structural=getattr(args, "adaptive_max_structural", 1024),
        adaptive_max_rerank=getattr(args, "adaptive_max_rerank", 2048),
        adaptive_overlap_threshold=getattr(args, "adaptive_overlap_threshold", 0.35),
        adaptive_min_route_recall=getattr(args, "adaptive_min_route_recall", 128),
        adaptive_gate=getattr(args, "adaptive_gate", "overlap"),
        adaptive_global_margin=getattr(args, "adaptive_global_margin", 0.0),
        adaptive_metric_fast_path=getattr(args, "adaptive_metric_fast_path", "skip"),
        adaptive_max_tp_fraction=getattr(args, "adaptive_max_tp_fraction", 0.15),
        adaptive_rescue_quality_overlap=getattr(
            args, "adaptive_rescue_quality_overlap", 0.75
        ),
        adaptive_rescue_near_margin_fraction=getattr(
            args, "adaptive_rescue_near_margin_fraction", 0.80
        ),
        adaptive_preserve_global_top50=getattr(
            args, "adaptive_preserve_global_top50", False
        ),
        metric_prefilter_candidates=getattr(args, "metric_prefilter_candidates", 256),
        metric_prefilter_high_quality_candidates=getattr(
            args, "metric_prefilter_high_quality_candidates", 50
        ),
        metric_prefilter_global_protect=getattr(args, "metric_prefilter_global_protect", 64),
        metric_prefilter_structural_protect=getattr(args, "metric_prefilter_structural_protect", 32),
        metric_prefilter_embedding_weight=getattr(
            args, "metric_prefilter_embedding_weight", 1.0
        ),
        metric_prefilter_hash_weight=getattr(
            args, "metric_prefilter_hash_weight", 0.0
        ),
        metric_prefilter_hash_score=getattr(
            args, "metric_prefilter_hash_score", "hamming"
        ),
        metric_prefilter_soft_temperature=getattr(
            args, "metric_prefilter_soft_temperature", 1.0
        ),
        dual_metric_search_candidates=getattr(
            args, "dual_metric_search_candidates", 256
        ),
        dual_metric_candidates=getattr(args, "dual_metric_candidates", 96),
        dual_metric_tp_candidates=getattr(
            args, "dual_metric_tp_candidates", 128
        ),
        dual_metric_high_quality_tp_candidates=getattr(
            args, "dual_metric_high_quality_tp_candidates", 64
        ),
        dual_metric_hard_tp_candidates=getattr(
            args, "dual_metric_hard_tp_candidates", 160
        ),
        dual_metric_hard_margin_fraction=getattr(
            args, "dual_metric_hard_margin_fraction", 0.50
        ),
        dual_metric_head_protect=getattr(
            args, "dual_metric_head_protect", 48
        ),
        dual_metric_structural_protect=getattr(
            args, "dual_metric_structural_protect", 16
        ),
        dual_metric_embedding_weight=getattr(
            args, "dual_metric_embedding_weight", 0.40
        ),
        dual_metric_hash_weight=getattr(
            args, "dual_metric_hash_weight", 0.40
        ),
        dual_metric_topology_weight=getattr(
            args, "dual_metric_topology_weight", 0.20
        ),
        seed=args.selection_seed,
    )


def _build_index(args, gallery, selected_rows, scale_dir):
    from hash_global import (
        HashGlobalIndex,
        encode_soft_codes,
        load_hash_head,
    )

    embeddings = gallery["embeddings"][selected_rows]
    trajectories = gallery["trajectories"][selected_rows]
    scale_embeddings = gallery.get("scale_embeddings")
    if scale_embeddings is not None:
        scale_embeddings = scale_embeddings[selected_rows]
    head, head_metadata = load_hash_head(args.hash_model_file)
    hash_dir = osp.join(scale_dir, "hash_index")
    topo_dir = osp.join(scale_dir, "topo_index")
    start = perf_counter()
    hash_index = HashGlobalIndex(_hash_config(args, head)).build(
        embeddings,
        head,
        encode_device=args.hash_device,
        metadata={"hash_training": head_metadata},
    )
    hash_seconds = perf_counter() - start
    hash_index.save(hash_dir)
    methods = {
        item.strip()
        for item in str(getattr(args, "methods", "topo-route-hash")).split(",")
        if item.strip()
    }
    topology_methods = {
        "topo-route-hash",
        "adaptive-topo-hash",
        "adaptive-dual-metric",
        "adaptive-dual-metric-no-rerank",
    }
    needs_topology = bool(methods.intersection(topology_methods))
    topo_config = _topo_config(args)
    if needs_topology:
        hierarchy = load_hierarchy_arrays(args.hierarchy_dir)
        topo_index = AdaptiveTopoHashIndex(hash_index, hierarchy, topo_config).build(
            trajectories, scale_embeddings=scale_embeddings
        )
    else:
        # Global Hash and Exact do not consume path postings. Avoid building
        # and saving a full topology index for hash-only large-scale runs.
        topo_index = SimpleNamespace(
            hash_index=hash_index,
            metric_hash_index=None,
            trajectories=trajectories,
            config=topo_config,
            build_seconds=0.0,
        )
    metric_hash_seconds = 0.0
    metric_hash_index = None
    metric_hash_model_file = getattr(args, "metric_hash_model_file", None)
    if metric_hash_model_file and needs_topology:
        metric_head, metric_head_metadata = load_hash_head(metric_hash_model_file)
        if metric_head.config.bits != head.config.bits:
            raise ValueError(
                "Candidate and metric hash heads must use the same bit count."
            )
        metric_start = perf_counter()
        metric_hash_index = HashGlobalIndex(_hash_config(args, metric_head)).build(
            embeddings,
            metric_head,
            encode_device=args.hash_device,
            metadata={"hash_training": metric_head_metadata, "role": "metric"},
        )
        topo_index.metric_hash_index = metric_hash_index
        if args.metric_prefilter_hash_score == "soft":
            topo_index.metric_prefilter_soft_codes = encode_soft_codes(
                metric_head,
                embeddings,
                batch_size=args.encode_batch_size,
                device=args.hash_device,
                temperature=args.metric_prefilter_soft_temperature,
            )
        else:
            topo_index.metric_prefilter_codes = metric_hash_index.codes
        metric_hash_seconds = perf_counter() - metric_start
    dual_metric_weight_file = getattr(args, "dual_metric_weight_file", None)
    if dual_metric_weight_file and needs_topology:
        topo_index.set_dual_metric_weight_payload(
            load_weight_payload(dual_metric_weight_file, metric_name=args.rerank_metric)
        )
    if needs_topology:
        topo_index.save(topo_dir)
    np.save(osp.join(scale_dir, "selected_gallery_rows.npy"), selected_rows)
    return topo_index, {
        "hash_seconds": float(hash_seconds),
        "hash_backend": hash_index.config.backend,
        "topo_seconds": float(topo_index.build_seconds),
        "metric_hash_seconds": float(metric_hash_seconds),
        "metric_hash_backend": (
            metric_hash_index.config.backend if metric_hash_index is not None else None
        ),
        "total_seconds": float(
            hash_seconds + metric_hash_seconds + topo_index.build_seconds
        ),
        "hash_disk_mb": float(_directory_size(hash_dir) / (1024 ** 2)),
        "topo_disk_mb": (
            float(_directory_size(topo_dir) / (1024 ** 2))
            if needs_topology
            else 0.0
        ),
    }


def _query_global_hash_tp(
    index, query_trajectory, query_code, top_k, metric_backend, exclude_ids=None
):
    """Rerank the entire Hamming candidate pool by online TP, without gating."""
    if metric_backend is None or metric_backend.name != "TP":
        raise ValueError("Global Hash TP reranking requires a TP metric backend.")
    if getattr(index, "trajectories", None) is None:
        raise ValueError("Global Hash TP reranking requires Gallery trajectories.")
    start = perf_counter()
    candidate_ids, _, details = index.hash_index.hamming_search(
        query_code,
        candidate_limit=index.hash_index.config.hamming_candidates,
        minimum_candidates=top_k,
        exclude_ids=exclude_ids,
    )
    search_ms = (perf_counter() - start) * 1000.0
    start = perf_counter()
    # Compute from trajectories and the point-distance matrix, never from the
    # evaluation-only Query x Gallery TP table. This cost is timed per query.
    distances = (
        np.asarray(metric_backend.distances(
            query_trajectory, index.trajectories[candidate_ids]
        ), dtype=np.float32)
        if len(candidate_ids)
        else np.empty(0, dtype=np.float32)
    )
    finite = np.flatnonzero(np.isfinite(distances))
    # Tie-break by Gallery row ID, matching exact TP ground truth rather than
    # the incoming Hamming order. Invalid distances do not enter the output.
    order = finite[np.lexsort((candidate_ids[finite], distances[finite]))][:top_k]
    rerank_ms = (perf_counter() - start) * 1000.0
    return candidate_ids[order], distances[order], {
        **details,
        "candidate_ids": candidate_ids,
        "rerank_candidate_count": int(len(candidate_ids)),
        "invalid_tp_candidate_count": int(len(candidate_ids) - len(finite)),
        "hamming_search_ms": search_ms,
        "tp_rerank_ms": rerank_ms,
    }


def _evaluate_method(
    method,
    index,
    query_embeddings,
    query_trajectories,
    query_source_ids,
    gallery_embeddings,
    gallery_source_ids,
    gallery_positive_mask,
    query_labels=None,
    query_codes=None,
    top_k=50,
    exact_max_queries=0,
    query_scale_embeddings=None,
    query_scale_weights=None,
    code_encoding_ms_per_query=0.0,
    query_metadata=None,
    metric_backend=None,
    diagnostic_state=None,
    query_metric_codes=None,
    query_metric_hash_codes=None,
    global_hash_rerank_mode="full",
):
    # Backward compatibility: older callers passed query_codes in the
    # query_labels position and had no separate label array.
    if query_labels is not None:
        candidate_labels = np.asarray(query_labels)
        candidate_codes = np.asarray(query_codes) if query_codes is not None else None
        if candidate_labels.dtype == np.uint8 and (
            candidate_codes is None or candidate_codes.ndim == 0
        ):
            # Original positional order: ..., positive_mask, query_codes,
            # top_k, exact_max_queries.
            old_codes = query_labels
            old_top_k = int(candidate_codes.item()) if candidate_codes is not None else int(top_k)
            old_exact = int(top_k) if candidate_codes is not None else int(exact_max_queries)
            query_codes = old_codes
            query_labels = None
            exact_max_queries = old_exact
            top_k = old_top_k
    if query_codes is None:
        raise TypeError("query_codes is required for hash-based retrieval methods.")
    total = len(query_embeddings)
    if method == "exact" and exact_max_queries > 0:
        total = min(total, exact_max_queries)
    predictions = []
    latencies = []
    global_tp_details = []
    candidate_lists = {
        "route": [], "structural": [], "global": [], "union": [], "metric": []
    }
    candidate_query_ids = {name: [] for name in candidate_lists}
    details_counts = {
        "route_before_cap": [],
        "route_after_cap": [],
        "l3_posting_candidates": [],
        "l2_posting_candidates": [],
        "l2_intersection_candidates": [],
        "l2_fallback_added": [],
        "expanded_l3_tokens": [],
        "structural_candidate_count": [],
        "global_candidate_count": [],
        "union_candidate_count": [],
        "struct_global_overlap": [],
        "final_global_only": [],
        "final_struct_only": [],
        "final_overlap": [],
        "top1_from_struct_only": [],
        "adaptive_stage": [],
        "adaptive_difficult": [],
        "adaptive_overlap_ratio": [],
        "adaptive_global_fast_path": [],
        "adaptive_rescue_quality": [],
        "adaptive_rescue_near_margin": [],
        "metric_rerank_skipped": [],
        "metric_prefilter_applied": [],
        "metric_prefilter_input_count": [],
        "metric_prefilter_candidate_count": [],
        "metric_prefilter_budget": [],
        "metric_prefilter_protected_global": [],
        "metric_prefilter_protected_structural": [],
        "adaptive_tp_budget_forced_fast_path": [],
        "adaptive_tp_budget_selected": [],
        "dual_metric_search_count": [],
        "dual_metric_union_count": [],
        "dual_metric_tp_budget": [],
        "dual_metric_protected_global": [],
        "dual_metric_protected_metric": [],
        "dual_metric_protected_structural": [],
        "dual_metric_metric_full_scan": [],
    }
    # A batch-level TP budget prevents a growing gallery from turning an
    # increasing fraction of queries into expensive topology/metric rescues.
    # We prefetch the cheap Global result once, rank low-confidence queries by
    # their Hamming margin, and reuse the cache inside query_adaptive.
    adaptive_global_caches = None
    adaptive_tp_allowed = None
    adaptive_low_confidence = None
    adaptive_probe_ms = np.zeros(total, dtype=np.float64)
    adaptive_methods = {
        "adaptive-topo-hash",
        "adaptive-dual-metric",
        "adaptive-dual-metric-no-rerank",
    }
    if method in adaptive_methods and metric_backend is not None:
        cfg = index.config
        dual_metric_method = method in {
            "adaptive-dual-metric", "adaptive-dual-metric-no-rerank"
        }
        if (
            cfg.adaptive_metric_fast_path == "skip"
            and cfg.adaptive_gate in {"global", "hybrid"}
            and cfg.adaptive_max_tp_fraction < 1.0
        ):
            baseline_limit = int(getattr(index.hash_index.config, "hamming_candidates", 0))
            global_search_limit = int(cfg.adaptive_max_global)
            if cfg.adaptive_preserve_global_top50 or dual_metric_method:
                global_search_limit = max(global_search_limit, baseline_limit)
            initial_limit = (
                int(cfg.adaptive_initial_global)
                if cfg.adaptive_preserve_global_top50 or dual_metric_method
                else global_search_limit
            )
            adaptive_global_caches = []
            margins = np.full(total, np.inf, dtype=np.float64)
            for query_id in range(total):
                exclude_ids = None
                if getattr(index, "exclude_self", False):
                    exclude_ids = np.flatnonzero(
                        np.asarray(gallery_source_ids) == query_source_ids[query_id]
                    ).astype(np.int64, copy=False)
                probe_start = perf_counter()
                ids, distances, global_details = index.hash_index.hamming_search(
                    query_codes[query_id],
                    candidate_limit=initial_limit,
                    minimum_candidates=top_k,
                    exclude_ids=exclude_ids,
                )
                adaptive_probe_ms[query_id] = (perf_counter() - probe_start) * 1000.0
                margins[query_id], confident = index._global_confidence(
                    distances, cfg.adaptive_initial_global
                )
                adaptive_global_caches.append(
                    {
                        "global_ids": np.asarray(ids, dtype=np.int64),
                        "global_distances": np.asarray(distances, dtype=np.int32),
                        "global_details": global_details,
                    }
                )
            adaptive_low_confidence, adaptive_tp_allowed = (
                _select_low_confidence_queries(
                    margins,
                    cfg.adaptive_global_margin,
                    cfg.adaptive_max_tp_fraction,
                )
            )
    full_scan = 0
    exact_gallery = None
    if method == "exact":
        exact_gallery = np.asarray(gallery_embeddings, dtype=np.float32)
        if index.hash_index.config.normalize or index.hash_index.config.metric == "cosine":
            exact_gallery = exact_gallery / np.maximum(
                np.linalg.norm(exact_gallery, axis=1, keepdims=True), 1e-12
            )
    try:
        from tqdm import tqdm

        iterator = tqdm(range(total), desc=f"{method} queries")
    except ImportError:
        iterator = range(total)
    for query_id in iterator:
        exclude_ids = None
        if getattr(index, "exclude_self", False):
            # Candidate IDs are gallery row indices, whereas source IDs may
            # be arbitrary labels. Exclude every gallery row representing the
            # current query source, including duplicate source rows.
            exclude_ids = np.flatnonzero(
                np.asarray(gallery_source_ids) == query_source_ids[query_id]
            ).astype(np.int64, copy=False)
        start = perf_counter()
        if method == "topo-route-hash":
            ids, _, details = index.query(
                query_embeddings[query_id],
                query_trajectories[query_id],
                query_codes[query_id],
                top_k=top_k,
                query_scale_embeddings=(
                    query_scale_embeddings[query_id]
                    if query_scale_embeddings is not None else None
                ),
                query_scale_weights=(
                    query_scale_weights[query_id]
                    if query_scale_weights is not None else None
                ),
                exclude_ids=exclude_ids,
            )
        elif method in adaptive_methods:
            no_rerank_method = method == "adaptive-dual-metric-no-rerank"
            force_global_fast_path = bool(
                (no_rerank_method and adaptive_low_confidence is not None
                 and not adaptive_low_confidence[query_id])
                or (
                    not no_rerank_method
                    and adaptive_low_confidence is not None
                    and adaptive_low_confidence[query_id]
                    and not adaptive_tp_allowed[query_id]
                )
            )
            query_kwargs = {
                "top_k": top_k,
                "metric_backend": metric_backend,
                "query_scale_embeddings": (
                    query_scale_embeddings[query_id]
                    if query_scale_embeddings is not None else None
                ),
                "query_scale_weights": (
                    query_scale_weights[query_id]
                    if query_scale_weights is not None else None
                ),
                "exclude_ids": exclude_ids,
                "global_cache_override": (
                    adaptive_global_caches[query_id]
                    if adaptive_global_caches is not None else None
                ),
                "force_global_fast_path": force_global_fast_path,
                "force_metric_rerank": bool(
                    not no_rerank_method
                    and
                    adaptive_tp_allowed is not None
                    and adaptive_tp_allowed[query_id]
                ),
                "dual_metric_no_rerank": no_rerank_method,
                "metric_query_code": (
                    query_metric_codes[query_id]
                    if query_metric_codes is not None else None
                ),
            }
            if method in {
                "adaptive-dual-metric", "adaptive-dual-metric-no-rerank"
            }:
                if query_metric_hash_codes is None:
                    raise ValueError(
                        "adaptive-dual-metric requires query_metric_hash_codes."
                    )
                ids, _, details = index.query_adaptive_dual_metric(
                    query_embeddings[query_id],
                    query_trajectories[query_id],
                    query_codes[query_id],
                    query_metric_hash_codes[query_id],
                    **query_kwargs,
                )
            else:
                ids, _, details = index.query_adaptive(
                    query_embeddings[query_id],
                    query_trajectories[query_id],
                    query_codes[query_id],
                    **query_kwargs,
                )
        elif method == "global-hash":
            if global_hash_rerank_mode == "tp":
                ids, _, details = _query_global_hash_tp(
                    index, query_trajectories[query_id], query_codes[query_id],
                    top_k, metric_backend, exclude_ids=exclude_ids,
                )
                global_tp_details.append({
                    name: details[name] for name in (
                        "rerank_candidate_count", "invalid_tp_candidate_count",
                        "hamming_search_ms", "tp_rerank_ms",
                    )
                })
            else:
                ids, _, details = index.hash_index.query(
                    query_embeddings[query_id],
                    top_k=top_k,
                    exclude_ids=exclude_ids,
                    return_details=True,
                    query_code=query_codes[query_id],
                    rerank_mode=global_hash_rerank_mode,
                )
            candidate_lists["global"].append(details["candidate_ids"])
            candidate_query_ids["global"].append(query_id)
            full_scan += int(details["used_full_scan"])
        elif method == "exact":
            query = np.asarray(query_embeddings[query_id], dtype=np.float32)
            if index.hash_index.config.normalize or index.hash_index.config.metric == "cosine":
                query = query / max(float(np.linalg.norm(query)), 1e-12)
            if index.hash_index.config.metric == "cosine":
                distance = 1.0 - exact_gallery @ query
            else:
                difference = exact_gallery - query.reshape(1, -1)
                distance = np.einsum("ij,ij->i", difference, difference)
            if exclude_ids is not None and len(exclude_ids):
                for exclude_id in exclude_ids:
                    exclude_id = int(exclude_id)
                    if 0 <= exclude_id < len(distance):
                        distance[exclude_id] = np.inf
            ids = _topk_indices(distance, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        if method in {"topo-route-hash", *adaptive_methods}:
            candidate_lists["route"].append(details["route_ids"])
            candidate_lists["structural"].append(details["structural_candidate_ids"])
            candidate_lists["global"].append(details["global_candidate_ids"])
            candidate_lists["union"].append(details["union_candidate_ids"])
            for name in ("route", "structural", "global", "union"):
                candidate_query_ids[name].append(query_id)
            if details.get("metric_candidate_ids") is not None:
                candidate_lists["metric"].append(details["metric_candidate_ids"])
                candidate_query_ids["metric"].append(query_id)
            for name in details_counts:
                if name.startswith("adaptive_") and method not in adaptive_methods:
                    continue
                if name not in details:
                    continue
                value = details[name]
                details_counts[name].append(
                    float(value) if name == "adaptive_overlap_ratio" else int(value)
                )
            full_scan += int(details["global_used_full_scan"])
            if diagnostic_state is not None and method in adaptive_methods:
                global_top50 = details.get("global_top50_ids")
                metric_pool = details.get("metric_candidate_ids")
                diagnostic_state.setdefault("queries", 0)
                diagnostic_state["queries"] += 1
                diagnostic_state.setdefault("global_top50", []).append(
                    None
                    if global_top50 is None
                    else np.asarray(global_top50, dtype=np.int64)
                )
                diagnostic_state.setdefault("metric_pool", []).append(
                    None
                    if metric_pool is None
                    else np.asarray(metric_pool, dtype=np.int64)
                )
                diagnostic_state.setdefault("final", []).append(
                    np.asarray(ids, dtype=np.int64)
                )
                diagnostic_state.setdefault("tp_rerank_applied", []).append(
                    bool(details.get("tp_rerank_applied", metric_pool is not None))
                )
        latencies.append(
            (perf_counter() - start) * 1000.0
            + (adaptive_probe_ms[query_id] if method in adaptive_methods else 0.0)
        )
        predictions.append(np.asarray(ids, dtype=np.int64))

    latency = np.asarray(latencies, dtype=np.float64)
    encoding_overhead = (
        0.0 if method == "exact" else float(code_encoding_ms_per_query)
    )
    end_to_end_mean = (
        float(latency.mean() + encoding_overhead) if len(latency) else 0.0
    )
    if query_labels is not None:
        metrics = _row_retrieval_metrics(predictions, query_labels[:total])
    else:
        metrics = source_retrieval_metrics(
            predictions,
            query_source_ids[:total],
            gallery_source_ids,
            ks=tuple(sorted({top_k, *(k for k in (1, 10, 50) if k <= top_k)})),
            gallery_positive_mask=gallery_positive_mask,
        )

    candidate_recall = {
        name: {
            **(
                _candidate_label_stats(
                    lists,
                    query_labels[:total],
                    candidate_query_ids[name],
                )
                if query_labels is not None
                else _candidate_source_stats(
                    lists,
                    np.asarray(query_source_ids)[candidate_query_ids[name]],
                    gallery_source_ids,
                    gallery_positive_mask=gallery_positive_mask,
                )
            ),
            "evaluated_queries": int(len(lists)),
        }
        for name, lists in candidate_lists.items()
        if lists
    }
    if query_labels is not None and candidate_lists["metric"]:
        metric_query_ids = np.asarray(
            candidate_query_ids["metric"], dtype=np.int64
        )
        union_by_query = {
            int(query_id): candidates
            for query_id, candidates in zip(
                candidate_query_ids["union"], candidate_lists["union"]
            )
        }
        aligned_union = [union_by_query[int(query_id)] for query_id in metric_query_ids]
        candidate_recall["union-tp-queries"] = {
            **_candidate_label_stats(
                aligned_union, query_labels[:total], metric_query_ids
            ),
            "evaluated_queries": int(len(metric_query_ids)),
        }
        candidate_recall["metric-retention"] = {
            **_candidate_retention_stats(
                aligned_union,
                candidate_lists["metric"],
                query_labels[:total],
                metric_query_ids,
            ),
            "evaluated_queries": int(len(metric_query_ids)),
        }

    result = {
        "method": method,
        "queries": int(total),
        "metrics": metrics,
        "latency_ms": {
            "mean": end_to_end_mean,
            "p50": float(np.percentile(latency, 50) + encoding_overhead)
            if len(latency) else 0.0,
            "p95": float(np.percentile(latency, 95) + encoding_overhead)
            if len(latency) else 0.0,
            "retrieval_mean": float(latency.mean()) if len(latency) else 0.0,
            "code_encoding_per_query": encoding_overhead,
        },
        "qps": float(1000.0 / end_to_end_mean) if end_to_end_mean else 0.0,
        "candidate_recall": candidate_recall,
        "candidate_counts": {
            **{
                f"{name}_mean": float(np.mean(values))
                for name, values in details_counts.items()
                if values
            },
            "global_full_scan_queries": int(full_scan),
        },
        "final_provenance": (
            {
                name: _distribution_stats(values)
                for name, values in details_counts.items()
                if name.startswith("final_") or name == "top1_from_struct_only"
            }
            if method in {"topo-route-hash", *adaptive_methods}
            else None
        ),
    }
    if method == "global-hash" and global_hash_rerank_mode == "tp":
        counts = [item["rerank_candidate_count"] for item in global_tp_details]
        result["tp_rerank"] = {
            "queries": len(global_tp_details),
            "fraction": len(global_tp_details) / total if total else 0.0,
            "gated": False,
            "candidate_limit": int(index.hash_index.config.hamming_candidates),
            "candidate_count_mean": float(np.mean(counts)) if counts else 0.0,
            "candidate_count_min": min(counts, default=0),
            "candidate_count_max": max(counts, default=0),
            "invalid_candidate_count": sum(
                item["invalid_tp_candidate_count"] for item in global_tp_details
            ),
            "distance_source": "online-trajectories",
            "tie_break": "gallery-row-id",
            "hamming_search_mean_ms": float(np.mean([
                item["hamming_search_ms"] for item in global_tp_details
            ])) if global_tp_details else 0.0,
            "tp_rerank_mean_ms": float(np.mean([
                item["tp_rerank_ms"] for item in global_tp_details
            ])) if global_tp_details else 0.0,
        }
    if method in adaptive_methods:
        eligible_count = int(
            np.count_nonzero(adaptive_low_confidence)
            if adaptive_low_confidence is not None else 0
        )
        selected_count = int(
            np.count_nonzero(adaptive_tp_allowed)
            if adaptive_tp_allowed is not None else 0
        )
        result["adaptive_tp_budget"] = {
            "max_fraction": float(index.config.adaptive_max_tp_fraction),
            "max_queries": int(
                np.floor(total * index.config.adaptive_max_tp_fraction + 1e-12)
            ),
            "low_confidence_queries": eligible_count,
            "selected_queries": selected_count,
            "forced_fast_path_queries": max(eligible_count - selected_count, 0),
            "selected_fraction": float(selected_count / max(total, 1)),
        }
        if adaptive_low_confidence is not None:
            eligible_ids = np.flatnonzero(adaptive_low_confidence).astype(
                np.int64, copy=False
            )
            selected_ids = np.flatnonzero(
                adaptive_low_confidence & adaptive_tp_allowed
            ).astype(np.int64, copy=False)
            result["adaptive_tp_budget"].update({
                "low_confidence_query_ids": eligible_ids.tolist(),
                "selected_query_ids": selected_ids.tolist(),
                "low_confidence_latency_mean_ms": (
                    float(np.mean(latency[eligible_ids]))
                    if len(eligible_ids) else None
                ),
                "selected_latency_mean_ms": (
                    float(np.mean(latency[selected_ids]))
                    if len(selected_ids) else None
                ),
            })
    query_metadata = query_metadata or {}
    if "perturbation_types" in query_metadata and "severities" in query_metadata:
        perturbation_types = np.asarray(query_metadata["perturbation_types"])[:total]
        severities = np.asarray(query_metadata["severities"], dtype=np.float32)[:total]
        groups = []
        group_keys = sorted(
            {
                (str(kind), round(float(severity), 6))
                for kind, severity in zip(perturbation_types, severities)
            }
        )
        for kind, severity in group_keys:
            indices = np.flatnonzero(
                (perturbation_types == kind) & np.isclose(severities, severity)
            )
            group_predictions = [predictions[index] for index in indices]
            group = {
                "perturbation_type": kind,
                "severity": float(severity),
                "queries": int(len(indices)),
                "metrics": source_retrieval_metrics(
                    group_predictions,
                    query_source_ids[indices],
                    gallery_source_ids,
                    ks=tuple(
                        sorted({top_k, *(k for k in (1, 10, 50) if k <= top_k)})
                    ),
                    gallery_positive_mask=gallery_positive_mask,
                ),
            }
            for name in (
                "effective_severities",
                "actual_replacement_counts",
                "path_valid_ratios",
            ):
                if name not in query_metadata:
                    continue
                values = np.asarray(query_metadata[name], dtype=np.float64)[:total][
                    indices
                ]
                finite = values[np.isfinite(values)]
                group[f"{name}_mean"] = (
                    float(finite.mean()) if len(finite) else None
                )
            groups.append(group)
        result["groups"] = groups
    else:
        result["groups"] = []
    return result, predictions


def _positive_hit_flags(
    predictions, query_source_ids, gallery_source_ids, positive_mask, cutoff
):
    flags = []
    for prediction, source_id in zip(predictions, query_source_ids):
        rows = np.asarray(prediction, dtype=np.int64)[:cutoff]
        flags.append(
            bool(
                np.any(
                    (gallery_source_ids[rows] == source_id) & positive_mask[rows]
                )
            )
        )
    return np.asarray(flags, dtype=bool)


def _method_comparison(
    predictions_by_method,
    query_source_ids,
    gallery_source_ids,
    positive_mask,
    top_k,
):
    topology_method = next(
        (
            method
            for method in ("adaptive-topo-hash", "topo-route-hash")
            if method in predictions_by_method
        ),
        None,
    )
    if topology_method is None or "global-hash" not in predictions_by_method:
        return None
    topo = predictions_by_method[topology_method]
    global_predictions = predictions_by_method["global-hash"]
    total = min(len(topo), len(global_predictions))
    result = {}
    for cutoff in sorted({top_k, *(k for k in (1, 10, 50) if k <= top_k)}):
        topo_hit = _positive_hit_flags(
            topo[:total], query_source_ids[:total], gallery_source_ids, positive_mask, cutoff
        )
        global_hit = _positive_hit_flags(
            global_predictions[:total],
            query_source_ids[:total],
            gallery_source_ids,
            positive_mask,
            cutoff,
        )
        result[f"struct_rescue_at_{cutoff}"] = float(
            np.mean(topo_hit & ~global_hit)
        )
        result[f"global_only_hit_at_{cutoff}"] = float(
            np.mean(global_hit & ~topo_hit)
        )
    return result


def _global_top50_diagnostics(state, tp_truth_ids=None):
    global_top50 = state.get("global_top50", []) if state else []
    metric_pools = state.get("metric_pool", []) if state else []
    final_predictions = state.get("final", []) if state else []
    total = int(
        state.get("queries", min(len(global_top50), len(metric_pools), len(final_predictions)))
        if state else 0
    )
    tp_flags = state.get("tp_rerank_applied", []) if state else []
    if len(tp_flags) < total:
        tp_flags = list(tp_flags) + [True] * (total - len(tp_flags))
    eligible = [
        index
        for index in range(min(total, len(global_top50), len(metric_pools), len(final_predictions)))
        if tp_flags[index]
        and global_top50[index] is not None
        and metric_pools[index] is not None
    ]
    if not eligible:
        return {
            "queries": int(total),
            "tp_rerank_queries": 0,
            "tp_rerank_fraction": 0.0 if total else None,
            "global_top50_pool_retention": None,
            "global_top50_final_retention": None,
            "tp_truth50_pool_recall": None,
        }

    pool_retention = []
    final_retention = []
    for index in eligible:
        protected = global_top50[index]
        pool = metric_pools[index]
        final = final_predictions[index]
        protected = set(np.asarray(protected, dtype=np.int64).reshape(-1).tolist())
        pool = set(np.asarray(pool, dtype=np.int64).reshape(-1).tolist())
        final = set(np.asarray(final, dtype=np.int64).reshape(-1).tolist())
        denominator = max(len(protected), 1)
        pool_retention.append(len(protected & pool) / denominator)
        final_retention.append(len(protected & final) / denominator)

    truth_pool_recall = None
    truth_queries = 0
    if tp_truth_ids is not None:
        values = []
        truth_total = min(total, len(tp_truth_ids))
        for index in eligible:
            if index >= truth_total:
                continue
            pool = metric_pools[index]
            truth = tp_truth_ids[index]
            truth = np.asarray(truth, dtype=np.int64).reshape(-1)
            truth = truth[truth >= 0][:50]
            if not len(truth):
                continue
            truth_queries += 1
            pool = set(np.asarray(pool, dtype=np.int64).reshape(-1).tolist())
            values.append(len(pool & set(truth.tolist())) / len(truth))
        truth_pool_recall = float(np.mean(values)) if values else None

    return {
        "queries": int(total),
        "tp_rerank_queries": int(len(eligible)),
        "tp_rerank_fraction": float(len(eligible) / max(total, 1)),
        "tp_truth_queries": int(truth_queries),
        "global_top50_pool_retention": float(np.mean(pool_retention)),
        "global_top50_final_retention": float(np.mean(final_retention)),
        "tp_truth50_pool_recall": truth_pool_recall,
    }


def _build_metric_backend(args):
    metric = getattr(args, "rerank_metric", "embedding")
    if metric == "embedding":
        return None
    return TrajectoryMetricBackend(
        metric,
        point_distance_file=args.tp_point_distance_file,
        edge_file=getattr(args, "edge_file", None),
        node_count=getattr(args, "node_count", 0),
        max_trajectory_length=getattr(args, "max_trajectory_length", 0),
        neterp_anchor_node=getattr(args, "neterp_anchor_node", None),
        num_threads=args.tp_num_threads,
    )


def run_scale(
    args,
    gallery,
    queries,
    gallery_size,
    tp_distance_table=None,
    metric_backend=None,
):
    from hash_global import encode_binary_codes, encode_soft_codes

    rerank_metric = getattr(args, "rerank_metric", "embedding")
    scale_dir = osp.join(args.output_dir, f"gallery_{gallery_size}")
    os.makedirs(scale_dir, exist_ok=True)
    positive_mask = (
        gallery.get("is_initial")
        if args.positive_match_mode == "initial-only"
        else None
    )
    if args.positive_match_mode == "initial-only" and positive_mask is None:
        raise ValueError("initial-only requires an is_initial array in the Gallery.")
    selected_rows = select_gallery_rows(
        gallery["source_ids"],
        queries["source_ids"],
        gallery_size,
        seed=args.selection_seed,
        mode=args.gallery_selection,
        positives_per_query=args.positives_per_query,
        positive_mask=positive_mask,
    )
    selected_source_ids = gallery["source_ids"][selected_rows]
    selected_positive_mask = (
        np.ones(len(selected_rows), dtype=bool)
        if positive_mask is None
        else positive_mask[selected_rows]
    )
    composition = gallery_composition(
        selected_source_ids,
        queries["source_ids"],
        positives_per_query=args.positives_per_query,
        positive_mask=selected_positive_mask,
    )
    index, build_report = _build_index(args, gallery, selected_rows, scale_dir)
    query_embeddings = queries["embeddings"]
    query_scale_embeddings = queries.get("scale_embeddings")
    query_scale_weights = queries.get("scale_weights")
    query_trajectories = queries["trajectories"]
    query_source_ids = queries["source_ids"]
    query_labels = _payload_labels(queries)
    query_metadata = {
        key: np.asarray(value)
        for key, value in queries.get("metadata", {}).items()
    }
    if args.exclude_self is None:
        args.exclude_self = _same_file(gallery["path"], queries["path"])
    index.exclude_self = bool(args.exclude_self)
    if args.max_queries > 0:
        query_embeddings = query_embeddings[: args.max_queries]
        if query_scale_embeddings is not None:
            query_scale_embeddings = query_scale_embeddings[: args.max_queries]
        if query_scale_weights is not None:
            query_scale_weights = query_scale_weights[: args.max_queries]
        query_trajectories = query_trajectories[: args.max_queries]
        query_source_ids = query_source_ids[: args.max_queries]
        if query_labels is not None:
            query_labels = query_labels[: args.max_queries]
        query_metadata = {
            key: value[: args.max_queries] for key, value in query_metadata.items()
        }
    code_start = perf_counter()
    query_codes = encode_binary_codes(
        index.hash_index.head,
        query_embeddings,
        batch_size=args.encode_batch_size,
        device=args.query_code_device,
    )
    code_ms = (perf_counter() - code_start) * 1000.0
    query_metric_codes = None
    query_metric_hash_codes = None
    if index.metric_hash_index is not None:
        metric_code_start = perf_counter()
        metric_head = index.metric_hash_index.head
        query_metric_hash_codes = encode_binary_codes(
            metric_head,
            query_embeddings,
            batch_size=args.encode_batch_size,
            device=args.query_code_device,
        )
        if args.metric_prefilter_hash_score == "soft":
            query_metric_codes = encode_soft_codes(
                metric_head,
                query_embeddings,
                batch_size=args.encode_batch_size,
                device=args.query_code_device,
                temperature=args.metric_prefilter_soft_temperature,
            )
        else:
            query_metric_codes = query_metric_hash_codes
        code_ms += (perf_counter() - metric_code_start) * 1000.0
    code_ms_per_query = code_ms / max(len(query_embeddings), 1)
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    results = []
    predictions_by_method = {}
    diagnostic_states = {}
    for method in methods:
        diagnostic_state = (
            {}
            if method in {
                "adaptive-topo-hash",
                "adaptive-dual-metric",
                "adaptive-dual-metric-no-rerank",
            }
            and bool(getattr(args, "adaptive_preserve_global_top50", False))
            else None
        )
        method_result, predictions = _evaluate_method(
            method,
            index,
            query_embeddings,
            query_trajectories,
            query_source_ids,
            gallery["embeddings"][selected_rows] if method == "exact" else None,
            selected_source_ids,
            selected_positive_mask,
            query_labels,
            query_codes,
            args.top_k,
            args.exact_max_queries,
            query_scale_embeddings=query_scale_embeddings,
            query_scale_weights=query_scale_weights,
            code_encoding_ms_per_query=code_ms_per_query,
            query_metadata=query_metadata,
            metric_backend=metric_backend,
            diagnostic_state=diagnostic_state,
            query_metric_codes=query_metric_codes,
            query_metric_hash_codes=query_metric_hash_codes,
            global_hash_rerank_mode=getattr(args, "global_hash_rerank_mode", "full"),
        )
        method_result["rerank_metric"] = (
            rerank_metric
            if method in {
                "adaptive-topo-hash",
                "adaptive-dual-metric",
                "adaptive-dual-metric-no-rerank",
            }
            else None
        )
        if method == "global-hash":
            method_result["hash_rerank_mode"] = getattr(
                args, "global_hash_rerank_mode", "full"
            )
            if method_result["hash_rerank_mode"] == "tp":
                method_result["rerank_metric"] = "TP"
        results.append(method_result)
        predictions_by_method[method] = predictions
        if diagnostic_state is not None:
            diagnostic_states[method] = diagnostic_state
    comparison = _method_comparison(
        predictions_by_method,
        query_source_ids,
        selected_source_ids,
        selected_positive_mask,
        args.top_k,
    )
    tp_report = None
    tp_ids = None
    if args.tp_eval:
        if tp_distance_table is not None:
            start = perf_counter()
            selected_columns = np.asarray(
                [args.tp_max_row_position.get(int(row), -1) for row in selected_rows],
                dtype=np.int64,
            )
            if np.any(selected_columns < 0):
                raise ValueError(
                    "Cannot reuse the max TP distance table because this scale "
                    "selected a Gallery row outside the max-scale selection."
                )
            tp_ids, _, table_report = tp_ground_truth_from_distance_table(
                tp_distance_table, selected_columns, top_k=args.tp_top_k
            )
            tp_report = {
                "enabled": True,
                "queries": int(len(tp_ids)),
                "gallery_rows": int(len(selected_rows)),
                "top_k": int(args.tp_top_k),
                "compute_seconds": float(perf_counter() - start),
                "cache_reused": True,
                "cache_file": getattr(args, "tp_distance_table_file", None),
                "derived_from_distance_table": True,
                "distance_table_gallery_rows": int(tp_distance_table.shape[1]),
                **table_report,
            }
        else:
            tp_cache_file = resolve_tp_cache_file(
                args.tp_ground_truth_file,
                scale_dir,
                gallery_size,
                multi_scale=getattr(args, "tp_multi_scale", False),
            )
            tp_ids, _, tp_report = compute_tp_ground_truth(
                queries["trajectories"],
                gallery["trajectories"][selected_rows],
                args.tp_point_distance_file,
                top_k=args.tp_top_k,
                max_queries=args.tp_max_queries,
                num_threads=args.tp_num_threads,
                device=args.tp_device,
                gpu_batch_size=args.tp_gpu_batch_size,
                cache_file=tp_cache_file,
                force_recompute=args.tp_force_recompute,
            )
        for method_result in results:
            method_result["tp_metrics"] = tp_retrieval_metrics(
                predictions_by_method[method_result["method"]], tp_ids
            )
    for method_result in results:
        method = method_result["method"]
        if method in diagnostic_states:
            method_result["global_top50_diagnostics"] = _global_top50_diagnostics(
                diagnostic_states[method], tp_ids
            )
    payload = {
        "protocol": "topology_aware_multi_resolution_path_hash",
        "gallery_size": int(gallery_size),
        "queries": int(len(query_embeddings)),
        "gallery_selection": args.gallery_selection,
        "positive_match_mode": args.positive_match_mode,
        "exclude_self": bool(args.exclude_self),
        "rerank_metric": rerank_metric,
        "gallery_composition": composition,
        "topo_config": asdict(index.config),
        "rerank_weights": {
            "embedding": float(index.config.embedding_weight),
            "hierarchy_embedding": float(index.config.hierarchy_embedding_weight),
            "topology": float(
                1.0 - index.config.embedding_weight
                if index.config.hierarchy_embedding_weight <= 0.0
                else (
                    1.0 - index.config.embedding_weight
                    - index.config.hierarchy_embedding_weight
                    if index.config.topology_weight < 0.0
                    else index.config.topology_weight
                )
            ),
        },
        "build": build_report,
        "query_code_encoding_ms": float(code_ms),
        "methods": results,
        "comparison": comparison,
        "tp_evaluation": tp_report,
    }
    result_file = osp.join(scale_dir, "result.json")
    _json_dump(result_file, payload)
    print(
        f"gallery={gallery_size}, queries={len(query_embeddings)}, "
        f"build_s={build_report['total_seconds']:.3f}, result={result_file}"
    )
    for method in results:
        metric_keys = ("HR@10", "HR@50", "R10@50", "R1@1", "R1@10", "R1@50")
        recall = ", ".join(
            f"{key}={method['metrics'][key]:.4f}"
            for key in metric_keys
            if key in method["metrics"]
        )
        if not recall:
            recall = ", ".join(
                f"{key}={value:.4f}"
                for key, value in method["metrics"].items()
                if key.startswith("id_recall_at_")
            )
        print(
            f"  method={method['method']}, {recall}, "
            f"mean_ms={method['latency_ms']['mean']:.3f}, qps={method['qps']:.2f}"
        )
        if method.get("tp_metrics"):
            tp_metrics = ", ".join(
                f"{key}={value:.4f}"
                for key, value in method["tp_metrics"].items()
                if key != "queries"
            )
            print(f"    TP ({method['tp_metrics']['queries']} queries): {tp_metrics}")
        budget = method.get("adaptive_tp_budget")
        if budget:
            print(
                "    Adaptive TP budget "
                f"(max_fraction={budget['max_fraction']:.4f}, "
                f"max_queries={budget['max_queries']}, "
                f"low_confidence={budget['low_confidence_queries']}, "
                f"selected={budget['selected_queries']}, "
                f"forced_fast_path={budget['forced_fast_path_queries']}):"
            )
        diagnostics = method.get("global_top50_diagnostics")
        if diagnostics:
            values = []
            for key in (
                "global_top50_pool_retention",
                "global_top50_final_retention",
                "tp_truth50_pool_recall",
            ):
                value = diagnostics.get(key)
                values.append(
                    f"{key}={value:.4f}" if value is not None else f"{key}=n/a"
                )
            print(
                f"    Global Top-50 diagnostics ({diagnostics['queries']} queries, "
                f"TP={diagnostics.get('tp_rerank_queries', 0)}, "
                f"fraction={diagnostics.get('tp_rerank_fraction', 0.0):.4f}): "
                + ", ".join(values)
            )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TopoRouteHash: L3/L2 postings, L1 path MinHash, Global Hash "
            "fallback, and structure-aware rank fusion. Optional continuous "
            "multi-scale reranking uses precomputed scale_embeddings."
        )
    )
    parser.add_argument("--gallery-file", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--gallery-source-ids", default=None)
    parser.add_argument("--query-source-ids", default=None)
    parser.add_argument(
        "--gallery-source-id-mode", choices=["stored", "query-row"], default="stored"
    )
    exclusion = parser.add_mutually_exclusive_group()
    exclusion.add_argument("--exclude-self", dest="exclude_self", action="store_true")
    exclusion.add_argument("--no-exclude-self", dest="exclude_self", action="store_false")
    parser.set_defaults(exclude_self=None)
    parser.add_argument("--hash-model-file", required=True)
    parser.add_argument(
        "--metric-hash-model-file",
        default=None,
        help="Independent metric-head checkpoint used by adaptive-dual-metric.",
    )
    parser.add_argument(
        "--dual-metric-weight-file",
        default=None,
        help="Optional learned dual-metric fusion-weight JSON.",
    )
    parser.add_argument("--hierarchy-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--gallery-size", type=int, default=0)
    sizes.add_argument("--gallery-sizes", default=None)
    parser.add_argument(
        "--gallery-selection",
        choices=["source-balanced", "fixed-positive", "prefix"],
        default="fixed-positive",
    )
    parser.add_argument(
        "--positive-match-mode",
        choices=["same-source", "initial-only"],
        default="initial-only",
    )
    parser.add_argument("--positives-per-query", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--methods", default="topo-route-hash,global-hash,exact"
    )
    parser.add_argument(
        "--candidate-source",
        choices=["hybrid", "global-only"],
        default="hybrid",
        help=(
            "Candidate generation for topo-route-hash. 'global-only' skips "
            "L3/L2 posting retrieval and only reranks Global Hash candidates."
        ),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--exact-max-queries", type=int, default=200)

    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument(
        "--rerank-metric",
        choices=["embedding", "TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
        default="embedding",
        help=(
            "Final metric for adaptive reranking, or TP for Global Hash tp mode. "
            "The selected metric is computed online and included in query latency."
        ),
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--probe-radius", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--hamming-candidates", type=int, default=1024)
    parser.add_argument(
        "--hash-backend", choices=["mih", "hnsw", "flat"], default="mih",
        help="Hash retrieval: MIH probing, Hamming HNSW, or exact packed XOR/popcount scan.",
    )
    parser.add_argument(
        "--global-hash-rerank-mode",
        choices=["full", "hash-only", "tp"],
        default="full",
        help=(
            "Rank Global Hash results by continuous embedding distance ('full') "
            "or return Hamming results directly ('hash-only'), or rerank every "
            "query's Hamming pool by online TP ('tp', requires --rerank-metric TP)."
        ),
    )
    parser.add_argument("--hash-flat-block-size", type=int, default=65536)
    parser.add_argument("--hash-hnsw-m", type=int, default=16)
    parser.add_argument("--hash-hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hash-hnsw-ef-search", type=int, default=256)
    parser.add_argument(
        "--embedding-storage", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--hash-device", default="auto")
    parser.add_argument("--query-code-device", default="cpu")
    parser.add_argument("--edge-file", default=None)
    parser.add_argument("--node-count", type=int, default=0)
    parser.add_argument("--max-trajectory-length", type=int, default=0)
    parser.add_argument("--neterp-anchor-node", type=int, default=None)

    parser.add_argument("--qgram", type=int, default=2)
    parser.add_argument("--minhash-size", type=int, default=8)
    parser.add_argument("--postings-per-level", type=int, default=4)
    parser.add_argument("--l3-expand-hops", type=int, default=1)
    parser.add_argument("--min-route-candidates", type=int, default=1024)
    parser.add_argument("--max-route-candidates", type=int, default=4096)
    parser.add_argument("--structural-candidates", type=int, default=512)
    parser.add_argument("--global-candidates", type=int, default=1024)
    parser.add_argument("--max-rerank-candidates", type=int, default=1536)
    parser.add_argument("--adaptive-initial-global", type=int, default=512)
    parser.add_argument("--adaptive-initial-structural", type=int, default=256)
    parser.add_argument("--adaptive-max-global", type=int, default=2048)
    parser.add_argument("--adaptive-max-structural", type=int, default=1024)
    parser.add_argument("--adaptive-max-rerank", type=int, default=2048)
    parser.add_argument("--adaptive-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--adaptive-min-route-recall", type=int, default=128)
    parser.add_argument(
        "--adaptive-gate", choices=["overlap", "global", "hybrid"],
        default="overlap",
    )
    parser.add_argument("--adaptive-global-margin", type=float, default=0.05)
    parser.add_argument(
        "--adaptive-metric-fast-path",
        choices=["skip", "metric"],
        default="skip",
        help=(
            "For adaptive-topo-hash, skip metric reranking on the confident "
            "Global path or run the configured metric there as well."
        ),
    )
    parser.add_argument(
        "--adaptive-max-tp-fraction",
        type=float,
        default=0.15,
        help=(
            "Maximum fraction of queries allowed to enter the expensive trajectory "
            "metric branch when adaptive-metric-fast-path=skip. Select the lowest "
            "Global-margin queries first; 1.0 disables the cap."
        ),
    )
    parser.add_argument(
        "--adaptive-preserve-global-top50",
        action="store_true",
        help=(
            "Protect the Global Hash embedding Top-50 inside Adaptive metric rerank "
            "pools. With adaptive-metric-fast-path=skip, only difficult "
            "queries use the configured metric pool. Requires a non-embedding "
            "--rerank-metric."
        ),
    )
    parser.add_argument("--adaptive-rescue-quality-overlap", type=float, default=0.75)
    parser.add_argument("--adaptive-rescue-near-margin-fraction", type=float, default=0.80)
    parser.add_argument("--metric-prefilter-candidates", type=int, default=64)
    parser.add_argument("--metric-prefilter-high-quality-candidates", type=int, default=50)
    parser.add_argument("--metric-prefilter-global-protect", type=int, default=32)
    parser.add_argument("--metric-prefilter-structural-protect", type=int, default=16)
    parser.add_argument("--metric-prefilter-embedding-weight", type=float, default=1.0)
    parser.add_argument("--metric-prefilter-hash-weight", type=float, default=0.0)
    parser.add_argument(
        "--metric-prefilter-hash-score", choices=["hamming", "soft"], default="hamming"
    )
    parser.add_argument("--metric-prefilter-soft-temperature", type=float, default=1.0)
    parser.add_argument("--dual-metric-search-candidates", type=int, default=256)
    parser.add_argument("--dual-metric-candidates", type=int, default=96)
    parser.add_argument("--dual-metric-high-quality-tp-candidates", type=int, default=64)
    parser.add_argument("--dual-metric-tp-candidates", type=int, default=128)
    parser.add_argument("--dual-metric-hard-tp-candidates", type=int, default=160)
    parser.add_argument("--dual-metric-hard-margin-fraction", type=float, default=0.50)
    parser.add_argument("--dual-metric-head-protect", type=int, default=48)
    parser.add_argument("--dual-metric-structural-protect", type=int, default=16)
    parser.add_argument("--dual-metric-embedding-weight", type=float, default=0.40)
    parser.add_argument("--dual-metric-hash-weight", type=float, default=0.40)
    parser.add_argument("--dual-metric-topology-weight", type=float, default=0.20)
    parser.add_argument("--l3-weight", type=float, default=0.45)
    parser.add_argument("--l2-weight", type=float, default=0.35)
    parser.add_argument("--l1-weight", type=float, default=0.20)
    parser.add_argument("--embedding-weight", type=float, default=0.80)
    parser.add_argument(
        "--hierarchy-embedding-weight", "--hierarchy-rerank-weight",
        dest="hierarchy_embedding_weight", type=float, default=0.0,
        help=(
            "Weight of continuous multi-scale hierarchy distances in the final "
            "rank fusion. Requires scale_embeddings in both payloads."
        ),
    )
    parser.add_argument(
        "--topology-weight", type=float, default=-1.0,
        help=(
            "Topology signature rank weight when hierarchy reranking is enabled. "
            "Default -1 derives the remaining weight."
        ),
    )
    add_tp_arguments(parser)
    parser.add_argument(
        "--tp-reuse-max-distance-table",
        action="store_true",
        help=(
            "Compute/load Query x max-Gallery TP distances once and derive each "
            "scale's exact TP Top-K by table lookup."
        ),
    )
    parser.add_argument(
        "--tp-distance-table-file",
        default=None,
        help="Optional .npy cache for --tp-reuse-max-distance-table.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods).difference(METHODS))
    if not methods or unknown:
        raise ValueError(f"Unknown methods: {unknown}; expected a subset of {METHODS}.")
    adaptive_methods = {
        "adaptive-topo-hash",
        "adaptive-dual-metric",
        "adaptive-dual-metric-no-rerank",
    }
    dual_metric_methods = {
        "adaptive-dual-metric",
        "adaptive-dual-metric-no-rerank",
    }
    global_tp = args.global_hash_rerank_mode == "tp"
    if global_tp and ("global-hash" not in methods or args.rerank_metric != "TP"):
        raise ValueError(
            "--global-hash-rerank-mode tp requires --methods global-hash "
            "(or a method list containing it) and --rerank-metric TP."
        )
    if (args.rerank_metric != "embedding"
            and not adaptive_methods.intersection(methods) and not global_tp):
        raise ValueError(
            "Online metric reranking requires an adaptive method or "
            "--global-hash-rerank-mode tp with --rerank-metric TP."
        )
    if dual_metric_methods.intersection(methods) and not args.metric_hash_model_file:
        raise ValueError(
            "adaptive-dual-metric requires --metric-hash-model-file."
        )
    if args.dual_metric_weight_file and not dual_metric_methods.intersection(methods):
        raise ValueError(
            "--dual-metric-weight-file requires an adaptive-dual-metric method."
        )
    if args.adaptive_preserve_global_top50:
        if not adaptive_methods.intersection(methods):
            raise ValueError(
                "--adaptive-preserve-global-top50 requires an adaptive method."
            )
        if args.rerank_metric == "embedding":
            raise ValueError(
                "--adaptive-preserve-global-top50 requires a non-embedding "
                "--rerank-metric (for example TP or DITA)."
            )
        if min(
            args.hamming_candidates,
            args.metric_prefilter_candidates,
            args.metric_prefilter_high_quality_candidates,
            args.adaptive_max_rerank,
        ) < 50:
            raise ValueError(
                "Global Top-50 protection requires both metric prefilter budgets "
                "and adaptive_max_rerank to be at least 50."
            )
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if "topo-route-hash" in methods and args.top_k > args.global_candidates:
        raise ValueError("top_k cannot exceed global_candidates for topo-route-hash.")
    if adaptive_methods.intersection(methods) and args.top_k > args.adaptive_max_global:
        raise ValueError(
            "top_k cannot exceed adaptive_max_global for adaptive methods."
        )
    if "global-hash" in methods and args.top_k > args.hamming_candidates:
        raise ValueError("top_k cannot exceed hamming_candidates for global-hash.")
    gallery = load_embedding_payload(
        args.gallery_file, args.gallery_source_ids, role="gallery"
    )
    queries = load_embedding_payload(
        args.query_file, args.query_source_ids, role="query"
    )
    gallery = remap_gallery_source_ids(
        gallery, queries, mode=args.gallery_source_id_mode
    )
    if gallery["embeddings"].shape[1] != queries["embeddings"].shape[1]:
        raise ValueError("Gallery and Query embedding dimensions must match.")
    gallery_scales = gallery.get("scale_embeddings")
    query_scales = queries.get("scale_embeddings")
    if (gallery_scales is None) != (query_scales is None):
        raise ValueError(
            "Gallery and Query must either both contain scale_embeddings or both omit them."
        )
    if gallery_scales is not None:
        if gallery_scales.shape[1:] != query_scales.shape[1:]:
            raise ValueError(
                "Gallery and Query scale_embeddings must have matching [scales, dim] shapes; "
                f"got {gallery_scales.shape[1:]} and {query_scales.shape[1:]}."
            )
    sizes = parse_gallery_sizes(
        args.gallery_size, args.gallery_sizes, len(gallery["embeddings"])
    )
    args.tp_multi_scale = len(sizes) > 1
    os.makedirs(args.output_dir, exist_ok=True)
    tp_distance_table = None
    metric_backend = None
    results = []
    try:
        if args.tp_eval and args.tp_reuse_max_distance_table:
            max_size = int(sizes[-1])
            max_rows = select_gallery_rows(
                gallery["source_ids"],
                queries["source_ids"],
                max_size,
                seed=args.selection_seed,
                mode=args.gallery_selection,
                positives_per_query=args.positives_per_query,
                positive_mask=(
                    gallery.get("is_initial")
                    if args.positive_match_mode == "initial-only"
                    else None
                ),
            )
            table_file = args.tp_distance_table_file
            if not table_file:
                table_file = osp.join(
                    args.output_dir, f"tp_distance_table_gallery_{max_size}.npy"
                )
            args.tp_distance_table_file = table_file
            tp_distance_table, _ = compute_tp_distance_table(
                queries["trajectories"],
                gallery["trajectories"][max_rows],
                args.tp_point_distance_file,
                max_queries=args.tp_max_queries,
                num_threads=args.tp_num_threads,
                device=args.tp_device,
                gpu_batch_size=args.tp_gpu_batch_size,
                cache_file=table_file,
                force_recompute=args.tp_force_recompute,
            )
            args.tp_max_row_position = {
                int(row): position for position, row in enumerate(max_rows)
            }
        metric_backend = _build_metric_backend(args)
        if global_tp:
            print(
                "[global-hash] online TP rerank on every query, candidates=%d, "
                "top_k=%d, gated=false, threads=%d"
                % (args.hamming_candidates, args.top_k, args.tp_num_threads)
            )
        if metric_backend is not None and adaptive_methods.intersection(methods):
            print(
                "[adaptive] online rerank metric=%s, fast_path=%s, threads=%d"
                % (
                    args.rerank_metric,
                    args.adaptive_metric_fast_path,
                    args.tp_num_threads,
                )
            )
        for gallery_size in sizes:
            results.append(
                run_scale(
                    args,
                    gallery,
                    queries,
                    gallery_size,
                    tp_distance_table,
                    metric_backend,
                )
            )
            gc.collect()
    finally:
        if tp_distance_table is not None:
            close_mmap_array(tp_distance_table)
        if metric_backend is not None:
            metric_backend.close()
    summary = {
        "protocol": "topology_aware_multi_resolution_path_hash_sweep",
        "gallery_sizes": sizes,
        "rerank_metric": args.rerank_metric,
        "results": results,
    }
    summary_file = osp.join(args.output_dir, "scale_summary.json")
    _json_dump(summary_file, summary)
    print(f"Scale summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
