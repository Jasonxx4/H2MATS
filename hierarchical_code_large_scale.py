import argparse
import gc
import json
import os
import os.path as osp
from pathlib import Path
from time import perf_counter

import numpy as np


METHODS = ("hier-code", "code-scan", "exact")


def _topk_indices(distances, k):
    k = min(int(k), len(distances))
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k == len(distances):
        return np.argsort(distances, kind="stable")
    order = np.argpartition(distances, k - 1)[:k]
    return order[np.argsort(distances[order], kind="stable")]


def _directory_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def _json_dump(path, payload):
    out_dir = osp.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _load_source_ids(path, expected_size):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = ("source_ids", "query_source_ids", "gallery_source_ids")
        key = next((name for name in keys if name in data.files), None)
        if key is None:
            raise ValueError(f"Source-ID file {path} does not contain one of: {keys}.")
        source_ids = np.asarray(data[key], dtype=np.int64)
        data.close()
    else:
        source_ids = np.asarray(data, dtype=np.int64)
    source_ids = source_ids.reshape(-1)
    if len(source_ids) != expected_size:
        raise ValueError(
            f"Source-ID file has {len(source_ids)} rows; expected {expected_size}."
        )
    return source_ids


def load_embedding_payload(path, source_ids_file=None, role="gallery"):
    with np.load(path, allow_pickle=True) as data:
        required = {"embeddings", "trajectories"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Embedding file {path} is missing arrays: {missing}.")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        trajectories = np.asarray(data["trajectories"], dtype=object)
        scale_embeddings = (
            np.asarray(data["scale_embeddings"], dtype=np.float32)
            if "scale_embeddings" in data.files else None
        )
        scale_weights = (
            np.asarray(data["scale_weights"], dtype=np.float32)
            if "scale_weights" in data.files else None
        )
        coordinates = (
            np.asarray(data["coordinates"], dtype=object)
            if "coordinates" in data.files
            else None
        )
        if len(embeddings) != len(trajectories):
            raise ValueError(
                f"Embedding file {path} has {len(embeddings)} embeddings but "
                f"{len(trajectories)} trajectories."
            )
        if scale_embeddings is not None:
            if scale_embeddings.ndim != 3 or len(scale_embeddings) != len(embeddings):
                raise ValueError(
                    f"Embedding file {path} has invalid scale_embeddings shape "
                    f"{scale_embeddings.shape}; expected [N, S, D]."
                )
        if scale_weights is not None:
            if scale_weights.ndim != 2 or len(scale_weights) != len(embeddings):
                raise ValueError(
                    f"Embedding file {path} has invalid scale_weights shape "
                    f"{scale_weights.shape}; expected [N, S]."
                )
            if scale_embeddings is not None and scale_weights.shape[1] != scale_embeddings.shape[1]:
                raise ValueError(
                    f"Embedding file {path} has {scale_weights.shape[1]} scale weights "
                    f"but {scale_embeddings.shape[1]} scale embeddings."
                )

        source_ids = None
        for key in (f"{role}_source_ids", "source_ids"):
            if key in data.files:
                source_ids = np.asarray(data[key], dtype=np.int64).reshape(-1)
                break
        metadata = {}
        for key in (
            "perturbation_types",
            "severities",
            "variant_ids",
            "requested_replacement_counts",
            "actual_replacement_counts",
            "replaced_positions",
            "replacement_node_ids",
            "path_valid_ratios",
            "effective_severities",
        ):
            if key in data.files and len(data[key]) == len(embeddings):
                metadata[key] = np.asarray(data[key])
        is_initial = None
        for key in ("is_initial", "initial_mask", "is_clean"):
            if key in data.files:
                is_initial = np.asarray(data[key], dtype=bool).reshape(-1)
                break
        labels = None
        for key in ("labels", "vali_y", "train_y"):
            if key in data.files:
                labels = np.asarray(data[key], dtype=np.int64)
                break
        anchor_count = None
        if "anchor_count" in data.files:
            anchor_count = int(np.asarray(data["anchor_count"]).item())

    if source_ids_file:
        source_ids = _load_source_ids(source_ids_file, len(embeddings))
    elif source_ids is None:
        source_ids = np.arange(len(embeddings), dtype=np.int64)
    if len(source_ids) != len(embeddings):
        raise ValueError(
            f"Embedding file {path} has {len(embeddings)} rows but "
            f"{len(source_ids)} source IDs."
        )
    if is_initial is not None and len(is_initial) != len(embeddings):
        raise ValueError(
            f"Embedding file {path} has {len(embeddings)} rows but "
            f"{len(is_initial)} initial-trajectory flags."
        )
    if coordinates is not None and len(coordinates) != len(embeddings):
        raise ValueError(
            f"Embedding file {path} has {len(embeddings)} rows but "
            f"{len(coordinates)} coordinate rows."
        )
    return {
        "embeddings": embeddings,
        "scale_embeddings": scale_embeddings,
        "scale_weights": scale_weights,
        "trajectories": trajectories,
        "coordinates": coordinates,
        "source_ids": source_ids,
        "is_initial": is_initial,
        "metadata": metadata,
        "labels": labels,
        "anchor_count": anchor_count,
        "path": str(Path(path).resolve()),
    }


def remap_gallery_source_ids(gallery, queries, mode="stored"):
    """Support old perturbation files whose source IDs are Query row indices."""
    if mode == "stored":
        return gallery
    if mode != "query-row":
        raise ValueError(f"Unknown Gallery source-ID mode: {mode}")
    local_source_ids = np.asarray(gallery["source_ids"], dtype=np.int64).reshape(-1)
    query_source_ids = np.asarray(queries["source_ids"], dtype=np.int64).reshape(-1)
    if np.any((local_source_ids < 0) | (local_source_ids >= len(query_source_ids))):
        raise ValueError(
            "Gallery source IDs cannot be remapped as Query row indices: an ID is "
            "outside the Query range."
        )
    remapped = dict(gallery)
    remapped["source_ids"] = query_source_ids[local_source_ids]
    remapped["source_id_mode"] = "query-row"
    return remapped


def parse_gallery_sizes(gallery_size, gallery_sizes, available):
    if gallery_sizes:
        sizes = [int(item.strip()) for item in gallery_sizes.split(",") if item.strip()]
    elif gallery_size and gallery_size > 0:
        sizes = [int(gallery_size)]
    else:
        sizes = [int(available)]
    sizes = sorted(set(sizes))
    if not sizes or sizes[0] <= 0:
        raise ValueError("Gallery sizes must be positive integers.")
    if sizes[-1] > available:
        raise ValueError(
            f"Requested gallery_size={sizes[-1]} exceeds the master Gallery size "
            f"({available})."
        )
    return sizes


def select_gallery_rows(
    gallery_source_ids,
    query_source_ids,
    gallery_size,
    seed=2026,
    mode="source-balanced",
    positives_per_query=1,
    positive_mask=None,
):
    gallery_source_ids = np.asarray(gallery_source_ids, dtype=np.int64).reshape(-1)
    query_sources = np.unique(np.asarray(query_source_ids, dtype=np.int64).reshape(-1))
    gallery_size = int(gallery_size)
    if gallery_size <= 0 or gallery_size > len(gallery_source_ids):
        raise ValueError(
            f"gallery_size must be in [1, {len(gallery_source_ids)}], got {gallery_size}."
        )
    if mode == "prefix":
        return np.arange(gallery_size, dtype=np.int64)
    if mode not in ("source-balanced", "fixed-positive"):
        raise ValueError(f"Unknown gallery selection mode: {mode}")

    positives_per_query = int(positives_per_query)
    if positives_per_query <= 0:
        raise ValueError("positives_per_query must be a positive integer.")

    query_source_set = set(int(item) for item in query_sources)
    if positive_mask is None:
        positive_mask = np.ones(len(gallery_source_ids), dtype=bool)
        source_disjoint_distractors = True
    else:
        positive_mask = np.asarray(positive_mask, dtype=bool).reshape(-1)
        if len(positive_mask) != len(gallery_source_ids):
            raise ValueError(
                "positive_mask must have one value for every Gallery row."
            )
        source_disjoint_distractors = False
    rows_by_source = {}
    for row_id, source_id in enumerate(gallery_source_ids):
        source_id = int(source_id)
        if source_id in query_source_set and positive_mask[row_id]:
            rows_by_source.setdefault(source_id, []).append(row_id)

    represented_sources = sorted(rows_by_source)
    rng = np.random.default_rng(seed)
    if mode == "fixed-positive":
        missing_sources = [
            int(source_id)
            for source_id in query_sources
            if len(rows_by_source.get(int(source_id), ())) < positives_per_query
        ]
        if missing_sources:
            preview = ", ".join(str(item) for item in missing_sources[:10])
            raise ValueError(
                "fixed-positive selection requires at least "
                f"{positives_per_query} Gallery row(s) for every Query source; "
                f"{len(missing_sources)} source(s) are missing or insufficient "
                f"(first: {preview})."
            )

        positive_rows = []
        for source_id in query_sources:
            rows = np.asarray(rows_by_source[int(source_id)], dtype=np.int64)
            positive_rows.extend(rng.permutation(rows)[:positives_per_query])
        positive_rows = np.asarray(positive_rows, dtype=np.int64)
        if gallery_size < len(positive_rows):
            raise ValueError(
                f"gallery_size={gallery_size} is smaller than the fixed positive set "
                f"({len(positive_rows)} rows = {len(query_sources)} Query sources x "
                f"{positives_per_query})."
            )

        distractor_rows = np.asarray(
            [
                row_id
                for row_id, source_id in enumerate(gallery_source_ids)
                if (
                    int(source_id) not in query_source_set
                    if source_disjoint_distractors
                    else not (
                        int(source_id) in query_source_set
                        and positive_mask[row_id]
                    )
                )
            ],
            dtype=np.int64,
        )
        rng.shuffle(distractor_rows)
        required_distractors = gallery_size - len(positive_rows)
        if required_distractors > len(distractor_rows):
            raise ValueError(
                f"fixed-positive gallery_size={gallery_size} requires "
                f"{required_distractors} independent distractor rows, but only "
                f"{len(distractor_rows)} are available. Extra variants of Query "
                "sources are intentionally excluded. Add trajectories whose source_ids "
                "are disjoint from the Query source IDs, reduce Gallery size, or use "
                "source-balanced for the multi-positive protocol."
            )
        return np.concatenate(
            [positive_rows, distractor_rows[:required_distractors]]
        )

    if gallery_size < len(represented_sources):
        raise ValueError(
            f"gallery_size={gallery_size} cannot preserve one positive for each of the "
            f"{len(represented_sources)} represented Query sources."
        )

    anchors = np.asarray(
        [rows[int(rng.integers(0, len(rows)))] for rows in rows_by_source.values()],
        dtype=np.int64,
    )
    remaining_mask = np.ones(len(gallery_source_ids), dtype=bool)
    remaining_mask[anchors] = False
    remaining = np.flatnonzero(remaining_mask)
    rng.shuffle(remaining)
    return np.concatenate([anchors, remaining[: gallery_size - len(anchors)]])


def gallery_composition(
    selected_source_ids,
    query_source_ids,
    positives_per_query=1,
    positive_mask=None,
):
    selected_source_ids = np.asarray(selected_source_ids, dtype=np.int64).reshape(-1)
    query_sources = np.unique(np.asarray(query_source_ids, dtype=np.int64).reshape(-1))
    query_source_set = set(int(item) for item in query_sources)
    query_source_mask = np.asarray(
        [int(source_id) in query_source_set for source_id in selected_source_ids],
        dtype=bool,
    )
    if positive_mask is None:
        positive_mask = query_source_mask
    else:
        positive_mask = np.asarray(positive_mask, dtype=bool).reshape(-1)
        if len(positive_mask) != len(selected_source_ids):
            raise ValueError(
                "positive_mask must have one value for every selected Gallery row."
            )
        positive_mask = positive_mask & query_source_mask
    positive_ids = selected_source_ids[positive_mask]
    positive_counts = dict(zip(*np.unique(positive_ids, return_counts=True)))
    per_source_counts = np.asarray(
        [int(positive_counts.get(source_id, 0)) for source_id in query_sources],
        dtype=np.int64,
    )
    covered_sources = int(np.count_nonzero(per_source_counts > 0))
    positives_per_query = int(positives_per_query)
    expected_positive_rows = int(len(query_sources) * int(positives_per_query))
    return {
        "query_source_count": int(len(query_sources)),
        "covered_query_sources": int(covered_sources),
        "query_positive_coverage": float(covered_sources / max(len(query_sources), 1)),
        "positive_rows": int(np.count_nonzero(positive_mask)),
        "distractor_rows": int(np.count_nonzero(~positive_mask)),
        "distinct_distractor_sources": int(
            len(np.unique(selected_source_ids[~positive_mask]))
        ),
        "expected_positive_rows": expected_positive_rows,
        "same_source_extra_rows": int(
            np.maximum(per_source_counts - positives_per_query, 0).sum()
        ),
        "sources_with_unexpected_positive_count": int(
            np.count_nonzero(per_source_counts != positives_per_query)
        ),
    }


def source_retrieval_metrics(
    predictions,
    query_source_ids,
    gallery_source_ids,
    ks=(1, 10, 50),
    gallery_positive_mask=None,
):
    query_source_ids = np.asarray(query_source_ids, dtype=np.int64).reshape(-1)
    gallery_source_ids = np.asarray(gallery_source_ids, dtype=np.int64).reshape(-1)
    if len(predictions) != len(query_source_ids):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match Query count "
            f"({len(query_source_ids)})."
        )
    if gallery_positive_mask is None:
        gallery_positive_mask = np.ones(len(gallery_source_ids), dtype=bool)
    else:
        gallery_positive_mask = np.asarray(
            gallery_positive_mask, dtype=bool
        ).reshape(-1)
        if len(gallery_positive_mask) != len(gallery_source_ids):
            raise ValueError(
                "gallery_positive_mask must have one value for every Gallery row."
            )
    ks = tuple(sorted(set(int(k) for k in ks if int(k) > 0)))
    if not ks:
        raise ValueError("At least one positive recall cutoff is required.")

    ranks = np.zeros(len(predictions), dtype=np.int64)
    retrieved_positive_counts = np.zeros(len(predictions), dtype=np.int64)
    max_k = ks[-1]
    gallery_counts = dict(
        zip(
            *np.unique(
                gallery_source_ids[gallery_positive_mask], return_counts=True
            )
        )
    )
    positive_counts = np.asarray(
        [gallery_counts.get(source_id, 0) for source_id in query_source_ids],
        dtype=np.int64,
    )
    for query_id, prediction in enumerate(predictions):
        prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)[:max_k]
        if np.any((prediction < 0) | (prediction >= len(gallery_source_ids))):
            raise ValueError("Predictions contain a Gallery row outside the selected index.")
        matches = (
            gallery_source_ids[prediction] == query_source_ids[query_id]
        ) & gallery_positive_mask[prediction]
        positions = np.flatnonzero(matches)
        if len(positions):
            ranks[query_id] = int(positions[0]) + 1
            retrieved_positive_counts[query_id] = int(len(positions))

    reciprocal_ranks = np.zeros(len(ranks), dtype=np.float64)
    hit_mask = ranks > 0
    reciprocal_ranks[hit_mask] = 1.0 / ranks[hit_mask]
    positive_coverage = np.divide(
        retrieved_positive_counts,
        positive_counts,
        out=np.zeros_like(retrieved_positive_counts, dtype=np.float64),
        where=positive_counts > 0,
    )
    return {
        **{
            f"id_recall_at_{k}": float(np.mean((ranks > 0) & (ranks <= k)))
            for k in ks
        },
        f"mrr_at_{max_k}": float(reciprocal_ranks.mean()),
        f"positive_coverage_at_{max_k}": float(positive_coverage.mean()),
        "mean_rank_if_hit": float(ranks[hit_mask].mean()) if np.any(hit_mask) else None,
        "misses_at_max_k": int(np.count_nonzero(~hit_mask)),
    }


def _candidate_source_stats(
    candidate_lists,
    query_source_ids,
    gallery_source_ids,
    gallery_positive_mask=None,
):
    hits = []
    coverage = []
    if gallery_positive_mask is None:
        gallery_positive_mask = np.ones(len(gallery_source_ids), dtype=bool)
    else:
        gallery_positive_mask = np.asarray(
            gallery_positive_mask, dtype=bool
        ).reshape(-1)
        if len(gallery_positive_mask) != len(gallery_source_ids):
            raise ValueError(
                "gallery_positive_mask must have one value for every Gallery row."
            )
    gallery_counts = dict(
        zip(
            *np.unique(
                gallery_source_ids[gallery_positive_mask], return_counts=True
            )
        )
    )
    for candidates, source_id in zip(candidate_lists, query_source_ids):
        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
        matches = (
            (gallery_source_ids[candidates] == source_id)
            & gallery_positive_mask[candidates]
            if len(candidates)
            else []
        )
        matched = int(np.count_nonzero(matches))
        total = int(gallery_counts.get(int(source_id), 0))
        hits.append(matched > 0)
        coverage.append(matched / max(total, 1))
    return {
        "source_hit_rate": float(np.mean(hits)) if hits else 0.0,
        "positive_coverage": float(np.mean(coverage)) if coverage else 0.0,
    }


def _distribution_stats(values):
    """Return JSON-safe summary statistics for per-Query candidate counts."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _candidate_overlap_summary(struct_candidates, global_candidates):
    """Summarize unique A/B candidates and their per-Query overlap."""
    if len(struct_candidates) != len(global_candidates):
        raise ValueError(
            "Structural and global candidate lists must have the same Query count."
        )
    struct_only = []
    global_only = []
    overlap = []
    union_unique = []
    overlap_rate = []
    for struct_ids, global_ids in zip(struct_candidates, global_candidates):
        struct_ids = np.unique(np.asarray(struct_ids, dtype=np.int64).reshape(-1))
        global_ids = np.unique(np.asarray(global_ids, dtype=np.int64).reshape(-1))
        intersection = np.intersect1d(struct_ids, global_ids, assume_unique=True)
        a_only = np.setdiff1d(struct_ids, global_ids, assume_unique=True)
        b_only = np.setdiff1d(global_ids, struct_ids, assume_unique=True)
        union_count = int(len(struct_ids) + len(global_ids) - len(intersection))
        struct_only.append(len(a_only))
        global_only.append(len(b_only))
        overlap.append(len(intersection))
        union_unique.append(union_count)
        overlap_rate.append(len(intersection) / max(union_count, 1))
    return {
        "queries": int(len(struct_candidates)),
        "struct_only": _distribution_stats(struct_only),
        "global_only": _distribution_stats(global_only),
        "overlap": _distribution_stats(overlap),
        "union_unique": _distribution_stats(union_unique),
        "overlap_rate": _distribution_stats(overlap_rate),
    }


def _fusion_config(args):
    from hierarchical_code_fusion import HierarchicalCodeFusionConfig

    return HierarchicalCodeFusionConfig(
        routing_mode=args.routing_mode,
        pattern_level=args.pattern_level,
        path_max_len=args.path_max_len,
        l3_top_r=args.l3_top_r,
        l3_expand_hops=args.l3_expand_hops,
        path_top_p=args.path_top_p,
        l1_top_buckets=args.l1_top_buckets,
        l2_top_regions=args.l2_top_regions,
        min_route_candidates=args.min_route_candidates,
        max_route_candidates=args.max_route_candidates,
        bucket_max_size=args.bucket_max_size,
        struct_hamming_candidates=args.struct_hamming_candidates,
        code_scan_candidates=args.code_scan_candidates,
        max_rerank_candidates=args.max_rerank_candidates,
    )


def _build_indexes(args, gallery, scale_dir, selected_rows):
    from binary_codes import BinaryCodeIndexConfig, BinaryCodeIndex, load_code_encoder
    from hierarchical_code_fusion import HierarchicalCodeFusionIndex

    embeddings = gallery["embeddings"][selected_rows]
    trajectories = gallery["trajectories"][selected_rows]
    source_ids = gallery["source_ids"][selected_rows]
    code_index_dir = osp.join(scale_dir, "code_index")
    fusion_index_dir = osp.join(scale_dir, "fusion_index")

    head, head_metadata = load_code_encoder(args.code_model_file)
    code_config = BinaryCodeIndexConfig(
        bits=head.config.bits,
        subcode_bits=16,
        probe_radius=args.probe_radius,
        hamming_candidates=args.hamming_candidates,
        metric=args.metric,
        normalize=args.normalize,
        embedding_storage=args.embedding_storage,
        encode_batch_size=args.encode_batch_size,
    )
    code_index_start = perf_counter()
    code_index = BinaryCodeIndex(code_config).build(
        embeddings,
        head,
        encode_device=args.code_device,
        metadata={
            "source_embedding_file": gallery["path"],
            "selected_gallery_size": int(len(selected_rows)),
            "code_training": head_metadata,
        },
    )
    code_seconds = perf_counter() - code_index_start
    code_index.save(code_index_dir)

    fusion_start = perf_counter()
    fusion_index = HierarchicalCodeFusionIndex(code_index, _fusion_config(args)).build(
        trajectories,
        args.hierarchy_dir,
        metadata_info={
            "source_embedding_file": gallery["path"],
            "selected_gallery_size": int(len(selected_rows)),
        },
    )
    fusion_seconds = perf_counter() - fusion_start
    fusion_index.save(fusion_index_dir, code_index_dir)
    np.save(osp.join(scale_dir, "selected_gallery_rows.npy"), selected_rows)
    np.save(osp.join(scale_dir, "selected_gallery_source_ids.npy"), source_ids)
    return fusion_index, {
        "code_index_dir": code_index_dir,
        "fusion_index_dir": fusion_index_dir,
        "code_seconds": float(code_seconds),
        "fusion_seconds": float(fusion_seconds),
        "total_seconds": float(code_seconds + fusion_seconds),
        "code_disk_mb": float(_directory_size(code_index_dir) / (1024 ** 2)),
        "fusion_disk_mb": float(_directory_size(fusion_index_dir) / (1024 ** 2)),
    }


def _evaluate_method(
    method,
    index,
    query_embeddings,
    query_trajectories,
    query_source_ids,
    gallery_embeddings,
    gallery_source_ids,
    query_codes,
    code_encoding_ms_per_query,
    top_k,
    exact_max_queries,
    gallery_positive_mask=None,
):
    total = len(query_embeddings)
    if method == "exact" and exact_max_queries > 0:
        total = min(total, exact_max_queries)
    predictions = []
    latencies_ms = []
    candidate_lists = {
        "struct": [],
        "struct_unique": [],
        "code_scan": [],
        "union_before_cap": [],
        "union_rerank": [],
    }
    detail_counts = {
        "route_pool": [],
        "route_pool_before_cap": [],
        "struct": [],
        "struct_unique": [],
        "global": [],
        "union_before_cap": [],
        "rerank": [],
        "patterns": [],
        "l2": [],
        "l3": [],
        "max_l3_hop": [],
        "buckets": [],
        "struct_only": [],
        "global_only": [],
        "overlap": [],
        "union_unique": [],
        "overlap_rate": [],
    }
    full_scan_count = 0
    exact_gallery = None
    if method == "exact":
        exact_gallery = np.asarray(gallery_embeddings, dtype=np.float32)
        if index.code_index.config.normalize or index.code_index.config.metric == "cosine":
            norms = np.linalg.norm(exact_gallery, axis=1, keepdims=True)
            exact_gallery = exact_gallery / np.maximum(norms, 1e-12)

    query_ids = range(total)
    try:
        from tqdm import tqdm

        query_ids = tqdm(query_ids, desc=f"{method} queries")
    except ImportError:
        pass

    for query_id in query_ids:
        start = perf_counter()
        if method == "hier-code":
            ids, _, details = index.query(
                query_embeddings[query_id],
                query_trajectories[query_id],
                top_k=top_k,
                exclude_ids=None,
                return_details=True,
                query_code=query_codes[query_id],
            )
            candidate_lists["struct"].append(details["struct_candidate_ids"])
            candidate_lists["struct_unique"].append(
                details["struct_unique_candidate_ids"]
            )
            candidate_lists["code_scan"].append(details["global_candidate_ids"])
            # Compare the two routes before fusion removes Global IDs from the
            # structural route and refills its protected novelty quota.
            struct_ids = np.unique(
                np.asarray(details["struct_candidate_ids"], dtype=np.int64)
            )
            global_ids = np.unique(
                np.asarray(details["global_candidate_ids"], dtype=np.int64)
            )
            intersection = np.intersect1d(struct_ids, global_ids, assume_unique=True)
            struct_only_count = len(
                np.setdiff1d(struct_ids, global_ids, assume_unique=True)
            )
            global_only_count = len(
                np.setdiff1d(global_ids, struct_ids, assume_unique=True)
            )
            union_unique_count = len(struct_ids) + len(global_ids) - len(intersection)
            detail_counts["struct_only"].append(struct_only_count)
            detail_counts["global_only"].append(global_only_count)
            detail_counts["overlap"].append(len(intersection))
            detail_counts["union_unique"].append(union_unique_count)
            detail_counts["overlap_rate"].append(
                len(intersection) / max(union_unique_count, 1)
            )
            candidate_lists["union_before_cap"].append(
                details["union_before_cap_candidate_ids"]
            )
            candidate_lists["union_rerank"].append(details["candidate_ids"])
            detail_counts["route_pool"].append(details["route_pool_count"])
            detail_counts["route_pool_before_cap"].append(
                details.get("route_pool_before_cap", details["route_pool_count"])
            )
            detail_counts["struct"].append(details["struct_candidate_count"])
            detail_counts["struct_unique"].append(
                details["struct_unique_candidate_count"]
            )
            detail_counts["global"].append(details["global_candidate_count"])
            detail_counts["union_before_cap"].append(
                details["union_count_before_cap"]
            )
            detail_counts["rerank"].append(details["rerank_candidate_count"])
            detail_counts["patterns"].append(details["pattern_count"])
            detail_counts["l2"].append(details.get("l2_count", 0))
            detail_counts["l3"].append(details.get("l3_count", 0))
            detail_counts["max_l3_hop"].append(details.get("max_l3_hop", 0))
            detail_counts["buckets"].append(details["bucket_count"])
            full_scan_count += int(details["global_used_full_scan"])
        elif method == "code-scan":
            ids, _, details = index.code_index.query(
                query_embeddings[query_id],
                top_k=top_k,
                exclude_ids=None,
                return_details=True,
                query_code=query_codes[query_id],
            )
            candidate_lists["code_scan"].append(details["candidate_ids"])
            detail_counts["global"].append(details["hamming_candidate_count"])
            detail_counts["rerank"].append(details["hamming_candidate_count"])
            full_scan_count += int(details["used_full_scan"])
        elif method == "exact":
            query_embedding = np.asarray(
                query_embeddings[query_id], dtype=np.float32
            ).reshape(-1)
            if index.code_index.config.normalize or index.code_index.config.metric == "cosine":
                query_embedding = query_embedding / max(
                    float(np.linalg.norm(query_embedding)), 1e-12
                )
            if index.code_index.config.metric == "cosine":
                distances = 1.0 - exact_gallery @ query_embedding
            else:
                difference = exact_gallery - query_embedding.reshape(1, -1)
                distances = np.einsum("ij,ij->i", difference, difference)
            ids = _topk_indices(distances, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        latencies_ms.append((perf_counter() - start) * 1000.0)
        predictions.append(ids)

    source_metrics = source_retrieval_metrics(
        predictions,
        query_source_ids[:total],
        gallery_source_ids,
        ks=tuple(sorted({top_k, *(k for k in (1, 10, 50) if k <= top_k)})),
        gallery_positive_mask=gallery_positive_mask,
    )
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    encoding_overhead = 0.0 if method == "exact" else code_encoding_ms_per_query
    end_to_end_total_ms = float(latency_array.sum() + encoding_overhead * total)
    candidate_recall = {
        name: _candidate_source_stats(
            lists,
            query_source_ids[:total],
            gallery_source_ids,
            gallery_positive_mask=gallery_positive_mask,
        )
        for name, lists in candidate_lists.items()
        if lists
    }
    candidate_overlap = None
    if method == "hier-code":
        candidate_overlap = _candidate_overlap_summary(
            candidate_lists["struct"], candidate_lists["code_scan"]
        )
    candidate_counts = {
        **{
            f"{name}_mean": float(np.mean(values)) if values else 0.0
            for name, values in detail_counts.items()
            if values
        },
        "global_full_scan_queries": int(full_scan_count),
    }
    if candidate_overlap is not None:
        for name, summary in candidate_overlap.items():
            if name == "queries":
                continue
            candidate_counts.update(
                {
                    f"{name}_{stat}": value
                    for stat, value in summary.items()
                    if stat != "count"
                }
            )
    return {
        "method": method,
        "queries": int(total),
        "top_k": int(top_k),
        "metrics": source_metrics,
        "candidate_recall": candidate_recall,
        "candidate_overlap": candidate_overlap,
        "latency_ms": {
            "mean": float(latency_array.mean() + encoding_overhead) if total else 0.0,
            "p50": float(np.percentile(latency_array, 50) + encoding_overhead)
            if total else 0.0,
            "p95": float(np.percentile(latency_array, 95) + encoding_overhead)
            if total else 0.0,
            "p99": float(np.percentile(latency_array, 99) + encoding_overhead)
            if total else 0.0,
            "retrieval_mean": float(latency_array.mean()) if total else 0.0,
            "code_encoding_per_query": float(encoding_overhead),
        },
        "qps": float(total / (end_to_end_total_ms / 1000.0))
        if end_to_end_total_ms > 0 else 0.0,
        "candidate_counts": candidate_counts,
    }


def run_scale(args, gallery, queries, gallery_size):
    from binary_codes import encode_binary_codes

    scale_dir = osp.join(args.output_dir, f"gallery_{gallery_size}")
    os.makedirs(scale_dir, exist_ok=True)
    master_positive_mask = None
    if args.positive_match_mode == "initial-only":
        master_positive_mask = gallery.get("is_initial")
        if master_positive_mask is None:
            raise ValueError(
                "--positive-match-mode initial-only requires the Gallery NPZ to "
                "contain an is_initial boolean array. Build it with "
                "build_initial_only_gallery.py."
            )
    selected_rows = select_gallery_rows(
        gallery["source_ids"],
        queries["source_ids"],
        gallery_size,
        seed=args.selection_seed,
        mode=args.gallery_selection,
        positives_per_query=args.positives_per_query,
        positive_mask=master_positive_mask,
    )
    selected_source_ids = gallery["source_ids"][selected_rows]
    selected_positive_mask = (
        master_positive_mask[selected_rows]
        if master_positive_mask is not None
        else None
    )
    if selected_positive_mask is not None:
        np.save(
            osp.join(scale_dir, "selected_gallery_is_initial.npy"),
            selected_positive_mask,
        )
    composition = gallery_composition(
        selected_source_ids,
        queries["source_ids"],
        positives_per_query=args.positives_per_query,
        positive_mask=selected_positive_mask,
    )
    if (
        args.gallery_selection == "fixed-positive"
        or args.positive_match_mode == "initial-only"
    ):
        expected = args.positives_per_query * composition["query_source_count"]
        if (
            composition["positive_rows"] != expected
            or composition["same_source_extra_rows"] != 0
            or composition["sources_with_unexpected_positive_count"] != 0
            or composition["query_positive_coverage"] != 1.0
        ):
            raise RuntimeError(
                "Positive Gallery invariant failed: every Query source must have "
                "exactly --positives-per-query valid positive rows."
            )
    positive_source_ids = (
        selected_source_ids
        if selected_positive_mask is None
        else selected_source_ids[selected_positive_mask]
    )
    represented_sources, positive_counts = np.unique(
        positive_source_ids, return_counts=True
    )
    represented_set = set(int(item) for item in represented_sources)
    query_mask = np.asarray(
        [int(source_id) in represented_set for source_id in queries["source_ids"]],
        dtype=bool,
    )
    missing_sources = np.unique(queries["source_ids"][~query_mask])
    if len(missing_sources) and args.missing_positive_policy == "error":
        preview = ", ".join(str(int(item)) for item in missing_sources[:10])
        raise ValueError(
            f"Selected Gallery size {gallery_size} has no positive for "
            f"{len(missing_sources)} Query source IDs (first: {preview}). Use a master "
            "Gallery with complete source coverage, change the selection mode, or set "
            "--missing-positive-policy skip."
        )

    query_embeddings = queries["embeddings"][query_mask]
    query_trajectories = queries["trajectories"][query_mask]
    query_source_ids = queries["source_ids"][query_mask]
    if args.max_queries > 0:
        query_embeddings = query_embeddings[: args.max_queries]
        query_trajectories = query_trajectories[: args.max_queries]
        query_source_ids = query_source_ids[: args.max_queries]
    if not len(query_embeddings):
        raise ValueError("No Query with a positive Gallery item remains for evaluation.")

    index, build_report = _build_indexes(args, gallery, scale_dir, selected_rows)
    code_start = perf_counter()
    query_codes = encode_binary_codes(
        index.code_index.head,
        query_embeddings,
        batch_size=index.code_index.config.encode_batch_size,
        device=args.query_code_device,
    )
    code_encoding_ms = (perf_counter() - code_start) * 1000.0
    code_encoding_ms_per_query = code_encoding_ms / max(len(query_embeddings), 1)

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods).difference(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Expected a subset of {METHODS}.")
    exact_gallery_embeddings = (
        gallery["embeddings"][selected_rows] if "exact" in methods else None
    )
    method_results = [
        _evaluate_method(
            method,
            index,
            query_embeddings,
            query_trajectories,
            query_source_ids,
            exact_gallery_embeddings,
            selected_source_ids,
            query_codes,
            code_encoding_ms_per_query,
            args.top_k,
            args.exact_max_queries,
            selected_positive_mask,
        )
        for method in methods
    ]
    query_positive_counts = np.asarray(
        [
            positive_counts[np.searchsorted(represented_sources, source_id)]
            for source_id in query_source_ids
        ],
        dtype=np.int64,
    )
    result = {
        "protocol": (
            "large_scale_initial_trajectory_identification"
            if args.positive_match_mode == "initial-only"
            else (
                "large_scale_fixed_positive_source_identification"
                if args.gallery_selection == "fixed-positive"
                else "large_scale_paired_source_identification"
            )
        ),
        "gallery_size": int(gallery_size),
        "master_gallery_size": int(len(gallery["embeddings"])),
        "gallery_selection": args.gallery_selection,
        "positive_match_mode": args.positive_match_mode,
        "positives_per_query": (
            int(args.positives_per_query)
            if (
                args.gallery_selection == "fixed-positive"
                or args.positive_match_mode == "initial-only"
            )
            else None
        ),
        "selection_seed": int(args.selection_seed),
        "query_file": queries["path"],
        "gallery_file": gallery["path"],
        "gallery_source_id_mode": args.gallery_source_id_mode,
        "queries_available": int(len(queries["embeddings"])),
        "queries_evaluated": int(len(query_embeddings)),
        "missing_query_sources": int(len(missing_sources)),
        "gallery_composition": composition,
        "positive_multiplicity": {
            "min": int(query_positive_counts.min()),
            "mean": float(query_positive_counts.mean()),
            "p50": float(np.percentile(query_positive_counts, 50)),
            "max": int(query_positive_counts.max()),
        },
        "codes": {
            "bits": int(index.code_index.config.bits),
            "probe_radius": int(index.code_index.config.probe_radius),
            "hamming_candidates": int(index.code_index.config.hamming_candidates),
            "metric": index.code_index.config.metric,
        },
        "fusion": vars(index.config),
        "build": build_report,
        "query_code_encoding_ms": float(code_encoding_ms),
        "methods": method_results,
    }
    result_file = osp.join(scale_dir, "result.json")
    _json_dump(result_file, result)
    print(
        f"gallery={gallery_size}, queries={len(query_embeddings)}, "
        f"build_s={build_report['total_seconds']:.3f}, result={result_file}"
    )
    print(
        f"  positives={composition['positive_rows']}, "
        f"distractors={composition['distractor_rows']}, "
        f"positive_coverage={composition['query_positive_coverage']:.4f}, "
        f"same_source_extra_rows={composition['same_source_extra_rows']}"
    )
    for method_result in method_results:
        metrics = method_result["metrics"]
        recalls = ", ".join(
            f"{key}={value:.4f}"
            for key, value in metrics.items()
            if key.startswith("id_recall_at_")
        )
        print(
            f"  method={method_result['method']}, {recalls}, "
            f"mean_ms={method_result['latency_ms']['mean']:.3f}, "
            f"p95_ms={method_result['latency_ms']['p95']:.3f}, "
            f"qps={method_result['qps']:.2f}"
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate hierarchy + code-scan fusion over configurable "
            "Gallery sizes using independent paired Query/Gallery files."
        )
    )
    parser.add_argument("--gallery-file", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--gallery-source-ids", default=None)
    parser.add_argument("--query-source-ids", default=None)
    parser.add_argument(
        "--gallery-source-id-mode",
        choices=["stored", "query-row"],
        default="stored",
        help="Use query-row to remap legacy perturbation Gallery IDs through Query source_ids.",
    )
    parser.add_argument("--code-model-file", required=True)
    parser.add_argument("--hierarchy-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--gallery-size", type=int, default=0)
    size_group.add_argument(
        "--gallery-sizes",
        default=None,
        help="Comma-separated scale sweep, for example 50000,100000,200000,500000.",
    )
    parser.add_argument(
        "--gallery-selection",
        choices=["source-balanced", "fixed-positive", "prefix"],
        default="source-balanced",
        help=(
            "source-balanced preserves one positive for every represented Query source; "
            "fixed-positive keeps exactly --positives-per-query valid rows per Query "
            "source and fills the remainder with eligible distractors; "
            "prefix uses the first N master Gallery rows."
        ),
    )
    parser.add_argument(
        "--positives-per-query",
        type=int,
        default=1,
        help="Number of positives per Query source in fixed-positive mode.",
    )
    parser.add_argument(
        "--positive-match-mode",
        choices=["same-source", "initial-only"],
        default="same-source",
        help=(
            "same-source counts every Gallery row with the Query source ID; "
            "initial-only additionally requires Gallery is_initial=True."
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--missing-positive-policy", choices=["error", "skip"], default="error"
    )
    parser.add_argument("--methods", default="hier-code")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--exact-max-queries", type=int, default=200)

    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--probe-radius", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--hamming-candidates", type=int, default=2000)
    parser.add_argument(
        "--embedding-storage", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--code-device", default="auto")
    parser.add_argument("--query-code-device", default="cpu")

    parser.add_argument(
        "--routing-mode", default="hierarchical", choices=["legacy", "hierarchical"]
    )
    parser.add_argument("--pattern-level", default="l3", choices=["l2", "l3"])
    parser.add_argument("--path-max-len", type=int, default=3)
    parser.add_argument("--l3-top-r", type=int, default=4)
    parser.add_argument("--l3-expand-hops", type=int, default=1)
    parser.add_argument("--path-top-p", type=int, default=16)
    parser.add_argument("--l1-top-buckets", type=int, default=4)
    parser.add_argument("--l2-top-regions", type=int, default=16)
    parser.add_argument("--min-route-candidates", type=int, default=2048)
    parser.add_argument("--max-route-candidates", type=int, default=4096)
    parser.add_argument("--bucket-max-size", type=int, default=512)
    parser.add_argument("--struct-hamming-candidates", type=int, default=512)
    parser.add_argument("--code-scan-candidates", type=int, default=1024)
    parser.add_argument("--max-rerank-candidates", type=int, default=1536)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.positives_per_query <= 0:
        raise ValueError("--positives-per-query must be positive.")
    if args.positive_match_mode == "initial-only" and args.positives_per_query != 1:
        raise ValueError(
            "--positive-match-mode initial-only requires --positives-per-query 1."
        )
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods).difference(METHODS))
    if not methods or unknown:
        raise ValueError(
            f"--methods must contain a non-empty subset of {METHODS}; unknown={unknown}."
        )
    if "hier-code" in methods and args.top_k > min(
        args.code_scan_candidates, args.max_rerank_candidates
    ):
        raise ValueError(
            "For hier-code, --top-k cannot exceed --code-scan-candidates or "
            "--max-rerank-candidates."
        )
    if "code-scan" in methods and args.top_k > args.hamming_candidates:
        raise ValueError(
            "For code-scan, --top-k cannot exceed --hamming-candidates."
        )
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
        raise ValueError(
            f"Gallery dimension {gallery['embeddings'].shape[1]} does not match "
            f"Query dimension {queries['embeddings'].shape[1]}."
        )
    sizes = parse_gallery_sizes(
        args.gallery_size, args.gallery_sizes, len(gallery["embeddings"])
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for gallery_size in sizes:
        results.append(run_scale(args, gallery, queries, gallery_size))
        gc.collect()
    summary = {
        "protocol": (
            "large_scale_initial_trajectory_identification_sweep"
            if args.positive_match_mode == "initial-only"
            else (
                "large_scale_fixed_positive_source_identification_sweep"
                if args.gallery_selection == "fixed-positive"
                else "large_scale_paired_source_identification_sweep"
            )
        ),
        "gallery_sizes": sizes,
        "gallery_selection": args.gallery_selection,
        "positive_match_mode": args.positive_match_mode,
        "positives_per_query": (
            int(args.positives_per_query)
            if (
                args.gallery_selection == "fixed-positive"
                or args.positive_match_mode == "initial-only"
            )
            else None
        ),
        "nested_selection": True,
        "results": results,
    }
    summary_file = osp.join(args.output_dir, "scale_summary.json")
    _json_dump(summary_file, summary)
    print(f"Scale summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
