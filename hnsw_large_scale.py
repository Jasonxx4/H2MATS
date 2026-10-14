"""Standalone HNSW baseline for the independent Query/Gallery benchmark."""

import argparse
import gc
import json
import os
import os.path as osp
from time import perf_counter

import numpy as np

from hierarchical_code_large_scale import (
    _json_dump,
    _topk_indices,
    gallery_composition,
    load_embedding_payload,
    parse_gallery_sizes,
    remap_gallery_source_ids,
    select_gallery_rows,
    source_retrieval_metrics,
)
from candidate_selection import _payload_labels, _row_retrieval_metrics, _same_file
from tp_retrieval_evaluation import (
    add_tp_arguments,
    close_mmap_array,
    compute_tp_distance_table,
    compute_tp_ground_truth,
    resolve_tp_cache_file,
    tp_ground_truth_from_distance_table,
    tp_retrieval_metrics,
)


def _load_hnswlib():
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError(
            "The hnsw method requires hnswlib. Install it with "
            "'pip install hnswlib' in the evaluation environment."
        ) from exc
    return hnswlib


class HNSWIndex:
    def __init__(
        self,
        vectors,
        metric="l2",
        m=16,
        ef_construction=200,
        ef_search=80,
        num_threads=0,
        seed=2026,
    ):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("HNSW vectors must be a non-empty 2-D array.")
        if metric not in {"l2", "cosine"}:
            raise ValueError("metric must be 'l2' or 'cosine'.")
        hnswlib = _load_hnswlib()
        self.metric = metric
        self.vectors = vectors
        self.ef_search = int(ef_search)
        self.num_threads = int(num_threads)
        space = "cosine" if metric == "cosine" else "l2"
        self.index = hnswlib.Index(space=space, dim=vectors.shape[1])
        self.index.init_index(
            max_elements=len(vectors),
            ef_construction=int(ef_construction),
            M=int(m),
            random_seed=int(seed),
        )
        if self.num_threads > 0:
            self.index.set_num_threads(self.num_threads)
        labels = np.arange(len(vectors), dtype=np.int64)
        if self.num_threads > 0:
            self.index.add_items(vectors, labels, num_threads=self.num_threads)
        else:
            self.index.add_items(vectors, labels)
        self.index.set_ef(max(int(ef_search), 1))

    def query(self, vector, top_k):
        top_k = min(int(top_k), len(self.vectors))
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self.metric == "cosine":
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        labels, distances = self.index.knn_query(
            vector.reshape(1, -1), k=top_k
        )
        return labels[0].astype(np.int64), distances[0].astype(np.float32)

    def save(self, path):
        self.index.save_index(path)


def vector_recall_metrics(approx_predictions, exact_predictions, ks=(1, 10, 50)):
    total = min(len(approx_predictions), len(exact_predictions))
    result = {"queries": int(total)}
    for cutoff in sorted(set(int(k) for k in ks if int(k) > 0)):
        overlap = []
        exact_top1_hits = []
        for approx, exact in zip(
            approx_predictions[:total], exact_predictions[:total]
        ):
            approx_rows = np.asarray(approx, dtype=np.int64)[:cutoff]
            exact_rows = np.asarray(exact, dtype=np.int64)[:cutoff]
            denominator = max(len(exact_rows), 1)
            overlap.append(
                len(np.intersect1d(approx_rows, exact_rows, assume_unique=False))
                / denominator
            )
            exact_top1_hits.append(
                bool(len(exact_rows) and exact_rows[0] in set(approx_rows.tolist()))
            )
        result[f"vector_recall_at_{cutoff}"] = (
            float(np.mean(overlap)) if overlap else 0.0
        )
        result[f"exact_top1_recall_at_{cutoff}"] = (
            float(np.mean(exact_top1_hits)) if exact_top1_hits else 0.0
        )
    return result


def _evaluate(
    method,
    query_embeddings,
    query_source_ids,
    gallery_embeddings,
    gallery_source_ids,
    gallery_positive_mask,
    top_k,
    exact_max_queries,
    hnsw_config,
    index_file=None,
    query_labels=None,
    exclude_self=False,
    return_predictions=False,
):
    total = len(query_embeddings)
    if method == "exact" and exact_max_queries > 0:
        total = min(total, int(exact_max_queries))
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    gallery_for_search = gallery
    if hnsw_config["metric"] == "cosine":
        gallery_for_search = gallery / np.maximum(
            np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12
        )
    hnsw = None
    build_seconds = 0.0
    if method == "hnsw":
        start = perf_counter()
        hnsw = HNSWIndex(gallery_for_search, **hnsw_config)
        build_seconds = perf_counter() - start
        if index_file:
            hnsw.save(index_file)
    predictions = []
    latencies = []
    for query_id in range(total):
        start = perf_counter()
        query = np.asarray(query_embeddings[query_id], dtype=np.float32).reshape(-1)
        if method == "hnsw":
            ids, _ = hnsw.query(query, top_k + int(bool(exclude_self)))
        elif method == "exact":
            if hnsw_config["metric"] == "cosine":
                query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
                distances = 1.0 - gallery_for_search @ query_norm
            else:
                difference = gallery - query.reshape(1, -1)
                distances = np.einsum("ij,ij->i", difference, difference)
            if exclude_self and 0 <= int(query_source_ids[query_id]) < len(distances):
                distances[int(query_source_ids[query_id])] = np.inf
            ids = _topk_indices(distances, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        if exclude_self:
            ids = ids[ids != int(query_source_ids[query_id])][:top_k]
        else:
            ids = ids[:top_k]
        latencies.append((perf_counter() - start) * 1000.0)
        predictions.append(np.asarray(ids, dtype=np.int64))
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
    latency = np.asarray(latencies, dtype=np.float64)
    result = {
        "method": method,
        "queries": int(total),
        "exclude_self": bool(exclude_self),
        "metrics": metrics,
        "latency_ms": {
            "mean": float(latency.mean()) if len(latency) else 0.0,
            "p50": float(np.percentile(latency, 50)) if len(latency) else 0.0,
            "p95": float(np.percentile(latency, 95)) if len(latency) else 0.0,
        },
        "qps": float(1000.0 / latency.mean()) if len(latency) and latency.mean() else 0.0,
        "build_seconds": float(build_seconds),
        "embedding_memory_mb": float(gallery.nbytes / (1024.0 ** 2)),
        "index_disk_mb": (
            float(osp.getsize(index_file) / (1024.0 ** 2))
            if index_file and osp.exists(index_file)
            else None
        ),
    }
    return (result, predictions) if return_predictions else result


def run_scale(args, gallery, queries, gallery_size, tp_distance_table=None):
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
    query_embeddings = queries["embeddings"]
    query_source_ids = queries["source_ids"]
    query_labels = _payload_labels(queries)
    exclude_self = args.exclude_self
    if exclude_self is None:
        exclude_self = _same_file(gallery["path"], queries["path"])
    if args.max_queries > 0:
        query_embeddings = query_embeddings[: args.max_queries]
        query_source_ids = query_source_ids[: args.max_queries]
        if query_labels is not None:
            query_labels = query_labels[: args.max_queries]
    gallery_embeddings = gallery["embeddings"][selected_rows]
    results = []
    predictions_by_method = {}
    for method in args.methods:
        index_file = (
            osp.join(scale_dir, "hnsw_index.bin") if method == "hnsw" else None
        )
        result, predictions = _evaluate(
            method,
            query_embeddings,
            query_source_ids,
            gallery_embeddings,
            selected_source_ids,
            selected_positive_mask,
            args.top_k,
            args.exact_max_queries,
            {
                "metric": args.metric,
                "m": args.hnsw_m,
                "ef_construction": args.hnsw_ef_construction,
                "ef_search": args.hnsw_ef_search,
                "num_threads": args.hnsw_num_threads,
                "seed": args.selection_seed,
            },
            index_file=index_file,
            query_labels=query_labels,
            exclude_self=bool(exclude_self),
            return_predictions=True,
        )
        results.append(result)
        predictions_by_method[method] = predictions
    vector_comparison = None
    if {"hnsw", "exact"}.issubset(predictions_by_method):
        vector_comparison = vector_recall_metrics(
            predictions_by_method["hnsw"],
            predictions_by_method["exact"],
            ks=tuple(sorted({args.top_k, *(k for k in (1, 10, 50) if k <= args.top_k)})),
        )
    tp_report = None
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
                tp_distance_table,
                selected_columns,
                top_k=args.tp_top_k,
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
        for result in results:
            result["tp_metrics"] = tp_retrieval_metrics(
                predictions_by_method[result["method"]], tp_ids
            )
    payload = {
        "protocol": "hnsw_vs_exact_source_identification",
        "gallery_size": int(gallery_size),
        "queries": int(len(query_embeddings)),
        "gallery_selection": args.gallery_selection,
        "positive_match_mode": args.positive_match_mode,
        "exclude_self": bool(exclude_self),
        "gallery_composition": composition,
        "hnsw_config": {
            "metric": args.metric,
            "m": args.hnsw_m,
            "ef_construction": args.hnsw_ef_construction,
            "ef_search": args.hnsw_ef_search,
            "num_threads": args.hnsw_num_threads,
        },
        "methods": results,
        "hnsw_vs_exact": vector_comparison,
        "tp_evaluation": tp_report,
    }
    result_file = osp.join(scale_dir, "result.json")
    _json_dump(result_file, payload)
    print(f"gallery={gallery_size}, queries={len(query_embeddings)}, result={result_file}")
    for result in results:
        metric_keys = ("HR@10", "HR@50", "R10@50", "R1@1", "R1@10", "R1@50")
        metrics = ", ".join(
            f"{key}={result['metrics'][key]:.4f}"
            for key in metric_keys
            if key in result["metrics"]
        )
        if not metrics:
            metrics = ", ".join(
                f"{key}={value:.4f}"
                for key, value in result["metrics"].items()
                if key.startswith("id_recall_at_")
            )
        print(
            f"  method={result['method']}, {metrics}, "
            f"mean_ms={result['latency_ms']['mean']:.3f}, qps={result['qps']:.2f}"
        )
        if result.get("tp_metrics"):
            tp_metrics = ", ".join(
                f"{key}={value:.4f}"
                for key, value in result["tp_metrics"].items()
                if key != "queries"
            )
            print(f"    TP ({result['tp_metrics']['queries']} queries): {tp_metrics}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a simple full-gallery HNSW ANN with exact vector search."
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
    parser.add_argument("--methods", default="hnsw,exact")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--exact-max-queries", type=int, default=200)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=20)
    parser.add_argument("--hnsw-ef-search", type=int, default=1024)
    parser.add_argument(
        "--hnsw-num-threads",
        type=int,
        default=0,
        help="HNSW build/query threads; 0 uses the hnswlib default.",
    )
    add_tp_arguments(parser)
    parser.add_argument(
        "--tp-reuse-max-distance-table",
        action="store_true",
        help=(
            "For multi-scale TP evaluation, compute/cache Query x max-Gallery TP "
            "distances once and derive each smaller scale's exact Top-K by table lookup."
        ),
    )
    parser.add_argument(
        "--tp-distance-table-file",
        default=None,
        help="Optional .npy cache for --tp-reuse-max-distance-table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(args.methods).difference({"hnsw", "exact"}))
    if unknown or not args.methods:
        raise ValueError(f"Unknown methods: {unknown}; expected hnsw and/or exact.")
    if args.top_k <= 0 or args.positives_per_query <= 0:
        raise ValueError("top_k and positives_per_query must be positive.")
    if "hnsw" in args.methods and args.hnsw_ef_search < args.top_k:
        raise ValueError("hnsw-ef-search must be greater than or equal to top-k.")
    if args.positive_match_mode == "initial-only" and args.positives_per_query != 1:
        raise ValueError("initial-only requires positives_per_query=1.")
    gallery = load_embedding_payload(
        args.gallery_file, args.gallery_source_ids, role="gallery"
    )
    queries = load_embedding_payload(args.query_file, args.query_source_ids, role="query")
    gallery = remap_gallery_source_ids(
        gallery, queries, mode=args.gallery_source_id_mode
    )
    if gallery["embeddings"].shape[1] != queries["embeddings"].shape[1]:
        raise ValueError("Gallery and Query embedding dimensions must match.")
    sizes = parse_gallery_sizes(
        args.gallery_size, args.gallery_sizes, len(gallery["embeddings"])
    )
    args.tp_multi_scale = len(sizes) > 1
    os.makedirs(args.output_dir, exist_ok=True)
    tp_distance_table = None
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
            table_file = osp.join(args.output_dir, f"tp_distance_table_gallery_{max_size}.npy")
        args.tp_distance_table_file = table_file
        tp_distance_table, table_report = compute_tp_distance_table(
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
        max_row_position = {int(row): position for position, row in enumerate(max_rows)}
        args.tp_max_row_position = max_row_position
        print(
            "TP distance table ready: "
            f"gallery={max_size}, queries={table_report['queries']}, "
            f"cache_reused={table_report['cache_reused']}, file={table_file}"
        )
    results = []
    for gallery_size in sizes:
        results.append(run_scale(args, gallery, queries, gallery_size, tp_distance_table))
        gc.collect()
    if tp_distance_table is not None:
        close_mmap_array(tp_distance_table)
    summary_file = osp.join(args.output_dir, "scale_summary.json")
    _json_dump(
        summary_file,
        {"protocol": "hnsw_vs_exact_source_identification_sweep", "results": results},
    )
    print(f"Scale summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
