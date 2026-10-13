"""Build and evaluate Adaptive TopoHash directly from a trained encoder checkpoint.

The script reuses the checkpoint embedding export and post-hoc hash-head
training pipeline, then evaluates Global Hash, legacy TopoHash, and the
adaptive Global-core/topology-rescue methods on the same test split.  The
``adaptive-dual-metric`` method keeps the Candidate Hash search and adds an
independent Metric-Head retrieval only for scheduled low-confidence queries.
"""

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from dual_metric_fusion import (
    fit_weight_payload,
    load_weight_payload,
    save_weight_payload,
)
from hier_hash_large_scale import _json_dump, load_embedding_payload
from topo_route_hash_large_scale import (
    METHODS,
    _build_index,
    _evaluate_method,
    _hash_config,
    _row_retrieval_metrics,
    source_retrieval_metrics,
)
from trajectory_metric_backend import METRIC_NAMES, TrajectoryMetricBackend


def _is_embedding_file(path):
    if not path:
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return "embeddings" in data.files
    except (OSError, ValueError, TypeError):
        return False


def _run_hash_export(args, assets_dir, retrain_only=False):
    hash_script = osp.join(osp.dirname(__file__), "run_hash_global_all.py")
    command = [
        sys.executable,
        hash_script,
        "--model", args.checkpoint,
        "--dataset", args.dataset,
        "--dataset-mode", args.dataset_mode,
        "--distance-type", args.distance_type,
        "--output-dir", assets_dir,
        "--bits", str(args.bits),
        "--hash-device", args.hash_device,
        "--embedding-batch-size", str(args.embedding_batch_size),
        "--epochs", str(args.hash_epochs),
        "--hash-batch-size", str(args.hash_batch_size),
        "--positive-pool", str(args.hash_positive_pool),
        "--positives-per-anchor", str(args.hash_positives_per_anchor),
        "--negatives-per-anchor", str(args.hash_negatives_per_anchor),
        "--negative-strategy", args.hash_negative_strategy,
        "--hard-negative-pool", str(args.hash_hard_negative_pool),
        "--hard-negative-fraction", str(args.hash_hard_negative_fraction),
        "--metric-distance-weight", str(args.hash_metric_distance_weight),
        "--metric-distance-scale", str(args.hash_metric_distance_scale),
        "--embedding-distance-weight", str(args.hash_embedding_distance_weight),
        "--in-batch-negative-weight", str(args.hash_in_batch_negative_weight),
        "--graded-rank-weight", str(args.hash_graded_rank_weight),
        "--graded-rank-margin", str(args.hash_graded_rank_margin),
        "--graded-rank-cutoffs", args.hash_graded_rank_cutoffs,
        "--cutoff-rank-weight", str(args.hash_cutoff_rank_weight),
        "--cutoff-rank", str(args.hash_cutoff_rank),
        "--cutoff-rank-margin", str(args.hash_cutoff_rank_margin),
        "--cutoff-rank-window", str(args.hash_cutoff_rank_window),
        "--cutoff-samples", str(args.hash_cutoff_samples),
        "--candidate-embedding-distance-weight",
        str(args.candidate_hash_embedding_distance_weight),
        "--candidate-in-batch-negative-weight",
        str(args.candidate_hash_in_batch_negative_weight),
        "--skip-eval",
    ]
    if args.dual_hash_heads:
        command.append("--dual-hash-heads")
    if args.validation_file:
        command.extend(["--validation-file", args.validation_file])
    if args.hash_metric_distance_file:
        command.extend(["--metric-distance-file", args.hash_metric_distance_file])
    if args.hash_metric_candidate_teacher_file:
        command.extend([
            "--metric-candidate-teacher-file",
            args.hash_metric_candidate_teacher_file,
        ])
    if args.hash_filter_same_source_negatives:
        command.append("--filter-same-source-negatives")
    if args.test_file:
        command.extend(["--test-file", args.test_file])
    if args.cuda is not None:
        command.extend(["--cuda", str(args.cuda)])
    if args.seq_encoder:
        command.extend(["--seq-encoder", args.seq_encoder])
    if args.graph_encoder:
        command.extend(["--graph-encoder", args.graph_encoder])
    if args.coord_encoder:
        command.extend(["--coord-encoder", args.coord_encoder])
    if args.hmt_backbone:
        command.extend(["--hmt-backbone", args.hmt_backbone])
    if args.hmt_scale_aligned_graph:
        command.append("--hmt-scale-aligned-graph")
    if args.coord_fusion:
        command.extend(["--coord-fusion", args.coord_fusion])
    if args.normalize:
        command.append("--normalize")
    if args.force:
        command.append("--force-all")
    else:
        if args.reexport_embeddings:
            command.append("--force-export")
        if retrain_only:
            command.append("--force-train")
    print("[adaptive] exporting embeddings and preparing hash assets")
    subprocess.run(command, check=True, cwd=osp.dirname(__file__))
    return (
        osp.join(assets_dir, "test_embeddings.npz"),
        osp.join(
            assets_dir,
            (
                f"hash_head_{args.bits}_metric.pt"
                if args.dual_hash_heads else f"hash_head_{args.bits}.pt"
            ),
        ),
    )


def _load_truth(path):
    if not path:
        return None
    if not osp.isfile(path):
        raise FileNotFoundError(f"Truth file not found: {path}")
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            for key in ("ids", "truth_ids", "labels"):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.int64)
        raise ValueError(f"Truth NPZ {path} must contain ids/truth_ids/labels.")
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)


def _format_result_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _print_table(title, rows, columns):
    if not rows:
        return
    widths = {
        column: max(
            len(column),
            *(len(_format_result_value(row.get(column, ""))) for row in rows),
        )
        for column in columns
    }
    print(f"\n[results] {title}")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        fields = []
        for column in columns:
            value = _format_result_value(row.get(column, ""))
            fields.append(
                value.ljust(widths[column])
                if column in {"method", "pool"}
                else value.rjust(widths[column])
            )
        print("  ".join(fields))


def _print_result_summary(results):
    metric_names = list(dict.fromkeys(
        name
        for result in results
        for name in result.get("metrics", {})
    ))
    metric_rows = [
        {
            "method": result["method"],
            "queries": result.get("queries", 0),
            **result.get("metrics", {}),
        }
        for result in results
    ]
    _print_table(
        "retrieval metrics",
        metric_rows,
        ["method", "queries", *metric_names],
    )

    truth_names = list(dict.fromkeys(
        name
        for result in results
        for name in result.get("metric_truth", {})
    ))
    truth_rows = [
        {"method": result["method"], **result["metric_truth"]}
        for result in results
        if result.get("metric_truth")
    ]
    if truth_rows:
        _print_table("external metric truth", truth_rows, ["method", *truth_names])

    latency_rows = []
    for result in results:
        latency = result.get("latency_ms", {})
        latency_rows.append({
            "method": result["method"],
            "mean_ms": latency.get("mean", 0.0),
            "p50_ms": latency.get("p50", 0.0),
            "p95_ms": latency.get("p95", 0.0),
            "qps": result.get("qps", 0.0),
        })
    _print_table(
        "latency and throughput",
        latency_rows,
        ["method", "mean_ms", "p50_ms", "p95_ms", "qps"],
    )

    candidate_rows = []
    candidate_names = []
    for result in results:
        for pool, stats in result.get("candidate_recall", {}).items():
            candidate_names.extend(stats)
            candidate_rows.append({
                "method": result["method"],
                "pool": pool,
                **stats,
            })
    candidate_names = list(dict.fromkeys(candidate_names))
    if candidate_rows:
        _print_table(
            "candidate coverage",
            candidate_rows,
            ["method", "pool", *candidate_names],
        )

    for result in results:
        counts = result.get("candidate_counts", {})
        if not counts:
            continue
        values = ", ".join(
            f"{name}={_format_result_value(value)}"
            for name, value in counts.items()
        )
        print(f"[results] {result['method']} candidate counts: {values}")


def _pack_trajectory_rows(trajectories, query_indices):
    """Pack variable-length trajectories without relying on object arrays."""
    values = []
    offsets = [0]
    for query_index in query_indices:
        trajectory = np.asarray(trajectories[int(query_index)], dtype=np.int64)
        trajectory = trajectory.reshape(-1)
        values.append(trajectory)
        offsets.append(offsets[-1] + len(trajectory))
    return (
        np.concatenate(values) if values else np.empty(0, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
    )


def _subset_metrics(
    predictions,
    query_indices,
    labels,
    query_source_ids,
    gallery_source_ids,
    top_k,
):
    query_indices = np.asarray(query_indices, dtype=np.int64)
    subset_predictions = [predictions[int(index)] for index in query_indices]
    if labels is not None:
        return _row_retrieval_metrics(
            subset_predictions, np.asarray(labels)[query_indices]
        )
    return source_retrieval_metrics(
        subset_predictions,
        np.asarray(query_source_ids)[query_indices],
        np.asarray(gallery_source_ids),
        ks=tuple(sorted({top_k, *(k for k in (1, 10, 50) if k <= top_k)})),
        gallery_positive_mask=np.ones(len(gallery_source_ids), dtype=bool),
    )


def _write_low_confidence_comparison(
    args,
    results,
    predictions,
    trajectories,
    query_source_ids,
    gallery_source_ids,
    labels,
):
    """Persist low-confidence trajectories and Global-vs-TP comparisons."""
    adaptive_method = next(
        (
            method
            for method in ("adaptive-dual-metric", "adaptive-topo-hash")
            if method in predictions
        ),
        None,
    )
    no_rerank_method = "adaptive-dual-metric-no-rerank"
    if adaptive_method is None or "global-hash" not in predictions:
        return None
    has_no_rerank = no_rerank_method in predictions
    adaptive_result = next(
        result for result in results if result["method"] == adaptive_method
    )
    budget = adaptive_result.get("adaptive_tp_budget", {})
    eligible_indices = np.asarray(
        budget.get("low_confidence_query_ids", []), dtype=np.int64
    )
    reranked_indices = np.asarray(
        budget.get("selected_query_ids", []), dtype=np.int64
    )
    total = min(len(query_source_ids), len(predictions[adaptive_method]))
    eligible_indices = eligible_indices[(eligible_indices >= 0) & (eligible_indices < total)]
    reranked_indices = reranked_indices[(reranked_indices >= 0) & (reranked_indices < total)]
    if not len(eligible_indices):
        return {
            "adaptive_method": adaptive_method,
            "eligible_queries": 0,
            "reranked_queries": 0,
            "artifact": None,
        }

    subset_definitions = {
        "eligible": eligible_indices,
        "reranked": reranked_indices,
    }
    subset_report = {}
    for subset_name, subset_indices in subset_definitions.items():
        if not len(subset_indices):
            subset_report[subset_name] = {"queries": 0, "methods": {}}
            continue
        method_report = {}
        method_specs = [
            ("global-hash", "global-hash-direct"),
            (adaptive_method, "topology-tp-rerank"),
        ]
        if has_no_rerank:
            method_specs.append((no_rerank_method, "dual-hash-no-rerank"))
        for method_name, method_label in method_specs:
            method_result = next(
                result for result in results if result["method"] == method_name
            )
            method_report[method_label] = {
                "queries": int(len(subset_indices)),
                "metrics": _subset_metrics(
                    predictions[method_name],
                    subset_indices,
                    labels,
                    query_source_ids,
                    gallery_source_ids,
                    args.top_k,
                ),
                "full_method_mean_ms": method_result["latency_ms"]["mean"],
            }
        global_metrics = method_report["global-hash-direct"]["metrics"]
        rerank_metrics = method_report["topology-tp-rerank"]["metrics"]
        method_report["delta_rerank_minus_global"] = {
            key: float(rerank_metrics[key] - global_metrics[key])
            for key in rerank_metrics
            if key in global_metrics and key != "queries"
        }
        if has_no_rerank:
            no_rerank_metrics = method_report["dual-hash-no-rerank"]["metrics"]
            method_report["delta_no_rerank_minus_global"] = {
                key: float(no_rerank_metrics[key] - global_metrics[key])
                for key in no_rerank_metrics
                if key in global_metrics and key != "queries"
            }
            method_report["delta_rerank_minus_no_rerank"] = {
                key: float(rerank_metrics[key] - no_rerank_metrics[key])
                for key in rerank_metrics
                if key in no_rerank_metrics and key != "queries"
            }
        subset_report[subset_name] = {
            "queries": int(len(subset_indices)),
            "methods": method_report,
        }

    trajectory_values, trajectory_offsets = _pack_trajectory_rows(
        trajectories, eligible_indices
    )
    eligible_position = {
        int(query_index): position
        for position, query_index in enumerate(eligible_indices)
    }
    reranked_positions = np.asarray(
        [eligible_position[int(index)] for index in reranked_indices if int(index) in eligible_position],
        dtype=np.int64,
    )
    artifact_file = osp.join(args.output_dir, "low_confidence_comparison.npz")
    save_payload = {
        "eligible_query_indices": eligible_indices,
        "eligible_source_ids": np.asarray(query_source_ids)[eligible_indices],
        "eligible_trajectory_values": trajectory_values,
        "eligible_trajectory_offsets": trajectory_offsets,
        "reranked_positions_in_eligible": reranked_positions,
        "global_hash_predictions": np.asarray(
            [predictions["global-hash"][int(index)] for index in eligible_indices],
            dtype=np.int64,
        ),
        "topology_tp_predictions": np.asarray(
            [predictions[adaptive_method][int(index)] for index in eligible_indices],
            dtype=np.int64,
        ),
    }
    if has_no_rerank:
        save_payload["dual_hash_no_rerank_predictions"] = np.asarray(
            [predictions[no_rerank_method][int(index)] for index in eligible_indices],
            dtype=np.int64,
        )
    if labels is not None:
        save_payload["eligible_truth_labels"] = np.asarray(labels)[eligible_indices]
    np.savez_compressed(artifact_file, **save_payload)
    report = {
        "adaptive_method": adaptive_method,
        "definition": {
            "eligible": "queries below the adaptive confidence threshold",
            "reranked": "eligible queries actually sent to Topology + TP reranking",
        },
        "eligible_queries": int(len(eligible_indices)),
        "reranked_queries": int(len(reranked_indices)),
        "subset_report": subset_report,
        "artifact": osp.abspath(artifact_file),
    }
    if has_no_rerank:
        report["definition"]["dual_hash_no_rerank"] = (
            "eligible queries sent through Global + Metric-Head + topology "
            "candidate fusion without TP reranking"
        )
    return report


def _metric_args(args, trajectories):
    metric = args.rerank_metric
    if metric == "embedding":
        return None
    node_count = args.node_count
    if node_count <= 0:
        node_count = max(int(np.max(np.concatenate(trajectories))) + 1, 1)
    max_length = args.max_trajectory_length
    if max_length <= 0:
        max_length = max(len(item) for item in trajectories)
    return TrajectoryMetricBackend(
        metric,
        point_distance_file=args.point_distance_file,
        edge_file=args.edge_file,
        node_count=node_count,
        max_trajectory_length=max_length,
        neterp_anchor_node=args.neterp_anchor_node,
        num_threads=args.tp_num_threads,
    )


def _attach_metric_hash(index, payload, model_file, args, topo_args):
    """Attach the independent metric head and encode validation/query codes."""
    from hash_global import (
        HashGlobalIndex,
        encode_binary_codes,
        encode_soft_codes,
        load_hash_head,
    )

    metric_head, _ = load_hash_head(model_file)
    if metric_head.config.bits != index.hash_index.config.bits:
        raise ValueError(
            "Candidate and metric-prefilter hash heads must use the same bit count."
        )
    index.metric_hash_index = HashGlobalIndex(
        _hash_config(topo_args, metric_head)
    ).build(
        payload["embeddings"],
        metric_head,
        encode_device=args.hash_device,
        metadata={"role": "adaptive-dual-metric"},
    )
    packed_codes = encode_binary_codes(
        metric_head,
        payload["embeddings"],
        batch_size=args.encode_batch_size,
        device=args.query_code_device,
    )
    if args.metric_prefilter_hash_score == "soft":
        score_codes = encode_soft_codes(
            metric_head,
            payload["embeddings"],
            batch_size=args.encode_batch_size,
            device=args.query_code_device,
            temperature=args.metric_prefilter_soft_temperature,
        )
        index.metric_prefilter_soft_codes = score_codes
    else:
        score_codes = packed_codes
        index.metric_prefilter_codes = packed_codes
    return score_codes, packed_codes


def _fit_or_load_dual_metric_weights(
    args,
    topo_args,
    validation_embedding_file,
    metric_hash_model_file,
):
    if not args.learn_dual_metric_weights:
        return None, None
    if not metric_hash_model_file:
        raise ValueError(
            "Learned dual-metric weights require an independent metric hash head."
        )
    mode = "learned-bucket" if args.dual_metric_weight_per_bucket else "learned-global"
    weight_file = args.dual_metric_weight_file or osp.join(
        args.output_dir,
        f"dual_metric_weights_{args.rerank_metric.lower()}_{mode}.json",
    )
    if osp.isfile(weight_file) and not args.refit_dual_metric_weights:
        payload = load_weight_payload(weight_file, metric_name=args.rerank_metric)
        if payload.get("mode") != mode:
            raise ValueError(
                f"Learned weight mode mismatch: requested {mode}, "
                f"file contains {payload.get('mode')}."
            )
        print(f"[adaptive] loaded learned dual-metric weights: {weight_file}")
        return payload, weight_file
    if not osp.isfile(validation_embedding_file):
        raise FileNotFoundError(
            "Learned dual-metric weights require validation embeddings: "
            f"{validation_embedding_file}"
        )

    from hash_global import encode_binary_codes

    validation = load_embedding_payload(validation_embedding_file, role="query")
    validation_rows = np.arange(len(validation["embeddings"]), dtype=np.int64)
    fit_index_dir = osp.join(args.output_dir, "dual_metric_weight_fit_index")
    fit_index, _ = _build_index(
        topo_args, validation, validation_rows, fit_index_dir
    )
    fit_index.exclude_self = bool(args.exclude_self)
    candidate_codes = encode_binary_codes(
        fit_index.hash_index.head,
        validation["embeddings"],
        batch_size=args.encode_batch_size,
        device=args.query_code_device,
    )
    metric_score_codes, metric_hash_codes = _attach_metric_hash(
        fit_index, validation, metric_hash_model_file, args, topo_args
    )
    validation_anchor_count = validation.get("anchor_count")
    if validation_anchor_count is None:
        validation_anchor_count = len(validation["embeddings"])
    query_ids = np.arange(
        min(int(validation_anchor_count), len(validation["embeddings"])),
        dtype=np.int64,
    )
    if 0 < args.dual_metric_weight_max_queries < len(query_ids):
        rng = np.random.default_rng(args.dual_metric_weight_seed)
        query_ids = np.sort(
            rng.choice(
                query_ids,
                size=args.dual_metric_weight_max_queries,
                replace=False,
            )
        )
    metric_backend = _metric_args(args, validation["trajectories"])
    records = []
    try:
        try:
            from tqdm import tqdm

            iterator = tqdm(query_ids, desc="learn dual-metric weights")
        except ImportError:
            iterator = query_ids
        source_ids = np.asarray(validation["source_ids"])
        for query_id in iterator:
            query_id = int(query_id)
            exclude_ids = None
            if args.exclude_self:
                exclude_ids = np.flatnonzero(
                    source_ids == source_ids[query_id]
                ).astype(np.int64, copy=False)
            _, _, details = fit_index.query_adaptive_dual_metric(
                validation["embeddings"][query_id],
                validation["trajectories"][query_id],
                candidate_codes[query_id],
                metric_hash_codes[query_id],
                top_k=args.top_k,
                metric_backend=metric_backend,
                metric_query_code=metric_score_codes[query_id],
                exclude_ids=exclude_ids,
                dual_metric_no_rerank=True,
                collect_dual_metric_features=True,
            )
            fusion = details.get("dual_metric_fusion_features")
            if fusion is None or not len(fusion["universe_ids"]):
                continue
            universe_ids = np.asarray(fusion["universe_ids"], dtype=np.int64)
            oracle_distances = metric_backend.distances(
                validation["trajectories"][query_id],
                validation["trajectories"][universe_ids],
            )
            records.append({
                "rank_features": fusion["rank_features"],
                "protected_mask": fusion["protected_mask"],
                "oracle_distances": oracle_distances,
                "bucket": fusion["bucket"],
                "global_margin": float(details["adaptive_global_margin"]),
            })
    finally:
        metric_backend.close()

    eligible = [
        record for record in records
        if record["global_margin"] < args.adaptive_global_margin
    ]
    rerank_budget = min(
        int(np.floor(len(query_ids) * args.adaptive_max_tp_fraction + 1e-12)),
        len(eligible),
    )
    rerank_samples = sorted(
        eligible, key=lambda record: record["global_margin"]
    )[:rerank_budget]
    requested = {
        item.strip() for item in args.methods.split(",") if item.strip()
    }
    samples_by_variant = {
        "rerank": (
            rerank_samples if "adaptive-dual-metric" in requested else []
        ),
        "no_rerank": (
            eligible
            if "adaptive-dual-metric-no-rerank" in requested else []
        ),
    }
    if not any(samples_by_variant.values()):
        raise ValueError(
            "No low-confidence validation queries are available to learn "
            "dual-metric weights. Increase --dual-metric-weight-max-queries, "
            "--adaptive-global-margin, or --adaptive-max-tp-fraction."
        )
    initial_weights = (
        args.dual_metric_embedding_weight,
        args.dual_metric_hash_weight,
        args.dual_metric_topology_weight,
    )
    payload = fit_weight_payload(
        samples_by_variant,
        args.rerank_metric,
        initial_weights,
        per_bucket=args.dual_metric_weight_per_bucket,
        top_k=args.top_k,
        epochs=args.dual_metric_weight_epochs,
        learning_rate=args.dual_metric_weight_learning_rate,
        max_pairs_per_query=args.dual_metric_weight_max_pairs_per_query,
        weight_floor=args.dual_metric_weight_floor,
        seed=args.dual_metric_weight_seed,
    )
    payload["training"].update({
        "validation_embedding_file": osp.abspath(validation_embedding_file),
        "eligible_queries": int(len(eligible)),
        "rerank_queries": int(len(rerank_samples)),
    })
    save_weight_payload(weight_file, payload)
    print(f"[adaptive] fitted learned dual-metric weights: {weight_file}")
    return payload, weight_file


def run(args):
    requested_methods = {
        item.strip() for item in args.methods.split(",") if item.strip()
    }
    if args.learn_dual_metric_weights and not requested_methods.intersection({
        "adaptive-dual-metric", "adaptive-dual-metric-no-rerank"
    }):
        raise ValueError(
            "--learn-dual-metric-weights requires an adaptive-dual-metric method."
        )
    if args.learn_dual_metric_weights:
        if args.dual_metric_weight_epochs <= 0:
            raise ValueError("--dual-metric-weight-epochs must be positive.")
        if args.dual_metric_weight_learning_rate <= 0.0:
            raise ValueError(
                "--dual-metric-weight-learning-rate must be positive."
            )
        if args.dual_metric_weight_max_queries < 0:
            raise ValueError(
                "--dual-metric-weight-max-queries cannot be negative."
            )
        if args.dual_metric_weight_max_pairs_per_query <= 0:
            raise ValueError(
                "--dual-metric-weight-max-pairs-per-query must be positive."
            )
        if not 0.0 <= args.dual_metric_weight_floor < 1.0 / 3.0:
            raise ValueError(
                "--dual-metric-weight-floor must be in [0, 1/3)."
            )
    if requested_methods.intersection({
        "adaptive-dual-metric", "adaptive-dual-metric-no-rerank"
    }):
        if args.rerank_metric == "embedding":
            raise ValueError(
                "adaptive-dual-metric requires a non-embedding --rerank-metric."
            )
        if args.hamming_candidates < 50:
            raise ValueError(
                "adaptive-dual-metric requires at least 50 Global Hamming candidates."
            )
        dual_budgets = (
            args.dual_metric_high_quality_tp_candidates,
            args.dual_metric_tp_candidates,
            args.dual_metric_hard_tp_candidates,
        )
        if min(dual_budgets) < args.top_k:
            raise ValueError(
                "All adaptive-dual-metric TP budgets must be >= --top-k."
            )
        if args.dual_metric_search_candidates < args.dual_metric_candidates:
            raise ValueError(
                "--dual-metric-search-candidates must be >= "
                "--dual-metric-candidates."
            )
    if args.adaptive_preserve_global_top50:
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
                "Global Top-50 protection requires the Hamming, metric prefilter, "
                "and adaptive rerank budgets to be at least 50."
            )
    os.makedirs(args.output_dir, exist_ok=True)
    if args.hierarchy_dir is None:
        args.hierarchy_dir = osp.join(
            "dataset", args.dataset, "hierarchy_louvain"
        )
    if not osp.isdir(args.hierarchy_dir):
        raise FileNotFoundError(
            f"Hierarchy directory not found: {args.hierarchy_dir}. "
            "Pass --hierarchy-dir to override it."
        )
    if args.point_distance_file is None:
        args.point_distance_file = osp.join(
            "ground_truth", args.dataset, "Point_dis_matrix.npy"
        )
    if args.edge_file is None:
        args.edge_file = osp.join("dataset", args.dataset, "edge_weight.csv")
    if args.neterp_anchor_node is None and args.dataset == "beijing":
        args.neterp_anchor_node = 72526
    if args.reuse_hash_assets_dir:
        assets_dir = args.reuse_hash_assets_dir
        validation_embedding_file = osp.join(
            assets_dir, "validation_embeddings.npz"
        )
        test_file = osp.join(assets_dir, "test_embeddings.npz")
        hash_model_file = osp.join(
            assets_dir,
            (
                f"hash_head_{args.bits}_metric.pt"
                if args.dual_hash_heads else f"hash_head_{args.bits}.pt"
            ),
        )
        default_candidate_model_file = osp.join(
            assets_dir,
            (
                f"hash_head_{args.bits}_candidate.pt"
                if args.dual_hash_heads else f"hash_head_{args.bits}.pt"
            ),
        )
        default_metric_model_file = (
            hash_model_file if args.dual_hash_heads else None
        )
        external_validation_embeddings = _is_embedding_file(args.validation_file)
        required_assets = []
        if not args.reexport_embeddings:
            required_assets.append(test_file)
            if args.retrain_hash:
                if not external_validation_embeddings:
                    required_assets.append(validation_embedding_file)
            else:
                required_assets.append(
                    args.candidate_hash_model_file or default_candidate_model_file
                )
                metric_file = (
                    args.metric_prefilter_hash_model_file or default_metric_model_file
                )
                if metric_file:
                    required_assets.append(metric_file)
        missing = [path for path in required_assets if not osp.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Reusable hash assets are incomplete; missing: {missing}"
            )
        print(f"[adaptive] reusing hash assets: {assets_dir}")
        if args.retrain_hash:
            print(
                "[adaptive] --retrain-hash enabled: forcing metric and candidate "
                "hash-head training"
            )
        if args.retrain_hash or args.reexport_embeddings:
            test_file, hash_model_file = _run_hash_export(
                args, assets_dir, retrain_only=args.retrain_hash
            )
    else:
        assets_dir = osp.join(args.output_dir, "hash_assets")
        test_file, hash_model_file = _run_hash_export(args, assets_dir)
    validation_embedding_file = (
        args.validation_file
        if _is_embedding_file(args.validation_file)
        else osp.join(assets_dir, "validation_embeddings.npz")
    )
    trained_hash_model_file = hash_model_file
    default_candidate_model_file = osp.join(
        assets_dir,
        (
            f"hash_head_{args.bits}_candidate.pt"
            if args.dual_hash_heads else f"hash_head_{args.bits}.pt"
        ),
    )
    candidate_hash_model_file = (
        args.candidate_hash_model_file or default_candidate_model_file
    )
    metric_prefilter_hash_model_file = (
        args.metric_prefilter_hash_model_file
        or (trained_hash_model_file if args.dual_hash_heads else None)
    )
    for role, path in (
        ("candidate", candidate_hash_model_file),
        ("metric prefilter", metric_prefilter_hash_model_file),
    ):
        if path and not osp.isfile(path):
            raise FileNotFoundError(f"{role} hash model not found: {path}")
    payload = load_embedding_payload(test_file, role="query")
    if args.max_queries > 0:
        limit = min(args.max_queries, len(payload["embeddings"]))
        for key in ("embeddings", "trajectories", "source_ids", "labels"):
            if key in payload and payload[key] is not None:
                payload[key] = payload[key][:limit]
        for key in ("scale_embeddings", "scale_weights"):
            if payload.get(key) is not None:
                payload[key] = payload[key][:limit]

    selected_rows = np.arange(len(payload["embeddings"]), dtype=np.int64)
    topo_args = SimpleNamespace(
        hash_model_file=candidate_hash_model_file,
        hash_device=args.hash_device,
        hierarchy_dir=args.hierarchy_dir,
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
        hierarchy_embedding_weight=0.0,
        topology_weight=-1.0,
        adaptive_initial_global=args.adaptive_initial_global,
        adaptive_initial_structural=args.adaptive_initial_structural,
        adaptive_max_global=args.adaptive_max_global,
        adaptive_max_structural=args.adaptive_max_structural,
        adaptive_max_rerank=args.adaptive_max_rerank,
        adaptive_overlap_threshold=args.adaptive_overlap_threshold,
        adaptive_min_route_recall=args.adaptive_min_route_recall,
        adaptive_gate=args.adaptive_gate,
        adaptive_global_margin=args.adaptive_global_margin,
        adaptive_metric_fast_path=args.adaptive_metric_fast_path,
        adaptive_max_tp_fraction=args.adaptive_max_tp_fraction,
        adaptive_preserve_global_top50=args.adaptive_preserve_global_top50,
        adaptive_rescue_quality_overlap=args.adaptive_rescue_quality_overlap,
        adaptive_rescue_near_margin_fraction=args.adaptive_rescue_near_margin_fraction,
        metric_prefilter_candidates=args.metric_prefilter_candidates,
        metric_prefilter_high_quality_candidates=args.metric_prefilter_high_quality_candidates,
        metric_prefilter_global_protect=args.metric_prefilter_global_protect,
        metric_prefilter_structural_protect=args.metric_prefilter_structural_protect,
        metric_prefilter_embedding_weight=args.metric_prefilter_embedding_weight,
        metric_prefilter_hash_weight=args.metric_prefilter_hash_weight,
        metric_prefilter_hash_score=args.metric_prefilter_hash_score,
        metric_prefilter_soft_temperature=args.metric_prefilter_soft_temperature,
        dual_metric_search_candidates=args.dual_metric_search_candidates,
        dual_metric_candidates=args.dual_metric_candidates,
        dual_metric_tp_candidates=args.dual_metric_tp_candidates,
        dual_metric_high_quality_tp_candidates=(
            args.dual_metric_high_quality_tp_candidates
        ),
        dual_metric_hard_tp_candidates=args.dual_metric_hard_tp_candidates,
        dual_metric_hard_margin_fraction=args.dual_metric_hard_margin_fraction,
        dual_metric_head_protect=args.dual_metric_head_protect,
        dual_metric_structural_protect=args.dual_metric_structural_protect,
        dual_metric_embedding_weight=args.dual_metric_embedding_weight,
        dual_metric_hash_weight=args.dual_metric_hash_weight,
        dual_metric_topology_weight=args.dual_metric_topology_weight,
        selection_seed=2026,
        probe_radius=args.probe_radius,
        hamming_candidates=args.hamming_candidates,
        hash_backend=args.hash_backend,
        hash_hnsw_m=args.hash_hnsw_m,
        hash_hnsw_ef_construction=args.hash_hnsw_ef_construction,
        hash_hnsw_ef_search=args.hash_hnsw_ef_search,
        metric=args.embedding_metric,
        normalize=args.normalize,
        embedding_storage=args.embedding_storage,
        encode_batch_size=args.encode_batch_size,
    )
    index_dir = osp.join(args.output_dir, "adaptive_index")
    index, build_report = _build_index(
        topo_args, payload, selected_rows, index_dir
    )
    index.exclude_self = bool(args.exclude_self)
    print(
        "[adaptive] exclude_self=%s, rerank_metric=%s, embedding_weight=%.3f"
        % (args.exclude_self, args.rerank_metric, args.embedding_weight)
    )
    if args.rerank_metric == "TP":
        try:
            import numba

            numba_status = f"enabled ({numba.get_num_threads()} threads)"
        except ImportError:
            numba_status = "unavailable; Python/NumPy fallback"
        print(f"[adaptive] TP backend: flattened exact_tp_distances, Numba {numba_status}")
    metric_backend = _metric_args(args, payload["trajectories"])
    from hash_global import encode_binary_codes, encode_soft_codes

    query_codes = encode_binary_codes(
        index.hash_index.head,
        payload["embeddings"],
        batch_size=args.encode_batch_size,
        device=args.query_code_device,
    )
    query_metric_codes = None
    query_metric_hash_codes = None
    if metric_prefilter_hash_model_file:
        query_metric_codes, query_metric_hash_codes = _attach_metric_hash(
            index,
            payload,
            metric_prefilter_hash_model_file,
            args,
            topo_args,
        )
        print(
            "[adaptive] dual hash heads: candidate=%s, metric-prefilter=%s"
            % (candidate_hash_model_file, metric_prefilter_hash_model_file)
        )
        print(
            "[adaptive] metric prefilter hash score: %s"
            % args.metric_prefilter_hash_score
        )
    elif (
        args.metric_prefilter_hash_score == "soft"
        and args.metric_prefilter_hash_weight > 0.0
    ):
        index.metric_prefilter_soft_codes = encode_soft_codes(
            index.hash_index.head,
            payload["embeddings"],
            batch_size=args.encode_batch_size,
            device=args.query_code_device,
            temperature=args.metric_prefilter_soft_temperature,
        )
        query_metric_codes = index.metric_prefilter_soft_codes
        print("[adaptive] soft metric prefilter uses the candidate head fallback")
    learned_weight_payload, learned_weight_file = (
        _fit_or_load_dual_metric_weights(
            args,
            topo_args,
            validation_embedding_file,
            metric_prefilter_hash_model_file,
        )
    )
    if learned_weight_payload is not None:
        index.set_dual_metric_weight_payload(learned_weight_payload)
    source_ids = payload["source_ids"]
    labels = payload.get("labels")
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods).difference(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if set(methods).intersection({
        "adaptive-dual-metric", "adaptive-dual-metric-no-rerank"
    }):
        if metric_backend is None:
            raise ValueError(
                "adaptive-dual-metric requires a non-embedding --rerank-metric."
            )
        if index.metric_hash_index is None or query_metric_hash_codes is None:
            raise ValueError(
                "adaptive-dual-metric requires --dual-hash-heads or an explicit "
                "--metric-prefilter-hash-model-file."
            )
    results = []
    predictions = {}
    for method in methods:
        result, ids = _evaluate_method(
            method,
            index,
            payload["embeddings"],
            payload["trajectories"],
            source_ids,
            payload["embeddings"] if method == "exact" else None,
            source_ids,
            np.ones(len(source_ids), dtype=bool),
            labels,
            query_codes,
            args.top_k,
            args.exact_max_queries,
            query_scale_embeddings=payload.get("scale_embeddings"),
            query_scale_weights=payload.get("scale_weights"),
            metric_backend=metric_backend,
            query_metric_codes=query_metric_codes,
            query_metric_hash_codes=query_metric_hash_codes,
        )
        result["rerank_metric"] = (
            args.rerank_metric
            if method in {"adaptive-topo-hash", "adaptive-dual-metric"}
            else None
        )
        results.append(result)
        predictions[method] = ids

    truth = _load_truth(args.truth_file)
    if truth is not None:
        for result in results:
            result["metric_truth"] = _row_retrieval_metrics(
                predictions[result["method"]], truth
            )
    low_confidence_comparison = _write_low_confidence_comparison(
        args,
        results,
        predictions,
        payload["trajectories"],
        source_ids,
        source_ids,
        labels,
    )
    if low_confidence_comparison is not None:
        print(
            "[results] low-confidence comparison: "
            f"eligible={low_confidence_comparison['eligible_queries']}, "
            f"reranked={low_confidence_comparison['reranked_queries']}, "
            f"artifact={low_confidence_comparison.get('artifact') or 'n/a'}"
        )
        low_confidence_rows = []
        low_confidence_delta_rows = []
        for subset_name, subset in low_confidence_comparison.get(
            "subset_report", {}
        ).items():
            for method_name, method_report in subset.get("methods", {}).items():
                if method_name == "delta_rerank_minus_global":
                    continue
                if method_name in {
                    "delta_no_rerank_minus_global",
                    "delta_rerank_minus_no_rerank",
                }:
                    continue
                metrics = method_report.get("metrics", {})
                low_confidence_rows.append({
                    "subset": subset_name,
                    "method": method_name,
                    "queries": method_report["queries"],
                    **metrics,
                })
            delta = subset.get("methods", {}).get("delta_rerank_minus_global")
            if delta:
                low_confidence_delta_rows.append({
                    "subset": subset_name,
                    **delta,
                })
        if low_confidence_rows:
            _print_table(
                "low-confidence retrieval metrics",
                low_confidence_rows,
                [
                    "subset", "method", "queries", "HR@10", "HR@50",
                    "R10@50", "R1@1", "R1@10", "R1@50",
                ],
            )
        if low_confidence_delta_rows:
            _print_table(
                "low-confidence delta (topology-tp-rerank - global-hash-direct)",
                low_confidence_delta_rows,
                [
                    "subset", "HR@10", "HR@50", "R10@50", "R1@1",
                    "R1@10", "R1@50",
                ],
            )
            low_confidence_no_rerank_delta_rows = []
            for subset_name, subset in low_confidence_comparison.get(
                "subset_report", {}
            ).items():
                delta = subset.get("methods", {}).get(
                    "delta_no_rerank_minus_global"
                )
                if delta:
                    low_confidence_no_rerank_delta_rows.append({
                        "subset": subset_name,
                        **delta,
                    })
            if low_confidence_no_rerank_delta_rows:
                _print_table(
                    "low-confidence delta (dual-hash-no-rerank - global-hash-direct)",
                    low_confidence_no_rerank_delta_rows,
                    [
                        "subset", "HR@10", "HR@50", "R10@50", "R1@1",
                        "R1@10", "R1@50",
                    ],
                )
    output = {
        "protocol": "adaptive_topo_hash_checkpoint_eval",
        "checkpoint": osp.abspath(args.checkpoint),
        "embedding_file": osp.abspath(test_file),
        "hash_model_file": osp.abspath(candidate_hash_model_file),
        "trained_hash_model_file": osp.abspath(trained_hash_model_file),
        "metric_prefilter_hash_model_file": (
            osp.abspath(metric_prefilter_hash_model_file)
            if metric_prefilter_hash_model_file else None
        ),
        "rerank_metric": args.rerank_metric,
        "dual_metric_learned_weights": {
            "enabled": bool(args.learn_dual_metric_weights),
            "scope": (
                "bucket" if args.dual_metric_weight_per_bucket else "global"
            ),
            "file": (
                osp.abspath(learned_weight_file)
                if learned_weight_file is not None else None
            ),
            "payload": learned_weight_payload,
        },
        "exclude_self": bool(args.exclude_self),
        "methods": results,
        "low_confidence_comparison": low_confidence_comparison,
        "build": build_report,
        "adaptive_config": {
            key: getattr(args, key)
            for key in (
                "adaptive_initial_global", "adaptive_initial_structural",
                "adaptive_max_global", "adaptive_max_structural",
                "adaptive_max_rerank", "adaptive_overlap_threshold",
                "adaptive_min_route_recall",
                "adaptive_gate", "adaptive_global_margin",
                "adaptive_metric_fast_path",
                "adaptive_max_tp_fraction",
                "adaptive_preserve_global_top50",
                "adaptive_rescue_quality_overlap",
                "adaptive_rescue_near_margin_fraction",
                "metric_prefilter_candidates", "metric_prefilter_global_protect",
                "metric_prefilter_structural_protect",
                "metric_prefilter_high_quality_candidates",
                "metric_prefilter_embedding_weight",
                "metric_prefilter_hash_weight",
                "metric_prefilter_hash_score",
                "metric_prefilter_soft_temperature",
                "dual_metric_search_candidates",
                "dual_metric_candidates",
                "dual_metric_tp_candidates",
                "dual_metric_high_quality_tp_candidates",
                "dual_metric_hard_tp_candidates",
                "dual_metric_hard_margin_fraction",
                "dual_metric_head_protect",
                "dual_metric_structural_protect",
                "dual_metric_embedding_weight",
                "dual_metric_hash_weight",
                "dual_metric_topology_weight",
                "learn_dual_metric_weights",
                "dual_metric_weight_file",
                "refit_dual_metric_weights",
                "dual_metric_weight_per_bucket",
                "dual_metric_weight_epochs",
                "dual_metric_weight_learning_rate",
                "dual_metric_weight_max_queries",
                "dual_metric_weight_max_pairs_per_query",
                "dual_metric_weight_floor",
                "dual_metric_weight_seed",
                "tp_num_threads",
            )
        },
        "ranking_config": {
            "embedding_metric": args.embedding_metric,
            "embedding_weight": float(args.embedding_weight),
            "topology_weight": float(1.0 - args.embedding_weight),
            "candidate_source": args.candidate_source,
        },
    }
    result_file = osp.join(args.output_dir, "adaptive_result.json")
    _json_dump(result_file, output)
    print(f"[done] result: {result_file}")
    _print_result_summary(results)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="beijing", choices=["beijing", "porto", "tdrive"])
    parser.add_argument("--dataset-mode", default="standard", choices=["standard", "50w", "full"])
    parser.add_argument("--distance-type", default="TP", choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"])
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--hash-metric-distance-file", default=None)
    parser.add_argument("--hash-metric-candidate-teacher-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reuse-hash-assets-dir",
        default=None,
        help=(
            "Reuse an existing directory containing test_embeddings.npz and "
            "hash_head_<bits>.pt instead of exporting/training again."
        ),
    )
    parser.add_argument(
        "--retrain-hash",
        action="store_true",
        help=(
            "Force retraining of the Candidate/Metric hash heads from the "
            "existing validation embeddings; the trajectory encoder and embeddings "
            "are reused. Combine with --reexport-embeddings if the checkpoint "
            "or encoder settings changed."
        ),
    )
    parser.add_argument(
        "--reexport-embeddings",
        action="store_true",
        help=(
            "Re-export validation/test embeddings into the reused asset directory. "
            "Use with --retrain-hash when the checkpoint or encoder settings changed."
        ),
    )
    parser.add_argument(
        "--dual-hash-heads",
        action="store_true",
        help=(
            "Train/load hash_head_<bits>_candidate.pt for global hash retrieval and "
            "hash_head_<bits>_metric.pt for learned metric prefiltering."
        ),
    )
    parser.add_argument(
        "--candidate-hash-model-file",
        default=None,
        help=(
            "Optional baseline hash head used for global hash candidate generation. "
            "Defaults to hash_head_<bits>.pt in the asset directory."
        ),
    )
    parser.add_argument(
        "--metric-prefilter-hash-model-file",
        default=None,
        help=(
            "Optional TP-distilled hash head used only for metric-prefilter "
            "Hamming scores. Enables the dual-head path."
        ),
    )
    parser.add_argument(
        "--hierarchy-dir",
        default=None,
        help="Hierarchy directory; defaults to dataset/<dataset>/hierarchy_louvain.",
    )
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--hash-device", default="cpu")
    parser.add_argument("--query-code-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--hash-epochs", type=int, default=30)
    parser.add_argument("--hash-batch-size", type=int, default=512)
    parser.add_argument("--hash-positive-pool", type=int, default=10)
    parser.add_argument("--hash-positives-per-anchor", type=int, default=1)
    parser.add_argument("--hash-negatives-per-anchor", type=int, default=1)
    parser.add_argument(
        "--hash-negative-strategy",
        choices=["random", "metric-hard", "mixed"],
        default="random",
    )
    parser.add_argument("--hash-hard-negative-pool", type=int, default=0)
    parser.add_argument("--hash-hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--hash-metric-distance-weight", type=float, default=0.0)
    parser.add_argument("--hash-metric-distance-scale", type=float, default=0.0)
    parser.add_argument("--hash-embedding-distance-weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate-hash-embedding-distance-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--candidate-hash-in-batch-negative-weight", type=float, default=0.25
    )
    parser.add_argument("--hash-in-batch-negative-weight", type=float, default=0.0)
    parser.add_argument("--hash-graded-rank-weight", type=float, default=0.0)
    parser.add_argument("--hash-graded-rank-margin", type=float, default=0.05)
    parser.add_argument("--hash-graded-rank-cutoffs", default="10,20,50")
    parser.add_argument("--hash-cutoff-rank-weight", type=float, default=0.0)
    parser.add_argument("--hash-cutoff-rank", type=int, default=192)
    parser.add_argument("--hash-cutoff-rank-margin", type=float, default=0.02)
    parser.add_argument("--hash-cutoff-rank-window", type=int, default=64)
    parser.add_argument("--hash-cutoff-samples", type=int, default=4)
    parser.add_argument("--hash-filter-same-source-negatives", action="store_true")
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--seq-encoder", default="hmt")
    parser.add_argument("--graph-encoder", default="hybrid_multilevel")
    parser.add_argument("--coord-encoder", default="transformer")
    parser.add_argument("--hmt-backbone", default="hybrid")
    parser.add_argument("--hmt-scale-aligned-graph", action="store_true", default=True)
    parser.add_argument("--coord-fusion", default="gated_attention")
    parser.add_argument("--embedding-metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--rerank-metric", choices=METRIC_NAMES, default="embedding")
    parser.add_argument("--point-distance-file", default=None)
    parser.add_argument("--edge-file", default=None)
    parser.add_argument("--node-count", type=int, default=0)
    parser.add_argument("--max-trajectory-length", type=int, default=0)
    parser.add_argument("--neterp-anchor-node", type=int, default=None)
    parser.add_argument(
        "--tp-num-threads",
        type=int,
        default=64,
        help="Numba CPU threads for online metric reranking; 0 uses the runtime default.",
    )
    parser.add_argument("--truth-file", default=None)
    parser.add_argument(
        "--methods",
        default="adaptive-topo-hash,global-hash,exact",
        help=(
            "Comma-separated methods. Use adaptive-dual-metric for the independent "
            "Metric-Head rescue path."
        ),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--exact-max-queries", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=0)
    self_group = parser.add_mutually_exclusive_group()
    self_group.add_argument(
        "--exclude-self", dest="exclude_self", action="store_true"
    )
    self_group.add_argument(
        "--include-self", dest="exclude_self", action="store_false"
    )
    parser.set_defaults(exclude_self=True)
    parser.add_argument("--candidate-source", choices=["hybrid", "global-only"], default="hybrid")
    parser.add_argument("--probe-radius", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--hamming-candidates", type=int, default=2048)
    parser.add_argument(
        "--hash-backend", choices=["mih", "hnsw"], default="mih",
        help="Use full-code Hamming HNSW instead of MIH hash-table probing.",
    )
    parser.add_argument("--hash-hnsw-m", type=int, default=16)
    parser.add_argument("--hash-hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hash-hnsw-ef-search", type=int, default=256)
    parser.add_argument("--embedding-storage", choices=["float16", "float32"], default="float16")
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--qgram", type=int, default=2)
    parser.add_argument("--minhash-size", type=int, default=8)
    parser.add_argument("--postings-per-level", type=int, default=4)
    parser.add_argument("--l3-expand-hops", type=int, default=1)
    parser.add_argument("--min-route-candidates", type=int, default=1024)
    parser.add_argument("--max-route-candidates", type=int, default=4096)
    parser.add_argument("--structural-candidates", type=int, default=512)
    parser.add_argument("--global-candidates", type=int, default=1024)
    parser.add_argument("--max-rerank-candidates", type=int, default=1536)
    parser.add_argument("--l3-weight", type=float, default=0.45)
    parser.add_argument("--l2-weight", type=float, default=0.35)
    parser.add_argument("--l1-weight", type=float, default=0.20)
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=1.0,
        help=(
            "Embedding rank weight for embedding reranking. The adaptive runner "
            "defaults to 1.0 so topology only rescues candidates."
        ),
    )
    parser.add_argument("--adaptive-initial-global", type=int, default=512)
    parser.add_argument("--adaptive-initial-structural", type=int, default=256)
    parser.add_argument("--adaptive-max-global", type=int, default=2048)
    parser.add_argument("--adaptive-max-structural", type=int, default=1024)
    parser.add_argument("--adaptive-max-rerank", type=int, default=2048)
    parser.add_argument("--adaptive-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--adaptive-min-route-recall", type=int, default=128)
    parser.add_argument(
        "--adaptive-gate",
        choices=["overlap", "global", "hybrid"],
        default="overlap",
        help=(
            "Adaptive gate. 'overlap' preserves the original topology gate; "
            "'global'/'hybrid' can use a Global-Hamming margin fast path."
        ),
    )
    parser.add_argument(
        "--adaptive-global-margin",
        type=float,
        default=0.05,
        help="Normalized Hamming margin required by global/hybrid fast gates.",
    )
    parser.add_argument(
        "--adaptive-metric-fast-path",
        choices=["skip", "metric"],
        default="skip",
        help=(
            "High-confidence Global path: skip expensive TP/DITA/LCRS "
            "metric (default) or run it for an accuracy comparison."
        ),
    )
    parser.add_argument(
        "--adaptive-max-tp-fraction",
        type=float,
        default=0.15,
        help=(
            "Maximum fraction of queries allowed into the expensive trajectory "
            "metric branch; "
            "lowest Global-margin queries are selected first."
        ),
    )
    parser.add_argument(
        "--adaptive-preserve-global-top50",
        action="store_true",
        help=(
            "Protect the Global Hash embedding Top-50 in adaptive metric rerank "
            "pools. Requires a non-embedding rerank metric and candidate budgets "
            "of at least 50."
        ),
    )
    parser.add_argument(
        "--adaptive-rescue-quality-overlap",
        type=float,
        default=0.75,
        help=(
            "Global/structural overlap required for the smaller high-quality "
            "low-confidence rescue budget."
        ),
    )
    parser.add_argument(
        "--adaptive-rescue-near-margin-fraction",
        type=float,
        default=0.80,
        help=(
            "Reserved for confidence-aware rescue scheduling; values near the "
            "margin threshold remain on the full metric budget."
        ),
    )
    parser.add_argument(
        "--metric-prefilter-candidates", type=int, default=64,
        help="Maximum candidates sent to TP/DITA/LCRS/NetERP after embedding prefilter.",
    )
    parser.add_argument(
        "--metric-prefilter-high-quality-candidates", type=int, default=50,
        help=(
            "Metric candidate budget for low-confidence queries whose Global "
            "and structural rescue pools strongly agree."
        ),
    )
    parser.add_argument(
        "--metric-prefilter-global-protect", type=int, default=32,
        help="Global candidates protected from metric prefilter truncation.",
    )
    parser.add_argument(
        "--metric-prefilter-structural-protect", type=int, default=16,
        help="Topology candidates protected from metric prefilter truncation.",
    )
    parser.add_argument(
        "--metric-prefilter-embedding-weight", type=float, default=1.0,
        help="Weight for normalized embedding distance in metric prefilter.",
    )
    parser.add_argument(
        "--metric-prefilter-hash-weight", type=float, default=0.0,
        help="Weight for learned hash Hamming rank in metric prefilter.",
    )
    parser.add_argument(
        "--metric-prefilter-hash-score", choices=["hamming", "soft"], default="hamming",
        help="Use packed Hamming or continuous tanh-code distance for hash prefiltering.",
    )
    parser.add_argument(
        "--metric-prefilter-soft-temperature", type=float, default=1.0,
        help="Temperature used when encoding continuous metric-head hash codes.",
    )
    parser.add_argument(
        "--dual-metric-search-candidates", type=int, default=256,
        help="Metric-Head MIH candidates retrieved before dual-route merge.",
    )
    parser.add_argument(
        "--dual-metric-candidates", type=int, default=96,
        help="Metric-Head candidates contributed to the dual-route union.",
    )
    parser.add_argument(
        "--dual-metric-tp-candidates", type=int, default=128,
        help="TP pool for ordinary low-confidence dual-metric queries.",
    )
    parser.add_argument(
        "--dual-metric-high-quality-tp-candidates", type=int, default=64,
        help="TP pool when Global/Topology agreement is high.",
    )
    parser.add_argument(
        "--dual-metric-hard-tp-candidates", type=int, default=160,
        help="TP pool for the hardest low-confidence queries.",
    )
    parser.add_argument(
        "--dual-metric-hard-margin-fraction", type=float, default=0.50,
        help="Use the hard TP budget below this fraction of the margin threshold.",
    )
    parser.add_argument(
        "--dual-metric-head-protect", type=int, default=48,
        help="Metric-Head candidates protected in the merged TP pool.",
    )
    parser.add_argument(
        "--dual-metric-structural-protect", type=int, default=16,
        help="Topology candidates protected in the merged TP pool.",
    )
    parser.add_argument(
        "--dual-metric-embedding-weight", type=float, default=0.40,
        help="Embedding rank weight in dual-route candidate fusion.",
    )
    parser.add_argument(
        "--dual-metric-hash-weight", type=float, default=0.40,
        help="Metric-Head hash rank weight in dual-route candidate fusion.",
    )
    parser.add_argument(
        "--dual-metric-topology-weight", type=float, default=0.20,
        help="Topology rank weight in dual-route candidate fusion.",
    )
    parser.add_argument(
        "--learn-dual-metric-weights",
        action="store_true",
        help=(
            "Learn metric-specific dual-route fusion weights from validation "
            "embeddings, or load them from --dual-metric-weight-file."
        ),
    )
    parser.add_argument(
        "--dual-metric-weight-file",
        default=None,
        help=(
            "Learned weight JSON. When omitted, a metric/scope-specific file is "
            "created in --output-dir and reused on later runs."
        ),
    )
    parser.add_argument(
        "--refit-dual-metric-weights",
        action="store_true",
        help="Refit learned weights even when the learned weight JSON exists.",
    )
    weight_scope = parser.add_mutually_exclusive_group()
    weight_scope.add_argument(
        "--dual-metric-weight-per-bucket",
        dest="dual_metric_weight_per_bucket",
        action="store_true",
        help="Learn separate high-quality, ordinary, and hard-query weights.",
    )
    weight_scope.add_argument(
        "--dual-metric-weight-global",
        dest="dual_metric_weight_per_bucket",
        action="store_false",
        help="Learn one weight vector shared by all low-confidence queries.",
    )
    parser.set_defaults(dual_metric_weight_per_bucket=True)
    parser.add_argument("--dual-metric-weight-epochs", type=int, default=200)
    parser.add_argument(
        "--dual-metric-weight-learning-rate", type=float, default=0.05
    )
    parser.add_argument(
        "--dual-metric-weight-max-queries", type=int, default=512,
        help="Maximum validation queries used for fusion-weight fitting; 0 uses all.",
    )
    parser.add_argument(
        "--dual-metric-weight-max-pairs-per-query", type=int, default=512
    )
    parser.add_argument(
        "--dual-metric-weight-floor",
        type=float,
        default=0.005,
        help="Minimum learned value for each of the three fusion weights.",
    )
    parser.add_argument("--dual-metric-weight-seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
