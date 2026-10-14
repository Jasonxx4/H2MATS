"""Run the complete augmented Beijing TP embedding and ANN experiment.

This orchestrator intentionally keeps TP ground truth out of model training:
the TP table is passed only to the final candidate-selection/HNSW evaluation commands.
Gallery and Query embeddings are exported in the exact order used to construct
that table, and Query exports reuse the Gallery coordinate normalization stats.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DEFAULTS = {
    "beijing": {
        "hierarchy_dir": "dataset/beijing/hierarchy_louvain",
        "point_distance_file": "ground_truth/beijing/Point_dis_matrix.npy",
    },
    "porto": {
        "hierarchy_dir": "dataset/porto/hierarchy_louvain",
        "point_distance_file": "ground_truth/porto/Point_dis_matrix.npy",
    },
    "tdrive": {
        "hierarchy_dir": "dataset/tdrive/hierarchy_louvain",
        "point_distance_file": "ground_truth/tdrive/Point_dis_matrix.npy",
    },
}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _apply_dataset_defaults(args):
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.hierarchy_dir is None:
        args.hierarchy_dir = defaults["hierarchy_dir"]
    if args.validation_file is None:
        args.validation_file = (
            f"ground_truth/{args.dataset}/{args.distance_type}/spatial_vali_set.npz"
        )
    if args.tp_point_distance_file is None:
        args.tp_point_distance_file = defaults["point_distance_file"]
    return args


def _run(command, *, cwd=PROJECT_ROOT, dry_run=False):
    printable = " ".join(str(item) for item in command)
    print(f"\n$ {printable}")
    if dry_run:
        return
    subprocess.run([str(item) for item in command], cwd=str(cwd), check=True)


def _python():
    return sys.executable


def _validate_checkpoint(path, label):
    resolved = _resolve(path)
    if resolved.is_file():
        return resolved
    if resolved.is_dir() and any(resolved.glob("epoch_*.pt")):
        return resolved
    candidates = []
    parent = resolved.parent
    if parent.exists():
        candidates = sorted(str(item) for item in parent.glob("*.pt"))[:8]
    hint = f" Nearby .pt files: {candidates}" if candidates else ""
    raise FileNotFoundError(
        f"{label} checkpoint does not exist or contains no epoch_*.pt files: "
        f"{resolved}.{hint} Replace the example path with a real checkpoint."
    )


def _add_model_options(command, args, role="ours"):
    """Append the model-construction flags for one checkpoint.

    The two checkpoints in this experiment may come from different model
    families.  In particular, the legacy SPGMT checkpoint must not inherit
    the HMT/HybridGeo flags used by the current model.
    """
    prefix = "" if role == "ours" else f"{role}_"
    use_hybrid_geo = getattr(args, f"{prefix}use_hybrid_geo")
    use_si = getattr(args, f"{prefix}use_si")
    seq_encoder = getattr(args, f"{prefix}seq_encoder")
    coord_encoder = getattr(args, f"{prefix}coord_encoder")
    hmt_backbone = getattr(args, f"{prefix}hmt_backbone")
    graph_encoder = getattr(args, f"{prefix}graph_encoder")
    coord_fusion = getattr(args, f"{prefix}coord_fusion")

    if use_hybrid_geo is True:
        command.append("--use-hybrid-geo")
    elif use_hybrid_geo is False:
        command.append("--no-use-hybrid-geo")
    if use_si is True:
        command.append("--use-si")
    elif use_si is False:
        command.append("--no-use-si")
    if seq_encoder:
        command.extend(["--seq-encoder", seq_encoder])
    if coord_encoder:
        command.extend(["--coord-encoder", coord_encoder])
    if hmt_backbone:
        command.extend(["--hmt-backbone", hmt_backbone])
    if graph_encoder:
        command.extend(["--graph-encoder", graph_encoder])
    if coord_fusion:
        command.extend(["--coord-fusion", coord_fusion])
    return command


def _export_command(args, model, trajectories, coordinates, output, stats_file=None, role="ours"):
    model = _validate_checkpoint(model, f"{role} trajectory encoder")
    command = [
        _python(),
        "ann_pipeline.py",
        "build",
        "--dataset",
        args.dataset,
        "--distance-type",
        args.distance_type,
        "--model",
        str(model),
        "--trajectory-array",
        str(trajectories),
        "--coordinate-array",
        str(coordinates),
        "--embedding-file",
        str(output),
        "--batch-size",
        str(args.embedding_batch_size),
        "--cuda",
        str(args.cuda),
        "--compact-output",
        "--export-only",
    ]
    if stats_file is not None:
        command.extend(["--coordinate-stats-file", str(stats_file)])
    return _add_model_options(command, args, role=role)


def _export_validation_command(args, model, output):
    model = _validate_checkpoint(model, "Our trajectory encoder")
    command = [
        _python(),
        "ann_pipeline.py",
        "build",
        "--dataset",
        args.dataset,
        "--distance-type",
        args.distance_type,
        "--model",
        str(model),
        "--eval-file",
        str(_resolve(args.validation_file)),
        "--embedding-file",
        str(output),
        "--batch-size",
        str(args.embedding_batch_size),
        "--cuda",
        str(args.cuda),
        "--export-only",
    ]
    return _add_model_options(command, args, role="ours")


def _attach_metadata(args, embedding_file, metadata_file, output_file):
    return [
        _python(),
        "build_augmented_tp_benchmark.py",
        "attach-metadata",
        "--embedding-file",
        str(embedding_file),
        "--metadata-file",
        str(metadata_file),
        "--output-file",
        str(output_file),
        "--overwrite",
    ]


def _topology_command(args, gallery, query, code_model, output_dir):
    _apply_dataset_defaults(args)
    table = _resolve(args.tp_distance_table_file)
    methods = [args.topology_method]
    if args.topology_method == "adaptive-dual-metric":
        # Keep the fused-code candidate list as an ablation next to the
        # trajectory-metric reranked result.
        methods.append("adaptive-dual-metric-no-rerank")
    if args.include_code_scan:
        methods.append("code-scan")
    command = [
        _python(),
        "candidate_selection.py",
        "--gallery-file",
        str(gallery),
        "--query-file",
        str(query),
        "--code-model-file",
        str(code_model),
        "--hierarchy-dir",
        str(_resolve(args.hierarchy_dir)),
        "--output-dir",
        str(output_dir),
        "--gallery-sizes",
        args.gallery_sizes,
        "--gallery-selection",
        "prefix",
        "--positive-match-mode",
        "initial-only",
        "--positives-per-query",
        "1",
        "--methods",
        ",".join(methods),
        "--max-queries",
        str(args.query_count),
        "--top-k",
        str(args.top_k),
        "--candidate-source",
        args.candidate_source,
        "--probe-radius",
        str(args.probe_radius),
        "--hamming-candidates",
        str(args.hamming_candidates),
        "--code-backend",
        args.code_backend,
        "--code-flat-block-size",
        str(args.code_flat_block_size),
        "--structural-candidates",
        str(args.structural_candidates),
        "--global-candidates",
        str(args.global_candidates),
        "--max-rerank-candidates",
        str(args.max_rerank_candidates),
        "--min-route-candidates",
        str(args.min_route_candidates),
        "--max-route-candidates",
        str(args.max_route_candidates),
        "--rerank-metric",
        args.rerank_metric,
        "--tp-eval",
        "--tp-point-distance-file",
        str(_resolve(args.tp_point_distance_file)),
        "--tp-max-queries",
        str(args.query_count),
        "--tp-top-k",
        str(args.tp_top_k),
        "--tp-num-threads",
        str(args.tp_num_threads),
        "--tp-reuse-max-distance-table",
        "--tp-distance-table-file",
        str(table),
    ]
    command.extend([
        "--node-count", str(args.node_count),
        "--max-trajectory-length", str(args.max_trajectory_length),
    ])
    if args.edge_file:
        command.extend(["--edge-file", str(_resolve(args.edge_file))])
    if args.neterp_anchor_node is not None:
        command.extend(["--neterp-anchor-node", str(args.neterp_anchor_node)])
    adaptive_options = {
        "--adaptive-initial-global": args.adaptive_initial_global,
        "--adaptive-initial-structural": args.adaptive_initial_structural,
        "--adaptive-max-global": args.adaptive_max_global,
        "--adaptive-max-structural": args.adaptive_max_structural,
        "--adaptive-max-rerank": args.adaptive_max_rerank,
        "--adaptive-overlap-threshold": args.adaptive_overlap_threshold,
        "--adaptive-min-route-recall": args.adaptive_min_route_recall,
        "--adaptive-gate": args.adaptive_gate,
        "--adaptive-global-margin": args.adaptive_global_margin,
        "--adaptive-metric-fast-path": args.adaptive_metric_fast_path,
        "--adaptive-max-tp-fraction": args.adaptive_max_tp_fraction,
        "--adaptive-rescue-quality-overlap": args.adaptive_rescue_quality_overlap,
        "--adaptive-rescue-near-margin-fraction": args.adaptive_rescue_near_margin_fraction,
        "--metric-prefilter-candidates": args.metric_prefilter_candidates,
        "--metric-prefilter-high-quality-candidates": args.metric_prefilter_high_quality_candidates,
        "--metric-prefilter-global-protect": args.metric_prefilter_global_protect,
        "--metric-prefilter-structural-protect": args.metric_prefilter_structural_protect,
        "--metric-prefilter-embedding-weight": args.metric_prefilter_embedding_weight,
        "--metric-prefilter-code-weight": args.metric_prefilter_code_weight,
        "--metric-prefilter-code-score": args.metric_prefilter_code_score,
        "--metric-prefilter-soft-temperature": args.metric_prefilter_soft_temperature,
    }
    for flag, value in adaptive_options.items():
        command.extend([flag, str(value)])
    if args.adaptive_preserve_global_top50:
        command.append("--adaptive-preserve-global-top50")
    if args.topology_method == "adaptive-selection":
        command.extend(["--embedding-weight", str(args.topology_embedding_weight)])
    if args.metric_code_model_file:
        command.extend([
            "--metric-code-model-file",
            str(_resolve(args.metric_code_model_file)),
        ])
    if args.dual_metric_weight_file:
        command.extend([
            "--dual-metric-weight-file",
            str(_resolve(args.dual_metric_weight_file)),
        ])
    if args.topology_method == "adaptive-dual-metric":
        dual_metric_options = {
            "--dual-metric-search-candidates": args.dual_metric_search_candidates,
            "--dual-metric-candidates": args.dual_metric_candidates,
            "--dual-metric-high-quality-tp-candidates": (
                args.dual_metric_high_quality_tp_candidates
            ),
            "--dual-metric-tp-candidates": args.dual_metric_tp_candidates,
            "--dual-metric-hard-tp-candidates": args.dual_metric_hard_tp_candidates,
            "--dual-metric-hard-margin-fraction": args.dual_metric_hard_margin_fraction,
            "--dual-metric-head-protect": args.dual_metric_head_protect,
            "--dual-metric-structural-protect": args.dual_metric_structural_protect,
            "--dual-metric-embedding-weight": args.dual_metric_embedding_weight,
            "--dual-metric-code-weight": args.dual_metric_code_weight,
            "--dual-metric-topology-weight": args.dual_metric_topology_weight,
        }
        for flag, value in dual_metric_options.items():
            command.extend([flag, str(value)])
    command.extend([
        "--code-device",
        str(args.code_device),
        "--query-code-device",
        str(args.topology_query_code_device),
    ])
    return command


def _hnsw_command(args, gallery, query, output_dir):
    _apply_dataset_defaults(args)
    table = _resolve(args.tp_distance_table_file)
    return [
        _python(),
        "hnsw_large_scale.py",
        "--gallery-file",
        str(gallery),
        "--query-file",
        str(query),
        "--output-dir",
        str(output_dir),
        "--gallery-sizes",
        args.gallery_sizes,
        "--gallery-selection",
        "prefix",
        "--positive-match-mode",
        "initial-only",
        "--positives-per-query",
        "1",
        "--methods",
        "hnsw",
        "--metric",
        args.metric,
        "--max-queries",
        str(args.query_count),
        "--top-k",
        str(args.top_k),
        "--hnsw-m",
        str(args.hnsw_m),
        "--hnsw-ef-construction",
        str(args.hnsw_ef_construction),
        "--hnsw-ef-search",
        str(args.hnsw_ef_search),
        "--hnsw-num-threads",
        str(args.hnsw_num_threads),
        "--tp-eval",
        "--tp-max-queries",
        str(args.query_count),
        "--tp-top-k",
        str(args.tp_top_k),
        "--tp-point-distance-file",
        str(_resolve(args.tp_point_distance_file)),
        "--tp-reuse-max-distance-table",
        "--tp-distance-table-file",
        str(table),
    ]


def _write_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run(args):
    _apply_dataset_defaults(args)
    if args.adaptive_preserve_global_top50:
        if args.topology_method not in {
            "adaptive-selection", "adaptive-dual-metric"
        }:
            raise ValueError(
                "--adaptive-preserve-global-top50 requires "
                "an adaptive candidate selection method."
            )
        if args.rerank_metric == "embedding":
            raise ValueError(
                "--adaptive-preserve-global-top50 requires a trajectory "
                "--rerank-metric."
            )
        protected_budgets = [args.hamming_candidates, args.adaptive_max_rerank]
        if args.topology_method == "adaptive-dual-metric":
            protected_budgets.extend([
                args.dual_metric_high_quality_tp_candidates,
                args.dual_metric_tp_candidates,
                args.dual_metric_hard_tp_candidates,
            ])
        else:
            protected_budgets.extend([
                args.metric_prefilter_candidates,
                args.metric_prefilter_high_quality_candidates,
            ])
        if min(protected_budgets) < 50:
            raise ValueError(
                "Global Top-50 protection requires the Global, metric-rerank, "
                "and adaptive budgets to be at least 50."
            )
    metric_code_model_path = None
    if args.topology_method == "adaptive-dual-metric":
        if args.rerank_metric == "embedding":
            raise ValueError(
                "--topology-method adaptive-dual-metric requires a trajectory "
                "--rerank-metric."
            )
        if not args.metric_code_model_file:
            raise ValueError(
                "--topology-method adaptive-dual-metric requires "
                "--metric-code-model-file."
            )
        metric_code_model_path = _resolve(args.metric_code_model_file)
        if not metric_code_model_path.is_file():
            raise FileNotFoundError(
                f"Metric code encoder does not exist: {metric_code_model_path}"
            )
    model_path = _validate_checkpoint(args.model, "Our model")
    spgmt_model_path = (
        _validate_checkpoint(args.spgmt_model, "SPGMT model")
        if args.spgmt_model
        else None
    )
    benchmark_dir = _resolve(args.benchmark_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_nodes = benchmark_dir / "gallery_trajectories.npy"
    gallery_coordinates = benchmark_dir / "gallery_coordinates.npy"
    query_nodes = benchmark_dir / "query_trajectories.npy"
    query_coordinates = benchmark_dir / "query_coordinates.npy"
    gallery_metadata = benchmark_dir / "gallery_metadata.npz"
    query_metadata = benchmark_dir / "query_metadata.npz"
    for required in (
        gallery_nodes,
        gallery_coordinates,
        query_nodes,
        query_coordinates,
        gallery_metadata,
        query_metadata,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing benchmark input: {required}")

    our_gallery_raw = output_dir / "our_gallery_embeddings_raw.npz"
    our_query_raw = output_dir / "our_query_embeddings_raw.npz"
    our_gallery = output_dir / "our_gallery_embeddings.npz"
    our_query = output_dir / "our_query_embeddings.npz"
    spgmt_gallery_raw = output_dir / "spgmt_gallery_embeddings_raw.npz"
    spgmt_query_raw = output_dir / "spgmt_query_embeddings_raw.npz"
    spgmt_gallery = output_dir / "spgmt_gallery_embeddings.npz"
    spgmt_query = output_dir / "spgmt_query_embeddings.npz"
    coordinate_stats = output_dir / "coordinate_stats.npy"
    validation_embeddings = output_dir / "validation_embeddings.npz"
    code_model = _resolve(args.code_model_file) if args.code_model_file else output_dir / f"code_encoder_{args.code_bits}.pt"

    commands = []
    if not args.skip_export:
        commands.extend(
            [
                _export_command(
                    args,
                    model_path,
                    gallery_nodes,
                    gallery_coordinates,
                    our_gallery_raw,
                    role="ours",
                ),
                [
                    _python(),
                    "-c",
                    (
                        "import numpy as np; "
                        f"d=np.load(r'{our_gallery_raw}'); "
                        f"np.save(r'{coordinate_stats}', d['coordinate_stats'])"
                    ),
                ],
                _export_command(
                    args,
                    model_path,
                    query_nodes,
                    query_coordinates,
                    our_query_raw,
                    stats_file=coordinate_stats,
                    role="ours",
                ),
                _attach_metadata(args, our_gallery_raw, gallery_metadata, our_gallery),
                _attach_metadata(args, our_query_raw, query_metadata, our_query),
            ]
        )
        if args.spgmt_model:
            commands.extend(
                [
                    _export_command(
                        args,
                        spgmt_model_path,
                        gallery_nodes,
                        gallery_coordinates,
                        spgmt_gallery_raw,
                        stats_file=coordinate_stats,
                        role="spgmt",
                    ),
                    _export_command(
                        args,
                        spgmt_model_path,
                        query_nodes,
                        query_coordinates,
                        spgmt_query_raw,
                        stats_file=coordinate_stats,
                        role="spgmt",
                    ),
                    _attach_metadata(
                        args, spgmt_gallery_raw, gallery_metadata, spgmt_gallery
                    ),
                    _attach_metadata(args, spgmt_query_raw, query_metadata, spgmt_query),
                ]
            )

    if not args.skip_code_training and not code_model.exists():
        commands.append(_export_validation_command(args, model_path, validation_embeddings))
        commands.append(
            [
                _python(),
                "binary_code_pipeline.py",
                "train",
                "--embedding-file",
                str(validation_embeddings),
                "--model-file",
                str(code_model),
                "--bits",
                str(args.code_bits),
                "--epochs",
                str(args.code_epochs),
                "--batch-size",
                str(args.code_batch_size),
                "--device",
                args.code_device,
            ]
        )

    if not args.skip_topology:
        topology_dir_names = {
            "adaptive-selection": "result_adaptive_candidate_selection",
            "adaptive-dual-metric": "result_adaptive_dual_candidate_selection",
            "hierarchical-route": "result_hierarchical_route",
        }
        topology_output_dir = output_dir / topology_dir_names[args.topology_method]
        commands.append(
            _topology_command(
                args,
                our_gallery,
                our_query,
                code_model,
                topology_output_dir,
            )
        )
    if not args.skip_hnsw:
        hnsw_gallery = spgmt_gallery if args.spgmt_model else our_gallery
        hnsw_query = spgmt_query if args.spgmt_model else our_query
        if not args.spgmt_model:
            print("[warning] --spgmt-model is absent; HNSW will use our model embeddings.")
        commands.append(
            _hnsw_command(
                args,
                hnsw_gallery,
                hnsw_query,
                output_dir / "result_spgmt_hnsw",
            )
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(benchmark_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model": str(model_path.resolve()),
        "spgmt_model": str(spgmt_model_path.resolve()) if spgmt_model_path else None,
        "code_model": str(code_model.resolve()),
        "metric_code_model": (
            str(metric_code_model_path.resolve())
            if metric_code_model_path is not None
            else None
        ),
        "tp_distance_table": str(_resolve(args.tp_distance_table_file).resolve()),
        "gallery_sizes": args.gallery_sizes,
        "query_count": int(args.query_count),
        "top_k": int(args.top_k),
        "tp_top_k": int(args.tp_top_k),
        "rerank_metric": args.rerank_metric,
        "tp_point_distance_file": str(_resolve(args.tp_point_distance_file)),
        "tp_num_threads": int(args.tp_num_threads),
        "topology_method": args.topology_method,
        "topology_methods": (
            ["adaptive-dual-metric", "adaptive-dual-metric-no-rerank"]
            if args.topology_method == "adaptive-dual-metric"
            else [args.topology_method]
        ) + (["code-scan"] if args.include_code_scan else []),
        "include_code_scan": bool(args.include_code_scan),
        "adaptive_preserve_global_top50": bool(
            args.adaptive_preserve_global_top50
        ),
        "topology_embedding_weight": float(args.topology_embedding_weight),
        "dual_metric_config": {
            "weight_file": (
                str(_resolve(args.dual_metric_weight_file))
                if args.dual_metric_weight_file
                else None
            ),
            "search_candidates": int(args.dual_metric_search_candidates),
            "candidates": int(args.dual_metric_candidates),
            "high_quality_tp_candidates": int(
                args.dual_metric_high_quality_tp_candidates
            ),
            "tp_candidates": int(args.dual_metric_tp_candidates),
            "hard_tp_candidates": int(args.dual_metric_hard_tp_candidates),
            "hard_margin_fraction": float(args.dual_metric_hard_margin_fraction),
            "head_protect": int(args.dual_metric_head_protect),
            "structural_protect": int(args.dual_metric_structural_protect),
            "embedding_weight": float(args.dual_metric_embedding_weight),
            "code_weight": float(args.dual_metric_code_weight),
            "topology_weight": float(args.dual_metric_topology_weight),
        },
        "hnsw_num_threads": int(args.hnsw_num_threads),
        "code_backend": args.code_backend,
        "code_flat_block_size": int(args.code_flat_block_size),
        "adaptive_config": {
            "adaptive_initial_global": int(args.adaptive_initial_global),
            "adaptive_initial_structural": int(args.adaptive_initial_structural),
            "adaptive_max_global": int(args.adaptive_max_global),
            "adaptive_max_structural": int(args.adaptive_max_structural),
            "adaptive_max_rerank": int(args.adaptive_max_rerank),
            "adaptive_overlap_threshold": float(args.adaptive_overlap_threshold),
            "adaptive_min_route_recall": int(args.adaptive_min_route_recall),
            "adaptive_gate": args.adaptive_gate,
            "adaptive_global_margin": float(args.adaptive_global_margin),
            "adaptive_metric_fast_path": args.adaptive_metric_fast_path,
            "adaptive_max_tp_fraction": float(args.adaptive_max_tp_fraction),
            "adaptive_preserve_global_top50": bool(
                args.adaptive_preserve_global_top50
            ),
            "adaptive_rescue_quality_overlap": float(args.adaptive_rescue_quality_overlap),
            "adaptive_rescue_near_margin_fraction": float(args.adaptive_rescue_near_margin_fraction),
            "metric_prefilter_candidates": int(args.metric_prefilter_candidates),
            "metric_prefilter_high_quality_candidates": int(args.metric_prefilter_high_quality_candidates),
            "metric_prefilter_global_protect": int(args.metric_prefilter_global_protect),
            "metric_prefilter_structural_protect": int(args.metric_prefilter_structural_protect),
        },
        "commands": [[str(item) for item in command] for command in commands],
    }
    _write_manifest(output_dir / "experiment_manifest.json", manifest)
    print(f"Manifest written to: {output_dir / 'experiment_manifest.json'}")
    for command in commands:
        _run(command, dry_run=args.dry_run)
    return manifest


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="Our trajectory encoder checkpoint.")
    parser.add_argument(
        "--spgmt-model",
        default=None,
        help="SPGMT baseline checkpoint. If omitted, HNSW uses our embeddings.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="beijing")
    parser.add_argument(
        "--distance-type",
        choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
        default="TP",
    )
    parser.add_argument("--hierarchy-dir", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--code-model-file", default=None)
    parser.add_argument(
        "--metric-code-model-file",
        default=None,
        help=(
            "Independent Metric-Head checkpoint for adaptive-dual-metric. "
            "Pair this with --code-model-file for the base encoder."
        ),
    )
    parser.add_argument("--dual-metric-weight-file", default=None)
    parser.add_argument("--code-bits", type=int, default=128)
    parser.add_argument("--code-epochs", type=int, default=30)
    parser.add_argument("--code-batch-size", type=int, default=512)
    parser.add_argument("--code-device", default="auto")
    parser.add_argument(
        "--topology-query-code-device",
        default="cpu",
        help="Device used to encode Query binary codes during candidate selection evaluation.",
    )
    parser.add_argument("--cuda", default="0")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help=(
            "Embedding batch size. Small values are safer because the coordinate "
            "Transformer allocates attention memory quadratically in trajectory length."
        ),
    )
    parser.add_argument("--gallery-sizes", default="500000,1000000")
    parser.add_argument("--query-count", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tp-top-k", type=int, default=50)
    parser.add_argument(
        "--rerank-metric",
        choices=["embedding", "TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
        default="embedding",
        help=(
            "Online final rerank metric for low-confidence adaptive queries. "
            "TP rerank time is included in candidate selection query latency."
        ),
    )
    parser.add_argument(
        "--tp-point-distance-file",
        default=None,
    )
    parser.add_argument(
        "--tp-num-threads",
        type=int,
        default=0,
        help="Numba CPU threads for online metric reranking and TP evaluation.",
    )
    parser.add_argument("--edge-file", default=None)
    parser.add_argument("--node-count", type=int, default=0)
    parser.add_argument("--max-trajectory-length", type=int, default=0)
    parser.add_argument("--neterp-anchor-node", type=int, default=None)
    parser.add_argument("--tp-distance-table-file", required=True)
    parser.add_argument(
        "--topology-method",
        choices=[
            "adaptive-selection",
            "adaptive-dual-metric",
            "hierarchical-route",
        ],
        default="adaptive-selection",
        help=(
            "Candidate-selection method used for our model. adaptive-dual-metric uses "
            "an independent Metric Head and adds a no-rerank ablation; "
            "hierarchical-route selects the legacy fixed-budget method."
        ),
    )
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument(
        "--candidate-source",
        choices=["hybrid", "global-only"],
        default="hybrid",
    )
    parser.add_argument("--probe-radius", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--hamming-candidates", type=int, default=2048)
    parser.add_argument("--code-backend", choices=["mih", "hnsw", "flat"], default="mih")
    parser.add_argument("--code-flat-block-size", type=int, default=65536)
    parser.add_argument("--structural-candidates", type=int, default=512)
    parser.add_argument("--global-candidates", type=int, default=1024)
    parser.add_argument("--max-rerank-candidates", type=int, default=1536)
    parser.add_argument("--min-route-candidates", type=int, default=1024)
    parser.add_argument("--max-route-candidates", type=int, default=4096)
    parser.add_argument(
        "--include-code-scan",
        action="store_true",
        help="Evaluate Code-Scan in the same candidate selection process for comparison.",
    )
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=80)
    parser.add_argument(
        "--hnsw-num-threads",
        type=int,
        default=1,
        help=(
            "HNSW build/query threads. Default 1 makes latency and build-time "
            "comparisons explicit and reproducible."
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
    )
    parser.add_argument("--adaptive-global-margin", type=float, default=0.05)
    parser.add_argument(
        "--adaptive-metric-fast-path",
        choices=["skip", "metric"],
        default="skip",
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
            "Protect the Code-Scan embedding Top-50 in Adaptive metric rerank pools. "
            "With adaptive-metric-fast-path=skip, only difficult "
            "queries use the configured metric; emit Top-50 retention diagnostics."
        ),
    )
    parser.add_argument("--adaptive-rescue-quality-overlap", type=float, default=0.75)
    parser.add_argument("--adaptive-rescue-near-margin-fraction", type=float, default=0.80)
    parser.add_argument("--metric-prefilter-candidates", type=int, default=64)
    parser.add_argument("--metric-prefilter-high-quality-candidates", type=int, default=50)
    parser.add_argument("--metric-prefilter-global-protect", type=int, default=32)
    parser.add_argument("--metric-prefilter-structural-protect", type=int, default=16)
    parser.add_argument("--metric-prefilter-embedding-weight", type=float, default=1.0)
    parser.add_argument("--metric-prefilter-code-weight", type=float, default=0.0)
    parser.add_argument(
        "--metric-prefilter-code-score",
        choices=["hamming", "soft"],
        default="hamming",
    )
    parser.add_argument("--metric-prefilter-soft-temperature", type=float, default=1.0)
    parser.add_argument("--dual-metric-search-candidates", type=int, default=256)
    parser.add_argument("--dual-metric-candidates", type=int, default=64)
    parser.add_argument(
        "--dual-metric-high-quality-tp-candidates", type=int, default=64
    )
    parser.add_argument("--dual-metric-tp-candidates", type=int, default=128)
    parser.add_argument("--dual-metric-hard-tp-candidates", type=int, default=160)
    parser.add_argument("--dual-metric-hard-margin-fraction", type=float, default=0.50)
    parser.add_argument("--dual-metric-head-protect", type=int, default=48)
    parser.add_argument("--dual-metric-structural-protect", type=int, default=16)
    parser.add_argument("--dual-metric-embedding-weight", type=float, default=0.96)
    parser.add_argument("--dual-metric-code-weight", type=float, default=0.02)
    parser.add_argument("--dual-metric-topology-weight", type=float, default=0.02)
    parser.add_argument(
        "--topology-embedding-weight",
        type=float,
        default=1.0,
        help=(
            "Embedding-rank weight of adaptive candidate selection. With the default 1.0, "
            "topology only rescues candidates and does not replace the code-scan "
            "embedding ranking."
        ),
    )
    parser.add_argument("--use-hybrid-geo", dest="use_hybrid_geo", action="store_true", default=True)
    parser.add_argument("--no-use-hybrid-geo", dest="use_hybrid_geo", action="store_false")
    parser.add_argument("--use-si", dest="use_si", action="store_true", default=None)
    parser.add_argument("--no-use-si", dest="use_si", action="store_false")
    parser.add_argument("--seq-encoder", default="hmt")
    parser.add_argument("--coord-encoder", default="transformer")
    parser.add_argument("--hmt-backbone", default="hybrid")
    parser.add_argument("--graph-encoder", default=None)
    parser.add_argument("--coord-fusion", default=None)
    parser.add_argument(
        "--spgmt-use-hybrid-geo",
        dest="spgmt_use_hybrid_geo",
        action="store_true",
        default=False,
        help="Use HybridGeo when exporting the SPGMT checkpoint (legacy SPGMT defaults to false).",
    )
    parser.add_argument(
        "--no-spgmt-use-hybrid-geo",
        "--spgmt-no-use-hybrid-geo",
        dest="spgmt_use_hybrid_geo",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--spgmt-use-si",
        dest="spgmt_use_si",
        action="store_true",
        default=True,
        help="Enable the spatial-index/KNN graph branch for the SPGMT checkpoint.",
    )
    parser.add_argument(
        "--no-spgmt-use-si",
        "--spgmt-no-use-si",
        dest="spgmt_use_si",
        action="store_false",
        help="Disable the spatial-index/KNN graph branch for the SPGMT checkpoint.",
    )
    parser.add_argument("--spgmt-seq-encoder", default="transformer")
    parser.add_argument("--spgmt-coord-encoder", default="lstm")
    parser.add_argument("--spgmt-hmt-backbone", default=None)
    parser.add_argument("--spgmt-graph-encoder", default=None)
    parser.add_argument("--spgmt-coord-fusion", default=None)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-code-training", action="store_true")
    parser.add_argument("--skip-topology", action="store_true")
    parser.add_argument("--skip-hnsw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
