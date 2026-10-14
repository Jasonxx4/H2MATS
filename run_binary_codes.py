import argparse
import json
import os
import os.path as osp
from copy import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from ann_pipeline import (
    apply_dataset,
    apply_model_overrides,
    new_default_config,
    refresh_distance_paths,
    resolve_checkpoint,
    setup_logging,
)
from binary_codes import (
    BinaryCodeIndexConfig,
    BinaryCodeIndex,
    CodeEncoderConfig,
    CodeTrainConfig,
    load_code_encoder,
    save_code_encoder,
    train_code_encoder,
)
from binary_code_pipeline import (
    _load_npz_payload,
    eval_command,
    load_metric_candidate_teacher,
    load_metric_distances,
)
from trajectory_embedding import export_spatial_embeddings


def _needs_refresh(output_path, input_paths, force=False):
    if force or not osp.exists(output_path):
        return True
    output_mtime = osp.getmtime(output_path)
    return any(osp.exists(path) and osp.getmtime(path) > output_mtime for path in input_paths)


def _is_embedding_file(path):
    """Return whether ``path`` is a pre-exported embedding NPZ."""
    try:
        with np.load(path, allow_pickle=True) as data:
            return "embeddings" in data.files
    except (OSError, ValueError, TypeError):
        return False


def _write_json(path, payload):
    out_dir = osp.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _configure_model(args):
    config = apply_dataset(new_default_config(), args.dataset)
    config.distance_type = args.distance_type
    refresh_distance_paths(config)
    if args.dataset_mode is not None:
        config.set_dataset_mode(args.dataset_mode)
    apply_model_overrides(config, args)
    return config


def _export_embeddings(config, checkpoint, eval_file, output_file, batch_size, force):
    if not osp.isfile(eval_file):
        raise FileNotFoundError(f"Cannot find evaluation split: {eval_file}")
    if not _needs_refresh(output_file, [checkpoint, eval_file], force=force):
        print(f"[reuse] embeddings: {output_file}")
        return
    print(f"[export] {eval_file} -> {output_file}")
    export_spatial_embeddings(
        config=config,
        load_model=checkpoint,
        eval_file=eval_file,
        output_file=output_file,
        batch_size=batch_size,
    )


def _train_head(args, validation_embedding_file, model_file, code_device, force):
    model_inputs = [validation_embedding_file]
    if args.metric_candidate_teacher_file:
        model_inputs.append(args.metric_candidate_teacher_file)
    if args.metric_distance_file:
        model_inputs.append(args.metric_distance_file)
    if not _needs_refresh(model_file, model_inputs, force=force):
        print(f"[reuse] code encoder: {model_file}")
        return load_code_encoder(model_file)[1]
    print(f"[train] code encoder: {model_file}")
    payload = _load_npz_payload(validation_embedding_file, require_labels=True)
    if getattr(args, "ignore_embedded_metric_distances", False):
        payload.pop("metric_distances", None)
        payload.pop("metric_distance_key", None)
    candidate_teacher_report = None
    if args.metric_candidate_teacher_file:
        teacher_labels, teacher_distances, candidate_teacher_report = (
            load_metric_candidate_teacher(
                args.metric_candidate_teacher_file, len(payload["embeddings"])
            )
        )
        payload["labels"] = teacher_labels
        payload["metric_distances"] = teacher_distances
        payload["metric_distance_key"] = "candidate_teacher_tp_distances"
    elif args.metric_distance_file:
        payload["metric_distances"] = load_metric_distances(args.metric_distance_file)
    head_config = CodeEncoderConfig(
        input_dim=payload["embeddings"].shape[1],
        bits=args.bits,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    train_config = CodeTrainConfig(
        metric=args.metric,
        epochs=args.epochs,
        batch_size=args.code_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_pool=args.positive_pool,
        positives_per_anchor=args.positives_per_anchor,
        negatives_per_anchor=args.negatives_per_anchor,
        negative_strategy=args.negative_strategy,
        hard_negative_pool=args.hard_negative_pool,
        hard_negative_fraction=args.hard_negative_fraction,
        metric_distance_weight=args.metric_distance_weight,
        metric_distance_scale=args.metric_distance_scale,
        embedding_distance_weight=args.embedding_distance_weight,
        filter_same_source_negatives=args.filter_same_source_negatives,
        in_batch_negative_weight=args.in_batch_negative_weight,
        graded_rank_weight=args.graded_rank_weight,
        graded_rank_margin=args.graded_rank_margin,
        graded_rank_cutoffs=tuple(
            int(value) for value in args.graded_rank_cutoffs.split(",") if value
        ),
        cutoff_rank_weight=args.cutoff_rank_weight,
        cutoff_rank=args.cutoff_rank,
        cutoff_rank_margin=args.cutoff_rank_margin,
        cutoff_rank_window=args.cutoff_rank_window,
        cutoff_samples=args.cutoff_samples,
        margin=args.margin,
        ranking_weight=args.ranking_weight,
        quantization_weight=args.quantization_weight,
        balance_weight=args.balance_weight,
        independence_weight=args.independence_weight,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=code_device,
    )
    head, report = train_code_encoder(
        payload["embeddings"],
        payload["labels"],
        head_config,
        train_config,
        metric_distances=payload.get("metric_distances"),
        source_ids=payload.get("source_ids"),
        anchor_count=payload.get("anchor_count"),
    )
    report["source_embedding_file"] = str(Path(validation_embedding_file).resolve())
    if candidate_teacher_report is not None:
        report["metric_candidate_teacher_file"] = str(
            Path(args.metric_candidate_teacher_file).resolve()
        )
        report["metric_candidate_teacher"] = candidate_teacher_report
    if payload.get("metric_distance_key"):
        report["metric_distance_key"] = payload["metric_distance_key"]
    elif args.metric_distance_file:
        report["metric_distance_file"] = str(Path(args.metric_distance_file).resolve())
    save_code_encoder(head, model_file, metadata=report)
    _write_json(model_file + ".json", report)
    return report


def _build_index(args, test_embedding_file, model_file, index_dir, code_device, force):
    config_file = osp.join(index_dir, "code_config.json")
    required_files = [
        config_file,
        osp.join(index_dir, "code_encoder.pt"),
        osp.join(index_dir, "codes.npy"),
        osp.join(index_dir, "embeddings.npy"),
        osp.join(index_dir, "table_offsets.npy"),
        osp.join(index_dir, "table_postings.npy"),
    ]
    force = force or not all(osp.isfile(path) for path in required_files)
    if not _needs_refresh(config_file, [test_embedding_file, model_file], force=force):
        print(f"[reuse] code-scan index: {index_dir}")
        return
    print(f"[build] code-scan index: {index_dir}")
    payload = _load_npz_payload(test_embedding_file)
    head, head_metadata = load_code_encoder(model_file)
    index_config = BinaryCodeIndexConfig(
        bits=head.config.bits,
        subcode_bits=16,
        probe_radius=args.probe_radius,
        hamming_candidates=args.hamming_candidates,
        metric=args.metric,
        normalize=args.normalize,
        embedding_storage=args.embedding_storage,
        encode_batch_size=args.encode_batch_size,
    )
    metadata = {
        "source_embedding_file": str(Path(test_embedding_file).resolve()),
        "source_code_model": str(Path(model_file).resolve()),
        "code_training": head_metadata,
    }
    index = BinaryCodeIndex(index_config).build(
        payload["embeddings"],
        head,
        encode_device=code_device,
        metadata=metadata,
    )
    index.save(index_dir)


def _evaluate_all(args, test_embedding_file, index_dir, result_file, code_device):
    print(f"[eval] all test queries -> {result_file}")
    eval_args = SimpleNamespace(
        embedding_file=test_embedding_file,
        index_dir=index_dir,
        no_mmap=args.no_mmap,
        max_queries=args.max_queries,
        exclude_self=True,
        top_k=args.top_k,
        truth_k=args.truth_k,
        result_file=result_file,
        query_code_device=code_device,
        rerank_mode=args.rerank_mode,
        small_rerank_candidates=args.small_rerank_candidates,
    )
    return eval_command(eval_args)


def run(args):
    setup_logging()
    config = _configure_model(args)
    checkpoint = resolve_checkpoint(args.model)
    output_dir = args.output_dir or osp.join(
        "ann_artifacts",
        config.dataset,
        config.distance_type,
        "binary_codes",
    )
    os.makedirs(output_dir, exist_ok=True)

    validation_file = args.validation_file or config.spatial_vali_set
    test_file = args.test_file or config.spatial_test_set
    validation_embedding_file = osp.join(output_dir, "validation_embeddings.npz")
    test_embedding_file = osp.join(output_dir, "test_embeddings.npz")
    if args.dual_code_encoders:
        model_file = osp.join(output_dir, f"code_encoder_{args.bits}_metric.pt")
        candidate_model_file = osp.join(
            output_dir, f"code_encoder_{args.bits}_base.pt"
        )
    else:
        model_file = osp.join(output_dir, f"code_encoder_{args.bits}.pt")
        candidate_model_file = model_file
    index_dir = osp.join(output_dir, f"index_{args.bits}")
    result_file = args.result_file or osp.join(output_dir, f"result_all_{args.bits}.json")
    force_export = args.force_all or args.force_export
    force_train = args.force_all or args.force_train
    force_build = args.force_all or args.force_build
    code_device = args.code_device
    if code_device is None:
        code_device = f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu"

    if _is_embedding_file(validation_file):
        validation_embedding_file = validation_file
        print(f"[reuse] pre-exported head training embeddings: {validation_file}")
    else:
        _export_embeddings(
            config,
            checkpoint,
            validation_file,
            validation_embedding_file,
            args.embedding_batch_size,
            force_export,
        )
    _export_embeddings(
        config,
        checkpoint,
        test_file,
        test_embedding_file,
        args.embedding_batch_size,
        force_export,
    )
    training_report = _train_head(
        args,
        validation_embedding_file,
        model_file,
        code_device,
        force_train,
    )
    candidate_training_report = None
    if args.dual_code_encoders:
        candidate_args = copy(args)
        candidate_args.ignore_embedded_metric_distances = True
        candidate_args.metric_distance_file = None
        candidate_args.metric_candidate_teacher_file = None
        candidate_args.metric_distance_weight = 0.0
        candidate_args.metric_distance_scale = 0.0
        candidate_args.embedding_distance_weight = (
            args.candidate_embedding_distance_weight
        )
        candidate_args.graded_rank_weight = 0.0
        candidate_args.cutoff_rank_weight = 0.0
        candidate_args.in_batch_negative_weight = (
            args.candidate_in_batch_negative_weight
        )
        print(
            "[train] base encoder uses embedding supervision only; "
            "TP distillation remains isolated in the metric encoder"
        )
        candidate_training_report = _train_head(
            candidate_args,
            validation_embedding_file,
            candidate_model_file,
            code_device,
            force_train,
        )
    _build_index(
        args,
        test_embedding_file,
        candidate_model_file,
        index_dir,
        code_device,
        force_build,
    )
    result = None
    if args.skip_eval:
        print("[skip] code-scan evaluation")
    else:
        result = _evaluate_all(args, test_embedding_file, index_dir, result_file, code_device)

    manifest = {
        "dataset": config.dataset,
        "dataset_mode": config.dataset_mode,
        "distance_type": config.distance_type,
        "checkpoint": str(Path(checkpoint).resolve()),
        "validation_file": str(Path(validation_file).resolve()),
        "test_file": str(Path(test_file).resolve()),
        "validation_embedding_file": str(Path(validation_embedding_file).resolve()),
        "test_embedding_file": str(Path(test_embedding_file).resolve()),
        "code_model_file": str(Path(candidate_model_file).resolve()),
        "base_code_model_file": str(Path(candidate_model_file).resolve()),
        "metric_prefilter_code_model_file": (
            str(Path(model_file).resolve()) if args.dual_code_encoders else None
        ),
        "index_dir": str(Path(index_dir).resolve()),
        "result_file": str(Path(result_file).resolve()),
        "skip_eval": bool(args.skip_eval),
        "model_options": {
            "use_hybrid_geo": config.gtraj.get("useHybridGeo"),
            "graph_encoder": config.gtraj.get("graphEncoder"),
            "seq_encoder": config.gtraj.get("seqEncoder"),
            "coord_encoder": config.gtraj.get("coordEncoder"),
            "hmt_backbone": config.gtraj.get("hmtBackbone"),
            "hmt_scale_aligned_graph": config.gtraj.get("hmtScaleAlignedGraph"),
            "coord_fusion": config.gtraj.get("coordFusion"),
            "cuda": config.cuda,
        },
        "code_config": asdict(BinaryCodeIndexConfig(
            bits=args.bits,
            probe_radius=args.probe_radius,
            hamming_candidates=args.hamming_candidates,
            metric=args.metric,
            normalize=args.normalize,
            embedding_storage=args.embedding_storage,
            encode_batch_size=args.encode_batch_size,
        )),
        "code_training_best_validation_loss": training_report.get("best_validation_loss"),
        "base_code_training_best_validation_loss": (
            candidate_training_report.get("best_validation_loss")
            if candidate_training_report is not None else None
        ),
        "rerank": {
            "mode": args.rerank_mode,
            "small_rerank_candidates": args.small_rerank_candidates,
        },
        "evaluation": result,
    }
    manifest_file = osp.join(output_dir, "run_manifest.json")
    _write_json(manifest_file, manifest)
    print(f"[done] manifest: {manifest_file}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export embeddings, train a post-hoc code encoder, build code-scan, "
            "and query every test trajectory."
        )
    )
    parser.add_argument("--model", required=True, help="Checkpoint file or epoch_*.pt directory.")
    parser.add_argument("--dataset", default="beijing", choices=["beijing", "porto", "tdrive"])
    parser.add_argument(
        "--dataset-mode", default="standard", choices=["standard", "50w", "full"]
    )
    parser.add_argument(
        "--distance-type",
        default="TP",
        choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
    )
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--metric-distance-file", default=None)
    parser.add_argument(
        "--metric-candidate-teacher-file",
        default=None,
        help=(
            "Optional ragged validation union-candidate TP teacher NPZ. When "
            "provided, it replaces the narrow label-aligned distance source."
        ),
    )
    parser.add_argument(
        "--dual-code-encoders",
        action="store_true",
        help=(
            "Train separate base and TP-metric code encoders. The base "
            "encoder builds the code-scan index; the metric encoder is reserved for prefiltering."
        ),
    )

    parser.add_argument("--cuda", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--test-batch", type=int, default=None)
    parser.add_argument("--seq-encoder", choices=["transformer", "mamba", "bimamba", "hmt"], default=None)
    parser.add_argument(
        "--graph-encoder",
        choices=["hybrid_multilevel", "simple_gcn"],
        default=None,
    )
    parser.add_argument("--coord-encoder", choices=["lstm", "mamba", "transformer"], default=None)
    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-d-conv", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument(
        "--hmt-backbone",
        choices=["conv", "mamba", "bimamba", "transformer", "hybrid"],
        default=None,
    )
    parser.add_argument("--hmt-layers", type=int, default=None)
    parser.add_argument("--hmt-kernel-size", type=int, default=None)
    parser.add_argument("--hmt-dropout", type=float, default=None)
    parser.add_argument("--hmt-transformer-heads", type=int, default=None)
    parser.add_argument(
        "--hmt-scale-aligned-graph",
        dest="hmt_scale_aligned_graph",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-hmt-scale-aligned-graph",
        dest="hmt_scale_aligned_graph",
        action="store_false",
    )
    parser.add_argument("--coord-fusion", choices=["gated_attention", "late_concat"], default=None)
    parser.add_argument("--use-hybrid-geo", dest="use_hybrid_geo", action="store_true", default=None)
    parser.add_argument("--no-use-hybrid-geo", dest="use_hybrid_geo", action="store_false")

    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--code-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-pool", type=int, default=10)
    parser.add_argument("--positives-per-anchor", type=int, default=1)
    parser.add_argument("--negatives-per-anchor", type=int, default=1)
    parser.add_argument(
        "--negative-strategy",
        choices=["random", "metric-hard", "mixed"],
        default="random",
    )
    parser.add_argument("--hard-negative-pool", type=int, default=0)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--metric-distance-weight", type=float, default=0.0)
    parser.add_argument("--metric-distance-scale", type=float, default=0.0)
    parser.add_argument("--embedding-distance-weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate-embedding-distance-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--candidate-in-batch-negative-weight", type=float, default=0.25
    )
    parser.add_argument("--filter-same-source-negatives", action="store_true")
    parser.add_argument("--in-batch-negative-weight", type=float, default=0.0)
    parser.add_argument("--graded-rank-weight", type=float, default=0.0)
    parser.add_argument("--graded-rank-margin", type=float, default=0.05)
    parser.add_argument("--graded-rank-cutoffs", default="10,20,50")
    parser.add_argument("--cutoff-rank-weight", type=float, default=0.0)
    parser.add_argument("--cutoff-rank", type=int, default=192)
    parser.add_argument("--cutoff-rank-margin", type=float, default=0.02)
    parser.add_argument("--cutoff-rank-window", type=int, default=64)
    parser.add_argument("--cutoff-samples", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--quantization-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--independence-weight", type=float, default=0.001)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--code-device", default=None)

    parser.add_argument("--probe-radius", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--hamming-candidates", type=int, default=2000)
    parser.add_argument("--embedding-storage", choices=["float16", "float32"], default="float16")
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--truth-k", type=int, default=50)
    parser.add_argument(
        "--rerank-mode",
        choices=["full", "code-only", "small-rerank"],
        default="full",
        help=(
            "Final ranking mode: full keeps the original continuous rerank, "
            "code-only returns Hamming-ranked results, small-rerank reranks only "
            "a reduced Hamming candidate set."
        ),
    )
    parser.add_argument(
        "--small-rerank-candidates",
        type=int,
        default=None,
        help="Hamming candidates kept before continuous rerank when --rerank-mode=small-rerank.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Number of test queries; 0 queries every test trajectory.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only export embeddings, train the code encoder, and build the index.",
    )
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
