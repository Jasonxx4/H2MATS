import argparse
import json
import os
import os.path as osp
from pathlib import Path
from time import perf_counter

import numpy as np

from binary_codes import (
    BinaryCodeIndexConfig,
    BinaryCodeIndex,
    CodeEncoderConfig,
    CodeTrainConfig,
    encode_binary_codes,
    load_code_encoder,
    save_code_encoder,
    train_code_encoder,
)


def _load_npz_payload(path, require_labels=False):
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(f"Embedding file {path} does not contain an 'embeddings' array.")
    labels = None
    for key in ("labels", "vali_y", "train_y"):
        if key in data:
            labels = np.asarray(data[key], dtype=np.int64)
            break
    if require_labels and labels is None:
        raise ValueError(
            f"Embedding file {path} does not contain labels. Expected one of: "
            "labels, vali_y, train_y."
        )
    payload = {
        "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
        "labels": labels,
    }
    if "anchor_count" in data:
        payload["anchor_count"] = int(np.asarray(data["anchor_count"]).item())
    if "source_ids" in data:
        payload["source_ids"] = np.asarray(data["source_ids"], dtype=np.int64)
    for key in (
        "metric_distances",
        "label_distances",
        "metric_label_distances",
        "train_dis",
        "vali_dis",
    ):
        if key in data:
            candidate = np.asarray(data[key], dtype=np.float32)
            if labels is not None and (
                candidate.shape == labels.shape
                or candidate.shape == (len(payload["embeddings"]), len(payload["embeddings"]))
            ):
                payload["metric_distances"] = candidate
                payload["metric_distance_key"] = key
                break
    return payload


def load_metric_distances(path):
    if not path:
        return None
    if str(path).lower().endswith(".npy"):
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "metric_distances",
            "label_distances",
            "metric_label_distances",
            "distances",
            "train_dis",
            "vali_dis",
        ):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32)
    raise ValueError(f"Metric distance file {path} does not contain a supported array.")


def load_metric_candidate_teacher(path, embedding_count):
    """Load ragged candidate IDs/distances and align them to embedding rows."""
    with np.load(path, allow_pickle=False) as data:
        required = {"query_ids", "offsets", "candidate_ids", "tp_distances"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Candidate teacher {path} is missing arrays: {missing}.")
        query_ids = np.asarray(data["query_ids"], dtype=np.int64).reshape(-1)
        offsets = np.asarray(data["offsets"], dtype=np.int64).reshape(-1)
        candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int64).reshape(-1)
        distances = np.asarray(data["tp_distances"], dtype=np.float32).reshape(-1)
    if len(offsets) != len(query_ids) + 1 or len(offsets) == 0:
        raise ValueError("Candidate teacher offsets must have len(query_ids) + 1 entries.")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("Candidate teacher offsets must start at zero and be monotonic.")
    if offsets[-1] != len(candidate_ids) or len(candidate_ids) != len(distances):
        raise ValueError("Candidate teacher flattened arrays do not match offsets.")
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("Candidate teacher query_ids must be unique.")
    if np.any((query_ids < 0) | (query_ids >= int(embedding_count))):
        raise ValueError("Candidate teacher query_ids are outside the embedding rows.")
    counts = np.diff(offsets)
    width = int(counts.max()) if len(counts) else 0
    if width <= 0:
        raise ValueError("Candidate teacher contains no candidates.")
    labels = np.full((int(embedding_count), width), -1, dtype=np.int64)
    metric_distances = np.full(
        (int(embedding_count), width), np.nan, dtype=np.float32
    )
    for query_id, start, stop in zip(query_ids, offsets[:-1], offsets[1:]):
        count = int(stop - start)
        row_ids = candidate_ids[start:stop]
        if np.any((row_ids < 0) | (row_ids >= int(embedding_count))):
            raise ValueError(
                f"Candidate teacher row {query_id} contains an invalid candidate ID."
            )
        labels[int(query_id), :count] = row_ids
        metric_distances[int(query_id), :count] = distances[start:stop]
    return labels, metric_distances, {
        "query_count": int(len(query_ids)),
        "candidate_count": int(len(candidate_ids)),
        "candidate_count_mean": float(np.mean(counts)),
        "candidate_count_min": int(np.min(counts)),
        "candidate_count_max": int(np.max(counts)),
    }


def _json_dump(path, payload):
    out_dir = osp.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _directory_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def _retrieval_metrics(predictions, labels):
    metrics = np.zeros((len(labels), 6), dtype=np.float32)
    for idx, truth in enumerate(labels):
        prediction = np.asarray(predictions[idx], dtype=np.int64)
        truth = np.asarray(truth, dtype=np.int64)
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
    return metrics.mean(axis=0) if len(metrics) else np.zeros(6, dtype=np.float32)


def train_command(args):
    payload = _load_npz_payload(args.embedding_file, require_labels=True)
    if args.metric_distance_file:
        payload["metric_distances"] = load_metric_distances(args.metric_distance_file)
    embeddings = payload["embeddings"]
    head_config = CodeEncoderConfig(
        input_dim=embeddings.shape[1],
        bits=args.bits,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    train_config = CodeTrainConfig(
        metric=args.metric,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_pool=args.positive_pool,
        margin=args.margin,
        ranking_weight=args.ranking_weight,
        quantization_weight=args.quantization_weight,
        balance_weight=args.balance_weight,
        independence_weight=args.independence_weight,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
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
    )
    head, report = train_code_encoder(
        embeddings,
        payload["labels"],
        head_config,
        train_config,
        metric_distances=payload.get("metric_distances"),
        source_ids=payload.get("source_ids"),
        anchor_count=payload.get("anchor_count"),
    )
    report["source_embedding_file"] = str(Path(args.embedding_file).resolve())
    if payload.get("metric_distance_key"):
        report["metric_distance_key"] = payload["metric_distance_key"]
    elif args.metric_distance_file:
        report["metric_distance_file"] = str(Path(args.metric_distance_file).resolve())
    save_code_encoder(head, args.model_file, metadata=report)
    report_file = args.report_file or args.model_file + ".json"
    _json_dump(report_file, report)
    print(f"Code encoder saved to: {args.model_file}")
    print(
        f"vectors={len(embeddings)}, bits={args.bits}, "
        f"best_validation_loss={report['best_validation_loss']:.6f}"
    )
    print(f"Training report saved to: {report_file}")


def build_command(args):
    payload = _load_npz_payload(args.embedding_file)
    head, head_metadata = load_code_encoder(args.model_file)
    config = BinaryCodeIndexConfig(
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
        "source_embedding_file": str(Path(args.embedding_file).resolve()),
        "source_code_model": str(Path(args.model_file).resolve()),
        "code_training": head_metadata,
    }
    index = BinaryCodeIndex(config).build(
        payload["embeddings"],
        head,
        encode_device=args.device,
        metadata=metadata,
    )
    index.save(args.index_dir)
    print(f"Code-scan index saved to: {args.index_dir}")
    print(
        f"vectors={len(index.embeddings)}, bits={config.bits}, "
        f"tables={config.bits // config.subcode_bits}, "
        f"build_seconds={index.build_seconds:.3f}, "
        f"disk_mb={_directory_size(args.index_dir) / (1024 ** 2):.2f}"
    )


def _same_file(left, right):
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def eval_command(args):
    if args.rerank_mode == "small-rerank":
        if args.small_rerank_candidates is None:
            raise ValueError("--small-rerank-candidates is required when --rerank-mode=small-rerank.")
        if args.small_rerank_candidates <= 0:
            raise ValueError("--small-rerank-candidates must be positive.")
    payload = _load_npz_payload(args.embedding_file, require_labels=True)
    index = BinaryCodeIndex.load(args.index_dir, mmap=not args.no_mmap)
    embeddings = payload["embeddings"]
    labels = payload["labels"]
    if embeddings.shape[1] != index.head.config.input_dim:
        raise ValueError(
            f"Query dimension {embeddings.shape[1]} does not match index dimension "
            f"{index.head.config.input_dim}."
        )
    total = len(embeddings) if args.max_queries <= 0 else min(args.max_queries, len(embeddings))
    source_file = index.metadata.get("source_embedding_file")
    exclude_self = args.exclude_self
    if exclude_self is None:
        exclude_self = _same_file(args.embedding_file, source_file)

    code_start = perf_counter()
    query_codes = encode_binary_codes(
        index.head,
        embeddings[:total],
        batch_size=index.config.encode_batch_size,
        device=getattr(args, "query_code_device", "cpu"),
    )
    code_encoding_ms = (perf_counter() - code_start) * 1000.0
    code_encoding_ms_per_query = code_encoding_ms / max(total, 1)

    predictions = []
    candidate_recalls = []
    latencies_ms = []
    mih_counts = []
    hamming_counts = []
    full_scan_count = 0
    query_ids = range(total)
    try:
        from tqdm import tqdm

        query_ids = tqdm(query_ids, desc="code-scan queries")
    except ImportError:
        pass
    for query_id in query_ids:
        exclude_ids = None
        if exclude_self:
            if "source_ids" in payload:
                exclude_ids = {int(payload["source_ids"][query_id])}
            else:
                exclude_ids = {query_id}
        start = perf_counter()
        ids, _, details = index.query(
            embeddings[query_id],
            top_k=args.top_k,
            exclude_ids=exclude_ids,
            return_details=True,
            query_code=query_codes[query_id],
            rerank_mode=args.rerank_mode,
            small_rerank_candidates=args.small_rerank_candidates,
        )
        latencies_ms.append((perf_counter() - start) * 1000.0)
        predictions.append(ids)
        truth = set(int(item) for item in labels[query_id, :args.truth_k])
        candidates = set(int(item) for item in details["candidate_ids"])
        candidate_recalls.append(len(truth & candidates) / max(len(truth), 1))
        mih_counts.append(details["mih_candidate_count"])
        hamming_counts.append(details["hamming_candidate_count"])
        full_scan_count += int(details["used_full_scan"])

    metrics = _retrieval_metrics(predictions, labels[:total])
    names = ["HR@10", "HR@50", "R10@50", "R1@1", "R1@10", "R1@50"]
    retrieval_mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    result = {
        "method": "code-scan",
        "queries": int(total),
        "top_k": int(args.top_k),
        "truth_k": int(args.truth_k),
        "exclude_self": bool(exclude_self),
        "metrics": {name: float(value) for name, value in zip(names, metrics)},
        "candidate_recall": float(np.mean(candidate_recalls)) if candidate_recalls else 0.0,
        "latency_ms": {
            "mean": retrieval_mean_ms + code_encoding_ms_per_query,
            "p50": (
                float(np.percentile(latencies_ms, 50)) + code_encoding_ms_per_query
                if latencies_ms else 0.0
            ),
            "p95": (
                float(np.percentile(latencies_ms, 95)) + code_encoding_ms_per_query
                if latencies_ms else 0.0
            ),
            "retrieval_mean": retrieval_mean_ms,
            "code_encoding_total": float(code_encoding_ms),
            "code_encoding_per_query": float(code_encoding_ms_per_query),
        },
        "candidate_counts": {
            "mih_mean": float(np.mean(mih_counts)) if mih_counts else 0.0,
            "hamming_mean": float(np.mean(hamming_counts)) if hamming_counts else 0.0,
            "full_scan_queries": int(full_scan_count),
        },
        "rerank": {
            "mode": args.rerank_mode,
            "small_rerank_candidates": (
                int(args.small_rerank_candidates)
                if args.small_rerank_candidates is not None else None
            ),
        },
        "index": {
            "vectors": int(len(index.embeddings)),
            "bits": int(index.config.bits),
            "probe_radius": int(index.config.probe_radius),
            "hamming_candidates": int(index.config.hamming_candidates),
            "metric": index.config.metric,
            "disk_mb": float(_directory_size(args.index_dir) / (1024 ** 2)),
        },
    }
    print(
        f"queries={total}, candidate_recall@{args.truth_k}={result['candidate_recall']:.4f}, "
        f"mean_ms={result['latency_ms']['mean']:.3f}, "
        f"p95_ms={result['latency_ms']['p95']:.3f}"
    )
    print(", ".join(f"{name}={result['metrics'][name]:.4f}" for name in names))
    print(
        f"mih_candidates={result['candidate_counts']['mih_mean']:.1f}, "
        f"rerank_candidates={result['candidate_counts']['hamming_mean']:.1f}, "
        f"full_scan_queries={full_scan_count}"
    )
    if args.result_file:
        _json_dump(args.result_file, result)
        print(f"Result saved to: {args.result_file}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary code learning and code-scan retrieval for trajectory embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a post-hoc binary code encoder.")
    train.add_argument("--embedding-file", required=True)
    train.add_argument("--model-file", required=True)
    train.add_argument("--report-file", default=None)
    train.add_argument("--metric-distance-file", default=None)
    train.add_argument("--bits", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--positive-pool", type=int, default=10)
    train.add_argument("--positives-per-anchor", type=int, default=1)
    train.add_argument("--negatives-per-anchor", type=int, default=1)
    train.add_argument(
        "--negative-strategy",
        choices=["random", "metric-hard", "mixed"],
        default="random",
        help="Sample random negatives, metric-ranked hard negatives, or a mixture.",
    )
    train.add_argument(
        "--hard-negative-pool",
        type=int,
        default=0,
        help="Number of ranked labels after positive-pool eligible as hard negatives; 0 means all.",
    )
    train.add_argument("--hard-negative-fraction", type=float, default=0.5)
    train.add_argument(
        "--metric-distance-weight",
        type=float,
        default=0.0,
        help="Optional weight for metric_distances/label_distances stored in the NPZ.",
    )
    train.add_argument("--metric-distance-scale", type=float, default=0.0)
    train.add_argument("--embedding-distance-weight", type=float, default=1.0)
    train.add_argument("--filter-same-source-negatives", action="store_true")
    train.add_argument("--in-batch-negative-weight", type=float, default=0.0)
    train.add_argument("--graded-rank-weight", type=float, default=0.0)
    train.add_argument("--graded-rank-margin", type=float, default=0.05)
    train.add_argument("--graded-rank-cutoffs", default="10,20,50")
    train.add_argument("--cutoff-rank-weight", type=float, default=0.0)
    train.add_argument("--cutoff-rank", type=int, default=192)
    train.add_argument("--cutoff-rank-margin", type=float, default=0.02)
    train.add_argument("--cutoff-rank-window", type=int, default=64)
    train.add_argument("--cutoff-samples", type=int, default=4)
    train.add_argument("--margin", type=float, default=0.1)
    train.add_argument("--ranking-weight", type=float, default=1.0)
    train.add_argument("--quantization-weight", type=float, default=0.1)
    train.add_argument("--balance-weight", type=float, default=0.01)
    train.add_argument("--independence-weight", type=float, default=0.001)
    train.add_argument("--temperature-start", type=float, default=1.0)
    train.add_argument("--temperature-end", type=float, default=0.1)
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--device", default="auto")

    build = subparsers.add_parser("build", help="Build a multi-index code index.")
    build.add_argument("--embedding-file", required=True)
    build.add_argument("--model-file", required=True)
    build.add_argument("--index-dir", required=True)
    build.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    build.add_argument("--normalize", action="store_true")
    build.add_argument("--probe-radius", type=int, choices=[0, 1, 2], default=1)
    build.add_argument("--hamming-candidates", type=int, default=2000)
    build.add_argument("--embedding-storage", choices=["float16", "float32"], default="float16")
    build.add_argument("--encode-batch-size", type=int, default=4096)
    build.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("eval", help="Evaluate an existing code-scan index.")
    evaluate.add_argument("--embedding-file", required=True)
    evaluate.add_argument("--index-dir", required=True)
    evaluate.add_argument("--top-k", type=int, default=50)
    evaluate.add_argument("--truth-k", type=int, default=50)
    evaluate.add_argument("--max-queries", type=int, default=0)
    evaluate.add_argument("--result-file", default=None)
    evaluate.add_argument("--no-mmap", action="store_true")
    evaluate.add_argument("--query-code-device", default="cpu")
    evaluate.add_argument(
        "--rerank-mode",
        choices=["full", "code-only", "small-rerank"],
        default="full",
        help=(
            "Final ranking mode: full keeps the original continuous rerank, "
            "code-only returns Hamming-ranked results, small-rerank reranks only "
            "a reduced Hamming candidate set."
        ),
    )
    evaluate.add_argument(
        "--small-rerank-candidates",
        type=int,
        default=None,
        help="Hamming candidates kept before continuous rerank when --rerank-mode=small-rerank.",
    )
    exclusion = evaluate.add_mutually_exclusive_group()
    exclusion.add_argument("--exclude-self", dest="exclude_self", action="store_true")
    exclusion.add_argument("--no-exclude-self", dest="exclude_self", action="store_false")
    evaluate.set_defaults(exclude_self=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "build":
        build_command(args)
    elif args.command == "eval":
        eval_command(args)


if __name__ == "__main__":
    main()
