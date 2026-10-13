"""Learn simplex-constrained weights for dual-metric candidate fusion."""

import json
import os
import os.path as osp

import numpy as np


BUCKETS = ("high_quality", "ordinary", "hard")
VARIANTS = ("rerank", "no_rerank")


def _normalized_weights(values):
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(weights) != 3 or not np.all(np.isfinite(weights)):
        raise ValueError("Dual-metric weights must contain three finite values.")
    if np.min(weights) < 0.0 or float(weights.sum()) <= 0.0:
        raise ValueError("Dual-metric weights must be non-negative and non-zero.")
    return (weights / weights.sum()).astype(np.float32)


def validate_weight_payload(payload, metric_name=None):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Learned dual-metric weight file must use version 1.")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Learned dual-metric weight file has no metrics mapping.")
    if metric_name is not None and metric_name not in metrics:
        raise ValueError(f"No learned dual-metric weights for {metric_name}.")
    for name, metric_config in metrics.items():
        if not isinstance(metric_config, dict):
            raise ValueError(f"Invalid learned weight configuration for {name}.")
        for variant in VARIANTS:
            variant_config = metric_config.get(variant)
            if not isinstance(variant_config, dict):
                raise ValueError(f"Missing {variant} learned weights for {name}.")
            for bucket in BUCKETS:
                _normalized_weights(variant_config.get(bucket))
    return payload


def load_weight_payload(path, metric_name=None):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return validate_weight_payload(payload, metric_name=metric_name)


def save_weight_payload(path, payload):
    validate_weight_payload(payload)
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _simplex_weights(logits, weight_floor):
    scale = 1.0 - 3.0 * float(weight_floor)
    return float(weight_floor) + scale * _softmax(logits)


def _sigmoid(values):
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _pair_arrays(samples, variant, top_k, max_pairs_per_query, seed):
    rng = np.random.default_rng(seed)
    deltas = []
    buckets = []
    for sample in samples:
        features = np.asarray(sample["rank_features"], dtype=np.float32)
        distances = np.asarray(sample["oracle_distances"], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 3:
            raise ValueError("rank_features must have shape [candidates, 3].")
        if len(features) != len(distances):
            raise ValueError("Rank features and oracle distances are not aligned.")
        finite_ids = np.flatnonzero(np.isfinite(distances))
        if len(finite_ids) <= top_k:
            continue
        oracle_order = finite_ids[np.argsort(distances[finite_ids], kind="stable")]
        positive_mask = np.zeros(len(features), dtype=bool)
        positive_mask[oracle_order[: min(top_k, len(oracle_order))]] = True
        eligible = np.isfinite(distances)
        if variant == "rerank":
            protected = np.asarray(sample["protected_mask"], dtype=bool)
            if len(protected) != len(features):
                raise ValueError("protected_mask is not aligned with rank features.")
            eligible &= ~protected
        positive_ids = np.flatnonzero(eligible & positive_mask)
        negative_ids = np.flatnonzero(eligible & ~positive_mask)
        if not len(positive_ids) or not len(negative_ids):
            continue
        pair_count = min(
            int(max_pairs_per_query), len(positive_ids) * len(negative_ids)
        )
        positive_choice = rng.choice(positive_ids, size=pair_count, replace=True)
        # Concentrate supervision near the oracle Top-K boundary.
        negative_order = negative_ids[np.argsort(distances[negative_ids], kind="stable")]
        negative_pool = negative_order[: min(len(negative_order), max(top_k * 2, 16))]
        negative_choice = rng.choice(negative_pool, size=pair_count, replace=True)
        deltas.append(features[positive_choice] - features[negative_choice])
        buckets.append(
            np.full(pair_count, BUCKETS.index(sample["bucket"]), dtype=np.int64)
        )
    if not deltas:
        raise ValueError(f"No valid {variant} training pairs were produced.")
    return np.concatenate(deltas), np.concatenate(buckets)


def _fit_variant(
    samples,
    variant,
    initial_weights,
    per_bucket,
    top_k,
    epochs,
    learning_rate,
    max_pairs_per_query,
    weight_floor,
    seed,
):
    deltas, bucket_ids = _pair_arrays(
        samples, variant, top_k, max_pairs_per_query, seed
    )
    generator = np.random.default_rng(seed + 17)
    order = generator.permutation(len(deltas))
    validation_count = int(round(len(order) * 0.2)) if len(order) >= 10 else 0
    validation_ids = order[:validation_count]
    training_ids = order[validation_count:]
    initial = _normalized_weights(initial_weights).astype(np.float64)
    rows = len(BUCKETS) if per_bucket else 1
    floor_scale = 1.0 - 3.0 * weight_floor
    initial_softmax = np.maximum((initial - weight_floor) / floor_scale, 1e-8)
    initial_softmax /= initial_softmax.sum()
    logits = np.repeat(np.log(initial_softmax)[None, :], rows, axis=0)
    first_moment = np.zeros_like(logits)
    second_moment = np.zeros_like(logits)
    best_logits = logits.copy()
    best_validation = float("inf")
    batch_size = min(4096, max(len(training_ids), 1))
    step = 0
    temperature = 0.10
    for _ in range(int(epochs)):
        shuffled = generator.permutation(training_ids)
        for start in range(0, len(shuffled), batch_size):
            ids = shuffled[start:start + batch_size]
            row_ids = bucket_ids[ids] if per_bucket else np.zeros(len(ids), dtype=np.int64)
            softmax_table = _softmax(logits)
            table = _simplex_weights(logits, weight_floor)
            weights = table[row_ids]
            score_delta = np.einsum("ij,ij->i", deltas[ids], weights)
            score_gradient = _sigmoid(score_delta / temperature) / temperature
            weight_gradients = np.zeros_like(logits)
            for row_id in range(rows):
                selected = row_ids == row_id
                if np.any(selected):
                    weight_gradients[row_id] = np.mean(
                        deltas[ids][selected] * score_gradient[selected, None],
                        axis=0,
                    )
            weight_gradients += 0.02 * (table - initial) / table.size
            scaled_gradients = (1.0 - 3.0 * weight_floor) * weight_gradients
            logit_gradients = softmax_table * (
                scaled_gradients
                - np.sum(
                    scaled_gradients * softmax_table, axis=1, keepdims=True
                )
            )
            step += 1
            first_moment = 0.9 * first_moment + 0.1 * logit_gradients
            second_moment = 0.999 * second_moment + 0.001 * logit_gradients ** 2
            first_unbiased = first_moment / (1.0 - 0.9 ** step)
            second_unbiased = second_moment / (1.0 - 0.999 ** step)
            logits -= learning_rate * first_unbiased / (
                np.sqrt(second_unbiased) + 1e-8
            )
        check_ids = validation_ids if len(validation_ids) else training_ids
        check_rows = (
            bucket_ids[check_ids]
            if per_bucket else np.zeros(len(check_ids), dtype=np.int64)
        )
        check_weights = _simplex_weights(logits, weight_floor)[check_rows]
        check_delta = np.einsum(
            "ij,ij->i", deltas[check_ids], check_weights
        )
        validation_loss = float(
            np.mean(np.logaddexp(0.0, check_delta / temperature))
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_logits = logits.copy()
    learned = _simplex_weights(best_logits, weight_floor).astype(np.float32)
    if not per_bucket:
        learned = np.repeat(learned, len(BUCKETS), axis=0)
    return {
        bucket: learned[position].astype(float).tolist()
        for position, bucket in enumerate(BUCKETS)
    }, {
        "pairs": int(len(deltas)),
        "validation_pairs": int(len(validation_ids)),
        "best_validation_loss": float(best_validation),
    }


def fit_weight_payload(
    samples_by_variant,
    metric_name,
    initial_weights,
    *,
    per_bucket,
    top_k=50,
    epochs=200,
    learning_rate=0.05,
    max_pairs_per_query=512,
    weight_floor=0.005,
    seed=2026,
):
    if not 0.0 <= float(weight_floor) < 1.0 / 3.0:
        raise ValueError("weight_floor must be in [0, 1/3).")
    metric_config = {}
    reports = {}
    fallback = {
        bucket: _normalized_weights(initial_weights).astype(float).tolist()
        for bucket in BUCKETS
    }
    for variant in VARIANTS:
        samples = samples_by_variant.get(variant, [])
        if samples:
            try:
                metric_config[variant], reports[variant] = _fit_variant(
                    samples,
                    variant,
                    initial_weights,
                    per_bucket,
                    top_k,
                    epochs,
                    learning_rate,
                    max_pairs_per_query,
                    weight_floor,
                    seed + VARIANTS.index(variant) * 1009,
                )
            except ValueError as error:
                metric_config[variant] = fallback
                reports[variant] = {
                    "pairs": 0,
                    "fallback": True,
                    "reason": str(error),
                }
        else:
            metric_config[variant] = fallback
            reports[variant] = {"pairs": 0, "fallback": True}
    payload = {
        "version": 1,
        "mode": "learned-bucket" if per_bucket else "learned-global",
        "metrics": {metric_name: metric_config},
        "training": {
            "top_k": int(top_k),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "max_pairs_per_query": int(max_pairs_per_query),
            "weight_floor": float(weight_floor),
            "queries": {
                key: int(len(value)) for key, value in samples_by_variant.items()
            },
            "reports": reports,
        },
    }
    return validate_weight_payload(payload, metric_name=metric_name)
