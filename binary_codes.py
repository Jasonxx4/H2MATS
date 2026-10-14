import json
import os
import os.path as osp
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _load_hnswlib():
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError(
            "The hnsw code backend requires hnswlib. "
            "Install it with 'pip install hnswlib'."
        ) from exc
    return hnswlib


def _binary_vectors(codes):
    """Represent packed bits as 0/1 floats; squared L2 equals Hamming."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError("codes must be a 2-D packed-code matrix.")
    return np.unpackbits(codes, axis=1, bitorder="little").astype(np.float32)


@dataclass
class CodeEncoderConfig:
    input_dim: int
    bits: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1

    def validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.bits <= 0 or self.bits % 16 != 0:
            raise ValueError("bits must be a positive multiple of 16.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class BinaryCodeEncoder(nn.Module):
    """Post-hoc MLP that maps a continuous trajectory embedding to binary code bits."""

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer("input_mean", torch.zeros(config.input_dim))
        self.register_buffer("input_std", torch.ones(config.input_dim))
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.bits),
        )

    @torch.no_grad()
    def fit_input_normalization(self, embeddings):
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected embeddings with shape [N, {self.config.input_dim}], "
                f"got {tuple(embeddings.shape)}."
            )
        self.input_mean.copy_(embeddings.mean(dim=0).to(self.input_mean.device))
        std = embeddings.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.input_std.copy_(std.to(self.input_std.device))

    def forward(self, embeddings):
        embeddings = (embeddings - self.input_mean) / self.input_std
        return self.network(embeddings)

    def soft_codes(self, embeddings, temperature=1.0):
        return torch.tanh(self(embeddings) / max(float(temperature), 1e-4))


@dataclass
class CodeTrainConfig:
    metric: str = "l2"
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    positive_pool: int = 10
    positives_per_anchor: int = 1
    negatives_per_anchor: int = 1
    negative_strategy: str = "random"
    hard_negative_pool: int = 0
    hard_negative_fraction: float = 0.5
    metric_distance_weight: float = 0.0
    metric_distance_scale: float = 0.0
    embedding_distance_weight: float = 1.0
    filter_same_source_negatives: bool = False
    in_batch_negative_weight: float = 0.0
    graded_rank_weight: float = 0.0
    graded_rank_margin: float = 0.05
    graded_rank_cutoffs: tuple = (10, 20, 50)
    cutoff_rank_weight: float = 0.0
    cutoff_rank: int = 192
    cutoff_rank_margin: float = 0.02
    cutoff_rank_window: int = 64
    cutoff_samples: int = 4
    margin: float = 0.1
    ranking_weight: float = 1.0
    quantization_weight: float = 0.1
    balance_weight: float = 0.01
    independence_weight: float = 0.001
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    validation_fraction: float = 0.1
    seed: int = 2026
    device: str = "auto"

    def validate(self):
        if self.metric not in {"l2", "cosine"}:
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.positive_pool <= 0:
            raise ValueError("positive_pool must be positive.")
        if self.positives_per_anchor <= 0 or self.negatives_per_anchor <= 0:
            raise ValueError("positive/negative samples per anchor must be positive.")
        if self.negative_strategy not in {"random", "metric-hard", "mixed"}:
            raise ValueError(
                "negative_strategy must be random, metric-hard, or mixed."
            )
        if self.hard_negative_pool < 0:
            raise ValueError("hard_negative_pool cannot be negative.")
        if not 0.0 <= self.hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0, 1].")
        if self.metric_distance_weight < 0.0:
            raise ValueError("metric_distance_weight cannot be negative.")
        if self.embedding_distance_weight < 0.0:
            raise ValueError("embedding_distance_weight cannot be negative.")
        if self.metric_distance_scale < 0.0:
            raise ValueError("metric_distance_scale cannot be negative.")
        if self.in_batch_negative_weight < 0.0:
            raise ValueError("in_batch_negative_weight cannot be negative.")
        if self.graded_rank_weight < 0.0:
            raise ValueError("graded_rank_weight cannot be negative.")
        if self.graded_rank_margin < 0.0:
            raise ValueError("graded_rank_margin cannot be negative.")
        cutoffs = tuple(int(value) for value in self.graded_rank_cutoffs)
        if len(cutoffs) < 2 or any(value <= 0 for value in cutoffs):
            raise ValueError("graded_rank_cutoffs must contain at least two positives.")
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise ValueError("graded_rank_cutoffs must be strictly increasing.")
        self.graded_rank_cutoffs = cutoffs
        if self.cutoff_rank_weight < 0.0:
            raise ValueError("cutoff_rank_weight cannot be negative.")
        if self.cutoff_rank <= 0:
            raise ValueError("cutoff_rank must be positive.")
        if self.cutoff_rank_margin < 0.0:
            raise ValueError("cutoff_rank_margin cannot be negative.")
        if self.cutoff_rank_window <= 0:
            raise ValueError("cutoff_rank_window must be positive.")
        if self.cutoff_samples <= 0:
            raise ValueError("cutoff_samples must be positive.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")


def _resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def _as_float32_matrix(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got {embeddings.shape}.")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Embedding matrix must be non-empty, got {embeddings.shape}.")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain NaN or infinity.")
    return embeddings


def _valid_positive_lists(labels, count, positive_pool):
    labels = np.asarray(labels)
    if labels.ndim != 2 or len(labels) != count:
        raise ValueError(f"labels must have shape [N, K] aligned with embeddings, got {labels.shape}.")
    positives = []
    valid_anchors = []
    for anchor, row in enumerate(labels):
        row = np.asarray(row, dtype=np.int64).reshape(-1)
        row = row[(row >= 0) & (row < count) & (row != anchor)]
        if len(row):
            row = np.asarray(list(dict.fromkeys(int(item) for item in row)), dtype=np.int64)
            row = row[:positive_pool]
            positives.append(row)
            valid_anchors.append(anchor)
    if not valid_anchors:
        raise ValueError("No valid positive IDs were found in labels.")
    return np.asarray(valid_anchors, dtype=np.int64), positives


def _prepare_metric_lists(
    labels, count, positive_pool, metric_distances=None, anchor_count=None
):
    """Keep metric-ranked IDs and optional distances aligned after filtering."""
    labels = np.asarray(labels, dtype=np.int64)
    anchor_count = count if anchor_count is None else int(anchor_count)
    if anchor_count <= 0 or anchor_count > count:
        raise ValueError(
            f"anchor_count must be in [1, {count}], got {anchor_count}."
        )
    if labels.ndim != 2 or len(labels) != anchor_count:
        raise ValueError(
            "labels must have shape [anchor_count, K]; "
            f"got {labels.shape} for anchor_count={anchor_count}."
        )
    if metric_distances is not None:
        metric_distances = np.asarray(metric_distances, dtype=np.float32)
        if metric_distances.shape != labels.shape:
            raise ValueError(
                "metric_distances must have the same shape as labels; "
                f"got {metric_distances.shape} and {labels.shape}."
            )

    anchors, positives, ranked = [], [], []
    positive_distances, ranked_distances = [], []
    for anchor, row in enumerate(labels):
        seen = set()
        row_ids, row_distances = [], []
        for column, item in enumerate(row):
            item = int(item)
            if item < 0 or item >= count or item == anchor or item in seen:
                continue
            seen.add(item)
            row_ids.append(item)
            row_distances.append(
                float(metric_distances[anchor, column])
                if metric_distances is not None
                else np.nan
            )
        if row_ids:
            ranked_ids = np.asarray(row_ids, dtype=np.int64)
            ranked_dist = np.asarray(row_distances, dtype=np.float32)
            if metric_distances is not None:
                # The label file may contain the correct candidate set without
                # being ordered by the supplied trajectory metric.  Make TP
                # (or the selected metric) the authoritative teacher order.
                metric_order = np.argsort(
                    np.where(np.isfinite(ranked_dist), ranked_dist, np.inf),
                    kind="stable",
                )
                ranked_ids = ranked_ids[metric_order]
                ranked_dist = ranked_dist[metric_order]
            anchors.append(anchor)
            positives.append(ranked_ids[:positive_pool])
            ranked.append(ranked_ids)
            positive_distances.append(ranked_dist[:positive_pool])
            ranked_distances.append(ranked_dist)
    if not anchors:
        raise ValueError("No valid positive IDs were found in labels.")
    return (
        np.asarray(anchors, dtype=np.int64),
        positives,
        ranked,
        positive_distances,
        ranked_distances,
    )


def _sample_triplets(
    anchor_positions,
    anchors,
    positives,
    ranked,
    positive_distances,
    ranked_distances,
    count,
    rng,
    positives_per_anchor=1,
    negatives_per_anchor=1,
    negative_strategy="random",
    hard_negative_pool=0,
    hard_negative_fraction=0.5,
    source_ids=None,
):
    selected_anchors = anchors[anchor_positions]
    selected_positives = np.empty(
        (len(selected_anchors), positives_per_anchor), dtype=np.int64
    )
    selected_negatives = np.empty(
        (len(selected_anchors), negatives_per_anchor), dtype=np.int64
    )
    selected_positive_distances = np.full(selected_positives.shape, np.nan, dtype=np.float32)
    selected_negative_distances = np.full(selected_negatives.shape, np.nan, dtype=np.float32)
    all_ids = np.arange(count, dtype=np.int64)
    for out_idx, position in enumerate(anchor_positions):
        positive_ids = positives[int(position)]
        positive_dist = positive_distances[int(position)]
        positive_choices = rng.integers(len(positive_ids), size=positives_per_anchor)
        selected_positives[out_idx] = positive_ids[positive_choices]
        selected_positive_distances[out_idx] = positive_dist[positive_choices]
        anchor = int(anchors[position])
        forbidden = set(int(item) for item in positive_ids)
        forbidden.add(anchor)
        if source_ids is not None:
            forbidden.update(
                int(item) for item in np.flatnonzero(source_ids == source_ids[anchor])
            )
        ranked_ids = ranked[int(position)]
        ranked_dist = ranked_distances[int(position)]
        forbidden_array = np.fromiter(forbidden, dtype=np.int64)
        hard_mask = ~np.isin(ranked_ids, forbidden_array)
        hard_ids = ranked_ids[hard_mask]
        hard_dist = ranked_dist[hard_mask]
        if hard_negative_pool > 0:
            hard_ids = hard_ids[:hard_negative_pool]
            hard_dist = hard_dist[:hard_negative_pool]
        random_forbidden = forbidden
        if negative_strategy != "random":
            random_forbidden = forbidden.union(int(item) for item in ranked_ids)
        allowed = all_ids[
            ~np.isin(all_ids, np.fromiter(random_forbidden, dtype=np.int64))
        ]
        if len(allowed) == 0 and len(hard_ids) == 0:
            allowed = all_ids[~np.isin(all_ids, forbidden_array)]
        if len(allowed) == 0 and len(hard_ids) == 0:
            raise ValueError("Cannot sample a random metric negative from this embedding set.")

        for negative_column in range(negatives_per_anchor):
            use_hard = (
                len(hard_ids) > 0
                and (
                    negative_strategy == "metric-hard"
                    or len(allowed) == 0
                    or (
                        negative_strategy == "mixed"
                        and rng.random() < hard_negative_fraction
                    )
                )
            )
            if use_hard:
                hard_index = int(rng.integers(len(hard_ids)))
                selected_negatives[out_idx, negative_column] = hard_ids[hard_index]
                selected_negative_distances[out_idx, negative_column] = hard_dist[hard_index]
            else:
                selected_negatives[out_idx, negative_column] = allowed[
                    rng.integers(len(allowed))
                ]
    return (
        selected_anchors,
        selected_positives,
        selected_negatives,
        selected_positive_distances,
        selected_negative_distances,
    )


def _estimate_l2_scale(embeddings, seed, sample_size=10000):
    if len(embeddings) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    size = min(sample_size, max(len(embeddings), 2))
    left = rng.integers(len(embeddings), size=size)
    right = rng.integers(len(embeddings), size=size)
    distances = np.linalg.norm(embeddings[left] - embeddings[right], axis=1)
    positive = distances[distances > 1e-12]
    return max(float(np.median(positive)) if len(positive) else 1.0, 1e-6)


def _teacher_distance(left, right, metric, l2_scale):
    if metric == "cosine":
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        return ((1.0 - torch.sum(left * right, dim=-1)) * 0.5).clamp(0.0, 1.0)
    distance = torch.linalg.vector_norm(left - right, dim=-1)
    return 1.0 - torch.exp(-distance / max(float(l2_scale), 1e-6))


def _soft_hamming(left, right):
    bits = left.shape[-1]
    return ((bits - torch.sum(left * right, dim=-1)) / (2.0 * bits)).clamp(0.0, 1.0)


def semantic_code_loss(
    anchor,
    positive,
    negative,
    teacher_anchor,
    teacher_positive,
    teacher_negative,
    train_config,
    l2_scale,
    metric_positive_distance=None,
    metric_negative_distance=None,
    metric_distance_scale=1.0,
    graded_rank_codes=None,
    cutoff_positive_codes=None,
    cutoff_negative_codes=None,
):
    if positive.ndim == 2:
        positive = positive.unsqueeze(1)
        teacher_positive = teacher_positive.unsqueeze(1)
        if metric_positive_distance is not None:
            metric_positive_distance = metric_positive_distance.unsqueeze(1)
    if negative.ndim == 2:
        negative = negative.unsqueeze(1)
        teacher_negative = teacher_negative.unsqueeze(1)
        if metric_negative_distance is not None:
            metric_negative_distance = metric_negative_distance.unsqueeze(1)
    positive_code_distance = _soft_hamming(anchor.unsqueeze(1), positive)
    negative_code_distance = _soft_hamming(anchor.unsqueeze(1), negative)
    positive_target = _teacher_distance(
        teacher_anchor.unsqueeze(1), teacher_positive, train_config.metric, l2_scale
    )
    negative_target = _teacher_distance(
        teacher_anchor.unsqueeze(1), teacher_negative, train_config.metric, l2_scale
    )

    embedding_semantic = F.mse_loss(positive_code_distance, positive_target)
    embedding_semantic = embedding_semantic + F.mse_loss(
        negative_code_distance, negative_target
    )
    semantic = train_config.embedding_distance_weight * embedding_semantic
    metric_semantic = semantic.new_zeros(())
    if (
        train_config.metric_distance_weight > 0.0
        and metric_positive_distance is not None
        and metric_negative_distance is not None
    ):
        scale = max(float(metric_distance_scale), 1e-6)
        positive_metric_target = 1.0 - torch.exp(
            -torch.nan_to_num(metric_positive_distance, nan=0.0).clamp_min(0.0) / scale
        )
        negative_metric_target = 1.0 - torch.exp(
            -torch.nan_to_num(metric_negative_distance, nan=0.0).clamp_min(0.0) / scale
        )
        positive_mask = torch.isfinite(metric_positive_distance)
        negative_mask = torch.isfinite(metric_negative_distance)
        if positive_mask.any():
            metric_semantic = metric_semantic + F.smooth_l1_loss(
                positive_code_distance[positive_mask],
                positive_metric_target[positive_mask],
            )
        if negative_mask.any():
            metric_semantic = metric_semantic + F.smooth_l1_loss(
                negative_code_distance[negative_mask],
                negative_metric_target[negative_mask],
            )
        semantic = semantic + train_config.metric_distance_weight * metric_semantic

    hardest_positive = positive_code_distance.max(dim=1).values
    hardest_negative = negative_code_distance.min(dim=1).values
    ranking = F.relu(
        train_config.margin + hardest_positive - hardest_negative
    ).mean()
    graded_rank = ranking.new_zeros(())
    if train_config.graded_rank_weight > 0.0 and graded_rank_codes is not None:
        tier_distances = _soft_hamming(
            anchor.unsqueeze(1), graded_rank_codes
        )
        tier_losses = []
        for left in range(tier_distances.shape[1] - 1):
            for right in range(left + 1, tier_distances.shape[1]):
                tier_losses.append(
                    F.relu(
                        train_config.graded_rank_margin
                        + tier_distances[:, left]
                        - tier_distances[:, right]
                    )
                )
        if tier_losses:
            graded_rank = torch.stack(tier_losses, dim=0).mean()
    cutoff_rank = ranking.new_zeros(())
    if (
        train_config.cutoff_rank_weight > 0.0
        and cutoff_positive_codes is not None
        and cutoff_negative_codes is not None
    ):
        cutoff_positive_distance = _soft_hamming(
            anchor.unsqueeze(1), cutoff_positive_codes
        )
        cutoff_negative_distance = _soft_hamming(
            anchor.unsqueeze(1), cutoff_negative_codes
        )
        cutoff_rank = F.relu(
            train_config.cutoff_rank_margin
            + cutoff_positive_distance.max(dim=1).values
            - cutoff_negative_distance.min(dim=1).values
        ).mean()
    codes = torch.cat(
        (
            anchor,
            positive.reshape(-1, positive.shape[-1]),
            negative.reshape(-1, negative.shape[-1]),
        ),
        dim=0,
    )
    quantization = torch.square(torch.abs(codes) - 1.0).mean()
    balance = torch.square(codes.mean(dim=0)).mean()
    correlation = codes.transpose(0, 1) @ codes / max(codes.shape[0], 1)
    identity = torch.eye(correlation.shape[0], device=correlation.device, dtype=correlation.dtype)
    independence = torch.square(correlation - identity).mean()
    total = (
        semantic
        + train_config.ranking_weight * ranking
        + train_config.graded_rank_weight * graded_rank
        + train_config.cutoff_rank_weight * cutoff_rank
        + train_config.quantization_weight * quantization
        + train_config.balance_weight * balance
        + train_config.independence_weight * independence
    )
    return total, {
        "semantic": semantic.detach(),
        "embedding_semantic": embedding_semantic.detach(),
        "metric_semantic": metric_semantic.detach(),
        "ranking": ranking.detach(),
        "graded_rank": graded_rank.detach(),
        "cutoff_rank": cutoff_rank.detach(),
        "quantization": quantization.detach(),
        "balance": balance.detach(),
        "independence": independence.detach(),
    }


def train_code_encoder(
    embeddings,
    labels,
    head_config,
    train_config=None,
    metric_distances=None,
    source_ids=None,
    anchor_count=None,
):
    """Train a code encoder while keeping the upstream trajectory encoder frozen."""
    train_config = train_config or CodeTrainConfig()
    train_config.validate()
    embeddings = _as_float32_matrix(embeddings)
    anchor_count = len(embeddings) if anchor_count is None else int(anchor_count)
    if embeddings.shape[1] != head_config.input_dim:
        raise ValueError(
            f"Code encoder input_dim={head_config.input_dim} does not match embeddings "
            f"with dimension {embeddings.shape[1]}."
        )
    if metric_distances is not None:
        metric_distances = np.asarray(metric_distances, dtype=np.float32)
        labels_array = np.asarray(labels, dtype=np.int64)
        if metric_distances.shape == (len(embeddings), len(embeddings)):
            valid = (labels_array >= 0) & (labels_array < len(embeddings))
            safe_labels = np.where(valid, labels_array, 0)
            metric_distances = np.take_along_axis(
                metric_distances, safe_labels, axis=1
            )
            metric_distances = np.where(valid, metric_distances, np.nan).astype(
                np.float32, copy=False
            )
        elif metric_distances.shape != labels_array.shape:
            raise ValueError(
                "metric_distances must have shape [N, K] aligned with labels or "
                f"a dense [N, N] table; got {metric_distances.shape}."
            )
        metric_distances = np.where(
            np.isfinite(metric_distances) & (metric_distances >= 0.0),
            metric_distances,
            np.nan,
        ).astype(np.float32, copy=False)
    elif train_config.metric_distance_weight > 0.0:
        raise ValueError(
            "metric_distance_weight requires metric distances aligned with labels. "
            "Store them in the embedding NPZ or pass --metric-distance-file."
        )
    if source_ids is not None:
        source_ids = np.asarray(source_ids).reshape(-1)
        if len(source_ids) != len(embeddings):
            raise ValueError("source_ids must have one entry per embedding.")
    (
        anchors,
        positives,
        ranked,
        positive_distances,
        ranked_distances,
    ) = _prepare_metric_lists(
        labels,
        len(embeddings),
        train_config.positive_pool,
        metric_distances,
        anchor_count=anchor_count,
    )
    positive_sets = {
        int(anchor): set(int(item) for item in positive_ids)
        for anchor, positive_ids in zip(anchors, positives)
    }
    positions_by_anchor = {
        int(anchor): position for position, anchor in enumerate(anchors)
    }

    def sample_graded_ids(anchor_ids, local_rng):
        if train_config.graded_rank_weight <= 0.0:
            return None
        tier_count = len(train_config.graded_rank_cutoffs)
        sampled = np.empty((len(anchor_ids), tier_count), dtype=np.int64)
        for row, anchor_id in enumerate(anchor_ids):
            anchor_position = positions_by_anchor[int(anchor_id)]
            ranked_ids = ranked[anchor_position]
            ranked_metric_distances = ranked_distances[anchor_position]
            if metric_distances is not None:
                # Unreachable TP pairs are represented as NaN.  They do not
                # define a meaningful rank tier, so remove them while keeping
                # the finite TP order and let the final tier be shorter when
                # necessary (e.g. 49 valid candidates for a Top-50 label).
                finite = np.isfinite(ranked_metric_distances)
                ranked_ids = ranked_ids[finite]
            start = 0
            for tier, stop in enumerate(train_config.graded_rank_cutoffs):
                stop = min(int(stop), len(ranked_ids))
                tier_ids = ranked_ids[start:stop]
                if len(tier_ids) == 0:
                    raise ValueError(
                        "graded rank supervision requires labels covering every "
                        f"configured tier {train_config.graded_rank_cutoffs}; "
                        f"anchor {anchor_id} has only {len(ranked_ids)} valid labels."
                    )
                sampled[row, tier] = tier_ids[local_rng.integers(len(tier_ids))]
                start = stop
        return sampled

    def sample_cutoff_ids(anchor_ids, local_rng):
        if train_config.cutoff_rank_weight <= 0.0:
            return None
        positive = np.empty(
            (len(anchor_ids), train_config.cutoff_samples), dtype=np.int64
        )
        negative = np.empty_like(positive)
        boundary = int(train_config.cutoff_rank)
        window = int(train_config.cutoff_rank_window)
        for row, anchor_id in enumerate(anchor_ids):
            anchor_position = positions_by_anchor[int(anchor_id)]
            ranked_ids = ranked[anchor_position]
            ranked_metric_distances = ranked_distances[anchor_position]
            finite = np.isfinite(ranked_metric_distances)
            ranked_ids = ranked_ids[finite]
            keep_ids = ranked_ids[max(0, boundary - window):boundary]
            reject_ids = ranked_ids[boundary:boundary + window]
            if len(keep_ids) == 0 or len(reject_ids) == 0:
                raise ValueError(
                    f"cutoff rank supervision at {boundary} requires candidates "
                    f"on both sides; anchor {anchor_id} has {len(ranked_ids)} "
                    "finite metric candidates."
                )
            positive[row] = keep_ids[
                local_rng.integers(len(keep_ids), size=train_config.cutoff_samples)
            ]
            negative[row] = reject_ids[
                local_rng.integers(len(reject_ids), size=train_config.cutoff_samples)
            ]
        return positive, negative

    rng = np.random.default_rng(train_config.seed)
    eligible_positions = np.arange(len(anchors), dtype=np.int64)
    cutoff_eligible_anchor_count = 0
    if train_config.cutoff_rank_weight > 0.0:
        if metric_distances is None:
            raise ValueError(
                "cutoff_rank_weight requires finite metric distances on union candidates."
            )
        eligible_positions = np.asarray(
            [
                position
                for position, distances in enumerate(ranked_distances)
                if np.count_nonzero(np.isfinite(distances)) > train_config.cutoff_rank
            ],
            dtype=np.int64,
        )
        cutoff_eligible_anchor_count = len(eligible_positions)
        if not cutoff_eligible_anchor_count:
            raise ValueError(
                "No anchors contain candidates on both sides of cutoff rank "
                f"{train_config.cutoff_rank}."
            )
    order = rng.permutation(eligible_positions)
    validation_size = int(round(len(order) * train_config.validation_fraction))
    if validation_size and len(order) - validation_size == 0:
        validation_size = 0
    validation_positions = order[:validation_size]
    training_positions = order[validation_size:]
    if len(training_positions) == 0:
        raise ValueError("No training anchors remain after the validation split.")

    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    device = _resolve_device(train_config.device)
    head = BinaryCodeEncoder(head_config)
    head.fit_input_normalization(embeddings)
    head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    l2_scale = _estimate_l2_scale(embeddings, train_config.seed)
    finite_metric_distances = (
        metric_distances[np.isfinite(metric_distances)]
        if metric_distances is not None
        else np.empty(0, dtype=np.float32)
    )
    metric_distance_scale = train_config.metric_distance_scale
    if metric_distance_scale <= 0.0 and len(finite_metric_distances):
        positive_metric_distances = finite_metric_distances[finite_metric_distances > 1e-12]
        metric_distance_scale = (
            float(np.median(positive_metric_distances))
            if len(positive_metric_distances)
            else 1.0
        )
    if metric_distance_scale <= 0.0:
        metric_distance_scale = 1.0
    best_state = None
    best_validation = float("inf")
    history = []

    fixed_validation = None
    fixed_validation_graded = None
    fixed_validation_cutoff = None
    if len(validation_positions):
        fixed_validation = _sample_triplets(
            validation_positions,
            anchors,
            positives,
            ranked,
            positive_distances,
            ranked_distances,
            len(embeddings),
            np.random.default_rng(train_config.seed + 1),
            positives_per_anchor=train_config.positives_per_anchor,
            negatives_per_anchor=train_config.negatives_per_anchor,
            negative_strategy=train_config.negative_strategy,
            hard_negative_pool=train_config.hard_negative_pool,
            hard_negative_fraction=train_config.hard_negative_fraction,
            source_ids=(source_ids if train_config.filter_same_source_negatives else None),
        )
        fixed_validation_graded = sample_graded_ids(
            fixed_validation[0], np.random.default_rng(train_config.seed + 2)
        )
        fixed_validation_cutoff = sample_cutoff_ids(
            fixed_validation[0], np.random.default_rng(train_config.seed + 3)
        )

    def compute_batch(
        batch_ids, temperature, update, graded_ids=None, cutoff_ids=None
    ):
        (
            anchor_ids,
            positive_ids,
            negative_ids,
            positive_metric_distances,
            negative_metric_distances,
        ) = batch_ids
        teacher_anchor = torch.as_tensor(embeddings[anchor_ids], device=device)
        teacher_positive = torch.as_tensor(embeddings[positive_ids], device=device)
        teacher_negative = torch.as_tensor(embeddings[negative_ids], device=device)
        soft_anchor = head.soft_codes(teacher_anchor, temperature)
        soft_positive = head.soft_codes(teacher_positive, temperature)
        soft_negative = head.soft_codes(teacher_negative, temperature)
        graded_codes = None
        if graded_ids is not None:
            graded_embeddings = torch.as_tensor(embeddings[graded_ids], device=device)
            graded_codes = head.soft_codes(graded_embeddings, temperature)
        cutoff_positive_codes = None
        cutoff_negative_codes = None
        if cutoff_ids is not None:
            cutoff_positive_embeddings = torch.as_tensor(
                embeddings[cutoff_ids[0]], device=device
            )
            cutoff_negative_embeddings = torch.as_tensor(
                embeddings[cutoff_ids[1]], device=device
            )
            cutoff_positive_codes = head.soft_codes(
                cutoff_positive_embeddings, temperature
            )
            cutoff_negative_codes = head.soft_codes(
                cutoff_negative_embeddings, temperature
            )
        positive_metric_distances = torch.as_tensor(
            positive_metric_distances, dtype=torch.float32, device=device
        )
        negative_metric_distances = torch.as_tensor(
            negative_metric_distances, dtype=torch.float32, device=device
        )
        loss, parts = semantic_code_loss(
            soft_anchor,
            soft_positive,
            soft_negative,
            teacher_anchor,
            teacher_positive,
            teacher_negative,
            train_config,
            l2_scale,
            positive_metric_distances,
            negative_metric_distances,
            metric_distance_scale,
            graded_codes,
            cutoff_positive_codes,
            cutoff_negative_codes,
        )
        if train_config.in_batch_negative_weight > 0.0 and len(anchor_ids) > 1:
            pairwise_distance = _soft_hamming(
                soft_anchor.unsqueeze(1), soft_anchor.unsqueeze(0)
            )
            valid = np.ones((len(anchor_ids), len(anchor_ids)), dtype=bool)
            for row, anchor_id in enumerate(anchor_ids):
                known_positives = positive_sets.get(int(anchor_id), set())
                for column, candidate_id in enumerate(anchor_ids):
                    if row == column or int(candidate_id) in known_positives:
                        valid[row, column] = False
                    elif (
                        train_config.filter_same_source_negatives
                        and source_ids is not None
                        and source_ids[int(anchor_id)] == source_ids[int(candidate_id)]
                    ):
                        valid[row, column] = False
            valid = torch.as_tensor(valid, device=device)
            valid_rows = valid.any(dim=1)
            if valid_rows.any():
                masked_distance = pairwise_distance.masked_fill(~valid, float("inf"))
                in_batch_negative = masked_distance.min(dim=1).values
                positive_distance = _soft_hamming(
                    soft_anchor.unsqueeze(1), soft_positive
                ).max(dim=1).values
                in_batch_loss = F.relu(
                    train_config.margin
                    + positive_distance[valid_rows]
                    - in_batch_negative[valid_rows]
                ).mean()
                loss = loss + train_config.in_batch_negative_weight * in_batch_loss
                parts["in_batch_negative"] = in_batch_loss.detach()
        if update:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return float(loss.detach().cpu()), {key: float(value.cpu()) for key, value in parts.items()}

    for epoch in range(train_config.epochs):
        fraction = epoch / max(train_config.epochs - 1, 1)
        temperature = train_config.temperature_start * (
            train_config.temperature_end / train_config.temperature_start
        ) ** fraction
        head.train()
        shuffled = rng.permutation(training_positions)
        epoch_losses = []
        epoch_parts = []
        for start in range(0, len(shuffled), train_config.batch_size):
            positions = shuffled[start:start + train_config.batch_size]
            triplets = _sample_triplets(
                positions,
                anchors,
                positives,
                ranked,
                positive_distances,
                ranked_distances,
                len(embeddings),
                rng,
                positives_per_anchor=train_config.positives_per_anchor,
                negatives_per_anchor=train_config.negatives_per_anchor,
                negative_strategy=train_config.negative_strategy,
                hard_negative_pool=train_config.hard_negative_pool,
                hard_negative_fraction=train_config.hard_negative_fraction,
                source_ids=(source_ids if train_config.filter_same_source_negatives else None),
            )
            graded_ids = sample_graded_ids(triplets[0], rng)
            cutoff_ids = sample_cutoff_ids(triplets[0], rng)
            loss, parts = compute_batch(
                triplets,
                temperature,
                update=True,
                graded_ids=graded_ids,
                cutoff_ids=cutoff_ids,
            )
            epoch_losses.append(loss)
            epoch_parts.append(parts)

        validation_loss = float(np.mean(epoch_losses))
        validation_parts = epoch_parts
        if fixed_validation is not None:
            head.eval()
            validation_losses = []
            validation_parts = []
            with torch.no_grad():
                for start in range(0, len(fixed_validation[0]), train_config.batch_size):
                    batch = tuple(
                        item[start:start + train_config.batch_size] for item in fixed_validation
                    )
                    graded_ids = (
                        fixed_validation_graded[start:start + train_config.batch_size]
                        if fixed_validation_graded is not None else None
                    )
                    cutoff_ids = (
                        tuple(
                            item[start:start + train_config.batch_size]
                            for item in fixed_validation_cutoff
                        )
                        if fixed_validation_cutoff is not None else None
                    )
                    loss, parts = compute_batch(
                        batch,
                        temperature,
                        update=False,
                        graded_ids=graded_ids,
                        cutoff_ids=cutoff_ids,
                    )
                    validation_losses.append(loss)
                    validation_parts.append(parts)
            validation_loss = float(np.mean(validation_losses))
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        train_part_keys = set().union(*(parts.keys() for parts in epoch_parts))
        validation_part_keys = set().union(
            *(parts.keys() for parts in validation_parts)
        )
        train_part_means = {
            f"train_{key}": float(np.mean([
                parts[key] for parts in epoch_parts if key in parts
            ]))
            for key in train_part_keys
        }
        validation_part_means = {
            f"validation_{key}": float(np.mean([
                parts[key] for parts in validation_parts if key in parts
            ]))
            for key in validation_part_keys
        }
        history.append({
            "epoch": epoch + 1,
            "temperature": float(temperature),
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
            **train_part_means,
            **validation_part_means,
        })

    if best_state is not None:
        head.load_state_dict(best_state)
    head.cpu().eval()
    report = {
        "head_config": asdict(head_config),
        "train_config": asdict(train_config),
        "embedding_count": int(len(embeddings)),
        "anchor_count": int(anchor_count),
        "training_anchor_count": int(len(training_positions)),
        "validation_anchor_count": int(len(validation_positions)),
        "cutoff_eligible_anchor_count": int(cutoff_eligible_anchor_count),
        "l2_distance_scale": float(l2_scale),
        "metric_distance_scale": float(metric_distance_scale),
        "metric_distance_count": int(len(finite_metric_distances)),
        "best_validation_loss": float(best_validation),
        "history": history,
    }
    return head, report


def save_code_encoder(head, path, metadata=None):
    out_dir = osp.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "config": asdict(head.config),
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_code_encoder(path, map_location="cpu"):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid code-encoder checkpoint: {path}")
    head = BinaryCodeEncoder(CodeEncoderConfig(**checkpoint["config"]))
    head.load_state_dict(checkpoint["state_dict"])
    head.to(map_location).eval()
    return head, checkpoint.get("metadata", {})


def pack_binary_codes(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"Expected a 2-D code matrix, got {codes.shape}.")
    if codes.shape[1] % 16 != 0:
        raise ValueError("Code length must be a multiple of 16 bits.")
    bits = codes > 0
    return np.packbits(bits, axis=1, bitorder="little")


def hamming_distances(query_code, codes):
    query_code = np.asarray(query_code, dtype=np.uint8).reshape(1, -1)
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != query_code.shape[1]:
        raise ValueError(
            f"Packed code shapes do not match: query={query_code.shape}, gallery={codes.shape}."
        )
    return _POPCOUNT[np.bitwise_xor(codes, query_code)].sum(axis=1, dtype=np.int32)


def encode_binary_codes(head, embeddings, batch_size=4096, device="auto"):
    embeddings = _as_float32_matrix(embeddings)
    if embeddings.shape[1] != head.config.input_dim:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match code encoder "
            f"input dimension {head.config.input_dim}."
        )
    device = _resolve_device(device)
    head = head.to(device).eval()
    packed = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.as_tensor(embeddings[start:start + batch_size], device=device)
            bits = head(batch) >= 0
            packed.append(pack_binary_codes(bits.cpu().numpy()))
    head.cpu()
    if not packed:
        return np.empty((0, head.config.bits // 8), dtype=np.uint8)
    return np.concatenate(packed, axis=0).astype(np.uint8, copy=False)


def encode_soft_codes(head, embeddings, batch_size=4096, device="auto", temperature=1.0):
    """Encode continuous tanh codes for a score-only prefilter."""
    embeddings = _as_float32_matrix(embeddings)
    if embeddings.shape[1] != head.config.input_dim:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match code encoder "
            f"input dimension {head.config.input_dim}."
        )
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    device = _resolve_device(device)
    head = head.to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.as_tensor(embeddings[start:start + batch_size], device=device)
            chunks.append(head.soft_codes(batch, temperature).cpu().numpy())
    head.cpu()
    if not chunks:
        return np.empty((0, head.config.bits), dtype=np.float16)
    return np.concatenate(chunks, axis=0).astype(np.float16, copy=False)


@dataclass
class BinaryCodeIndexConfig:
    bits: int = 128
    subcode_bits: int = 16
    probe_radius: int = 1
    hamming_candidates: int = 2000
    metric: str = "l2"
    normalize: bool = False
    embedding_storage: str = "float16"
    encode_batch_size: int = 4096
    backend: str = "mih"
    flat_block_size: int = 65536
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 256

    def validate(self):
        if self.bits <= 0 or self.bits % 16 != 0:
            raise ValueError("bits must be a positive multiple of 16.")
        if self.subcode_bits != 16:
            raise ValueError("The current MIH implementation requires subcode_bits=16.")
        if self.bits % self.subcode_bits != 0:
            raise ValueError("bits must be divisible by subcode_bits.")
        if self.probe_radius < 0 or self.probe_radius > 2:
            raise ValueError("probe_radius must be 0, 1, or 2.")
        if self.hamming_candidates <= 0:
            raise ValueError("hamming_candidates must be positive.")
        if self.metric not in {"l2", "cosine"}:
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.embedding_storage not in {"float16", "float32"}:
            raise ValueError("embedding_storage must be float16 or float32.")
        if self.backend not in {"mih", "hnsw", "flat"}:
            raise ValueError("backend must be 'mih', 'hnsw', or 'flat'.")
        if self.flat_block_size <= 0:
            raise ValueError("flat_block_size must be positive.")
        if self.hnsw_m <= 0 or self.hnsw_ef_construction <= 0 or self.hnsw_ef_search <= 0:
            raise ValueError("HNSW parameters must be positive.")


@lru_cache(maxsize=None)
def _xor_masks(bit_count, distance):
    if distance == 0:
        return (0,)
    masks = []
    for positions in combinations(range(bit_count), distance):
        mask = 0
        for position in positions:
            mask |= 1 << position
        masks.append(mask)
    return tuple(masks)


@lru_cache(maxsize=None)
def _probe_masks(bit_count, radius):
    return np.asarray(
        [
            mask
            for distance in range(int(radius) + 1)
            for mask in _xor_masks(int(bit_count), distance)
        ],
        dtype=np.uint16,
    )


def _packed_subcodes(packed_codes):
    packed_codes = np.asarray(packed_codes, dtype=np.uint8)
    if packed_codes.ndim == 1:
        packed_codes = packed_codes.reshape(1, -1)
    if packed_codes.ndim != 2 or packed_codes.shape[1] % 2 != 0:
        raise ValueError(f"Invalid packed-code shape: {packed_codes.shape}")
    pairs = packed_codes.reshape(packed_codes.shape[0], -1, 2).astype(np.uint16)
    return pairs[:, :, 0] | (pairs[:, :, 1] << 8)


def _topk_indices(distances, k):
    k = min(int(k), len(distances))
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k == len(distances):
        return np.argsort(distances, kind="stable")
    order = np.argpartition(distances, k - 1)[:k]
    return order[np.argsort(distances[order], kind="stable")]


class BinaryCodeIndex:
    """Binary-code candidate retrieval (MIH, HNSW, or exact binary scan) plus reranking."""

    def __init__(self, config=None):
        self.config = config or BinaryCodeIndexConfig()
        self.config.validate()
        self.head = None
        self.embeddings = None
        self.codes = None
        self.table_offsets = None
        self.table_postings = None
        self.hnsw_index = None
        self.metadata = {}
        self.build_seconds = 0.0

    def build(self, embeddings, head, encode_device="auto", metadata=None):
        start = perf_counter()
        embeddings = _as_float32_matrix(embeddings)
        if head.config.bits != self.config.bits:
            raise ValueError(
                f"Code encoder produces {head.config.bits} bits, index expects {self.config.bits}."
            )
        if embeddings.shape[1] != head.config.input_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match code encoder "
                f"input dimension {head.config.input_dim}."
            )
        raw_embeddings = embeddings
        storage_embeddings = embeddings
        if self.config.normalize or self.config.metric == "cosine":
            norm = np.linalg.norm(storage_embeddings, axis=1, keepdims=True)
            storage_embeddings = storage_embeddings / np.maximum(norm, 1e-12)
        storage_dtype = np.float16 if self.config.embedding_storage == "float16" else np.float32
        self.embeddings = storage_embeddings.astype(storage_dtype)
        self.head = head.cpu().eval()
        self.codes = encode_binary_codes(
            self.head,
            raw_embeddings,
            batch_size=self.config.encode_batch_size,
            device=encode_device,
        )
        self.table_offsets = self.table_postings = self.hnsw_index = None
        if self.config.backend == "hnsw":
            self._build_hnsw()
        elif self.config.backend == "mih":
            self._build_mih_tables()
        self.metadata = dict(metadata or {})
        self.build_seconds = perf_counter() - start
        return self

    def _build_mih_tables(self):
        subcodes = _packed_subcodes(self.codes)
        table_count = subcodes.shape[1]
        vector_count = len(subcodes)
        posting_dtype = np.int32 if vector_count <= np.iinfo(np.int32).max else np.int64
        self.table_offsets = np.empty((table_count, 65537), dtype=np.int64)
        self.table_postings = np.empty((table_count, vector_count), dtype=posting_dtype)
        for table_idx in range(table_count):
            keys = subcodes[:, table_idx]
            order = np.argsort(keys, kind="stable").astype(posting_dtype, copy=False)
            counts = np.bincount(keys, minlength=65536)
            self.table_offsets[table_idx, 0] = 0
            np.cumsum(counts, out=self.table_offsets[table_idx, 1:])
            self.table_postings[table_idx] = order

    def _build_hnsw(self):
        if len(self.codes) == 0:
            raise ValueError("Cannot build an HNSW code index with no vectors.")
        hnswlib = _load_hnswlib()
        index = hnswlib.Index(space="l2", dim=int(self.config.bits))
        index.init_index(
            max_elements=len(self.codes),
            ef_construction=int(self.config.hnsw_ef_construction),
            M=int(self.config.hnsw_m),
        )
        index.add_items(
            _binary_vectors(self.codes),
            np.arange(len(self.codes), dtype=np.int64),
        )
        index.set_ef(max(int(self.config.hnsw_ef_search), int(self.config.hamming_candidates)))
        self.hnsw_index = index
        self.table_offsets = None
        self.table_postings = None

    def _mih_candidates(self, query_code):
        query_subcodes = _packed_subcodes(query_code)[0]
        chunks = []
        for table_idx, subcode in enumerate(query_subcodes):
            offsets = self.table_offsets[table_idx]
            postings = self.table_postings[table_idx]
            for radius in range(self.config.probe_radius + 1):
                for mask in _xor_masks(self.config.subcode_bits, radius):
                    key = int(subcode) ^ mask
                    start, end = int(offsets[key]), int(offsets[key + 1])
                    if end > start:
                        chunks.append(postings[start:end])
        if not chunks:
            return np.empty((0,), dtype=np.int64)
        return np.unique(np.concatenate(chunks)).astype(np.int64, copy=False)

    def _encode_query(self, query_embedding):
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query_embedding.shape[1] != self.head.config.input_dim:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} does not match index dimension "
                f"{self.head.config.input_dim}."
            )
        with torch.no_grad():
            logits = self.head(torch.as_tensor(query_embedding))
        return pack_binary_codes((logits >= 0).numpy())[0]

    def encode_query_code(self, query_embedding):
        return self._encode_query(query_embedding)

    def _flat_hamming_search(self, query_code, candidate_limit, exclude_ids):
        count = len(self.codes)
        distances = np.empty(count, dtype=np.int32)
        for start in range(0, count, self.config.flat_block_size):
            stop = min(start + self.config.flat_block_size, count)
            distances[start:stop] = hamming_distances(query_code, self.codes[start:stop])
        valid_excluded = exclude_ids[(exclude_ids >= 0) & (exclude_ids < count)]
        distances[valid_excluded] = self.config.bits + 1
        k = min(candidate_limit, count - len(valid_excluded))
        ids = np.empty(0, dtype=np.int64)
        if k > 0:
            # Hamming codes often tie. Break boundary ties by Gallery row ID
            # so block size and save/load cannot change the candidate set.
            threshold = np.partition(distances, k - 1)[k - 1]
            below = np.flatnonzero(distances < threshold)
            equal = np.flatnonzero(distances == threshold)[:k - len(below)]
            ids = np.concatenate((below, equal)).astype(np.int64, copy=False)
            ids = ids[np.lexsort((ids, distances[ids]))]
        return ids, distances[ids], {
            "mih_candidate_count": 0,
            "flat_scan_count": int(count),
            "hamming_candidate_count": int(len(ids)),
            "used_full_scan": True,
        }

    def hamming_search(self, query_code, candidate_limit=None, minimum_candidates=1,
                       exclude_ids=None):
        """Retrieve candidates with the selected backend, ranked by Hamming distance."""
        if self.codes is None:
            raise RuntimeError("The code index has not been built or loaded.")
        query_code = np.asarray(query_code, dtype=np.uint8).reshape(-1)
        if len(query_code) != self.config.bits // 8:
            raise ValueError(
                f"Packed query code has {len(query_code)} bytes; expected "
                f"{self.config.bits // 8}."
            )
        candidate_limit = self.config.hamming_candidates if candidate_limit is None else int(candidate_limit)
        if candidate_limit <= 0 or minimum_candidates < 0:
            raise ValueError("candidate_limit must be positive and minimum_candidates non-negative.")

        if exclude_ids is None:
            exclude_ids = np.empty((0,), dtype=np.int64)
        else:
            if isinstance(exclude_ids, (set, frozenset)):
                exclude_ids = list(exclude_ids)
            exclude_ids = np.asarray(exclude_ids, dtype=np.int64).reshape(-1)
        exclude_ids = np.unique(exclude_ids)
        if self.config.backend == "flat":
            return self._flat_hamming_search(query_code, candidate_limit, exclude_ids)
        if self.config.backend == "hnsw":
            if self.hnsw_index is None:
                raise RuntimeError("The HNSW code index has not been built or loaded.")
            valid_excluded = exclude_ids[
                (exclude_ids >= 0) & (exclude_ids < len(self.codes))
            ]
            required = min(
                int(minimum_candidates),
                max(len(self.codes) - len(valid_excluded), 0),
            )
            request = min(
                len(self.codes),
                max(candidate_limit, int(minimum_candidates)) + len(exclude_ids),
            )
            query_vector = _binary_vectors(query_code.reshape(1, -1))
            search_k = max(request, 1)
            minimum_search_k = max(required + len(valid_excluded), 1)
            labels = None
            retry_count = 0
            last_error = None

            # hnswlib can fail to fill a large k when a binary-code graph has
            # many duplicate vectors or weakly connected regions.  Increasing
            # ef usually fixes the issue.  If it still cannot fill the request,
            # reduce k gradually while preserving the caller's required pool.
            while search_k >= minimum_search_k:
                base_ef = max(int(self.config.hnsw_ef_search), int(search_k))
                ef_values = []
                for value in (base_ef, base_ef * 2, base_ef * 4):
                    value = min(int(value), len(self.codes))
                    if value not in ef_values:
                        ef_values.append(value)
                for ef_value in ef_values:
                    self.hnsw_index.set_ef(ef_value)
                    try:
                        labels, _ = self.hnsw_index.knn_query(
                            query_vector, k=search_k
                        )
                        last_error = None
                        break
                    except RuntimeError as exc:
                        message = str(exc)
                        if (
                            "contiguous 2D array" not in message
                            and "ef or M is too small" not in message
                        ):
                            raise
                        last_error = exc
                        retry_count += 1
                if labels is not None:
                    break
                if search_k == minimum_search_k:
                    break
                reduced_k = max(minimum_search_k, int(search_k * 0.75))
                if reduced_k >= search_k:
                    reduced_k = search_k - 1
                search_k = max(reduced_k, minimum_search_k)

            if labels is None:
                raise RuntimeError(
                    "HNSW could not return the minimum required code candidate "
                    f"pool (required={required}, requested={request}). Rebuild "
                    "the index with larger --code-hnsw-m and "
                    "--code-hnsw-ef-construction."
                ) from last_error
            candidates = labels[0].astype(np.int64, copy=False)
            # Recompute exact packed Hamming distances for the small returned
            # pool. HNSW uses equivalent binary L2 only for graph navigation.
            hnsw_distances = hamming_distances(query_code, self.codes[candidates])
            if len(exclude_ids):
                keep = ~np.isin(candidates, exclude_ids)
                candidates = candidates[keep]
                hnsw_distances = hnsw_distances[keep]
            used_full_scan = len(candidates) < required
            if used_full_scan:
                # hnswlib returns at most N labels; this only handles pathological
                # exclusion requests and keeps the public contract intact.
                all_ids = np.arange(len(self.codes), dtype=np.int64)
                if len(exclude_ids):
                    all_ids = all_ids[~np.isin(all_ids, exclude_ids)]
                full_distances = hamming_distances(query_code, self.codes[all_ids])
                order = _topk_indices(full_distances, min(candidate_limit, len(all_ids)))
                return all_ids[order], full_distances[order].astype(np.int32), {
                    "mih_candidate_count": 0,
                    "hnsw_candidate_count": int(len(candidates)),
                    "hnsw_requested_count": int(request),
                    "hnsw_query_k": int(search_k),
                    "hnsw_retry_count": int(retry_count),
                    "hamming_candidate_count": int(len(order)),
                    "used_full_scan": True,
                }
            order = _topk_indices(hnsw_distances, min(candidate_limit, len(candidates)))
            return candidates[order], hnsw_distances[order], {
                "mih_candidate_count": 0,
                "hnsw_candidate_count": int(len(candidates)),
                "hnsw_requested_count": int(request),
                "hnsw_query_k": int(search_k),
                "hnsw_retry_count": int(retry_count),
                "hamming_candidate_count": int(len(order)),
                "used_full_scan": False,
            }

        candidates = self._mih_candidates(query_code)
        if len(exclude_ids) and len(candidates):
            candidates = candidates[~np.isin(candidates, exclude_ids)]
        mih_candidate_count = int(len(candidates))
        valid_excluded = exclude_ids[(exclude_ids >= 0) & (exclude_ids < len(self.codes))]
        available_count = len(self.codes) - len(valid_excluded)
        required = min(int(minimum_candidates), max(available_count, 0))
        used_full_scan = False
        if len(candidates) < required:
            candidates = np.arange(len(self.codes), dtype=np.int64)
            if len(exclude_ids):
                candidates = candidates[~np.isin(candidates, exclude_ids)]
            used_full_scan = True

        code_distance = hamming_distances(query_code, self.codes[candidates])
        order = _topk_indices(code_distance, min(candidate_limit, len(candidates)))
        selected_ids = candidates[order].astype(np.int64, copy=False)
        selected_distances = code_distance[order].astype(np.int32, copy=False)
        return selected_ids, selected_distances, {
            "mih_candidate_count": mih_candidate_count,
            "hamming_candidate_count": int(len(selected_ids)),
            "used_full_scan": used_full_scan,
        }

    def rerank_continuous(self, query_embedding, candidate_ids, top_k):
        """Rerank an explicit candidate set with the index's continuous metric."""
        if self.embeddings is None:
            raise RuntimeError("The code index has not been built or loaded.")
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {query_embedding.shape[0]} does not match index dimension "
                f"{self.embeddings.shape[1]}."
            )
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        if np.any((candidate_ids < 0) | (candidate_ids >= len(self.embeddings))):
            raise ValueError("candidate_ids contains an ID outside the code index.")
        if self.config.normalize or self.config.metric == "cosine":
            query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        gallery = self.embeddings[candidate_ids].astype(np.float32)
        if self.config.metric == "cosine":
            continuous_distance = 1.0 - gallery @ query_embedding
        else:
            difference = gallery - query_embedding.reshape(1, -1)
            continuous_distance = np.einsum("ij,ij->i", difference, difference)
        final_order = _topk_indices(continuous_distance, top_k)
        return (
            candidate_ids[final_order].astype(np.int64, copy=False),
            continuous_distance[final_order].astype(np.float32, copy=False),
        )

    def query(self, query_embedding, top_k=50, exclude_ids=None, return_details=False,
              query_code=None, rerank_mode="full", small_rerank_candidates=None):
        if self.embeddings is None or self.codes is None:
            raise RuntimeError("The code index has not been built or loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if rerank_mode not in {"full", "code-only", "small-rerank"}:
            raise ValueError(f"Unsupported rerank_mode: {rerank_mode}")
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query_code is None:
            query_code = self._encode_query(query_embedding)
        else:
            query_code = np.asarray(query_code, dtype=np.uint8).reshape(-1)
            if len(query_code) != self.config.bits // 8:
                raise ValueError(
                    f"Packed query code has {len(query_code)} bytes; expected "
                    f"{self.config.bits // 8}."
                )
        candidate_limit = self.config.hamming_candidates
        if rerank_mode == "code-only":
            candidate_limit = top_k
        elif rerank_mode == "small-rerank":
            if small_rerank_candidates is None:
                raise ValueError("small_rerank_candidates is required for small-rerank mode.")
            candidate_limit = max(int(small_rerank_candidates), top_k)
        rerank_ids, hamming_distances_out, hamming_details = self.hamming_search(
            query_code,
            candidate_limit=candidate_limit,
            minimum_candidates=top_k,
            exclude_ids=exclude_ids,
        )
        if rerank_mode == "code-only":
            ids = rerank_ids[:top_k].astype(np.int64, copy=False)
            distances = hamming_distances_out[:top_k].astype(np.float32, copy=False)
        else:
            ids, distances = self.rerank_continuous(query_embedding, rerank_ids, top_k)
        if not return_details:
            return ids, distances
        return ids, distances, {
            "candidate_ids": rerank_ids,
            "rerank_mode": rerank_mode,
            "rerank_candidate_count": int(len(rerank_ids)),
            **hamming_details,
        }

    def save(self, index_dir):
        if self.embeddings is None or self.codes is None or self.head is None:
            raise RuntimeError("Cannot save an unbuilt code index.")
        os.makedirs(index_dir, exist_ok=True)
        np.save(osp.join(index_dir, "embeddings.npy"), self.embeddings)
        np.save(osp.join(index_dir, "codes.npy"), self.codes)
        if self.config.backend == "hnsw":
            self.hnsw_index.save_index(osp.join(index_dir, "code_hnsw.bin"))
        elif self.config.backend == "mih":
            np.save(osp.join(index_dir, "table_offsets.npy"), self.table_offsets)
            np.save(osp.join(index_dir, "table_postings.npy"), self.table_postings)
        save_code_encoder(self.head, osp.join(index_dir, "code_encoder.pt"), metadata=self.metadata)
        payload = {
            "format_version": 1,
            "config": asdict(self.config),
            "num_vectors": int(len(self.embeddings)),
            "embedding_dim": int(self.embeddings.shape[1]),
            "build_seconds": float(self.build_seconds),
            "metadata": self.metadata,
        }
        with open(osp.join(index_dir, "code_config.json"), "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def load(cls, index_dir, mmap=True):
        with open(osp.join(index_dir, "code_config.json"), "r", encoding="utf-8") as file:
            payload = json.load(file)
        index = cls(BinaryCodeIndexConfig(**payload["config"]))
        mmap_mode = "r" if mmap else None
        index.embeddings = np.load(osp.join(index_dir, "embeddings.npy"), mmap_mode=mmap_mode)
        index.codes = np.load(osp.join(index_dir, "codes.npy"), mmap_mode=mmap_mode)
        if index.config.backend == "hnsw":
            hnswlib = _load_hnswlib()
            index.hnsw_index = hnswlib.Index(space="l2", dim=int(index.config.bits))
            index.hnsw_index.load_index(osp.join(index_dir, "code_hnsw.bin"), max_elements=len(index.codes))
            index.hnsw_index.set_ef(max(int(index.config.hnsw_ef_search), int(index.config.hamming_candidates)))
        elif index.config.backend == "mih":
            index.table_offsets = np.load(osp.join(index_dir, "table_offsets.npy"), mmap_mode=mmap_mode)
            index.table_postings = np.load(osp.join(index_dir, "table_postings.npy"), mmap_mode=mmap_mode)
        index.head, _ = load_code_encoder(osp.join(index_dir, "code_encoder.pt"))
        index.metadata = payload.get("metadata", {})
        index.build_seconds = float(payload.get("build_seconds", 0.0))
        if len(index.embeddings) != len(index.codes):
            raise ValueError("Code index embeddings and codes contain different vector counts.")
        return index
