import math

import torch
import torch.nn as nn


def import_mamba():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            return Mamba
        except ImportError as exc:
            raise ImportError(
                "Mamba sequence encoder requires mamba-ssm. "
                "Install it in the training environment before setting seqEncoder='mamba' or 'bimamba'."
            ) from exc


class MambaSeqEncoder(nn.Module):
    def __init__(self, d_model, out_dim, num_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        Mamba = import_mamba()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(norm(x))
            x = self.dropout(x)
            x = x + residual
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return self.out_proj(x)


class BiMambaSeqEncoder(nn.Module):
    def __init__(self, d_model, out_dim, num_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.forward_encoder = MambaSeqEncoder(
            d_model=d_model,
            out_dim=out_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.backward_encoder = MambaSeqEncoder(
            d_model=d_model,
            out_dim=out_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.fusion = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x, padding_mask=None):
        out_f = self.forward_encoder(x, padding_mask)
        x_rev = torch.flip(x, dims=[1])
        mask_rev = torch.flip(padding_mask, dims=[1]) if padding_mask is not None else None
        out_b = self.backward_encoder(x_rev, mask_rev)
        out_b = torch.flip(out_b, dims=[1])
        return self.fusion(torch.cat([out_f, out_b], dim=-1))


class GatedTemporalConvEncoder(nn.Module):
    """Linear-complexity temporal mixer used when Mamba is unavailable."""

    def __init__(self, d_model, num_layers=2, kernel_size=5, dropout=0.1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.depthwise = nn.ModuleList([
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
            )
            for _ in range(num_layers)
        ])
        self.pointwise = nn.ModuleList([nn.Linear(d_model, d_model * 2) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for norm, depthwise, pointwise in zip(self.norms, self.depthwise, self.pointwise):
            residual = x
            mixed = depthwise(norm(x).transpose(1, 2)).transpose(1, 2)
            value, gate = pointwise(mixed).chunk(2, dim=-1)
            x = residual + self.dropout(value * torch.sigmoid(gate))
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class TransformerSeqEncoder(nn.Module):
    """Masked Transformer encoder with sinusoidal positions for trajectory tokens."""

    def __init__(self, d_model, num_layers=2, num_heads=4, dropout=0.1, ffn_multiplier=4):
        super().__init__()
        if num_heads < 1 or d_model % num_heads != 0:
            raise ValueError(
                f"Transformer num_heads={num_heads} must be a positive divisor of d_model={d_model}."
            )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _positional_encoding(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros((length, self.d_model), device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    def forward(self, x, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        original_mask = padding_mask
        padding_mask = padding_mask.clone()
        padding_mask[:, 0] = False

        positions = self._positional_encoding(x.size(1), x.device, x.dtype)
        x = self.dropout(x + positions.unsqueeze(0))
        x = x.masked_fill(original_mask.unsqueeze(-1), 0.0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x.masked_fill(original_mask.unsqueeze(-1), 0.0)


class HierarchySegmentPool(nn.Module):
    """Compress consecutive nodes assigned to the same hierarchy region."""

    def __init__(self, d_model):
        super().__init__()
        self.start_proj = nn.Linear(d_model, d_model, bias=False)
        self.end_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, region_ids, padding_mask):
        if x.ndim != 3 or region_ids.shape != x.shape[:2] or padding_mask.shape != x.shape[:2]:
            raise ValueError("Expected x=[B,L,D], region_ids=[B,L], and padding_mask=[B,L].")
        batch_size, sequence_length, d_model = x.shape
        if sequence_length == 0:
            raise ValueError("HierarchySegmentPool requires a non-empty padded sequence dimension.")

        valid = ~padding_mask
        starts = torch.zeros_like(valid)
        starts[:, 0] = valid[:, 0]
        starts[:, 1:] = valid[:, 1:] & (
            (~valid[:, :-1]) | (region_ids[:, 1:] != region_ids[:, :-1])
        )
        ends = torch.zeros_like(valid)
        ends[:, -1] = valid[:, -1]
        ends[:, :-1] = valid[:, :-1] & (
            (~valid[:, 1:]) | (region_ids[:, :-1] != region_ids[:, 1:])
        )

        segment_ids = starts.long().cumsum(dim=1) - 1
        actual_lengths = starts.sum(dim=1)
        has_valid_tokens = valid.any(dim=1)
        lengths = torch.where(
            has_valid_tokens,
            actual_lengths,
            torch.ones_like(actual_lengths),
        )
        max_segments = max(int(lengths.max().item()), 1)

        batch_offsets = (
            torch.arange(batch_size, device=x.device, dtype=torch.long).unsqueeze(1)
            * max_segments
        )
        flat_segment_ids = (segment_ids + batch_offsets).reshape(-1)
        flat_valid = valid.reshape(-1)
        valid_segment_ids = flat_segment_ids[flat_valid]
        valid_x = x.reshape(-1, d_model)[flat_valid]

        pooled_size = batch_size * max_segments
        sums = x.new_zeros((pooled_size, d_model))
        sums.index_add_(0, valid_segment_ids, valid_x)
        counts = x.new_zeros((pooled_size, 1))
        counts.index_add_(0, valid_segment_ids, x.new_ones((valid_x.size(0), 1)))
        means = sums / counts.clamp_min(1.0)

        flat_starts = starts.reshape(-1)
        flat_ends = ends.reshape(-1)
        start_context = self.start_proj(x.reshape(-1, d_model)[flat_starts])
        end_context = self.end_proj(x.reshape(-1, d_model)[flat_ends])
        # Autocast may keep x in FP32 while Linear emits FP16/BF16. index_add_
        # requires the accumulator and source to have exactly the same dtype.
        boundary_context = start_context.new_zeros((pooled_size, d_model))
        boundary_context.index_add_(
            0,
            flat_segment_ids[flat_starts],
            start_context,
        )
        boundary_context.index_add_(
            0,
            flat_segment_ids[flat_ends],
            end_context,
        )

        padded = self.norm(
            means + boundary_context.to(means.dtype)
        ).reshape(batch_size, max_segments, d_model)
        positions = torch.arange(max_segments, device=x.device).unsqueeze(0)
        mask = positions >= lengths.unsqueeze(1)
        padded = padded.masked_fill(mask.unsqueeze(-1), 0.0)
        padded = torch.where(
            has_valid_tokens.reshape(batch_size, 1, 1),
            padded,
            torch.zeros_like(padded),
        )
        return padded, mask, lengths


class HierarchyAlignedMultiScaleEncoder(nn.Module):
    """Model node, L1, L2, and L3 trajectory sequences with aligned scales."""

    def __init__(self, d_model, out_dim, num_layers=2, backbone="conv",
                 kernel_size=5, d_state=16, d_conv=4, expand=2, dropout=0.1,
                 transformer_heads=4):
        super().__init__()
        if backbone not in {"conv", "mamba", "bimamba", "transformer", "hybrid"}:
            raise ValueError(f"Unknown HMT backbone: {backbone}")
        self.d_model = d_model
        self.backbone = backbone
        self.scale_backbones = (
            ("conv", "conv", "transformer", "transformer")
            if backbone == "hybrid"
            else (backbone,) * 4
        )
        self.segment_pools = nn.ModuleList([HierarchySegmentPool(d_model) for _ in range(3)])
        self.scale_encoders = nn.ModuleList([
            self._make_scale_encoder(
                d_model=d_model,
                num_layers=num_layers,
                backbone=scale_backbone,
                kernel_size=kernel_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                transformer_heads=transformer_heads,
            )
            for scale_backbone in self.scale_backbones
        ])
        self.scale_attention = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(4)])
        self.bottom_up_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.bottom_up_gate = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(3)])
        self.top_down_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.top_down_gate = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(3)])
        self.mix_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.router = nn.Sequential(
            nn.Linear(d_model * 4 + 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 4),
        )
        self.out_proj = nn.Linear(d_model, out_dim)
        self.last_scale_embeddings = None
        self.last_scale_weights = None
        self.last_segment_lengths = None

    @staticmethod
    def _make_scale_encoder(d_model, num_layers, backbone, kernel_size,
                            d_state, d_conv, expand, dropout, transformer_heads):
        if backbone == "conv":
            return GatedTemporalConvEncoder(
                d_model=d_model,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        if backbone == "mamba":
            return MambaSeqEncoder(
                d_model=d_model,
                out_dim=d_model,
                num_layers=num_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
        if backbone == "bimamba":
            return BiMambaSeqEncoder(
                d_model=d_model,
                out_dim=d_model,
                num_layers=num_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
        if backbone == "transformer":
            return TransformerSeqEncoder(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=transformer_heads,
                dropout=dropout,
            )
        raise ValueError(f"Unknown scale backbone: {backbone}")

    @staticmethod
    def _masked_attention_pool(x, padding_mask, scorer):
        scores = scorer(x).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(padding_mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

    def _mix_scales(self, summaries):
        mixed = list(summaries.unbind(dim=1))
        for level in range(1, 4):
            candidate = self.bottom_up_proj[level - 1](mixed[level - 1])
            gate = torch.sigmoid(self.bottom_up_gate[level - 1](torch.cat([mixed[level], candidate], dim=-1)))
            mixed[level] = self.mix_norms[level](mixed[level] + gate * candidate)
        for level in range(2, -1, -1):
            candidate = self.top_down_proj[level](mixed[level + 1])
            gate = torch.sigmoid(self.top_down_gate[level](torch.cat([mixed[level], candidate], dim=-1)))
            mixed[level] = self.mix_norms[level](mixed[level] + gate * candidate)
        return torch.stack(mixed, dim=1)

    def forward(self, x, node_ids, hierarchy_data, padding_mask=None,
                return_details=False, scale_inputs=None):
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if x.ndim != 3 or node_ids.shape != x.shape[:2] or padding_mask.shape != x.shape[:2]:
            raise ValueError("Expected x=[B,L,D], node_ids=[B,L], and padding_mask=[B,L].")
        if scale_inputs is None:
            scale_inputs = (x,) * 4
        if len(scale_inputs) != 4 or any(item.shape != x.shape for item in scale_inputs):
            raise ValueError("scale_inputs must contain four tensors with the same shape as x.")

        safe_nodes = node_ids.clamp(min=0, max=max(hierarchy_data["node_to_l1"].numel() - 1, 0))
        region_sequences = [
            hierarchy_data["node_to_l1"][safe_nodes],
            hierarchy_data["node_to_l2"][safe_nodes],
            hierarchy_data["node_to_l3"][safe_nodes],
        ]
        node_lengths = (~padding_mask).sum(dim=1)
        scale_tokens = [scale_inputs[0]]
        scale_masks = [padding_mask]
        scale_lengths = [node_lengths]
        for pool, scale_x, region_ids in zip(
                self.segment_pools, scale_inputs[1:], region_sequences):
            tokens, mask, lengths = pool(scale_x, region_ids, padding_mask)
            scale_tokens.append(tokens)
            scale_masks.append(mask)
            scale_lengths.append(lengths)

        summaries = []
        for encoder, scorer, tokens, mask in zip(
                self.scale_encoders, self.scale_attention, scale_tokens, scale_masks):
            encoded = encoder(tokens, padding_mask=mask)
            summaries.append(self._masked_attention_pool(encoded, mask, scorer))
        summaries = torch.stack(summaries, dim=1)
        mixed = self._mix_scales(summaries)

        lengths = torch.stack(scale_lengths, dim=1)
        scale_ratios = lengths.to(x.dtype) / lengths[:, :1].clamp_min(1).to(x.dtype)
        router_input = torch.cat([mixed.flatten(start_dim=1), scale_ratios], dim=-1)
        scale_weights = torch.softmax(self.router(router_input), dim=-1)
        fused = torch.sum(mixed * scale_weights.unsqueeze(-1), dim=1)
        output = self.out_proj(fused)

        self.last_scale_embeddings = mixed
        self.last_scale_weights = scale_weights
        self.last_segment_lengths = lengths
        if return_details:
            return output, {
                "scale_embeddings": mixed,
                "scale_weights": scale_weights,
                "segment_lengths": lengths,
            }
        return output
