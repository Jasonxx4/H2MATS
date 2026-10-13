import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as segment_softmax

from graph_normalization import distance_weighted_gcn_norm
from hyperbolic_utils import expmap0, logmap0, poincare_dist
from model_network import SMNEncoder, TransformerModel
from sequence_encoders import (
    BiMambaSeqEncoder,
    HierarchyAlignedMultiScaleEncoder,
    MambaSeqEncoder,
)


def sample_negative_parent_indices(true_parent_ids, parent_count, num_negatives=1):
    if parent_count <= 1:
        return None
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive.")
    output_shape = true_parent_ids.shape
    if num_negatives > 1:
        output_shape = (*output_shape, num_negatives)
    offsets = torch.randint(
        1,
        parent_count,
        output_shape,
        device=true_parent_ids.device,
        dtype=true_parent_ids.dtype,
    )
    if num_negatives > 1:
        true_parent_ids = true_parent_ids.unsqueeze(-1)
    return torch.remainder(true_parent_ids + offsets, parent_count)


class WeightedGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_norm):
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, norm=edge_norm)
        return out + self.bias

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class TransitionWeightedGCNLayer(MessagePassing):
    """GCN message layer whose edge gate uses trajectory transition features."""

    def __init__(self, in_channels, out_channels, edge_feature_size=4):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_feature_size, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 1),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_norm, edge_attr):
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, norm=edge_norm, edge_attr=edge_attr)
        return out + self.bias

    def message(self, x_j, norm, edge_attr):
        # Keep length normalization as the base affinity and use transitions as a bounded gate.
        transition_gate = 1.0 + 0.5 * torch.tanh(self.edge_gate(edge_attr))
        return norm.view(-1, 1) * transition_gate * x_j


class EuclideanLevelGNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_levels=4, dropout=0.3,
                 use_transition_features=True, transition_residual_init=-2.0):
        super().__init__()
        self.dropout = dropout
        self.use_transition_features = use_transition_features
        self.transition_feature_size = 4
        self.conv1 = nn.ModuleList([
            WeightedGCNLayer(input_size, hidden_size)
            for _ in range(num_levels)
        ])
        self.conv2 = nn.ModuleList([
            WeightedGCNLayer(hidden_size, hidden_size)
            for _ in range(num_levels)
        ])
        self.transition_conv1 = nn.ModuleList([
            TransitionWeightedGCNLayer(
                input_size,
                hidden_size,
                edge_feature_size=self.transition_feature_size,
            )
            for _ in range(num_levels)
        ])
        self.transition_conv2 = nn.ModuleList([
            TransitionWeightedGCNLayer(
                hidden_size,
                hidden_size,
                edge_feature_size=self.transition_feature_size,
            )
            for _ in range(num_levels)
        ])
        self.transition_norm = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_levels)
        ])
        # Initialize the auxiliary flow branch conservatively so geometry remains primary.
        self.transition_residual_logits = nn.Parameter(
            torch.full((num_levels,), transition_residual_init)
        )

    def _transition_edge_attr(self, graph):
        transition_attr = getattr(graph, "transition_attr", None)
        if transition_attr is None:
            return None
        if transition_attr.ndim != 2 or transition_attr.size(1) != self.transition_feature_size:
            raise ValueError(
                "graph.transition_attr must have shape [E, 4], "
                f"got {tuple(transition_attr.shape)}."
            )
        if transition_attr.size(0) != graph.edge_index.size(1):
            raise ValueError(
                "graph.transition_attr must align with graph.edge_index: "
                f"{transition_attr.size(0)} != {graph.edge_index.size(1)}."
            )
        cache = getattr(graph, "gcn_transition_attr", None)
        if cache is None or cache.size(0) != graph.gcn_edge_index.size(1):
            self_loops = transition_attr.new_zeros(
                (graph.x.size(0), self.transition_feature_size)
            )
            cache = torch.cat([transition_attr, self_loops], dim=0)
            graph.gcn_transition_attr = cache
        return cache

    def forward(self, level_graphs):
        outputs = []
        for level, graph in enumerate(level_graphs):
            if not hasattr(graph, "gcn_edge_index") or not hasattr(graph, "gcn_edge_norm"):
                graph.gcn_edge_index, graph.gcn_edge_norm, graph.gcn_affinity_scale = (
                    distance_weighted_gcn_norm(
                        graph.edge_index,
                        graph.edge_attr,
                        graph.x.size(0),
                    )
                )
            x = F.relu(self.conv1[level](graph.x, graph.gcn_edge_index, graph.gcn_edge_norm))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2[level](x, graph.gcn_edge_index, graph.gcn_edge_norm))
            transition_attr = (
                self._transition_edge_attr(graph)
                if self.use_transition_features else None
            )
            if transition_attr is not None:
                flow = F.relu(self.transition_conv1[level](
                    graph.x,
                    graph.gcn_edge_index,
                    graph.gcn_edge_norm,
                    transition_attr,
                ))
                flow = F.dropout(flow, p=self.dropout, training=self.training)
                flow = F.relu(self.transition_conv2[level](
                    flow,
                    graph.gcn_edge_index,
                    graph.gcn_edge_norm,
                    transition_attr,
                ))
                flow = self.transition_norm[level](flow)
                x = x + torch.sigmoid(self.transition_residual_logits[level]) * flow
            outputs.append(x)
        return outputs


class SimpleTwoLayerGCN(nn.Module):
    """Two-layer distance-weighted GCN over the original road graph only."""

    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = WeightedGCNLayer(input_size, hidden_size)
        self.conv2 = WeightedGCNLayer(hidden_size, hidden_size)

    def forward(self, graph):
        if not hasattr(graph, "gcn_edge_index") or not hasattr(graph, "gcn_edge_norm"):
            graph.gcn_edge_index, graph.gcn_edge_norm, graph.gcn_affinity_scale = (
                distance_weighted_gcn_norm(
                    graph.edge_index,
                    graph.edge_attr,
                    graph.x.size(0),
                )
            )
        x = F.relu(self.conv1(graph.x, graph.gcn_edge_index, graph.gcn_edge_norm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.conv2(x, graph.gcn_edge_index, graph.gcn_edge_norm))


class HierarchyPropagationBlock(nn.Module):
    """One residual bottom-up/top-down hierarchy message-passing block."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.child_query = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)])
        self.child_key = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)])
        self.child_value = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.child_gate = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size) for _ in range(3)])
        self.parent_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.parent_gate = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size) for _ in range(3)])
        self.bottom_up_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(3)])
        self.top_down_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(3)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            for _ in range(4)
        ])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])

    @staticmethod
    def _attention_pool(child_x, child_to_parent, parent_x, query, key, value):
        """Aggregate variable-size child sets with a parent-conditioned attention score."""
        if child_x.numel() == 0:
            return parent_x.new_zeros(parent_x.shape)
        queries = query(parent_x)[child_to_parent]
        keys = key(child_x)
        scores = (queries * keys).sum(dim=-1) / math.sqrt(child_x.size(-1))
        weights = segment_softmax(scores, child_to_parent, num_nodes=parent_x.size(0))
        values = value(child_x) * weights.unsqueeze(-1)
        pooled = parent_x.new_zeros(parent_x.shape)
        pooled.index_add_(0, child_to_parent, values)
        return pooled

    def forward(self, states, child_to_parent):
        states = list(states)

        # Sequential bottom-up propagation lets a coarse region see already-updated children.
        for level in range(3):
            child = states[level]
            parent = states[level + 1]
            pooled = self._attention_pool(
                child,
                child_to_parent[level],
                parent,
                self.child_query[level],
                self.child_key[level],
                self.child_value[level],
            )
            candidate = F.gelu(pooled)
            gate = torch.sigmoid(
                self.child_gate[level](torch.cat([parent, candidate], dim=-1))
            )
            states[level + 1] = self.bottom_up_norm[level](
                parent + F.dropout(gate * candidate, p=self.dropout, training=self.training)
            )

        # Sequential top-down propagation returns the coarse context to individual roads.
        for level in range(2, -1, -1):
            child = states[level]
            parent = states[level + 1][child_to_parent[level]]
            candidate = F.gelu(self.parent_proj[level](parent))
            gate = torch.sigmoid(
                self.parent_gate[level](torch.cat([child, candidate], dim=-1))
            )
            states[level] = self.top_down_norm[level](
                child + F.dropout(gate * candidate, p=self.dropout, training=self.training)
            )

        return [
            norm(state + F.dropout(ffn(state), p=self.dropout, training=self.training))
            for state, ffn, norm in zip(states, self.ffn, self.ffn_norm)
        ]


class HyperbolicHierarchyGNN(nn.Module):
    def __init__(self, hidden_size, curvature=1.0, num_blocks=2, dropout=0.1):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive.")
        self.curvature = curvature
        self.num_blocks = num_blocks
        self.tangent_scale = hidden_size ** -0.5
        self.input_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.blocks = nn.ModuleList([
            HierarchyPropagationBlock(hidden_size, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(4)])

    def forward(self, euclidean_outputs, hierarchy_data):
        states = [norm(x) for norm, x in zip(self.input_norm, euclidean_outputs)]
        child_to_parent = [
            hierarchy_data["node_to_l1"],
            hierarchy_data["l1_to_l2"],
            hierarchy_data["l2_to_l3"],
        ]
        for block in self.blocks:
            states = block(states, child_to_parent)
        tangent = [
            self.tangent_scale * torch.tanh(proj(state))
            for proj, state in zip(self.output_proj, states)
        ]
        return [expmap0(x, c=self.curvature) for x in tangent]


class NodeAdaptiveScaleFusion(nn.Module):
    """Fuse four geometry-aware node views with node-dependent scale weights."""

    def __init__(self, embedding_size, dropout=0.2, scale_dropout=0.2):
        super().__init__()
        if not 0 <= scale_dropout < 1:
            raise ValueError("scale_dropout must be in [0, 1).")
        self.scale_dropout = scale_dropout
        self.geometry_gates = nn.ModuleList([
            nn.Linear(embedding_size * 2, embedding_size)
            for _ in range(4)
        ])
        self.geometry_norms = nn.ModuleList([
            nn.LayerNorm(embedding_size) for _ in range(4)
        ])
        self.scale_projections = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(4)
        ])
        self.scale_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_size * 2, embedding_size),
                nn.GELU(),
                nn.Linear(embedding_size, 1),
            )
            for _ in range(4)
        ])
        self.output_norm = nn.LayerNorm(embedding_size)
        self.output_dropout = nn.Dropout(dropout)
        self.last_scale_weights = None
        self.last_balance_weights = None

    def forward(self, euclidean_views, hyperbolic_views):
        scale_views = []
        for level, (euclidean, hyperbolic) in enumerate(zip(euclidean_views, hyperbolic_views)):
            gate = torch.sigmoid(
                self.geometry_gates[level](torch.cat([euclidean, hyperbolic], dim=-1))
            )
            scale_views.append(self.geometry_norms[level](
                gate * euclidean + (1.0 - gate) * hyperbolic
            ))

        base = scale_views[0]
        logits = torch.cat([
            scorer(torch.cat([view, base], dim=-1))
            for scorer, view in zip(self.scale_scorers, scale_views)
        ], dim=-1)
        # Keep the auxiliary balance path in FP32. In FP16, tiny softmax
        # probabilities can underflow to zero before the entropy-style loss.
        balance_weights = torch.softmax(logits.float(), dim=-1)
        if self.training and self.scale_dropout > 0:
            dropped = torch.rand(4, device=logits.device) < self.scale_dropout
            if bool(dropped.all()):
                dropped[0] = False
            logits = logits.masked_fill(dropped.unsqueeze(0), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        routed = sum(
            weight.unsqueeze(-1) * projection(view)
            for weight, projection, view in zip(
                weights.unbind(dim=-1), self.scale_projections, scale_views
            )
        )
        output = self.output_norm(base + self.output_dropout(routed))
        self.last_scale_weights = weights.detach()
        self.last_balance_weights = balance_weights
        return output, scale_views, weights


class HybridGeoTrajSimEncoder(nn.Module):
    GRAPH_STATE_PREFIXES = (
        "euclidean_gnn.",
        "hyperbolic_gnn.",
        "fusion.",
        "scale_geometry_gates.",
        "scale_specific_norms.",
        "scale_residual_logits",
        "simple_gcn.",
    )

    def __init__(self, feature_size, embedding_size, hidden_size, num_layers, dropout_rate,
                 concat, device, useLSTM, curvature=1.0,
                 seq_encoder='transformer', mamba_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 coord_encoder='lstm', hmt_backbone='conv', hmt_layers=2,
                 hmt_kernel_size=5, hmt_dropout=0.1, hmt_transformer_heads=4,
                 hmt_scale_aligned_graph=True, coord_fusion='gated_attention',
                 coord_transformer_checkpoint=True,
                 graph_encoder='hybrid_multilevel', hierarchy_blocks=2,
                 hierarchy_dropout=0.1, graph_scale_dropout=0.2,
                 use_transition_features=True, transition_residual_init=-2.0):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.useLSTM = useLSTM
        self.curvature = curvature
        self.seq_encoder_type = seq_encoder
        self.coord_encoder = coord_encoder
        self.graph_encoder_type = graph_encoder
        self.hmt_scale_aligned_graph = hmt_scale_aligned_graph
        self.coord_fusion = coord_fusion
        if graph_encoder not in {'hybrid_multilevel', 'simple_gcn'}:
            raise ValueError(f"Unknown graph encoder: {graph_encoder}")
        if graph_encoder == 'simple_gcn' and hmt_scale_aligned_graph:
            raise ValueError(
                "simple_gcn only produces an L0 graph embedding; set "
                "hmt_scale_aligned_graph=False."
            )
        if coord_fusion not in {'late_concat', 'gated_attention'}:
            raise ValueError(f"Unknown coordinate fusion: {coord_fusion}")

        if graph_encoder == 'simple_gcn':
            self.simple_gcn = SimpleTwoLayerGCN(feature_size, embedding_size)
        else:
            self.euclidean_gnn = EuclideanLevelGNN(
                feature_size,
                embedding_size,
                use_transition_features=use_transition_features,
                transition_residual_init=transition_residual_init,
            )
            self.hyperbolic_gnn = HyperbolicHierarchyGNN(
                embedding_size,
                curvature=curvature,
                num_blocks=hierarchy_blocks,
                dropout=hierarchy_dropout,
            )
            self.fusion = NodeAdaptiveScaleFusion(
                embedding_size,
                dropout=0.2,
                scale_dropout=graph_scale_dropout,
            )
        if seq_encoder == 'hmt' and hmt_scale_aligned_graph:
            self.scale_geometry_gates = nn.ModuleList([
                nn.Linear(embedding_size * 2, embedding_size)
                for _ in range(4)
            ])
            self.scale_specific_norms = nn.ModuleList([
                nn.LayerNorm(embedding_size)
                for _ in range(4)
            ])
            self.scale_residual_logits = nn.Parameter(
                torch.zeros(4)
            )

        if seq_encoder == 'transformer':
            self.trm_encoder = TransformerModel(hidden_size, embedding_size, 4, 512, 1, 0.3, device)
        elif seq_encoder == 'mamba':
            self.seq_encoder = MambaSeqEncoder(
                d_model=embedding_size,
                out_dim=hidden_size,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        elif seq_encoder == 'bimamba':
            self.seq_encoder = BiMambaSeqEncoder(
                d_model=embedding_size,
                out_dim=hidden_size,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        elif seq_encoder == 'hmt':
            self.seq_encoder = HierarchyAlignedMultiScaleEncoder(
                d_model=embedding_size,
                out_dim=hidden_size,
                num_layers=hmt_layers,
                backbone=hmt_backbone,
                kernel_size=hmt_kernel_size,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=hmt_dropout,
                transformer_heads=hmt_transformer_heads,
            )
        else:
            raise ValueError(f"Unknown sequence encoder: {seq_encoder}")
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        if useLSTM:
            self.smn = SMNEncoder(2,
                                  hidden_size,
                                  stard_LSTM=True,
                                  incell=True,
                                  device=self.device,
                                  seq_model=coord_encoder,
                                  mamba_layers=mamba_layers,
                                  mamba_d_state=mamba_d_state,
                                  mamba_d_conv=mamba_d_conv,
                                  mamba_expand=mamba_expand).to(self.device)
            if coord_encoder == 'transformer':
                self.smn.t2s_model.activation_checkpointing = coord_transformer_checkpoint
            self.out_linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
            if coord_fusion == 'gated_attention':
                self.coord_attention = nn.Linear(hidden_size, 1)
                self.coord_fusion_gate = nn.Linear(2 * hidden_size, 2 * hidden_size)
                self.coord_fusion_candidate = nn.Linear(2 * hidden_size, 2 * hidden_size)
                self.coord_fusion_norm = nn.LayerNorm(2 * hidden_size)
                nn.init.zeros_(self.coord_attention.weight)
                nn.init.zeros_(self.coord_attention.bias)
                nn.init.zeros_(self.coord_fusion_candidate.weight)
                nn.init.zeros_(self.coord_fusion_candidate.bias)

        self.last_hyperbolic_outputs = None
        self.last_euclidean_outputs = None
        self.last_hyperbolic_tangent_outputs = None
        self.last_hierarchy_data = None
        self.last_scale_embeddings = None
        self.last_scale_weights = None
        self.last_graph_scale_weights = None
        self.last_segment_lengths = None
        self.last_triplet_combined = None
        self.graph_frozen = False
        self.cache_frozen_graph = True
        self._cached_graph_node_embeddings = None

    def _graph_modules(self):
        if self.graph_encoder_type == 'simple_gcn':
            return [self.simple_gcn]
        modules = [self.euclidean_gnn, self.hyperbolic_gnn, self.fusion]
        if hasattr(self, "scale_geometry_gates"):
            modules.extend([self.scale_geometry_gates, self.scale_specific_norms])
        return modules

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "graph_frozen", False):
            for module in self._graph_modules():
                module.eval()
        return self

    def graph_state_dict(self):
        """Return only parameters owned by the graph representation pipeline."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith(self.GRAPH_STATE_PREFIXES)
        }

    def graph_parameters(self):
        """Yield parameters owned by the graph representation pipeline."""
        for name, parameter in self.named_parameters():
            if name.startswith(self.GRAPH_STATE_PREFIXES):
                yield parameter

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            raise TypeError("A graph checkpoint must contain a state-dict mapping.")
        for key in ("graph_state_dict", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
        if not isinstance(checkpoint, dict):
            raise TypeError("The selected checkpoint entry is not a state dict.")
        return {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
            if torch.is_tensor(value)
        }

    def load_pretrained_graph(self, checkpoint, map_location=None, strict=True):
        """Load graph weights from either a graph-only or full model checkpoint."""
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        state_dict = self._checkpoint_state_dict(checkpoint)
        graph_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(self.GRAPH_STATE_PREFIXES)
        }
        expected = set(self.graph_state_dict())
        provided = set(graph_state)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if not graph_state:
            raise ValueError("The checkpoint does not contain HybridGeo graph parameters.")
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Graph checkpoint mismatch. "
                f"Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}."
            )
        compatible_state = {
            key: value for key, value in graph_state.items() if key in expected
        }
        self.load_state_dict(compatible_state, strict=False)
        self.clear_graph_cache()
        return missing, unexpected

    def freeze_graph_encoder(self, cache=True):
        """Freeze graph parameters while leaving sequence encoders trainable."""
        self.graph_frozen = True
        self.cache_frozen_graph = cache
        for module in self._graph_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        if hasattr(self, "scale_residual_logits"):
            self.scale_residual_logits.requires_grad = False
        self.clear_graph_cache()

    def unfreeze_graph_encoder(self):
        """Unfreeze graph parameters and discard detached frozen representations."""
        self.graph_frozen = False
        self.cache_frozen_graph = False
        for module in self._graph_modules():
            for parameter in module.parameters():
                parameter.requires_grad = True
            module.train(self.training)
        if hasattr(self, "scale_residual_logits"):
            self.scale_residual_logits.requires_grad = True
        self.clear_graph_cache()

    def clear_graph_cache(self):
        self._cached_graph_node_embeddings = None
        self.last_euclidean_outputs = None
        self.last_hyperbolic_outputs = None
        self.last_hyperbolic_tangent_outputs = None
        self.last_hierarchy_data = None

    def prepare_frozen_graph(self, hierarchy_data):
        """Build deterministic graph state once for reuse by trajectory batches."""
        if not self.graph_frozen:
            raise RuntimeError("freeze_graph_encoder() must be called before preparing a graph cache.")
        for module in self._graph_modules():
            module.eval()
        with torch.no_grad():
            graph_node_embeddings = self.encode_graph(hierarchy_data).detach()
        if self.cache_frozen_graph:
            self._cached_graph_node_embeddings = graph_node_embeddings
        return graph_node_embeddings

    def get_graph_node_embeddings(self, hierarchy_data):
        if getattr(self, "graph_frozen", False):
            if (
                getattr(self, "cache_frozen_graph", True)
                and getattr(self, "_cached_graph_node_embeddings", None) is not None
            ):
                return self._cached_graph_node_embeddings
            return self.prepare_frozen_graph(hierarchy_data)
        return self.encode_graph(hierarchy_data)

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, length in enumerate(seq_lengths):
            if length <= max_len:
                mask[i, :length] = 0
        return mask.bool()

    @staticmethod
    def _safe_index(x, index):
        return x[index.clamp(min=0, max=max(x.size(0) - 1, 0))]

    def map_tangent_levels_to_original_nodes(
        self,
        euclidean_outputs,
        hyperbolic_tangent_outputs,
        hierarchy_data,
    ):
        level_indices = [
            torch.arange(euclidean_outputs[0].size(0), device=euclidean_outputs[0].device),
            hierarchy_data["node_to_l1"],
            hierarchy_data["node_to_l2"],
            hierarchy_data["node_to_l3"],
        ]
        euclidean_node_views = [
            self._safe_index(level_output, level_index)
            for level_output, level_index in zip(euclidean_outputs, level_indices)
        ]
        hyperbolic_node_views = [
            self._safe_index(level_output, level_index)
            for level_output, level_index in zip(hyperbolic_tangent_outputs, level_indices)
        ]
        return euclidean_node_views, hyperbolic_node_views

    def map_levels_to_original_nodes(self, euclidean_outputs, hyperbolic_outputs, hierarchy_data):
        hyperbolic_tangent_outputs = [
            logmap0(x, c=self.curvature)
            for x in hyperbolic_outputs
        ]
        return self.map_tangent_levels_to_original_nodes(
            euclidean_outputs,
            hyperbolic_tangent_outputs,
            hierarchy_data,
        )

    def build_scale_aligned_node_embeddings(
        self,
        global_node_embeddings,
        euclidean_node_views,
        hyperbolic_node_views,
    ):
        scale_embeddings = []
        residual_weights = torch.sigmoid(self.scale_residual_logits)
        for level, (euclidean_view, hyperbolic_view) in enumerate(
                zip(euclidean_node_views, hyperbolic_node_views)):
            geometry_input = torch.cat([euclidean_view, hyperbolic_view], dim=-1)
            geometry_gate = torch.sigmoid(self.scale_geometry_gates[level](geometry_input))
            scale_specific = (
                geometry_gate * euclidean_view
                + (1.0 - geometry_gate) * hyperbolic_view
            )
            scale_specific = self.scale_specific_norms[level](scale_specific)
            scale_embeddings.append(
                global_node_embeddings + residual_weights[level] * scale_specific
            )
        return scale_embeddings

    def build_scale_aligned_trajectory_inputs(
        self,
        global_trajectory_embeddings,
        traj_seqs,
        hierarchy_data,
    ):
        safe_nodes = traj_seqs.clamp(
            min=0,
            max=max(hierarchy_data["node_to_l1"].numel() - 1, 0),
        )
        level_indices = [
            safe_nodes,
            hierarchy_data["node_to_l1"][safe_nodes],
            hierarchy_data["node_to_l2"][safe_nodes],
            hierarchy_data["node_to_l3"][safe_nodes],
        ]
        euclidean_views = [
            self._safe_index(level_output, level_index)
            for level_output, level_index in zip(self.last_euclidean_outputs, level_indices)
        ]
        hyperbolic_views = [
            self._safe_index(level_output, level_index)
            for level_output, level_index in zip(
                self.last_hyperbolic_tangent_outputs,
                level_indices,
            )
        ]
        return self.build_scale_aligned_node_embeddings(
            global_trajectory_embeddings,
            euclidean_views,
            hyperbolic_views,
        )

    def fuse_to_original_nodes(self, euclidean_outputs, hyperbolic_outputs, hierarchy_data):
        euclidean_node_views, hyperbolic_node_views = self.map_levels_to_original_nodes(
            euclidean_outputs,
            hyperbolic_outputs,
            hierarchy_data,
        )
        fused, _, _ = self.fusion(euclidean_node_views, hyperbolic_node_views)
        return fused

    def encode_graph(self, hierarchy_data):
        if self.graph_encoder_type == 'simple_gcn':
            graph_node_embeddings = self.simple_gcn(hierarchy_data["level_graphs"][0])
            self.last_euclidean_outputs = None
            self.last_hyperbolic_outputs = None
            self.last_hyperbolic_tangent_outputs = None
            self.last_hierarchy_data = hierarchy_data
            self.last_graph_scale_weights = None
            return graph_node_embeddings
        euclidean_outputs = self.euclidean_gnn(hierarchy_data["level_graphs"])
        hyperbolic_outputs = self.hyperbolic_gnn(euclidean_outputs, hierarchy_data)
        hyperbolic_tangent_outputs = [
            logmap0(x, c=self.curvature)
            for x in hyperbolic_outputs
        ]
        euclidean_node_views, hyperbolic_node_views = self.map_tangent_levels_to_original_nodes(
            euclidean_outputs,
            hyperbolic_tangent_outputs,
            hierarchy_data,
        )
        graph_node_embeddings, _, _ = self.fusion(
            euclidean_node_views,
            hyperbolic_node_views,
        )
        self.last_euclidean_outputs = euclidean_outputs
        self.last_hyperbolic_outputs = hyperbolic_outputs
        self.last_hyperbolic_tangent_outputs = hyperbolic_tangent_outputs
        self.last_hierarchy_data = hierarchy_data
        self.last_graph_scale_weights = self.fusion.last_balance_weights
        return graph_node_embeddings

    def encode_trajectories(
        self,
        graph_node_embeddings,
        hierarchy_data,
        traj_seqs,
        coor_seqs,
        seq_lengths,
    ):
        seq_lengths = seq_lengths.to(self.device)
        traj_seqs = traj_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)

        embedded_seq_tensor = graph_node_embeddings[traj_seqs].to(self.device)
        trm_encoder_src_mask = self.obtain_trm_src_mask(seq_lengths)
        if self.seq_encoder_type == 'hmt':
            scale_inputs = None
            if self.hmt_scale_aligned_graph:
                scale_inputs = self.build_scale_aligned_trajectory_inputs(
                    embedded_seq_tensor,
                    traj_seqs,
                    hierarchy_data,
                )
            attrn_out, details = self.seq_encoder(
                embedded_seq_tensor,
                traj_seqs,
                hierarchy_data,
                padding_mask=trm_encoder_src_mask,
                return_details=True,
                scale_inputs=scale_inputs,
            )
            self.last_scale_embeddings = details["scale_embeddings"]
            self.last_scale_weights = details["scale_weights"]
            self.last_segment_lengths = details["segment_lengths"]
        elif self.seq_encoder_type == 'transformer':
            seq_outputs = self.trm_encoder(embedded_seq_tensor.transpose(1, 0), trm_encoder_src_mask).transpose(1, 0)
        else:
            seq_outputs = self.seq_encoder(embedded_seq_tensor, padding_mask=trm_encoder_src_mask)
        if self.seq_encoder_type != 'hmt':
            u = torch.tanh(torch.matmul(seq_outputs, self.w_omega))
            att = torch.matmul(u, self.u_omega).squeeze(-1)
            att = att.masked_fill(trm_encoder_src_mask, -1e10)
            att_score = F.softmax(att, dim=1).unsqueeze(2)
            scored_outputs = seq_outputs * att_score
            attrn_out = torch.sum(scored_outputs, dim=1)

        if self.useLSTM:
            anchor_embedding, coordinate_outputs = self.smn([coor_seqs, seq_lengths])
            out = self.fuse_trajectory_coordinate(
                attrn_out,
                anchor_embedding,
                coordinate_outputs.transpose(0, 1),
                trm_encoder_src_mask,
            )
            out = self.out_linear(out)
        else:
            out = attrn_out
        return out

    @staticmethod
    def _masked_attention_pool(x, padding_mask, scorer):
        scores = scorer(x).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(padding_mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

    def fuse_trajectory_coordinate(
        self,
        trajectory_embedding,
        coordinate_last_embedding,
        coordinate_outputs,
        padding_mask,
    ):
        if self.coord_fusion == 'late_concat':
            return torch.cat((trajectory_embedding, coordinate_last_embedding), dim=-1)

        coordinate_embedding = self._masked_attention_pool(
            coordinate_outputs,
            padding_mask,
            self.coord_attention,
        )
        joint = torch.cat((trajectory_embedding, coordinate_embedding), dim=-1)
        fusion_gate = torch.sigmoid(self.coord_fusion_gate(joint))
        fusion_candidate = torch.tanh(self.coord_fusion_candidate(joint))
        return self.coord_fusion_norm(joint + fusion_gate * fusion_candidate)

    def forward(self, hierarchy_data, traj_seqs, coor_seqs, seq_lengths):
        graph_node_embeddings = self.get_graph_node_embeddings(hierarchy_data)
        return self.encode_trajectories(
            graph_node_embeddings,
            hierarchy_data,
            traj_seqs,
            coor_seqs,
            seq_lengths,
        )

    @staticmethod
    def _pad_trajectory_batch(tensor, target_length):
        padding = target_length - tensor.size(1)
        if padding < 0:
            raise ValueError(
                f"target_length={target_length} is shorter than input length {tensor.size(1)}."
            )
        if padding == 0:
            return tensor
        if tensor.ndim == 2:
            return F.pad(tensor, (0, padding), value=0)
        if tensor.ndim == 3:
            return F.pad(tensor, (0, 0, 0, padding), value=0)
        raise ValueError(f"Expected a 2-D or 3-D trajectory tensor, got {tensor.ndim} dimensions.")

    def forward_triplet(self, hierarchy_data, anchor, positive, negative):
        """Encode a triplet with one graph pass and memory-aware sequence passes."""
        batches = (anchor, positive, negative)
        target_length = max(batch[0].size(1) for batch in batches)
        batch_sizes = [batch[0].size(0) for batch in batches]
        separate_tokens = sum(batch[0].numel() for batch in batches)
        combined_tokens = sum(batch_sizes) * target_length
        graph_node_embeddings = self.get_graph_node_embeddings(hierarchy_data)

        # Combining is beneficial for linear encoders, but triples the peak batch
        # of the raw-length coordinate Transformer.
        if getattr(self, 'coord_encoder', None) == 'transformer' or combined_tokens > separate_tokens * 1.5:
            self.last_triplet_combined = False
            return tuple(
                self.encode_trajectories(
                    graph_node_embeddings,
                    hierarchy_data,
                    batch[0],
                    batch[1],
                    batch[2],
                )
                for batch in batches
            )

        self.last_triplet_combined = True
        trajectories = torch.cat(
            [self._pad_trajectory_batch(batch[0], target_length) for batch in batches],
            dim=0,
        )
        coordinates = torch.cat(
            [self._pad_trajectory_batch(batch[1], target_length) for batch in batches],
            dim=0,
        )
        lengths = torch.cat([batch[2] for batch in batches], dim=0)

        embeddings = self.encode_trajectories(
            graph_node_embeddings,
            hierarchy_data,
            trajectories,
            coordinates,
            lengths,
        )
        return tuple(torch.split(embeddings, batch_sizes, dim=0))

    def hierarchy_loss(self, margin=1.0, max_samples=8192, negative_count=8):
        if self.last_hyperbolic_outputs is None or self.last_hierarchy_data is None:
            return torch.tensor(0.0, device=self.device)

        h0, h1, h2, h3 = self.last_hyperbolic_outputs
        node_to_l1 = self.last_hierarchy_data["node_to_l1"]
        l1_to_l2 = self.last_hierarchy_data["l1_to_l2"]
        l2_to_l3 = self.last_hierarchy_data["l2_to_l3"]

        losses = []
        pairs = [
            (h0, h1, node_to_l1),
            (h1, h2, l1_to_l2),
            (h2, h3, l2_to_l3),
        ]
        for child, parent, mapping in pairs:
            if mapping.numel() > max_samples:
                child_indices = torch.randperm(mapping.numel(), device=mapping.device)[:max_samples]
                child_sample = child[child_indices]
                parent_mapping = mapping[child_indices]
            else:
                child_sample = child
                parent_mapping = mapping
            true_parent = parent[parent_mapping]
            pos = poincare_dist(child_sample, true_parent, c=self.curvature)
            negative_indices = sample_negative_parent_indices(
                parent_mapping,
                parent.size(0),
                num_negatives=negative_count,
            )
            if negative_indices is None:
                losses.append(pos.mean())
            else:
                neg_parent = parent[negative_indices]
                neg = poincare_dist(
                    child_sample.unsqueeze(1),
                    neg_parent,
                    c=self.curvature,
                ).min(dim=1).values
                losses.append((pos + F.relu(margin + pos - neg)).mean())
        return torch.stack(losses).mean()

    def graph_scale_balance_loss(self):
        """Keep all hierarchy scales trainable before the router becomes selective."""
        if self.last_graph_scale_weights is None:
            return torch.tensor(0.0, device=self.device)
        usage = self.last_graph_scale_weights.float().mean(dim=0).clamp_min(1e-8)
        return torch.xlogy(usage, usage * usage.numel()).sum()
