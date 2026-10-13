import torch
from torch_geometric.utils import add_self_loops


def distance_weighted_gcn_norm(edge_index, edge_length, num_nodes, scale=None, eps=1e-12):
    """Convert edge distances to affinities and return weighted GCN normalization."""
    edge_index = edge_index.long()
    edge_length = edge_length.reshape(-1).to(dtype=torch.float32)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
    if edge_index.size(1) != edge_length.numel():
        raise ValueError(
            f"edge_index contains {edge_index.size(1)} edges but edge_length contains "
            f"{edge_length.numel()} values."
        )
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")

    if scale is None:
        positive = edge_length[edge_length > 0]
        scale = positive.median() if positive.numel() else edge_length.new_tensor(1.0)
    else:
        scale = torch.as_tensor(scale, dtype=edge_length.dtype, device=edge_length.device)
    scale = scale.clamp_min(eps)

    affinity = torch.exp(-edge_length.clamp_min(0.0) / scale)
    normalized_edge_index, affinity = add_self_loops(
        edge_index,
        affinity,
        fill_value=1.0,
        num_nodes=num_nodes,
    )
    source, target = normalized_edge_index
    weighted_degree = affinity.new_zeros(num_nodes)
    weighted_degree.index_add_(0, target, affinity)
    degree_inv_sqrt = weighted_degree.clamp_min(eps).pow(-0.5)
    normalized_weight = degree_inv_sqrt[source] * affinity * degree_inv_sqrt[target]
    return normalized_edge_index, normalized_weight, scale
