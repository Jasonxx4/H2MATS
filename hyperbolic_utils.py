import torch


def project(x, c=1.0, eps=1e-5):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1.0 - eps) / (c ** 0.5)
    projected = x / norm * max_norm
    return torch.where(norm > max_norm, projected, x)


def expmap0(u, c=1.0, eps=1e-5):
    norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(eps)
    sqrt_c = c ** 0.5
    mapped = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    return project(mapped, c=c, eps=eps)


def logmap0(y, c=1.0, eps=1e-5):
    y = project(y, c=c, eps=eps)
    norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(eps)
    sqrt_c = c ** 0.5
    return torch.atanh((sqrt_c * norm).clamp(max=1.0 - eps)) * y / (sqrt_c * norm)


def poincare_dist(x, y, c=1.0, eps=1e-5):
    x = project(x, c=c, eps=eps)
    y = project(y, c=c, eps=eps)
    x2 = torch.sum(x * x, dim=-1)
    y2 = torch.sum(y * y, dim=-1)
    diff2 = torch.sum((x - y) ** 2, dim=-1)
    denom = ((1.0 - c * x2) * (1.0 - c * y2)).clamp_min(eps)
    z = 1.0 + 2.0 * c * diff2 / denom
    return torch.acosh(z.clamp_min(1.0 + eps)) / (c ** 0.5)
