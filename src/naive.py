"""Materialized attention baseline: writes the N×N score matrix to memory."""

import torch


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """softmax(Q Kᵀ · scale) V with an explicit N×N intermediate."""
    dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
