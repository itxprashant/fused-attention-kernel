"""Fused attention kernel and naive baseline."""

import torch


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard materialized attention: softmax(QK^T / sqrt(d)) V."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)
    if causal:
        seq = scores.shape[-1]
        mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Fused Triton attention kernel entry point."""
    raise NotImplementedError
