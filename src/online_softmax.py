"""CPU reference for the online-softmax recurrence used by FlashAttention."""

from __future__ import annotations

import numpy as np


def online_softmax_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    block_n: int = 16,
    causal: bool = False,
    scale: float | None = None,
) -> np.ndarray:
    """Exact tiled attention: stream K/V blocks and keep (m, ℓ, acc).

    q, k, v have shape [..., N, D] (or [..., N_q, D] / [..., N_k, D]).
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    *leading, n_q, dim = q.shape
    n_k = k.shape[-2]
    if k.shape[:-2] != q.shape[:-2] or v.shape != k.shape or v.shape[-1] != dim:
        raise ValueError("q, k, v leading dims and head dim must match")
    if block_n < 1:
        raise ValueError("block_n must be positive")

    q = q.reshape(-1, n_q, dim)
    k = k.reshape(-1, n_k, dim)
    v = v.reshape(-1, n_k, dim)
    if scale is None:
        scale = dim**-0.5

    rows = np.arange(n_q)
    running_max = np.full((q.shape[0], n_q), -np.inf)
    running_sum = np.zeros((q.shape[0], n_q), dtype=np.float64)
    acc = np.zeros_like(q)

    for start in range(0, n_k, block_n):
        end = min(start + block_n, n_k)
        scores = np.einsum("bqd,bkd->bqk", q, k[:, start:end]) * scale
        if causal:
            cols = np.arange(start, end)
            scores = np.where(rows[:, None] >= cols[None, :], scores, -np.inf)
        block_max = np.max(scores, axis=-1)
        new_max = np.maximum(running_max, block_max)
        alpha = np.exp(running_max - new_max)
        alpha = np.where(np.isfinite(alpha), alpha, 0.0)
        weights = np.exp(scores - new_max[..., None])
        weights = np.where(np.isfinite(weights), weights, 0.0)
        running_sum = alpha * running_sum + weights.sum(axis=-1)
        acc = alpha[..., None] * acc + np.einsum("bqk,bkd->bqd", weights, v[:, start:end])
        running_max = new_max

    denom = np.where(running_sum > 0, running_sum, 1.0)
    out = acc / denom[..., None]
    return out.reshape(*leading, n_q, dim)
