"""FlashAttention-2 style fused attention kernel in Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.naive import naive_attention

__all__ = ["fused_attention", "flash_attention", "naive_attention"]

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def _autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_warps=warps, num_stages=stages)
        for block_m in (32, 64, 128)
        for block_n in (32, 64, 128)
        for warps in (4, 8)
        for stages in (2, 3, 4)
    ]


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    O,
    L,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    H,
    N_Q,
    N_K,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H

    q_base = Q + batch * stride_qb + head * stride_qh
    k_base = K + batch * stride_kb + head * stride_kh
    v_base = V + batch * stride_vb + head * stride_vh
    o_base = O + batch * stride_ob + head * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_Q, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Fold 1/ln2 into the scale so the kernel can use exp2.
    qk_scale = sm_scale * 1.4426950408889634

    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_K)
    else:
        hi = N_K

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        k_ptrs = k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=cols[None, :] < N_K, other=0.0)

        qk = tl.dot(q, k) * qk_scale

        mask = cols[None, :] < N_K
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cols[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=cols[:, None] < N_K, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_i = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(q.dtype), mask=offs_m[:, None] < N_Q)

    # Logsumexp in base-2 units; needed if a backward pass recomputes P.
    tl.store(L + pid_bh * N_Q + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < N_Q)


_attn_fwd_autotuned = triton.autotune(
    configs=_autotune_configs(),
    key=["N_Q", "N_K", "HEAD_DIM"],
)(_attn_fwd)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    autotune: bool = False,
) -> torch.Tensor:
    """Fused tiled attention. Never materializes the N×N score matrix.

    q, k, v: [B, H, N, D] (k/v may use a different N). D in {16, 32, 64, 128}.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [batch, heads, seq, dim]")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q, k, v batch/head/dim must match; k and v seq lengths must match")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"expected float16 or bfloat16, got {q.dtype}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("fused_attention requires CUDA tensors")

    batch, heads, n_q, dim = q.shape
    n_k = k.shape[2]
    if dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"HEAD_DIM must be one of {_SUPPORTED_HEAD_DIMS}, got {dim}")
    if sm_scale is None:
        sm_scale = dim**-0.5

    out = torch.empty_like(q)
    lse = torch.empty((batch * heads, n_q), device=q.device, dtype=torch.float32)
    kernel = _attn_fwd_autotuned if autotune else _attn_fwd
    grid = lambda meta: (triton.cdiv(n_q, meta["BLOCK_M"]), batch * heads)
    args = (
        q,
        k,
        v,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        heads,
        n_q,
        n_k,
        sm_scale,
    )
    if autotune:
        kernel[grid](*args, HEAD_DIM=dim, CAUSAL=causal)
    else:
        kernel[grid](
            *args,
            HEAD_DIM=dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            CAUSAL=causal,
            num_warps=4,
            num_stages=2,
        )
    return out


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Alias for fused_attention."""
    return fused_attention(q, k, v, causal=causal, **kwargs)


def autotune_best_config() -> triton.Config | None:
    """Winning autotune config from the last tuned launch, if any."""
    return getattr(_attn_fwd_autotuned, "best_config", None)
