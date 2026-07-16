import pytest
import torch
import torch.nn.functional as F

from src.attention import fused_attention, naive_attention


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)


def _rand_qkv(batch, heads, seq, dim, dtype, device, extra_dim=0):
    shape = (batch, heads, seq, dim + extra_dim)
    q, k, v = (torch.randn(*shape, device=device, dtype=dtype) for _ in range(3))
    if extra_dim:
        q, k, v = q[..., :dim], k[..., :dim], v[..., :dim]
    return q, k, v


@requires_cuda
@pytest.mark.parametrize("seq", [1, 64, 128, 1000])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_matches_sdpa_seq_lengths(seq, causal):
    torch.manual_seed(0)
    q, k, v = _rand_qkv(2, 4, seq, 64, torch.float16, "cuda")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = fused_attention(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@requires_cuda
@pytest.mark.parametrize("dim", [16, 32, 64, 128])
def test_fused_matches_sdpa_head_dims(dim):
    torch.manual_seed(1)
    q, k, v = _rand_qkv(1, 2, 128, dim, torch.float16, "cuda")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = fused_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_matches_sdpa_dtypes(dtype):
    torch.manual_seed(2)
    q, k, v = _rand_qkv(2, 4, 256, 64, dtype, "cuda")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = fused_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@requires_cuda
def test_fused_noncontiguous_inputs():
    torch.manual_seed(3)
    q, k, v = _rand_qkv(2, 4, 200, 64, torch.float16, "cuda", extra_dim=8)
    assert not q.is_contiguous()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = fused_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@requires_cuda
def test_fused_matches_naive():
    torch.manual_seed(4)
    q, k, v = _rand_qkv(1, 2, 96, 32, torch.float16, "cuda")
    ref = naive_attention(q, k, v, causal=True)
    out = fused_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


def test_naive_matches_sdpa_cpu():
    torch.manual_seed(5)
    q, k, v = (torch.randn(1, 2, 64, 32) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = naive_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_fused_rejects_unsupported_head_dim():
    q, k, v = _rand_qkv(1, 1, 32, 24, torch.float16, "cuda")
    with pytest.raises(ValueError, match="HEAD_DIM"):
        fused_attention(q, k, v)
