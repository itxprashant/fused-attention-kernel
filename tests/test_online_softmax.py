import numpy as np
import pytest

from src.online_softmax import online_softmax_attention


def _materialized(q, k, v, causal=False, scale=None):
    n_q, d = q.shape[-2], q.shape[-1]
    n_k = k.shape[-2]
    if scale is None:
        scale = d**-0.5
    scores = np.einsum("...qd,...kd->...qk", q, k) * scale
    if causal:
        q_idx = np.arange(n_q)
        k_idx = np.arange(n_k)
        scores = np.where(q_idx[..., None] >= k_idx, scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v


@pytest.mark.parametrize("block_n", [4, 7, 16, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_materialized_softmax(block_n, causal):
    rng = np.random.default_rng(0)
    n, d = 32, 8
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    v = rng.standard_normal((n, d))
    out = online_softmax_attention(q, k, v, block_n=block_n, causal=causal)
    ref = _materialized(q, k, v, causal=causal)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_batched_and_headed():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((2, 3, 17, 8))
    k = rng.standard_normal((2, 3, 17, 8))
    v = rng.standard_normal((2, 3, 17, 8))
    out = online_softmax_attention(q, k, v, block_n=5, causal=True)
    ref = _materialized(q, k, v, causal=True)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_n_not_multiple_of_block():
    rng = np.random.default_rng(2)
    q = rng.standard_normal((13, 4))
    k = rng.standard_normal((13, 4))
    v = rng.standard_normal((13, 4))
    out = online_softmax_attention(q, k, v, block_n=8)
    ref = _materialized(q, k, v)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
