"""Show that the block-wise (m, ℓ, acc) update equals softmax(S) @ V."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.online_softmax import online_softmax_attention


def materialized(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = q @ k.T * scale
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights @ v


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 24, 8
    q, k, v = (rng.standard_normal((n, d)) for _ in range(3))
    ref = materialized(q, k, v)
    for block_n in (4, 7, 24):
        out = online_softmax_attention(q, k, v, block_n=block_n)
        err = np.max(np.abs(out - ref))
        print(f"block_n={block_n:2d}  max_abs_err={err:.3e}")


if __name__ == "__main__":
    main()
