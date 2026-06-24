"""Benchmark fused vs naive attention."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attention import naive_attention


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, heads, seq, dim = 1, 8, 2048, 64

    q = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float16)

    out = naive_attention(q, k, v, causal=True)
    print(f"naive attention output shape: {out.shape}")


if __name__ == "__main__":
    main()
