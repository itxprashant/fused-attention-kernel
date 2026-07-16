"""Latency, TFLOP/s and peak-memory sweeps: naive vs fused vs SDPA."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attention import autotune_best_config, fused_attention, naive_attention

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def attention_tflops(ms: float, batch: int, heads: int, seq: int, dim: int, causal: bool) -> float:
    flops = 4.0 * batch * heads * seq * seq * dim
    if causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12


def _alloc(batch, heads, seq, dim, device, dtype):
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    return q, k, v


def _fns(q, k, v, causal, autotune):
    return {
        "naive": lambda: naive_attention(q, k, v, causal=causal),
        "fused": lambda: fused_attention(q, k, v, causal=causal, autotune=autotune),
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
    }


def bench_latency(fn, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=warmup, rep=50)


def bench_memory(fn) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def sweep(
    seqs: list[int],
    batch: int,
    heads: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    autotune: bool,
    impls: list[str],
) -> list[dict]:
    rows: list[dict] = []
    device = "cuda"
    for seq in seqs:
        print(f"N={seq}", flush=True)
        q, k, v = _alloc(batch, heads, seq, dim, device, dtype)
        fns = _fns(q, k, v, causal, autotune)
        for name in impls:
            fn = fns[name]
            try:
                ms = bench_latency(fn)
                mem = bench_memory(fn)
                tflops = attention_tflops(ms, batch, heads, seq, dim, causal)
                rows.append(
                    {
                        "impl": name,
                        "seq": seq,
                        "batch": batch,
                        "heads": heads,
                        "dim": dim,
                        "causal": int(causal),
                        "ms": ms,
                        "tflops": tflops,
                        "peak_bytes": mem,
                    }
                )
                extra = ""
                if name == "fused" and autotune:
                    cfg = autotune_best_config()
                    if cfg is not None:
                        extra = (
                            f"  config BLOCK_M={cfg.kwargs.get('BLOCK_M')} "
                            f"BLOCK_N={cfg.kwargs.get('BLOCK_N')} "
                            f"warps={cfg.num_warps} stages={cfg.num_stages}"
                        )
                print(
                    f"  {name:6s}  {ms:8.3f} ms  {tflops:6.1f} TFLOP/s  "
                    f"{mem / 1e6:8.1f} MB{extra}",
                    flush=True,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"  {name:6s}  OOM", flush=True)
                rows.append(
                    {
                        "impl": name,
                        "seq": seq,
                        "batch": batch,
                        "heads": heads,
                        "dim": dim,
                        "causal": int(causal),
                        "ms": "",
                        "tflops": "",
                        "peak_bytes": "",
                    }
                )
        del q, k, v
        torch.cuda.empty_cache()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["impl", "seq", "batch", "heads", "dim", "causal", "ms", "tflops", "peak_bytes"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _xy(rows, impl, field):
    xs, ys = [], []
    for row in rows:
        if row["impl"] != impl or row[field] == "":
            continue
        xs.append(int(row["seq"]))
        ys.append(float(row[field]))
    return xs, ys


def write_plots(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    impls = []
    for row in rows:
        if row["impl"] not in impls:
            impls.append(row["impl"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for impl in impls:
        xs, ys = _xy(rows, impl, "ms")
        if xs:
            ax.plot(xs, ys, marker="o", label=impl)
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("latency (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_n.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for impl in impls:
        xs, ys = _xy(rows, impl, "tflops")
        if xs:
            ax.plot(xs, ys, marker="o", label=impl)
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("TFLOP/s")
    ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tflops_vs_n.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for impl in impls:
        xs, bytes_ = _xy(rows, impl, "peak_bytes")
        if xs:
            ax.plot(xs, [b / 1e6 for b in bytes_], marker="o", label=impl)
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("peak allocated memory (MB)")
    ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "memory_vs_n.png", dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--quick", action="store_true", help="short sequence sweep")
    parser.add_argument(
        "--seqs",
        type=int,
        nargs="+",
        default=None,
        help="sequence lengths to sweep",
    )
    parser.add_argument(
        "--impls",
        nargs="+",
        default=["naive", "fused", "sdpa"],
        choices=["naive", "fused", "sdpa"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    seqs = args.seqs or ([256, 512, 1024] if args.quick else [512, 1024, 2048, 4096, 8192, 16384])
    print(
        f"device={torch.cuda.get_device_name(0)}  "
        f"B={args.batch} H={args.heads} D={args.dim} causal={args.causal}",
        flush=True,
    )
    rows = sweep(
        seqs,
        args.batch,
        args.heads,
        args.dim,
        args.causal,
        torch.float16,
        args.autotune,
        args.impls,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = "results_autotune" if args.autotune else "results"
    csv_path = RESULTS_DIR / f"{stem}.csv"
    write_csv(csv_path, rows)
    if not args.autotune:
        write_plots(rows, RESULTS_DIR)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
