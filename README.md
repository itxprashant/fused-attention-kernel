# Fused Attention Kernel

IO-aware fused attention implementation with tiling and online softmax — no materialized N×N attention matrix.

## Features

- Triton fused attention kernel (QK^T → softmax → PV in one pass)
- Online/streaming softmax for numerical stability
- Shared-memory tiling
- Benchmark harness vs naive PyTorch attention (speed + peak memory)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires CUDA-capable GPU and PyTorch with CUDA support.

## Usage

```bash
python benchmark/benchmark.py
```

## Stack

- Triton
- PyTorch
- Nsight Compute (optional profiling)
