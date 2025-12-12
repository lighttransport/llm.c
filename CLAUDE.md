# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llm.c is a pure C/CUDA implementation of LLM training (GPT-2/GPT-3). The philosophy is simple, readable code in the main tree with optimized implementations in dev/cuda/.

## Build Commands

Default target is A64FX cross-compilation. Use `TARGET=` to select platform:

```bash
# A64FX cross-compile (default)
make train_gpt2
make train_gpt2 TARGET=a64fx

# A64FX with custom sysroot
make train_gpt2 A64FX_SYSROOT=/path/to/aarch64-sysroot

# A64FX native (when running on A64FX)
make train_gpt2 TARGET=a64fx_native

# Intel/AMD x86_64
make train_gpt2 TARGET=intel

# GPU training binary (default BF16 precision)
make train_gpt2cu TARGET=intel

# GPU with specific precision
make train_gpt2cu TARGET=intel PRECISION=FP32
make train_gpt2cu TARGET=intel PRECISION=FP16
make train_gpt2cu TARGET=intel PRECISION=BF16

# Enable cuDNN Flash Attention (slower compile, faster runtime)
make train_gpt2cu TARGET=intel USE_CUDNN=1

# Build tests
make test_gpt2      # CPU tests
make test_gpt2cu    # GPU tests (requires TARGET=intel)

# Clean build artifacts
make clean
```

## A64FX SVE Optimizations

SVE-optimized kernels in `llmc/sve_kernels.h` (enabled automatically for a64fx/a64fx_native targets):
- matmul (blocked for cache efficiency)
- layernorm
- encoder (embedding lookup)
- gelu activation
- residual connections
- softmax

SVE fast math functions in `llmc/sve_math.h`:
- exp (using FSCALE instruction)
- log (polynomial approximation)
- sin/cos (Taylor series with range reduction)
- tan, tanh

Enable profiling with `-DSVE_MATH_PROFILE` to measure ops/sec.

Run with OpenMP for A64FX's 48 cores:
```bash
OMP_NUM_THREADS=48 ./train_gpt2
```

## Fugaku Job Submission

```bash
# Interactive session (1 hour, spot queue)
pjsub --interact job_interactive.sh --sparam "wait-time=600"

# Batch job
pjsub job_fugaku.sh

# SVE math benchmark
make TARGET=a64fx_native
./test_sve_math -n 1000000 -w 5 -i 100
```

## Running Tests

Tests validate against PyTorch reference outputs stored in `gpt2_124M_debug_state.bin`.

```bash
# Run GPU tests
./test_gpt2cu

# Test with specific options
./test_gpt2cu -r 0    # No activation recomputation
./test_gpt2cu -r 2    # Recompute LayerNorms
./test_gpt2cu -w 0    # No master weights
./test_gpt2cu -b 32   # Custom batch size
```

## Training

```bash
# Quick start - downloads models and data
./dev/download_starter_pack.sh

# CPU training
OMP_NUM_THREADS=8 ./train_gpt2

# GPU training
./train_gpt2cu

# Multi-GPU with MPI
mpirun -np 8 ./train_gpt2cu
```

Key training flags: `-i` (data dir), `-b` (batch size), `-t` (seq len), `-l` (learning rate), `-x` (max steps), `-r` (recompute level 0-2), `-f` (flash attention).

## Architecture

### Core Files
- `train_gpt2.c` - CPU reference implementation (~1,182 lines)
- `train_gpt2.cu` - Main GPU training code (~1,904 lines)
- `train_gpt2.py` - PyTorch reference for validation

### Library Headers (llmc/)
All headers are in `llmc/` - key ones:
- `dataloader.h` - Multi-GPU data loading from .bin files
- `tokenizer.h` - GPT-2 tokenizer
- `cuda_common.h` - CUDA error checking macros
- `encoder.cuh`, `layernorm.cuh`, `attention.cuh`, `matmul.cuh` - Layer kernels
- `adamw.cuh` - Optimizer
- `zero.cuh` - Multi-GPU ZeRO training
- `sve_kernels.h` - SVE-optimized kernels for A64FX (ARM)
- `sve_math.h` - SVE fast math (exp, log, sin, cos, tan, tanh) with profiling

### Development Code (dev/)
- `dev/cuda/` - Optimized CUDA kernel library with individual kernel files
- `dev/data/` - Dataset preprocessing scripts (tinyshakespeare.py, fineweb.py, etc.)
- `dev/eval/` - Evaluation and export scripts

### Training Scripts (scripts/)
Pre-configured training scripts for different model sizes: `run_gpt2_124M.sh`, `run_gpt2_350M.sh`, etc.

## Data Format

Tokenized data stored as binary .bin files with 256-int header:
- Header: magic (20240520), version (1), token count
- Body: uint16 token stream

Preprocess with: `python dev/data/tinyshakespeare.py` (or other dataset scripts)

## Model Structure

GPT2Config defines: max_seq_len, vocab_size, num_layers, num_heads, channels. ParameterTensors holds 16 weight types: embeddings (wte, wpe), layer norms, attention (qkv, proj), feedforward (fc, fcproj).

## Key Design Patterns

- Header-only libraries in llmc/ (no separate compilation)
- Forward/backward pass pairs for each layer
- PyTorch reference used to generate test fixtures
- Mixed precision: BF16 default with FP32 master weights
- Multi-GPU via NCCL with ZeRO-style gradient sharding
