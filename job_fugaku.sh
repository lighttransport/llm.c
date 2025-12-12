#!/bin/bash
#PJM -g hp250467
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=01:00:00"
#PJM -L "freq=2000"
#PJM -L "eco_state=0"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

# Fugaku Job Script for llm.c SVE Math Benchmark
# ================================================
#
# Submit: pjsub job_fugaku.sh
# Interactive: pjsub --interact job_fugaku.sh
#
# Resource groups:
#   small    - up to 384 nodes, 24h max
#   large    - 385-55296 nodes, 24h max
#   spot-int - interactive, 1h max (for debugging)
#
# freq options:
#   2000 - Normal mode (default)
#   2200 - Boost mode (eco_state=2 only)
#
# eco_state options:
#   0 - eco mode off (use 2 FPUs, default)
#   2 - boost-eco mode (allows freq=2200)

set -e

echo "========================================"
echo "Fugaku llm.c SVE Math Benchmark"
echo "========================================"
echo "Job ID: ${PJM_JOBID:-interactive}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Environment setup
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Change to work directory
cd ${PJM_O_WORKDIR:-/vol0006/mdt0/data/hp250467/work/llm.c}

echo ""
echo "Working directory: $(pwd)"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo ""

# Build (native compilation on A64FX)
echo "========================================"
echo "Building with TARGET=a64fx_native..."
echo "========================================"
make clean
make TARGET=a64fx_native

echo ""
echo "========================================"
echo "Running SVE Math Benchmark..."
echo "========================================"

# Run benchmark with different sizes
./test_sve_math -n 100000 -w 10 -i 100
./test_sve_math -n 1000000 -w 5 -i 50
./test_sve_math -n 10000000 -w 2 -i 10

echo ""
echo "========================================"
echo "Running train_gpt2 (if data available)..."
echo "========================================"

# Check if training data exists
if [ -f "gpt2_124M.bin" ] && [ -f "dev/data/tinyshakespeare/tiny_shakespeare_train.bin" ]; then
    echo "Training data found, running training..."
    ./train_gpt2
else
    echo "Training data not found. To prepare:"
    echo "  1. Run: ./dev/download_starter_pack.sh"
    echo "  2. Or manually download gpt2_124M.bin"
    echo "  3. Run: python dev/data/tinyshakespeare.py"
fi

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
