#!/bin/bash
#PJM -g hp250467
#PJM -L "node=1"
#PJM -L "rscgrp=spot-int"
#PJM -L "elapse=01:00:00"
#PJM -L "freq=2000"
#PJM -L "eco_state=0"
#PJM -j
#PJM -S

# Interactive Job Script for llm.c on Fugaku
# ===========================================
#
# Usage:
#   pjsub --interact job_interactive.sh --sparam "wait-time=600"
#
# Or without --no-check-directory if needed:
#   pjsub --interact job_interactive.sh --sparam "wait-time=600" --no-check-directory
#
# Resource options:
#   freq=2000 : normal mode
#   freq=2200 : boost mode (eco_state=2 only)
#   eco_state=0 : use 2 FPUs (default, full performance)
#   eco_state=2 : boost-eco mode

echo "========================================"
echo "Interactive Fugaku Session"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Environment setup for A64FX
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Change to work directory
WORKDIR="${PJM_O_WORKDIR:-/vol0006/mdt0/data/hp250467/work/llm.c}"
cd "$WORKDIR"

echo ""
echo "Working directory: $(pwd)"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo ""
echo "Quick commands:"
echo "  make TARGET=a64fx_native           # Build for A64FX"
echo "  ./test_sve_math                     # Run SVE math benchmark"
echo "  ./train_gpt2                        # Run training (if data available)"
echo ""
echo "========================================"

# Start interactive shell
exec bash --login
