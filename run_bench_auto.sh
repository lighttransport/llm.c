#!/bin/bash
# Auto-run benchmark script for interactive job on Fugaku

set -e

echo "========================================"
echo "SVE Math Benchmark on Fugaku"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

cd /vol0006/mdt0/data/hp250467/work/llm.c

export OMP_NUM_THREADS=1

echo ""
echo "Compiler: fcc (Fujitsu compiler)"
fcc --version 2>&1 | head -2

echo ""
echo "Building with TARGET=a64fx_fcc..."
make clean 2>&1
make TARGET=a64fx_fcc 2>&1 || { echo "Build failed"; exit 1; }

echo ""
echo "Binary check:"
file ./test_sve_math

echo ""
echo "Running SVE Math Benchmark (single thread, N=100000)..."
./test_sve_math -n 100000 -w 2 -i 10

echo ""
echo "Done at $(date)"

exit 0
