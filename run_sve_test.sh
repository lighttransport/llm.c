#!/bin/bash
# SVE Kernels Validation and Performance Test Script
# Builds both SVE and non-SVE versions and compares them

set -e

echo "========================================"
echo "SVE Kernels Test on Fugaku"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

cd /vol0006/mdt0/data/hp250467/work/llm.c

export OMP_NUM_THREADS=1

echo ""
echo "Compiler: fcc (Fujitsu compiler)"
fcc --version 2>&1 | head -2

# Build SVE version
echo ""
echo "========================================"
echo "Building SVE version..."
echo "========================================"
make clean 2>&1
make test_sve_kernels TARGET=a64fx_fcc SVE=1 2>&1 || { echo "SVE build failed"; exit 1; }
mv test_sve_kernels test_sve_kernels_sve

# Build non-SVE version
echo ""
echo "========================================"
echo "Building non-SVE (reference) version..."
echo "========================================"
make clean 2>&1
make test_sve_kernels TARGET=a64fx_fcc SVE=0 2>&1 || { echo "non-SVE build failed"; exit 1; }
mv test_sve_kernels test_sve_kernels_ref

echo ""
echo "Binary check:"
file ./test_sve_kernels_sve
file ./test_sve_kernels_ref

# Run validation (SVE version compares against internal reference)
echo ""
echo "========================================"
echo "Running VALIDATION (SVE vs Reference)"
echo "========================================"
./test_sve_kernels_sve -v -i 3 -B 2 -T 32 -C 256 -H 4

# Run performance comparison
echo ""
echo "========================================"
echo "Running PERFORMANCE (Reference only)"
echo "========================================"
./test_sve_kernels_ref -p -i 5 -B 4 -T 64 -C 768 -H 12

echo ""
echo "========================================"
echo "Running PERFORMANCE (SVE enabled)"
echo "========================================"
./test_sve_kernels_sve -p -i 5 -B 4 -T 64 -C 768 -H 12

echo ""
echo "========================================"
echo "Done at $(date)"
echo "========================================"

exit 0
