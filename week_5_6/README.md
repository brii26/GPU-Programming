# Module 4 Lab: CUDA Matrix Multiplication

**GPU Programming Course — Accelerated Computing**

**Time Limit: 2 Hours**

---

## Overview

In this lab you will implement two versions of matrix multiplication on the GPU using CUDA, then compare their performance.

The lab has **3 parts**. You should complete them in order:

| Part | Description | Time |
|---|---|---|
| **Part 1** | Basic Matrix Multiplication (global memory only) | ~40 min |
| **Part 2** | Tiled Matrix Multiplication (shared memory) | ~50 min |
| **Part 3** | Performance Benchmark & Analysis | ~30 min |

---

## Prerequisites

- SSH access to the lab server with CUDA installed
- Basic C/C++ programming skills
- Completed Module 4 lectures (4.1 through 4.5)

---

## Quick Start

```bash
# 1. Connect to your server
ssh student@<server-ip>

# 2. Navigate to the lab folder
cd Lab-MatrixMultiplication-EN

# 3. Generate test datasets
make datasets

# 4. Verify your GPU is detected
nvidia-smi

# 5. Start with Part 1!
nano Part1-BasicMatMul/matmul_basic.cu    # or use vim/code
```

---

## Folder Structure

```
Lab-MatrixMultiplication-EN/
├── README.md                          # This file
├── Makefile                           # Build system
├── common/
│   ├── matrix_io.h                    # Matrix I/O utilities
│   └── timer.h                        # GPU/CPU timing
├── datasets/
│   ├── generate_datasets.c            # Dataset generator
│   ├── small/                         # 64x64, 100x100, 128x128
│   ├── medium/                        # 256x256, 500x500, 512x512, 256x512(rect)
│   └── large/                         # 1024x1024, 2048x2048, 4096x4096
├── Part1-BasicMatMul/
│   └── matmul_basic.cu                # YOUR CODE
├── Part2-TiledMatMul/
│   └── matmul_tiled.cu                # YOUR CODE
├── Part3-Performance/
│   └── benchmark.cu                   # Benchmark tool (complete)
└── solutions/                         # INSTRUCTOR ONLY
    ├── matmul_basic_solution.cu
    └── matmul_tiled_solution.cu
```

---

## Part 1: Basic Matrix Multiplication (40 minutes)

**Objective:** Implement a basic dense matrix multiplication kernel where each thread computes one element of the output matrix.

**File to edit:** `Part1-BasicMatMul/matmul_basic.cu`

**What you need to do:**

1. **Implement the kernel** — Each thread calculates `C[row][col] = sum(A[row][k] * B[k][col])`
2. **Allocate device memory** — Use `cudaMalloc` for matrices A, B, C
3. **Copy data to device** — Use `cudaMemcpy` with `HostToDevice`
4. **Configure grid/block** — Use 16x16 thread blocks, calculate grid size
5. **Launch the kernel** — Call your kernel with the right parameters
6. **Copy result back** — Use `cudaMemcpy` with `DeviceToHost`
7. **Free device memory** — Use `cudaFree`

**Build and test:**

```bash
# Build
make part1

# Test with small matrix
bin/matmul_basic datasets/small/mat_64x64_A.bin datasets/small/mat_64x64_B.bin datasets/small/mat_64x64_C.bin

# Test with medium matrix
bin/matmul_basic datasets/medium/mat_256x256_A.bin datasets/medium/mat_256x256_B.bin datasets/medium/mat_256x256_C.bin

# Test with non-power-of-2 (boundary test)
bin/matmul_basic datasets/small/mat_100x100_A.bin datasets/small/mat_100x100_B.bin datasets/small/mat_100x100_C.bin

# Quick auto-test
make test
```

**Expected output:**
```
Matrix A: 64 x 64
Matrix B: 64 x 64
GPU Kernel Time: 0.xxx ms
Verification: PASSED ✓
```

### Questions for Part 1

Answer these in your lab report:

**Q1.** How many floating-point operations does your kernel perform for multiplying two NxN matrices? Show your calculation.

**Q2.** How many global memory reads does your kernel perform? How many writes?

**Q3.** What is the compute-to-global-memory-access ratio (arithmetic intensity) of your kernel?

**Q4.** If your GPU has 600 GB/s memory bandwidth and 10 TFLOPS peak compute, what percentage of peak compute can your kernel achieve? Why?

---

## Part 2: Tiled Matrix Multiplication (50 minutes)

**Objective:** Implement a tiled matrix multiplication kernel using shared memory to reduce global memory accesses.

**File to edit:** `Part2-TiledMatMul/matmul_tiled.cu`

**What you need to do:**

1. **Implement the tiled kernel** — This is the main challenge!
   - Declare shared memory arrays for tile A and tile B
   - Loop over phases (number of tiles along the K dimension)
   - Each thread loads one element of A and one element of B into shared memory
   - Handle boundary conditions (load 0 for out-of-bounds elements)
   - `__syncthreads()` after loading
   - Compute partial dot product from shared memory
   - `__syncthreads()` after computing
   - Write final result with boundary check
2. **Allocate device memory**
3. **Copy data to device**
4. **Configure grid/block** — Block size must be TILE_WIDTH x TILE_WIDTH
5. **Launch the tiled kernel**
6. **Copy result back**
7. **Free device memory**

**Key concepts:**
- `TILE_WIDTH = 16` (defined at top of file)
- Number of phases = `ceil(K / TILE_WIDTH)`
- Each phase loads one TILE_WIDTH x TILE_WIDTH tile from A and B
- Boundary check: if index is out of bounds, load 0.0 instead

**Build and test:**

```bash
# Build
make part2

# Test with small matrix
bin/matmul_tiled datasets/small/mat_64x64_A.bin datasets/small/mat_64x64_B.bin datasets/small/mat_64x64_C.bin

# IMPORTANT: Test with non-power-of-2 to verify boundary handling!
bin/matmul_tiled datasets/small/mat_100x100_A.bin datasets/small/mat_100x100_B.bin datasets/small/mat_100x100_C.bin

# Test with large matrix
bin/matmul_tiled datasets/large/mat_1024x1024_A.bin datasets/large/mat_1024x1024_B.bin datasets/large/mat_1024x1024_C.bin

# Test with rectangular matrix
bin/matmul_tiled datasets/medium/mat_256x512_A.bin datasets/medium/mat_256x512_B.bin datasets/medium/mat_256x512_C.bin
```

**Expected output:**
```
Matrix A: 1024 x 1024
Matrix B: 1024 x 1024
[Basic]  GPU Kernel Time: X.XXX ms
[Tiled]  GPU Kernel Time: Y.YYY ms
Tiled Speedup over Basic: Z.ZZx
Verification: PASSED ✓
```

### Questions for Part 2

**Q5.** What is the compute-to-global-memory-access ratio of the tiled kernel with TILE_WIDTH=16? Compare it to Part 1.

**Q6.** Why do we need TWO `__syncthreads()` calls in each phase? What happens if you remove each one?

**Q7.** Why do we load `0.0` for out-of-bounds elements instead of skipping the load?

**Q8.** If TILE_WIDTH=16, each block uses how many bytes of shared memory? How many blocks can fit on an SM with 48KB shared memory?

**Q9.** What would happen if you increased TILE_WIDTH to 32? What are the trade-offs?

---

## Part 3: Performance Benchmark (30 minutes)

**Objective:** Run the benchmark tool and analyze the performance characteristics.

```bash
# Build and run benchmark
make part3
bin/benchmark
```

This will print a performance comparison table across multiple matrix sizes.

### Questions for Part 3

**Q10.** Copy the benchmark results table into your report. At what matrix size does the GPU first become faster than the CPU? Why?

**Q11.** What is the effective GFLOPS achieved by each implementation at the largest matrix size? What percentage of the GPU's theoretical peak is this?

**Q12.** Does the speedup of tiled over basic increase, decrease, or stay constant as matrix size grows? Explain why.

**Q13.** Name two additional optimizations (beyond tiling) that could further improve performance.

---

## Submission

Submit the following files:

1. `Part1-BasicMatMul/matmul_basic.cu` — Your completed Part 1
2. `Part2-TiledMatMul/matmul_tiled.cu` — Your completed Part 2
3. **Lab Report (PDF)** containing:
   - Answers to all 13 questions
   - Benchmark results table
   - Screenshot of successful test runs

---

## Useful Commands

```bash
make help          # Show all available targets
make all           # Build everything
make test          # Quick test with small matrices
make clean         # Remove compiled files
nvidia-smi         # Check GPU status
nvcc --version     # Check CUDA version
```

## Troubleshooting

**"CUDA Error: invalid device function"** — Your GPU architecture may differ. Edit the Makefile and change `-arch=sm_60` to match your GPU (e.g., `sm_75` for Turing, `sm_86` for Ampere, `sm_89` for Ada Lovelace).

**"Verification: FAILED"** — Check your indexing carefully. Common mistakes: swapping row/col, wrong Width variable in 1D indexing, missing boundary checks.

**"out of memory"** — The 4096x4096 dataset needs ~200MB GPU memory. Use smaller datasets if your GPU has limited VRAM.

---

*Good luck!*
