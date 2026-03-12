# GPU Programming — Week 3: Lab Exercises
**IF4010 — Institut Teknologi Bandung**  
**Session 2 + Session 3**

> **Scope:** 2D indexing, stencil (blur), heat diffusion — as covered in Session 1 lecture.

---

## Setup

SSH into the GPU server and navigate to your working directory:

```bash
ssh <your-username>@<server-ip>
mkdir -p ~/week3_4
cd ~/week3_4
```

Copy the starter files provided by the lecturer, then compile all:

```bash
make all
./00_hello_cuda    # verify CUDA works before proceeding
```

Verify your GPU:
```bash
nvidia-smi
make gpu_info
```

---

## Exercise W3-0: Hello CUDA (First Run!)

**Step 1.** Run your first CUDA program to verify the setup works:

```bash
./00_hello_cuda
```

You should see:
- "Hello from CPU (host)!"
- "Hello from GPU! (thread 0 in block 0)"
- Your GPU device name and compute capability

If this runs successfully, your CUDA environment is ready. Proceed to the exercises below.

---

# SESSION 2 — 2D Indexing & Image Blur

> **Goal:** Get comfortable with 2D indexing and the stencil pattern using image blur.

---

## Exercise W3-1: Run & Understand the Demo

**Step 1.** Run the indexing demo and study the output:
```bash
./01_2d_indexing
```

**Questions** (answer in comments in your code or in `report_week3_4_NIM.docx`):
1. For a matrix of size 5×7 with block size (4,4), what is the grid dimension? Calculate manually before running.
2. In Demo 1, why do some block/thread combos not appear in the output even though they were launched?
3. What does `flat_idx = row * width + col` mean? Draw it on paper for a 3×4 matrix.

---

## Exercise W3-2: Matrix Addition Kernel

Open `01_2d_indexing.cu` and create a **new file** `ex_w3_2.cu`.

**Task:** Write a kernel that computes the element-wise **matrix addition** of two matrices:

```
C[row][col] = A[row][col] + B[row][col]
```

Requirements:
- Use a 2D block/grid layout (block size = 16×16)
- Handle non-square matrices (W ≠ H)
- Verify correctness with a small test (e.g., 3×5 matrix, fill A with 1s, B with 2s, check C is all 3s)
- Print the result

**Bonus:** Extend to element-wise multiplication `C = A * B`.

```bash
nvcc -O2 -arch=sm_86 -o ex_w3_2 ex_w3_2.cu && ./ex_w3_2
```

---

## Exercise W3-3: Image Blur — Run & Modify

**Step 1.** Run the blur demo and verify it works:
```bash
./02_blur
```

You should see:
- Input: sharp vertical edge (left=0, right=100)
- Output: blurred edge
- `[OK] Blur kernel correct!` — verification passes

**Step 2.** Modify `02_blur.cu` to implement a **3×3 blur** (9-point stencil):

```
out[i][j] = (C + N + S + E + W + NE + NW + SE + SW) / 9
```

- Skip border pixels (col 0, col W-1, row 0, row H-1) — you need a full 3×3 neighbourhood
- Update the verification in `main()` for the new formula
- Test with the same 8×6 image

**Hint:** For pixel (r,c), neighbours are:
- N: (r-1,c), S: (r+1,c), E: (r,c+1), W: (r,c-1)
- NE: (r-1,c+1), NW: (r-1,c-1), SE: (r+1,c+1), SW: (r+1,c-1)

```bash
make 02_blur && ./02_blur
```

---

# SESSION 3 — Heat Diffusion

> **Goal:** Run and explore the heat diffusion simulation.

---

## Exercise W3-4: Heat Diffusion Exploration

Run the heat diffusion simulation and observe the output:
```bash
./03_heat_diffusion
```

**Part A — Modify initial conditions:**

In `03_heat_diffusion.cu`, the `initGrid()` function places a circular hot spot at the centre.  
Modify it to place **two separate hot spots** instead (e.g., top-left and bottom-right quadrants).

Observe: Does the heat merge in the centre? How many steps does it take?

**Part B — Data parallelism: grid size vs GPU time:**

Change `GRID_W` and `GRID_H` (keep them equal) and measure the **Total GPU time** reported at the end:

| GRID_W × GRID_H | Total threads | GPU time (ms) | Throughput (Mstencils/s) |
|-----------------|---------------|---------------|---------------------------|
| 256 × 256       |               |               |                           |
| 512 × 512       |               |               |                           |
| 1024 × 1024     |               |               |                           |

**Questions:**
1. For each size, how many threads are launched per step? (Hint: `gridDim × blockDim`)
2. How does GPU time scale with the number of threads? (Linear? Sub-linear? Why?)
3. Fill in the throughput column. Does throughput increase or decrease with larger grids? Explain.

**Part C — Thread count and memory access:**

1. For `GRID_W = GRID_H = 256` and `BLOCK_DIM = 16×16`, calculate:
   - Number of blocks in the grid
   - Number of threads launched per step
   - Number of **interior** cells actually updated (boundary cells are skipped)

2. In the stencil kernel, each thread reads 5 values (centre, N, S, E, W) and writes 1. Neighbouring threads in the same row access **adjacent** memory locations. Why is this access pattern well-suited for the GPU? (Hint: coalescing)

**Part D — Stability:**

The simulation has a stability condition: `alpha * dt / h² < 0.25`.  
Change `ALPHA` to `0.5f` and `DT` to `0.5f` (so `c = 0.25`, which violates the condition).  
What happens? Why does the code refuse to run?  
In one sentence: what does numerical instability represent physically?

---

## Submission

Upload to the course portal before the deadline:

1. **Source files:**
   - `ex_w3_2.cu` (matrix addition kernel)
   - `02_blur.cu` (modified with 3×3 blur)
   - `03_heat_diffusion.cu` (modified with two hot spots)

2. **Report** `report_week3_4_NIM.docx`:
   - Answers to all numbered **Questions** above
   - Your measured results from W3-4 Part B (grid size vs GPU time table)
   - One paragraph: what was the most surprising result today, and why?

---

## Quick Reference

```bash
# Compile all (use make clean first if build fails)
make clean
make all

# Run Session 2 & 3 examples
./00_hello_cuda    # first: verify CUDA works
./01_2d_indexing
./02_blur
./03_heat_diffusion

# Compile a single file
nvcc -O2 -arch=sm_86 -o myprogram myprogram.cu

# Check GPU utilisation while running
watch -n 0.5 nvidia-smi
```

### 2D Index Formula (memorise this!)
```c
int col = blockIdx.x * blockDim.x + threadIdx.x;   // X → column
int row = blockIdx.y * blockDim.y + threadIdx.y;   // Y → row
int idx = row * width + col;                        // row-major flat index
if (col >= width || row >= height) return;          // ALWAYS check boundary
```

### Grid Size Formula
```c
dim3 block(BLOCK_W, BLOCK_H);
dim3 grid( (W + block.x - 1) / block.x,
           (H + block.y - 1) / block.y );
```
