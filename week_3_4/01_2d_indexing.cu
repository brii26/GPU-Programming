// =============================================================================
// 01_2d_indexing.cu
// Week 3: 2D Thread Indexing & Boundary Handling in CUDA
//
// Compile: nvcc -o 01_2d_indexing 01_2d_indexing.cu
// Run:     ./01_2d_indexing
//
// Topics covered:
//   1. Mapping 2D thread/block indices to flat 1D array indices
//   2. Boundary handling with guard conditions
//   3. Practical kernel: matrix scaling and gradient (Sobel-like)
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error-checking macro (always use this in production code)
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  ->  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// =============================================================================
// KERNEL 1: Print thread-to-index mapping
//   Shows exactly which thread handles which matrix element.
//   Only useful for small grids -- don't print in production!
// =============================================================================
__global__ void kernelPrintMapping(int width, int height) {
    // Step 1: Calculate the global column (x) and row (y) for THIS thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Step 2: Boundary check -- ALWAYS do this before accessing arrays
    if (col >= width || row >= height) return;

    // Step 3: Convert 2D (row, col) to flat 1D index (row-major order)
    int flat = row * width + col;

    printf("Block(%d,%d) Thread(%d,%d) -> [row=%d, col=%d] -> flat_idx=%d\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           row, col, flat);
}

// =============================================================================
// KERNEL 2: Scale every element of a 2D matrix
//   output[row][col] = input[row][col] * scale
// =============================================================================
__global__ void kernelScaleMatrix(const float *input, float *output,
                                   float scale, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard: blocks at the edge may have excess threads
    if (col >= width || row >= height) return;

    int idx = row * width + col;
    output[idx] = input[idx] * scale;
}

// =============================================================================
// KERNEL 3: Horizontal gradient (finite difference stencil)
//   grad[row][col] = input[row][col+1] - input[row][col-1]  (central diff)
//   Demonstrates STENCIL ACCESS -- reading neighbours of a cell.
//   Border pixels are clamped to 0.
// =============================================================================
__global__ void kernelGradientX(const float *input, float *grad,
                                 int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int idx = row * width + col;

    // Boundary: clamp border columns to 0 instead of reading out-of-bounds
    if (col == 0 || col == width - 1) {
        grad[idx] = 0.0f;
        return;
    }

    // Central difference: needs left and right neighbours
    grad[idx] = input[idx + 1] - input[idx - 1];
}

// =============================================================================
// HOST HELPERS
// =============================================================================
void printMatrix(const char *label, const float *data, int width, int height) {
    printf("\n  [%s] (%d rows x %d cols)\n", label, height, width);
    for (int r = 0; r < height; r++) {
        printf("    row %d: ", r);
        for (int c = 0; c < width; c++) {
            printf("%7.2f ", data[r * width + c]);
        }
        printf("\n");
    }
}

// Allocate host+device mirror, copy h->d, return both pointers
void allocMirror(int n, float **h, float **d, float initVal) {
    *h = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) (*h)[i] = initVal + i;
    CUDA_CHECK(cudaMalloc(d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d, *h, n * sizeof(float), cudaMemcpyHostToDevice));
}

// =============================================================================
// MAIN
// =============================================================================
int main(void) {
    printf("============================================================\n");
    printf("  Week 3 Demo: 2D Thread Indexing & Boundary Handling\n");
    printf("============================================================\n\n");

    // -------------------------------------------------------------------------
    // DEMO 1: Visualise thread-to-index mapping on a 5x7 matrix
    //         Grid = ceil(5/4) x ceil(7/4) = 2x2 blocks, each block = 4x4 threads
    //         (Aligns with exercise W3-1: calculate grid dimension manually)
    // -------------------------------------------------------------------------
    {
        printf(">>> DEMO 1: Thread index mapping for a 5x7 matrix\n");
        printf("    Grid=(2,2) blocks, Block=(4,4) threads\n\n");

        int W = 7, H = 5;
        dim3 block(4, 4);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y);

        kernelPrintMapping<<<grid, block>>>(W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("\n");
    }

    // -------------------------------------------------------------------------
    // DEMO 2: Scale a 4x6 matrix by 3.0f
    // -------------------------------------------------------------------------
    {
        printf(">>> DEMO 2: Scale a 4x6 matrix by 3.0\n");

        int W = 6, H = 4, N = W * H;
        float *h_in, *h_out, *d_in, *d_out;

        h_in  = (float *)malloc(N * sizeof(float));
        h_out = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

        CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(4, 4);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y);

        kernelScaleMatrix<<<grid, block>>>(d_in, d_out, 3.0f, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        printMatrix("Input", h_in, W, H);
        printMatrix("Output (x3.0)", h_out, W, H);

        free(h_in); free(h_out);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    // -------------------------------------------------------------------------
    // DEMO 3: Horizontal gradient on a 4x8 ramp image
    //         Left half=0, right half=100 (vertical edge in the centre)
    // -------------------------------------------------------------------------
    {
        printf("\n>>> DEMO 3: Horizontal gradient (central difference stencil)\n");

        int W = 8, H = 4, N = W * H;
        float *h_img  = (float *)malloc(N * sizeof(float));
        float *h_grad = (float *)malloc(N * sizeof(float));

        // Image: left half = 0, right half = 100 (sharp vertical edge)
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                h_img[r * W + c] = (c >= W / 2) ? 100.0f : 0.0f;

        float *d_img, *d_grad;
        CUDA_CHECK(cudaMalloc(&d_img,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_img, h_img, N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(4, 4);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y);

        kernelGradientX<<<grid, block>>>(d_img, d_grad, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_grad, d_grad, N * sizeof(float), cudaMemcpyDeviceToHost));

        printMatrix("Image (vertical edge at col 4)", h_img, W, H);
        printMatrix("Gradient X (edge appears at col 4)", h_grad, W, H);

        free(h_img); free(h_grad);
        CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_grad));
    }

    printf("\n============================================================\n");
    printf("  Key Takeaways:\n");
    printf("  1. col = blockIdx.x*blockDim.x + threadIdx.x\n");
    printf("  2. row = blockIdx.y*blockDim.y + threadIdx.y\n");
    printf("  3. flat_idx = row * width + col  (row-major)\n");
    printf("  4. ALWAYS guard: if (col>=width || row>=height) return;\n");
    printf("============================================================\n");

    cudaDeviceReset();  // Force CUDA context cleanup (prevents hang on exit)
    return 0;
}
