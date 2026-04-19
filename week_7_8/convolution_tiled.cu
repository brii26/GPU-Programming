// ============================================================
// Part 2: Tiled 2D Convolution with Shared Memory
// ============================================================
// Uses shared memory to load input tiles (with halo regions)
// cooperatively, reducing redundant global memory accesses.
//
// Module concepts exercised:
//   - Shared memory tiling with halo cells (Module 4)
//   - Tile boundary conditions (Module 4, Lecture 4-5)
//   - __syncthreads() barrier synchronization (Module 4)
//   - Control divergence in loading vs. computation (Module 5)
//   - Memory coalescing during cooperative loading (Module 6)
// ============================================================

#include "common.h"

#define TILE_SIZE 16    // Output tile: each block computes TILE_SIZE x TILE_SIZE output pixels
#define MAX_FILTER_SIZE 7  // Maximum supported filter size

// ============================================================
// TODO: Implement the tiled convolution kernel
// ============================================================
// Each thread block computes a TILE_SIZE × TILE_SIZE region of
// output pixels. To do so, it must load a larger region of the
// input image into shared memory (the tile + its halo).
//
// Key questions to think about:
//   - How large must the shared memory tile be?
//   - The shared tile has more elements than threads in the block.
//     How do you handle that?
//   - What value do you store for positions outside the image?
//   - When is it safe to start reading from shared memory?
//
// Review Module 4 (Lectures 4-4 and 4-5) for tiling with halo
// cells and the cooperative loading pattern.
//
// Parameters:
//   input       - Input image (H x W, row-major, global memory)
//   filter      - Convolution filter (K x K, global memory)
//   output      - Output image (H x W, row-major, global memory)
//   height, width, filterSize - Image and filter dimensions
// ============================================================
__global__
void convTiledKernel(const float *input, const float *filter, float *output,
                     int height, int width, int filterSize) {

    int half = filterSize / 2;
    int sharedWidth = TILE_SIZE + filterSize - 1;

    // Step 1: Shared memory tile sized for worst-case filter (MAX_FILTER_SIZE).
    //         Only [sharedWidth x sharedWidth] elements are used at runtime.
    __shared__ float tile[TILE_SIZE + MAX_FILTER_SIZE - 1][TILE_SIZE + MAX_FILTER_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Step 2: Top-left corner of the input region this block needs.
    //         Offset by -half so the halo is included.
    int inStartRow = blockIdx.y * TILE_SIZE - half;
    int inStartCol = blockIdx.x * TILE_SIZE - half;

    // Step 3: Cooperatively load shared memory using a strided loop.
    //         sharedWidth x sharedWidth > TILE_SIZE x TILE_SIZE, so each
    //         thread may load more than one element.
    for (int i = ty; i < sharedWidth; i += TILE_SIZE) {
        for (int j = tx; j < sharedWidth; j += TILE_SIZE) {
            int globalRow = inStartRow + i;
            int globalCol = inStartCol + j;
            if (globalRow >= 0 && globalRow < height && globalCol >= 0 && globalCol < width)
                tile[i][j] = input[globalRow * width + globalCol];
            else
                tile[i][j] = 0.0f;  // zero-pad out-of-bounds
        }
    }

    // Step 4: Wait until all threads have finished loading shared memory
    //         before any thread reads from it for computation.
    __syncthreads();

    // Step 5: Each thread computes one output pixel from shared memory.
    int outRow = blockIdx.y * TILE_SIZE + ty;
    int outCol = blockIdx.x * TILE_SIZE + tx;

    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
                sum += filter[fy * filterSize + fx] * tile[ty + fy][tx + fx];
            }
        }
        output[outRow * width + outCol] = sum;
    }
}

// ============================================================
// Host code (complete - do not modify)
// ============================================================
#ifndef BENCHMARK_MODE  // Excluded when included by benchmark.cu
int main(int argc, char *argv[]) {
    int height = 1024;
    int width = 1024;
    int filterSize = 5;

    if (argc >= 3) {
        height = atoi(argv[1]);
        width = atoi(argv[2]);
    }
    if (argc >= 4) {
        filterSize = atoi(argv[3]);
    }

    if (filterSize > MAX_FILTER_SIZE) {
        printf("Error: filter size %d exceeds MAX_FILTER_SIZE %d\n", filterSize, MAX_FILTER_SIZE);
        return 1;
    }

    printf("=== Part 2: Tiled Convolution with Shared Memory ===\n");
    printf("Image: %d x %d, Filter: %d x %d, Tile: %d x %d\n",
           height, width, filterSize, filterSize, TILE_SIZE, TILE_SIZE);
    printf("Shared memory tile: %d x %d = %d elements (%.1f KB)\n\n",
           TILE_SIZE + filterSize - 1, TILE_SIZE + filterSize - 1,
           (TILE_SIZE + filterSize - 1) * (TILE_SIZE + filterSize - 1),
           (TILE_SIZE + filterSize - 1) * (TILE_SIZE + filterSize - 1) * sizeof(float) / 1024.0f);

    int imageSize = height * width * sizeof(float);
    int filterMemSize = filterSize * filterSize * sizeof(float);
    float *h_input  = (float *)malloc(imageSize);
    float *h_output = (float *)malloc(imageSize);
    float *h_ref    = (float *)malloc(imageSize);
    float *h_filter = (float *)malloc(filterMemSize);

    generate_random_image(h_input, height, width, 42);
    generate_filter(h_filter, filterSize);

    printf("Computing CPU reference...\n");
    convolution_cpu(h_input, h_filter, h_ref, height, width, filterSize);

    float *d_input, *d_output, *d_filter;
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA(cudaMalloc(&d_filter, filterMemSize));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filterMemSize, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                 (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launching kernel: grid(%d,%d), block(%d,%d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    TIMER_CREATE(start, stop);
    TIMER_START(start);

    convTiledKernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output,
                                           height, width, filterSize);
    CHECK_CUDA(cudaGetLastError());

    float elapsed;
    TIMER_STOP(start, stop, elapsed);
    printf("Kernel time: %.3f ms\n", elapsed);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));

    int errors = verify_result(h_output, h_ref, height, width, 1e-4f);
    if (errors == 0) {
        printf("\n*** VERIFICATION PASSED ***\n");
    } else {
        printf("\n*** VERIFICATION FAILED *** (%d errors)\n", errors);
    }

    TIMER_DESTROY(start, stop);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_filter);
    free(h_input); free(h_output); free(h_ref); free(h_filter);

    return (errors == 0) ? 0 : 1;
}
#endif // BENCHMARK_MODE
