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

    // Shared memory tile dimensions (provided as a hint)
    int sharedWidth = TILE_SIZE + filterSize - 1;

    // Step 1: Declare a 2D shared memory array large enough to hold
    //         the input tile plus its halo region.
    //         Think: how big must it be in each dimension?


    // Step 2: Identify this thread's position within the block,
    //         and compute where this tile's input region starts
    //         in the global image. Remember the input region is
    //         shifted by -half relative to the output region.


    // Step 3: Cooperatively load the shared memory tile from global memory.
    //         Challenge: the shared tile has more elements than there are
    //         threads in the block (e.g., 20x20 = 400 elements but only
    //         16x16 = 256 threads). You need a strategy to cover all elements.
    //         Also handle boundary conditions: what value should you store
    //         for positions that fall outside the image?


    // Step 4: Synchronize all threads in the block.
    //         Why is this necessary before proceeding to computation?


    // Step 5: Each thread computes one output pixel by convolving the
    //         filter with the appropriate region of shared memory.
    //         Check that the output position is within image bounds
    //         before writing the result.

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
