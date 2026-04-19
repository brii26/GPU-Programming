// ============================================================
// Part 1: Naive 2D Convolution Kernel
// ============================================================
// Each thread computes ONE output pixel by reading directly
// from global memory. No shared memory optimization.
//
// Module concepts exercised:
//   - 2D thread indexing
//   - Boundary condition handling (zero-padding)
//   - Control divergence awareness at image edges
// ============================================================

#include "common.h"

#define BLOCK_SIZE 16  // Thread block dimensions (16x16 = 256 threads)

// ============================================================
// TODO: Implement the naive convolution kernel
// ============================================================
// Each thread computes ONE output pixel. Map each thread to a
// pixel position, apply the convolution formula from Section 2
// of the README, and write the result. Handle boundary pixels
// with zero-padding (out-of-bounds input = 0).
//
// All arrays are stored in row-major order (row * width + col).
//
// Parameters:
//   input       - Input image in global memory (H x W)
//   filter      - Convolution filter in global memory (K x K)
//   output      - Output image in global memory (H x W)
//   height      - Image height (H)
//   width       - Image width (W)
//   filterSize  - Filter dimension (K), always odd
// ============================================================
__global__
void convNaiveKernel(const float *input, const float *filter, float *output,
                     int height, int width, int filterSize) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    int half = filterSize / 2;
    float sum = 0.0f;

    for (int fy = 0; fy < filterSize; fy++) {
        for (int fx = 0; fx < filterSize; fx++) {
            int inRow = row - half + fy;
            int inCol = col - half + fx;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                sum += filter[fy * filterSize + fx] * input[inRow * width + inCol];
            }
        }
    }

    output[row * width + col] = sum;
}

// ============================================================
// Host code (complete - do not modify)
// ============================================================
#ifndef BENCHMARK_MODE  // Excluded when included by benchmark.cu
int main(int argc, char *argv[]) {
    // Default parameters
    int height = 1024;
    int width = 1024;
    int filterSize = 5;

    // Parse optional command-line arguments
    if (argc >= 3) {
        height = atoi(argv[1]);
        width = atoi(argv[2]);
    }
    if (argc >= 4) {
        filterSize = atoi(argv[3]);
    }

    printf("=== Part 1: Naive Convolution ===\n");
    printf("Image: %d x %d, Filter: %d x %d\n\n", height, width, filterSize, filterSize);

    // Allocate host memory
    int imageSize = height * width * sizeof(float);
    int filterMemSize = filterSize * filterSize * sizeof(float);
    float *h_input  = (float *)malloc(imageSize);
    float *h_output = (float *)malloc(imageSize);
    float *h_ref    = (float *)malloc(imageSize);
    float *h_filter = (float *)malloc(filterMemSize);

    // Generate test data
    generate_random_image(h_input, height, width, 42);
    generate_filter(h_filter, filterSize);

    // CPU reference
    printf("Computing CPU reference...\n");
    convolution_cpu(h_input, h_filter, h_ref, height, width, filterSize);

    // Allocate device memory
    float *d_input, *d_output, *d_filter;
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA(cudaMalloc(&d_filter, filterMemSize));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filterMemSize, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Launching kernel: grid(%d,%d), block(%d,%d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    TIMER_CREATE(start, stop);
    TIMER_START(start);

    convNaiveKernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output,
                                           height, width, filterSize);
    CHECK_CUDA(cudaGetLastError());

    float elapsed;
    TIMER_STOP(start, stop, elapsed);
    printf("Kernel time: %.3f ms\n", elapsed);

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));

    // Verify
    int errors = verify_result(h_output, h_ref, height, width, 1e-4f);
    if (errors == 0) {
        printf("\n*** VERIFICATION PASSED ***\n");
    } else {
        printf("\n*** VERIFICATION FAILED *** (%d errors)\n", errors);
    }

    // Cleanup
    TIMER_DESTROY(start, stop);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_filter);
    free(h_input); free(h_output); free(h_ref); free(h_filter);

    return (errors == 0) ? 0 : 1;
}
#endif // BENCHMARK_MODE
