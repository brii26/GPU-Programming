// ============================================================
// Part 3: Benchmark Harness
// ============================================================
// Runs both kernels across multiple configurations and reports
// execution times and speedup. Students run this as-is after
// completing Parts 1 and 2.
//
// Usage: ./benchmark [warmup_runs] [benchmark_runs]
//   Default: 3 warmup runs, 10 benchmark runs
// ============================================================

#include "common.h"
#include <float.h>

// ============================================================
// Include student kernel implementations
// ============================================================
// BENCHMARK_MODE tells the student files to skip their main()
// so only the kernel functions are compiled into this binary.
// Students: Do NOT modify this file. Implement the kernels in
// their respective .cu files.
// ============================================================
#define BENCHMARK_MODE
#include "convolution_naive.cu"
#include "convolution_tiled.cu"

// ============================================================
// Benchmark configuration
// ============================================================
typedef struct {
    int height;
    int width;
    int filterSize;
    const char *description;
    const char *imagePath;   // PGM image file (NULL = use random data)
} BenchConfig;

BenchConfig configs[] = {
    {  512,  512, 3, "Small image, small filter",  "images/benchmark_512x512.pgm"   },
    {  512,  512, 7, "Small image, large filter",  "images/benchmark_512x512.pgm"   },
    { 2048, 2048, 3, "Large image, small filter",  "images/benchmark_2048x2048.pgm" },
    { 2048, 2048, 7, "Large image, large filter",  "images/benchmark_2048x2048.pgm" },
    { 4096, 4096, 5, "Very large image",           "images/benchmark_4096x4096.pgm" },
    { 1000, 1000, 5, "Non-power-of-2 image",       "images/benchmark_1000x1000.pgm" },
};
int numConfigs = sizeof(configs) / sizeof(configs[0]);

float benchmark_kernel(void (*launcher)(const float*, const float*, float*,
                                        int, int, int, dim3, dim3),
                       const float *d_input, const float *d_filter, float *d_output,
                       int height, int width, int filterSize,
                       dim3 gridDim, dim3 blockDim,
                       int warmup, int runs) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        launcher(d_input, d_filter, d_output, height, width, filterSize, gridDim, blockDim);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Benchmark
    TIMER_CREATE(start, stop);
    float total = 0.0f;
    for (int i = 0; i < runs; i++) {
        float elapsed;
        TIMER_START(start);
        launcher(d_input, d_filter, d_output, height, width, filterSize, gridDim, blockDim);
        TIMER_STOP(start, stop, elapsed);
        total += elapsed;
    }
    TIMER_DESTROY(start, stop);
    return total / runs;
}

void launch_naive(const float *input, const float *filter, float *output,
                  int height, int width, int filterSize,
                  dim3 gridDim, dim3 blockDim) {
    convNaiveKernel<<<gridDim, blockDim>>>(input, filter, output,
                                           height, width, filterSize);
}

void launch_tiled(const float *input, const float *filter, float *output,
                  int height, int width, int filterSize,
                  dim3 gridDim, dim3 blockDim) {
    convTiledKernel<<<gridDim, blockDim>>>(input, filter, output,
                                           height, width, filterSize);
}

int main(int argc, char *argv[]) {
    int warmup = 3;
    int runs = 10;
    if (argc >= 2) warmup = atoi(argv[1]);
    if (argc >= 3) runs = atoi(argv[2]);

    printf("================================================================\n");
    printf("  2D Convolution Benchmark: Naive vs. Tiled\n");
    printf("  Warmup: %d runs, Benchmark: %d runs (averaged)\n", warmup, runs);
    printf("================================================================\n\n");

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Warp Size: %d\n\n", prop.warpSize);

    printf("%-35s | %12s | %12s | %8s\n",
           "Configuration", "Naive (ms)", "Tiled (ms)", "Speedup");
    printf("------------------------------------+--------------+--------------+----------\n");

    for (int c = 0; c < numConfigs; c++) {
        int H = configs[c].height;
        int W = configs[c].width;
        int K = configs[c].filterSize;

        int imageSize = H * W * sizeof(float);
        int filterMemSize = K * K * sizeof(float);

        // Allocate and initialize
        float *h_input = NULL;
        float *h_filter = (float *)malloc(filterMemSize);
        generate_filter(h_filter, K);

        // Try to load PGM image; fall back to random data if not found
        int imgH, imgW;
        if (configs[c].imagePath && read_pgm(configs[c].imagePath, &h_input, &imgH, &imgW)) {
            // Verify dimensions match
            if (imgH != H || imgW != W) {
                printf("  Warning: image %dx%d doesn't match config %dx%d, using random data\n",
                       imgW, imgH, W, H);
                free(h_input);
                h_input = (float *)malloc(imageSize);
                generate_random_image(h_input, H, W, 42);
            }
        } else {
            h_input = (float *)malloc(imageSize);
            generate_random_image(h_input, H, W, 42);
            printf("  (using random data - run 'python3 generate_sample.py' for real images)\n");
        }

        float *d_input, *d_output_naive, *d_output_tiled, *d_filter;
        CHECK_CUDA(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA(cudaMalloc(&d_output_naive, imageSize));
        CHECK_CUDA(cudaMalloc(&d_output_tiled, imageSize));
        CHECK_CUDA(cudaMalloc(&d_filter, filterMemSize));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filterMemSize, cudaMemcpyHostToDevice));

        // Grid/block dims
        dim3 blockDim_naive(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim_naive((W + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 blockDim_tiled(TILE_SIZE, TILE_SIZE);
        dim3 gridDim_tiled((W + TILE_SIZE - 1) / TILE_SIZE,
                           (H + TILE_SIZE - 1) / TILE_SIZE);

        // Benchmark
        float naive_ms = benchmark_kernel(launch_naive, d_input, d_filter,
                                          d_output_naive, H, W, K,
                                          gridDim_naive, blockDim_naive,
                                          warmup, runs);

        float tiled_ms = benchmark_kernel(launch_tiled, d_input, d_filter,
                                          d_output_tiled, H, W, K,
                                          gridDim_tiled, blockDim_tiled,
                                          warmup, runs);

        char configStr[64];
        snprintf(configStr, sizeof(configStr), "%dx%d, %dx%d filter", H, W, K, K);

        printf("%-35s | %10.3f   | %10.3f   | %6.2fx\n",
               configStr, naive_ms, tiled_ms, naive_ms / tiled_ms);

        // Cleanup
        cudaFree(d_input); cudaFree(d_output_naive);
        cudaFree(d_output_tiled); cudaFree(d_filter);
        free(h_input); free(h_filter);
    }

    printf("\n================================================================\n");
    printf("  Benchmark complete. Use these numbers for your Part 3 report.\n");
    printf("================================================================\n");

    return 0;
}
