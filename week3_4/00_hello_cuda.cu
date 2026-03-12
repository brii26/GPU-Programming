// =============================================================================
// 00_hello_cuda.cu
// Week 3: Your First CUDA Program — Hello from GPU!
//
// Compile: make 00_hello_cuda   or   nvcc -o 00_hello_cuda 00_hello_cuda.cu
// Run:     ./00_hello_cuda
//
// This minimal example:
//   1. Prints "Hello" from the CPU (host)
//   2. Launches a kernel that prints "Hello from GPU!" from a thread
//   3. Prints GPU device info
//
// Use this to verify your CUDA setup works before running other demos.
// =============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error-checking macro
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  ->  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// Kernel: runs on the GPU, one thread prints a message
// -----------------------------------------------------------------------------
__global__ void helloKernel(void) {
    printf("Hello from GPU! (thread %d in block %d)\n",
           threadIdx.x, blockIdx.x);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(void) {
    printf("=== Hello CUDA ===\n\n");

    // 1. Hello from CPU
    printf("Hello from CPU (host)!\n\n");

    // 2. Launch kernel: 1 block, 1 thread
    printf("Launching kernel...\n");
    helloKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for GPU to finish
    printf("\n");

    // 3. Print GPU device info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("\n[OK] CUDA setup works! You can now run 01_2d_indexing, 02_blur, etc.\n");

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
