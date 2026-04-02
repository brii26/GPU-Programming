#pragma once

#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

static inline void gpu_timer_start(GpuTimer *timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
    cudaEventRecord(timer->start, 0);
}

static inline float gpu_timer_stop(GpuTimer *timer) {
    cudaEventRecord(timer->stop, 0);
    cudaEventSynchronize(timer->stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, timer->start, timer->stop);

    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);

    return elapsed_ms;
}

static inline struct timespec cpu_timer_start(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

static inline float cpu_timer_stop(struct timespec start) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    float elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0f +
                       (end.tv_nsec - start.tv_nsec) / 1000000.0f;

    return elapsed_ms;
}
