#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ============================================================
// CUDA Error Checking Macro
// Use after every CUDA call: CHECK_CUDA(cudaMalloc(...));
// ============================================================
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Timer Macros (using CUDA events for accurate GPU timing)
// ============================================================
#define TIMER_CREATE(start, stop) \
    cudaEvent_t start, stop; \
    CHECK_CUDA(cudaEventCreate(&start)); \
    CHECK_CUDA(cudaEventCreate(&stop));

#define TIMER_START(start) \
    CHECK_CUDA(cudaEventRecord(start, 0));

#define TIMER_STOP(start, stop, elapsed) \
    CHECK_CUDA(cudaEventRecord(stop, 0)); \
    CHECK_CUDA(cudaEventSynchronize(stop)); \
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

#define TIMER_DESTROY(start, stop) \
    CHECK_CUDA(cudaEventDestroy(start)); \
    CHECK_CUDA(cudaEventDestroy(stop));

// ============================================================
// CPU Reference Implementation (for verification)
// ============================================================
void convolution_cpu(const float *input, const float *filter, float *output,
                     int height, int width, int filterSize) {
    int half = filterSize / 2;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
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
    }
}

// ============================================================
// Verification: compare GPU output against CPU reference
// ============================================================
int verify_result(const float *gpu_output, const float *cpu_output,
                  int height, int width, float tolerance) {
    int errors = 0;
    for (int i = 0; i < height * width; i++) {
        if (fabs(gpu_output[i] - cpu_output[i]) > tolerance) {
            if (errors < 10) {  // Print first 10 mismatches
                int row = i / width;
                int col = i % width;
                printf("  MISMATCH at (%d,%d): GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       row, col, gpu_output[i], cpu_output[i],
                       fabs(gpu_output[i] - cpu_output[i]));
            }
            errors++;
        }
    }
    return errors;
}

// ============================================================
// Generate test data
// ============================================================
void generate_random_image(float *image, int height, int width, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < height * width; i++) {
        image[i] = (float)(rand() % 256) / 255.0f;  // Pixel values [0, 1]
    }
}

void generate_filter(float *filter, int filterSize) {
    // Simple averaging filter (box blur), normalized
    float val = 1.0f / (filterSize * filterSize);
    for (int i = 0; i < filterSize * filterSize; i++) {
        filter[i] = val;
    }
}

// ============================================================
// PGM Image I/O (Portable GrayMap - no external libraries)
// ============================================================
// PGM is a simple grayscale image format:
//   P5 (binary) or P2 (ASCII), width, height, max value, then pixel data.
// Students can view PGM files in GIMP, Photoshop, IrfanView, or
// on Linux with: display image.pgm   (ImageMagick)
// ============================================================

// Read a PGM file into a float array [0.0, 1.0]
// Returns 1 on success, 0 on failure
// Caller must free(*data) after use
int read_pgm(const char *filename, float **data, int *height, int *width) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", filename);
        return 0;
    }

    // Read magic number
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) { fclose(fp); return 0; }

    int binary = 0;
    if (magic[0] == 'P' && magic[1] == '5') binary = 1;
    else if (magic[0] == 'P' && magic[1] == '2') binary = 0;
    else {
        fprintf(stderr, "Error: '%s' is not a PGM file (expected P2 or P5)\n", filename);
        fclose(fp); return 0;
    }

    // Skip comments
    int c = fgetc(fp);
    while (c == '#' || c == '\n' || c == '\r' || c == ' ') {
        if (c == '#') { while ((c = fgetc(fp)) != '\n' && c != EOF); }
        c = fgetc(fp);
    }
    ungetc(c, fp);

    int w, h, maxval;
    if (fscanf(fp, "%d %d %d", &w, &h, &maxval) != 3) {
        fprintf(stderr, "Error: invalid PGM header in '%s'\n", filename);
        fclose(fp); return 0;
    }
    fgetc(fp);  // consume single whitespace after maxval

    *width = w;
    *height = h;
    *data = (float *)malloc(w * h * sizeof(float));

    if (binary) {
        // P5: binary data
        unsigned char *buf = (unsigned char *)malloc(w * h);
        if (fread(buf, 1, w * h, fp) != (size_t)(w * h)) {
            fprintf(stderr, "Error: incomplete pixel data in '%s'\n", filename);
            free(buf); free(*data); fclose(fp); return 0;
        }
        for (int i = 0; i < w * h; i++) {
            (*data)[i] = (float)buf[i] / (float)maxval;
        }
        free(buf);
    } else {
        // P2: ASCII data
        for (int i = 0; i < w * h; i++) {
            int val;
            if (fscanf(fp, "%d", &val) != 1) {
                fprintf(stderr, "Error: incomplete pixel data in '%s'\n", filename);
                free(*data); fclose(fp); return 0;
            }
            (*data)[i] = (float)val / (float)maxval;
        }
    }

    fclose(fp);
    printf("Loaded PGM: %s (%d x %d, maxval=%d)\n", filename, w, h, maxval);
    return 1;
}

// Write a float array [0.0, 1.0] as a PGM file (P5 binary)
// Values are clamped to [0, 255]
int write_pgm(const char *filename, const float *data, int height, int width) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot create '%s'\n", filename);
        return 0;
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    unsigned char *buf = (unsigned char *)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        buf[i] = (unsigned char)(val * 255.0f + 0.5f);
    }
    fwrite(buf, 1, width * height, fp);

    free(buf);
    fclose(fp);
    printf("Saved PGM: %s (%d x %d)\n", filename, width, height);
    return 1;
}

// ============================================================
// Predefined Filters for Visual Experimentation
// ============================================================
// Use with the demo program to see what different filters do.
// Each function fills a pre-allocated K×K array.
// ============================================================

typedef enum {
    FILTER_BOX_BLUR,
    FILTER_GAUSSIAN,
    FILTER_SHARPEN,
    FILTER_EDGE_DETECT,
    FILTER_EMBOSS
} FilterType;

// Generate a named filter. Returns the required filterSize (always odd).
// The caller must pass a buffer of at least 7*7 = 49 floats.
int generate_named_filter(float *filter, FilterType type) {
    int K;
    switch (type) {
        case FILTER_BOX_BLUR: {
            // 5x5 box blur (averaging)
            K = 5;
            float val = 1.0f / (K * K);
            for (int i = 0; i < K * K; i++) filter[i] = val;
            break;
        }
        case FILTER_GAUSSIAN: {
            // 5x5 Gaussian approximation (sigma ~ 1.0)
            K = 5;
            float g[25] = {
                1,  4,  7,  4, 1,
                4, 16, 26, 16, 4,
                7, 26, 41, 26, 7,
                4, 16, 26, 16, 4,
                1,  4,  7,  4, 1
            };
            float sum = 0;
            for (int i = 0; i < 25; i++) sum += g[i];
            for (int i = 0; i < 25; i++) filter[i] = g[i] / sum;
            break;
        }
        case FILTER_SHARPEN: {
            // 3x3 sharpen
            K = 3;
            float s[9] = {
                 0, -1,  0,
                -1,  5, -1,
                 0, -1,  0
            };
            for (int i = 0; i < 9; i++) filter[i] = s[i];
            break;
        }
        case FILTER_EDGE_DETECT: {
            // 3x3 Laplacian edge detector
            K = 3;
            float e[9] = {
                -1, -1, -1,
                -1,  8, -1,
                -1, -1, -1
            };
            for (int i = 0; i < 9; i++) filter[i] = e[i];
            break;
        }
        case FILTER_EMBOSS: {
            // 3x3 emboss
            K = 3;
            float m[9] = {
                -2, -1, 0,
                -1,  1, 1,
                 0,  1, 2
            };
            for (int i = 0; i < 9; i++) filter[i] = m[i];
            break;
        }
        default:
            K = 3;
            for (int i = 0; i < 9; i++) filter[i] = 0;
            filter[4] = 1.0f;  // identity
            break;
    }
    return K;
}

const char* filter_name(FilterType type) {
    switch (type) {
        case FILTER_BOX_BLUR:    return "Box Blur (5x5)";
        case FILTER_GAUSSIAN:    return "Gaussian Blur (5x5)";
        case FILTER_SHARPEN:     return "Sharpen (3x3)";
        case FILTER_EDGE_DETECT: return "Edge Detect (3x3)";
        case FILTER_EMBOSS:      return "Emboss (3x3)";
        default:                 return "Unknown";
    }
}

#endif // COMMON_H
