// ============================================================
// Visual Demo: Apply Convolution to Real Images
// ============================================================
// Loads a PGM grayscale image, applies multiple filters using
// the tiled convolution kernel, and saves the results so you
// can visually see what each filter does.
//
// Usage:
//   ./conv_demo                          (uses sample_input.pgm)
//   ./conv_demo myimage.pgm              (uses your own image)
//   ./conv_demo myimage.pgm blur         (apply only one filter)
//
// Available filters: blur, gaussian, sharpen, edge, emboss, all
//
// Output files are saved as: output_<filtername>.pgm
// View them with: GIMP, Photoshop, IrfanView, or on Linux:
//   display output_blur.pgm              (ImageMagick)
//   eog output_blur.pgm                  (GNOME image viewer)
// ============================================================

#include "common.h"

#define TILE_SIZE 16
#define MAX_FILTER_SIZE 7

// Include the tiled kernel (reuse student's implementation)
#define BENCHMARK_MODE
#include "convolution_tiled.cu"

// ============================================================
// Apply one filter to an image and save the result
// ============================================================
void apply_filter(const float *d_input, float *d_output,
                  int height, int width,
                  FilterType ftype, const char *output_prefix) {

    float h_filter[49];  // max 7x7
    int K = generate_named_filter(h_filter, ftype);

    printf("\n--- Applying: %-25s ---\n", filter_name(ftype));
    printf("  Filter size: %dx%d\n", K, K);

    // Print filter weights
    printf("  Weights:\n");
    for (int fy = 0; fy < K; fy++) {
        printf("    ");
        for (int fx = 0; fx < K; fx++) {
            printf("%7.3f ", h_filter[fy * K + fx]);
        }
        printf("\n");
    }

    // Upload filter
    float *d_filter;
    int filterMemSize = K * K * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_filter, filterMemSize));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filterMemSize, cudaMemcpyHostToDevice));

    // Launch tiled kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                 (height + TILE_SIZE - 1) / TILE_SIZE);

    TIMER_CREATE(start, stop);
    TIMER_START(start);

    convTiledKernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output,
                                           height, width, K);
    CHECK_CUDA(cudaGetLastError());

    float elapsed;
    TIMER_STOP(start, stop, elapsed);
    printf("  Kernel time: %.3f ms\n", elapsed);
    TIMER_DESTROY(start, stop);

    // Download and save result
    int imageSize = height * width * sizeof(float);
    float *h_output = (float *)malloc(imageSize);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));

    // For edge detection / emboss, normalize output to [0,1] for visibility
    if (ftype == FILTER_EDGE_DETECT || ftype == FILTER_EMBOSS) {
        float minVal = h_output[0], maxVal = h_output[0];
        for (int i = 1; i < height * width; i++) {
            if (h_output[i] < minVal) minVal = h_output[i];
            if (h_output[i] > maxVal) maxVal = h_output[i];
        }
        float range = maxVal - minVal;
        if (range > 0.001f) {
            for (int i = 0; i < height * width; i++) {
                h_output[i] = (h_output[i] - minVal) / range;
            }
        }
        printf("  Output normalized: [%.3f, %.3f] -> [0, 1]\n", minVal, maxVal);
    }

    // Build output filename
    char outfile[256];
    const char *tag = "";
    switch (ftype) {
        case FILTER_BOX_BLUR:    tag = "blur";     break;
        case FILTER_GAUSSIAN:    tag = "gaussian";  break;
        case FILTER_SHARPEN:     tag = "sharpen";   break;
        case FILTER_EDGE_DETECT: tag = "edge";      break;
        case FILTER_EMBOSS:      tag = "emboss";    break;
    }
    snprintf(outfile, sizeof(outfile), "%s_%s.pgm", output_prefix, tag);

    write_pgm(outfile, h_output, height, width);
    free(h_output);
    cudaFree(d_filter);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[]) {
    const char *input_file = "sample_input.pgm";
    const char *filter_choice = "all";

    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) filter_choice = argv[2];

    printf("================================================================\n");
    printf("  Visual Convolution Demo\n");
    printf("================================================================\n");

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);

    // Load input image
    float *h_input;
    int height, width;
    if (!read_pgm(input_file, &h_input, &height, &width)) {
        printf("\nCould not load '%s'.\n", input_file);
        printf("To generate a sample image, run:\n");
        printf("  python3 generate_sample.py\n");
        printf("Then re-run this demo.\n");
        return 1;
    }

    printf("Image size: %d x %d (%d pixels, %.1f MB)\n",
           height, width, height * width,
           height * width * sizeof(float) / (1024.0f * 1024.0f));

    // Allocate device memory
    int imageSize = height * width * sizeof(float);
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));

    // Determine which filters to run
    int run_blur = 0, run_gauss = 0, run_sharp = 0, run_edge = 0, run_emboss = 0;

    if (strcmp(filter_choice, "all") == 0) {
        run_blur = run_gauss = run_sharp = run_edge = run_emboss = 1;
    } else if (strcmp(filter_choice, "blur") == 0)     run_blur = 1;
      else if (strcmp(filter_choice, "gaussian") == 0) run_gauss = 1;
      else if (strcmp(filter_choice, "sharpen") == 0)  run_sharp = 1;
      else if (strcmp(filter_choice, "edge") == 0)     run_edge = 1;
      else if (strcmp(filter_choice, "emboss") == 0)   run_emboss = 1;
      else {
        printf("Unknown filter '%s'. Options: blur, gaussian, sharpen, edge, emboss, all\n",
               filter_choice);
        return 1;
    }

    if (run_blur)   apply_filter(d_input, d_output, height, width, FILTER_BOX_BLUR,    "output");
    if (run_gauss)  apply_filter(d_input, d_output, height, width, FILTER_GAUSSIAN,     "output");
    if (run_sharp)  apply_filter(d_input, d_output, height, width, FILTER_SHARPEN,      "output");
    if (run_edge)   apply_filter(d_input, d_output, height, width, FILTER_EDGE_DETECT,  "output");
    if (run_emboss) apply_filter(d_input, d_output, height, width, FILTER_EMBOSS,       "output");

    printf("\n================================================================\n");
    printf("  Demo complete! Open the output_*.pgm files to see results.\n");
    printf("================================================================\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
