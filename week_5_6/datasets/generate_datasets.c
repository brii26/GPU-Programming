#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

/* Dataset configuration */
typedef struct {
    int rows_A;
    int cols_A;
    int rows_B;
    int cols_B;
    const char *name;
} DatasetConfig;

/* Function to create directory if it doesn't exist */
void create_directory(const char *path) {
    mkdir(path, 0755);
}

/* Write matrix to binary file */
void write_matrix_binary(const char *filename, int rows, int cols, float *matrix) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        exit(1);
    }

    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(matrix, sizeof(float), rows * cols, f);

    fclose(f);
}

/* Read matrix from binary file */
void read_matrix_binary(const char *filename, int *rows, int *cols, float **matrix) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        exit(1);
    }

    fread(rows, sizeof(int), 1, f);
    fread(cols, sizeof(int), 1, f);

    *matrix = (float *)malloc((*rows) * (*cols) * sizeof(float));
    fread(*matrix, sizeof(float), (*rows) * (*cols), f);

    fclose(f);
}

/* Generate random matrix with values between 0.0 and 1.0 */
void generate_random_matrix(int rows, int cols, float *matrix) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

/* Matrix multiplication: C = A * B */
void matrix_multiply(int rows_A, int cols_A, float *A,
                     int rows_B, int cols_B, float *B,
                     float *C) {
    /* Verify dimensions */
    if (cols_A != rows_B) {
        fprintf(stderr, "Error: Matrix dimensions incompatible for multiplication\n");
        exit(1);
    }

    int result_rows = rows_A;
    int result_cols = cols_B;

    /* Initialize C to zero */
    memset(C, 0, result_rows * result_cols * sizeof(float));

    /* Perform multiplication with optimized loop order for cache locality */
    for (int i = 0; i < result_rows; i++) {
        for (int k = 0; k < cols_A; k++) {
            float A_ik = A[i * cols_A + k];
            for (int j = 0; j < result_cols; j++) {
                C[i * result_cols + j] += A_ik * B[k * cols_B + j];
            }
        }
    }
}

/* Generate a single dataset */
void generate_dataset(DatasetConfig *config) {
    printf("Generating dataset: %s (%dx%d * %dx%d)...\n",
           config->name, config->rows_A, config->cols_A, config->rows_B, config->cols_B);
    fflush(stdout);

    int rows_A = config->rows_A;
    int cols_A = config->cols_A;
    int rows_B = config->rows_B;
    int cols_B = config->cols_B;

    /* Allocate matrices */
    float *A = (float *)malloc(rows_A * cols_A * sizeof(float));
    float *B = (float *)malloc(rows_B * cols_B * sizeof(float));
    float *C = (float *)malloc(rows_A * cols_B * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed for dataset %s\n", config->name);
        exit(1);
    }

    /* Generate random matrices */
    printf("  Generating random matrices...\n");
    fflush(stdout);
    generate_random_matrix(rows_A, cols_A, A);
    generate_random_matrix(rows_B, cols_B, B);

    /* Compute expected result */
    printf("  Computing matrix multiplication (this may take a moment for large matrices)...\n");
    fflush(stdout);
    matrix_multiply(rows_A, cols_A, A, rows_B, cols_B, B, C);

    /* Create filename strings and save */
    char filename_A[256], filename_B[256], filename_C[256];

    snprintf(filename_A, sizeof(filename_A), "%s_A.bin", config->name);
    snprintf(filename_B, sizeof(filename_B), "%s_B.bin", config->name);
    snprintf(filename_C, sizeof(filename_C), "%s_C.bin", config->name);

    printf("  Writing files: %s, %s, %s\n", filename_A, filename_B, filename_C);
    fflush(stdout);

    write_matrix_binary(filename_A, rows_A, cols_A, A);
    write_matrix_binary(filename_B, rows_B, cols_B, B);
    write_matrix_binary(filename_C, rows_A, cols_B, C);

    /* Free memory */
    free(A);
    free(B);
    free(C);

    printf("  Done!\n\n");
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    /* Initialize random number generator with fixed seed for reproducibility */
    srand(42);

    /* Check for command-line arguments */
    int include_4096 = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("Generate matrix datasets for CUDA lab\n\n");
            printf("OPTIONS:\n");
            printf("  --all       Include 4096x4096 dataset (very slow on CPU)\n");
            printf("  --help, -h  Show this help message\n");
            return 0;
        } else if (strcmp(argv[1], "--all") == 0) {
            include_4096 = 1;
        }
    }

    /* Create directories if they don't exist */
    printf("Creating directories...\n");
    create_directory("small");
    create_directory("medium");
    create_directory("large");
    printf("Done!\n\n");

    /* Define all datasets */
    DatasetConfig datasets[] = {
        /* Small datasets */
        {64, 64, 64, 64, "small/mat_64x64"},
        {128, 128, 128, 128, "small/mat_128x128"},
        {100, 100, 100, 100, "small/mat_100x100"},

        /* Medium datasets */
        {256, 256, 256, 256, "medium/mat_256x256"},
        {512, 512, 512, 512, "medium/mat_512x512"},
        {500, 500, 500, 500, "medium/mat_500x500"},
        {256, 512, 512, 256, "medium/mat_256x512"},

        /* Large datasets */
        {1024, 1024, 1024, 1024, "large/mat_1024x1024"},
        {2048, 2048, 2048, 2048, "large/mat_2048x2048"},
        {4096, 4096, 4096, 4096, "large/mat_4096x4096"},
    };

    int num_datasets = sizeof(datasets) / sizeof(datasets[0]);

    /* Exclude 4096x4096 by default unless --all is specified */
    if (!include_4096) {
        num_datasets = 9;  /* Exclude the last dataset (4096x4096) */
        printf("Generating %d datasets (use --all to include 4096x4096)...\n\n", num_datasets);
    } else {
        printf("Generating %d datasets (including 4096x4096 - this will be slow)...\n\n", num_datasets);
    }

    for (int i = 0; i < num_datasets; i++) {
        generate_dataset(&datasets[i]);
    }

    printf("All datasets generated successfully!\n");
    return 0;
}
