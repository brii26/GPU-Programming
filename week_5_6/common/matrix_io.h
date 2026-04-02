#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

static inline Matrix matrix_alloc(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)malloc(rows * cols * sizeof(float));
    if (m.data == NULL) {
        fprintf(stderr, "Error: Failed to allocate matrix of size %d x %d\n", rows, cols);
        exit(1);
    }
    return m;
}

static inline void matrix_free(Matrix *m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
    }
}

static inline void matrix_save_bin(const char *filename, Matrix *m) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }

    fwrite(&m->rows, sizeof(int), 1, f);
    fwrite(&m->cols, sizeof(int), 1, f);
    fwrite(m->data, sizeof(float), m->rows * m->cols, f);
    fclose(f);
}

static inline Matrix matrix_load_bin(const char *filename) {
    Matrix m;
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        m.data = NULL;
        m.rows = 0;
        m.cols = 0;
        return m;
    }

    fread(&m.rows, sizeof(int), 1, f);
    fread(&m.cols, sizeof(int), 1, f);
    m.data = (float *)malloc(m.rows * m.cols * sizeof(float));
    if (m.data == NULL) {
        fprintf(stderr, "Error: Failed to allocate matrix for loading\n");
        fclose(f);
        exit(1);
    }
    fread(m.data, sizeof(float), m.rows * m.cols, f);
    fclose(f);
    return m;
}

static inline void matrix_save_csv(const char *filename, Matrix *m) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }

    fprintf(f, "%d,%d\n", m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%e", m->data[i * m->cols + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static inline void matrix_print(Matrix *m, int max_display) {
    int display_rows = (m->rows < max_display) ? m->rows : max_display;
    int display_cols = (m->cols < max_display) ? m->cols : max_display;

    printf("Matrix: %d x %d\n", m->rows, m->cols);
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            printf("%8.4f ", m->data[i * m->cols + j]);
        }
        if (display_cols < m->cols) printf("...");
        printf("\n");
    }
    if (display_rows < m->rows) {
        printf("...\n");
    }
}

static inline int matrix_compare(Matrix *a, Matrix *b, float tolerance) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: Matrix dimensions do not match (%d x %d vs %d x %d)\n",
               a->rows, a->cols, b->rows, b->cols);
        return 0;
    }

    int mismatch_count = 0;
    int total_elements = a->rows * a->cols;

    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(a->data[i] - b->data[i]);
        if (diff > tolerance) {
            if (mismatch_count < 5) {
                int row = i / a->cols;
                int col = i % a->cols;
                printf("Mismatch at [%d,%d]: %.6e vs %.6e (diff: %.6e)\n",
                       row, col, a->data[i], b->data[i], diff);
            }
            mismatch_count++;
        }
    }

    if (mismatch_count > 0) {
        printf("Total mismatches: %d / %d\n", mismatch_count, total_elements);
        return 0;
    }
    return 1;
}

static inline Matrix matrix_multiply_cpu(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        fprintf(stderr, "Error: Cannot multiply matrices with dimensions %d x %d and %d x %d\n",
                A->rows, A->cols, B->rows, B->cols);
        Matrix empty = {NULL, 0, 0};
        return empty;
    }

    Matrix C = matrix_alloc(A->rows, B->cols);

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C.data[i * C.cols + j] = sum;
        }
    }

    return C;
}
