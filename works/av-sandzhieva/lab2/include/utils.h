#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline void generate_random_matrix(float* mat, int rows, int cols) {
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned>(time(nullptr)));
        seeded = true;
    }
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // значения от 0.0 до 10.0
    }
}

inline bool compare_matrices(const float* mat1, const float* mat2, int rows, int cols, float epsilon = 1e-5f) {
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(mat1[i] - mat2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

inline void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

#endif // UTILS_H
