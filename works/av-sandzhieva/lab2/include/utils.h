#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>


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
        mat[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
}


inline bool compare_matrices(const float* mat1, const float* mat2, int rows, int cols) {
    float epsilon = 1e-3f;
    float max_diff = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < rows * cols; ++i) {
        float diff = fabsf(mat1[i] - mat2[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > epsilon) {
            if (mismatches < 5) {
                std::cout << "Mismatch at [" << i/cols << "][" << i%cols << "]: "
                          << mat1[i] << " vs " << mat2[i] << " (diff=" << diff << ")\n";
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "All elements match! Max. diff(eps 1e-3f): " << std::fixed << std::setprecision(6) << max_diff << "\n";
        return true;
    } else {
        std::cout << "Finded " << mismatches << " mismatches. Max. diff: " << std::fixed << std::setprecision(6) << max_diff << "\n";
        return false;
    }
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
