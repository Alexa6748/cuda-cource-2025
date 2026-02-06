#ifndef GPU_MATRIX_MULT_H
#define GPU_MATRIX_MULT_H

__global__ void gpu_matrix_multiply_basic(const float* A, const float* B, float* C, int M, int N, int K);

__global__ void gpu_matrix_multiply_shared(const float* A, const float* B, float* C, int M, int N, int K);

#endif // GPU_MATRIX_MULT_H
