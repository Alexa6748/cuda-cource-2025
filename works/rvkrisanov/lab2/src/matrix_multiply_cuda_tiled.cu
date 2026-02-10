#include "../include/matrix_multiply.h"
#include "../include/error_check.cuh"

#define TILE_SIZE 32
#define THREAD_WORK_SIZE 8

__global__ void matrix_multiply_kernel_tiled(
    const float * __restrict__ matrix_a,
    const float * __restrict__ matrix_b,
    float * __restrict__ matrix_result,
    const int M,
    const int N,
    const int K)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    float thread_results[THREAD_WORK_SIZE] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row_start = blockIdx.y * TILE_SIZE;
    int block_col_start = blockIdx.x * TILE_SIZE;

    float acc = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int tiled_col = tile_idx * TILE_SIZE + tx;
        int tiled_row = tile_idx * TILE_SIZE + ty;

        tile_a[ty][tx] = (block_row_start + ty < M && tiled_col < N)
            ? matrix_a[(block_row_start + ty) * N + tiled_col]
            : 0.0f;

        tile_b[ty][tx] = (tiled_row < N && block_col_start + tx < K)
            ? matrix_b[tiled_row * K + block_col_start + tx]
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += tile_a[ty][k] * tile_b[k][tx];
        }

        __syncthreads();
    }

    if (block_row_start + ty < M && block_col_start + tx < K) {
        matrix_result[(block_row_start + ty) * K + block_col_start + tx] = acc;
    }
}

void launch_matrix_multiply_tiled(
    const float *d_matrix_a,
    const float *d_matrix_b,
    float *d_matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns,
    cudaStream_t stream)
{
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    
    dim3 grid_size(
        (matrix_b_columns + TILE_SIZE - 1) / TILE_SIZE,
        (matrix_a_rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiply_kernel_tiled<<<grid_size, block_size, 0, stream>>>(
        d_matrix_a, d_matrix_b, d_matrix_result,
        matrix_a_rows, matrix_a_columns, matrix_b_columns);

    CUDA_CHECK(cudaGetLastError());
}
