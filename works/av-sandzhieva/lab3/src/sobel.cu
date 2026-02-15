#include "../include/sobel.h"
#include "../include/utils.h"

__device__ __forceinline__ int clamp_pixel_index(int val, int max_val) {
    return max(0, min(val, max_val - 1));
}

__global__ void sobelFilter(cudaTextureObject_t tex_input, unsigned char* output, int width, int height)
{
    __shared__ unsigned char shared_tile[TILE_SIZE][TILE_SIZE+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = bx * BLOCK_SIZE + tx;
    int row = by * BLOCK_SIZE + ty;

    int local_col = tx + 1;
    int local_row = ty + 1;

    if (col < width && row < height) {
        shared_tile[local_row][local_col] = tex1Dfetch<unsigned char>(tex_input, row * width + col);
    } else {
        shared_tile[local_row][local_col] = 0;
    }


    if (tx == 0) {
        int cy = clamp_pixel_index(row, height);
        int left = clamp_pixel_index(col - 1, width);
        shared_tile[local_row][local_col - 1] = tex1Dfetch<unsigned char>(tex_input, cy * width + left);
    }
    if (tx == BLOCK_SIZE - 1) {
        int cy = clamp_pixel_index(row, height);
        int right = clamp_pixel_index(col + 1, width);
        shared_tile[local_row][local_col + 1] = tex1Dfetch<unsigned char>(tex_input, cy * width + right);
    }

    // Вертикальные гало
    if (ty == 0) {
        int top = clamp_pixel_index(row - 1, height);
        int cx = clamp_pixel_index(col, width);
        shared_tile[local_row - 1][local_col] = tex1Dfetch<unsigned char>(tex_input, top * width + cx);
    }
    if (ty == BLOCK_SIZE - 1) {
        int bottom = clamp_pixel_index(row + 1, height);
        int cx = clamp_pixel_index(col, width);
        shared_tile[local_row + 1][local_col] = tex1Dfetch<unsigned char>(tex_input, bottom * width + cx);
    }

    // Углы
    if (tx == 0 && ty == 0) {
        int top  = clamp_pixel_index(row - 1, height);
        int left = clamp_pixel_index(col - 1, width);
        shared_tile[local_row - 1][local_col - 1] = tex1Dfetch<unsigned char>(tex_input, top * width + left);
    }
    if (tx == BLOCK_SIZE - 1 && ty == 0) {
        int top   = clamp_pixel_index(row - 1, height);
        int right = clamp_pixel_index(col + 1, width);
        shared_tile[local_row - 1][local_col + 1] = tex1Dfetch<unsigned char>(tex_input, top * width + right);
    }
    if (tx == 0 && ty == BLOCK_SIZE - 1) {
        int bottom = clamp_pixel_index(row + 1, height);
        int left   = clamp_pixel_index(col - 1, width);
        shared_tile[local_row + 1][local_col - 1] = tex1Dfetch<unsigned char>(tex_input, bottom * width + left);
    }
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) {
        int bottom = clamp_pixel_index(row + 1, height);
        int right  = clamp_pixel_index(col + 1, width);
        shared_tile[local_row + 1][local_col + 1] = tex1Dfetch<unsigned char>(tex_input, bottom * width + right);
    }

    __syncthreads();

    if (col >= width || row >= height) return;

    // Граничные пиксели обнуляем
    if (col == 0 || col == width - 1 || row == 0 || row == height - 1) {
        output[row * width + col] = 0;
        return;
    }

    float gx = -shared_tile[local_row - 1][local_col - 1] + shared_tile[local_row - 1][local_col + 1]
               - 2.0f * shared_tile[local_row][local_col - 1] + 2.0f * shared_tile[local_row][local_col + 1]
               - shared_tile[local_row + 1][local_col - 1] + shared_tile[local_row + 1][local_col + 1];

    float gy = -shared_tile[local_row - 1][local_col - 1] - 2.0f * shared_tile[local_row - 1][local_col] - shared_tile[local_row - 1][local_col + 1]
               + shared_tile[local_row + 1][local_col - 1] + 2.0f * shared_tile[local_row + 1][local_col] + shared_tile[local_row + 1][local_col + 1];

    float magnitude = sqrtf(gx * gx + gy * gy);
    output[row * width + col] = (magnitude > 255.0f) ? 255 : static_cast<unsigned char>(magnitude);
}
