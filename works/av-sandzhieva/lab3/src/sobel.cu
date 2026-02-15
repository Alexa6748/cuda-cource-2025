#include "../include/sobel.h"
#include "../include/utils.h"

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height)
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

    auto clamp = [](int val, int max_val) -> int {
        return (val < 0 ? 0 : (val >= max_val ? max_val - 1 : val));
    };

    if (col < width && row < height) {
        shared_tile[local_row][local_col] = input[row * width + col];
    } else {
        shared_tile[local_row][local_col] = 0;
    }

    int cx = clamp(col, width);
    int cy = clamp(row, height);

    if (tx == 0) {
        int left = clamp(col - 1, width);
        shared_tile[local_row][local_col - 1] = input[cy * width + left];
    }
    if (tx == BLOCK_SIZE - 1) {
        int right = clamp(col + 1, width);
        shared_tile[local_row][local_col + 1] = input[cy * width + right];
    }

    // Вертикальные гало
    if (ty == 0) {
        int top = clamp(row - 1, height);
        shared_tile[local_row - 1][local_col] = input[top * width + cx];
    }
    if (ty == BLOCK_SIZE - 1) {
        int bottom = clamp(row + 1, height);
        shared_tile[local_row + 1][local_col] = input[bottom * width + cx];
    }

    // Углы
    if (tx == 0 && ty == 0) {
        shared_tile[local_row - 1][local_col - 1] = input[clamp(row - 1, height) * width + clamp(col - 1, width)];
    }
    if (tx == BLOCK_SIZE - 1 && ty == 0) {
        shared_tile[local_row - 1][local_col + 1] = input[clamp(row - 1, height) * width + clamp(col + 1, width)];
    }
    if (tx == 0 && ty == BLOCK_SIZE - 1) {
        shared_tile[local_row + 1][local_col - 1] = input[clamp(row + 1, height) * width + clamp(col - 1, width)];
    }
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) {
        shared_tile[local_row + 1][local_col + 1] = input[clamp(row + 1, height) * width + clamp(col + 1, width)];
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
