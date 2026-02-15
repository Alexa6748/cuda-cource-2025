#ifndef SOBEL_H
#define SOBEL_H

#include <cuda_runtime.h>

__global__ void sobelFilter(cudaTextureObject_t tex_input, unsigned char* output, int width, int height);

#endif // SOBEL_H
