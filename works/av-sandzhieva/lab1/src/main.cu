#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define W 48
#define H 24
#define R 20.0f

__global__ void draw(char *buf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float dx = x - W/2.0f;
    float dy = (y - H/2.0f) * 2.0f;

    float d = sqrtf(dx*dx + dy*dy);

    buf[y*W + x] = (fabsf(d - R) < 2.1f) ? '*' : ' ';
}

int main()
{
    char *buf, *dbuf;
    buf = (char*)malloc(W*H);
    cudaMalloc(&dbuf, W*H);

    dim3 block(16,8);
    dim3 grid((W+15)/16, (H+7)/8);

    draw<<<grid,block>>>(dbuf);
    cudaDeviceSynchronize();
    cudaMemcpy(buf, dbuf, W*H, cudaMemcpyDeviceToHost);

    for(int y=0; y<H; y++) {
        for(int x=0; x<W; x++)
            putchar(buf[y*W+x]);
        putchar('\n');
    }

    cudaFree(dbuf);
    free(buf);
    return 0;
}
