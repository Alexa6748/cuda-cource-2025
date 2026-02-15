#include "../include/utils.h"
#include "../include/io.h"
#include "../include/sobel.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <input.pgm|.bmp> <output.pgm|.bmp>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    int w, h;
    unsigned char* host_input = nullptr;

    const char* in_ext = strrchr(input_path, '.');
    if (in_ext && strcmp(in_ext, ".pgm") == 0) {
        host_input = readPGM(input_path, &w, &h);
    } else if (in_ext && strcmp(in_ext, ".bmp") == 0) {
        host_input = readBMP(input_path, &w, &h);
    } else {
        fprintf(stderr, "Unsupported input format. Only .pgm and .bmp are allowed.\n");
        return 1;
    }

    if (!host_input) return 1;

    printf("Loaded image: %d × %d\n", w, h);

    size_t data_size = static_cast<size_t>(w) * h * sizeof(unsigned char);
    unsigned char* host_output = static_cast<unsigned char*>(malloc(data_size));

    unsigned char *dev_input = nullptr, *dev_output = nullptr;
    cudaTextureObject_t tex_input = 0;
    CUDA_CHECK(cudaMalloc(&dev_input, data_size));
    CUDA_CHECK(cudaMalloc(&dev_output, data_size));

    CUDA_CHECK(cudaMemcpy(dev_input, host_input, data_size, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = dev_input;
    resDesc.res.linear.desc = cudaCreateChannelDesc<unsigned char>();
    resDesc.res.linear.sizeInBytes = data_size;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType; 
    CUDA_CHECK(cudaCreateTextureObject(&tex_input, &resDesc, &texDesc, nullptr));

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    sobelFilter<<<grid_dim, block_dim>>>(dev_input, dev_output, w, h);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    printf("Kernel execution time: %.2f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(host_output, dev_output, data_size, cudaMemcpyDeviceToHost));

    const char* out_ext = strrchr(output_path, '.');
    if (out_ext && strcmp(out_ext, ".pgm") == 0) {
        writePGM(output_path, host_output, w, h);
    } else if (out_ext && strcmp(out_ext, ".bmp") == 0) {
        writeBMP(output_path, host_output, w, h);
    } else {
        fprintf(stderr, "Unsupported output format. Only .pgm and .bmp are allowed.\n");
        free(host_output);
        free(host_input);
        CUDA_CHECK(cudaFree(dev_input));
        CUDA_CHECK(cudaFree(dev_output));
        return 1;
    }

    printf("Saved result to %s\n", output_path);

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_output));
    free(host_input);
    free(host_output);

    return 0;
}
