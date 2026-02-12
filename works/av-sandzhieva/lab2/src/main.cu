#include <iostream>
#include <cstdlib> 
#include <ctime>    
#include <cstring>  
#include <chrono>   
#include <cuda_runtime.h>

#include "../include/cpu_matrix_mult.h"
#include "../include/gpu_matrix_mult.h"
#include "../include/utils.h"

#define DEFAULT_SIZE 1024

int main(int argc, char** argv) {
    int M = DEFAULT_SIZE;
    int N = DEFAULT_SIZE;
    int K = DEFAULT_SIZE;
    std::string mode = "all";  

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (i + 2 < argc) {
            M = std::atoi(argv[i]);
            N = std::atoi(argv[i+1]);
            K = std::atoi(argv[i+2]);
            i += 2;
        } else {
            std::cerr << "Usage: ./program [--mode cpu|gpu_basic|gpu_shared|all] M N K\n";
            return 1;
        }
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "Invalid matrix dimensions.\n";
        return 1;
    }

    std::cout << "Matrix dimensions: A(" << M << "x" << N << "), B(" << N << "x" << K << "), C(" << M << "x" << K << ")\n";
    std::cout << "Mode: " << mode << "\n";

    size_t sizeA = static_cast<size_t>(M) * N * sizeof(float);
    size_t sizeB = static_cast<size_t>(N) * K * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * K * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_cpu = (float*)malloc(sizeC);
    float* h_C_gpu = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        std::cerr << "Host memory allocation failed.\n";
        return 1;
    }

    generate_random_matrix(h_A, M, N);
    generate_random_matrix(h_B, N, K);

    double cpu_time = 0.0;
    if (mode == "cpu" || mode == "all") {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_matrix_multiply(h_A, h_B, h_C_cpu, M, N, K);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cpu_time = elapsed.count();
        std::cout << "CPU time: " << cpu_time << " seconds\n";
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaEvent_t start, stop;
    float gpu_time_ms = 0.0f;

    if (mode != "cpu") {
        CUDA_CHECK(cudaMalloc(&d_A, sizeA));
        CUDA_CHECK(cudaMalloc(&d_B, sizeB));
        CUDA_CHECK(cudaMalloc(&d_C, sizeC));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    if (mode == "gpu_basic" || mode == "all") {
        dim3 threads_basic(16, 16);
        dim3 blocks_basic((K + threads_basic.x - 1) / threads_basic.x, (M + threads_basic.y - 1) / threads_basic.y);

        CUDA_CHECK(cudaEventRecord(start));
        gpu_matrix_multiply_basic<<<blocks_basic, threads_basic>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
        std::cout << "GPU basic time: " << gpu_time_ms / 1000.0f << " seconds\n";

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

        if (mode == "all") {
            if (compare_matrices(h_C_cpu, h_C_gpu, M, K)) {
                std::cout << "Basic GPU results match CPU.\n";
            } else {
                std::cout << "Basic GPU results do NOT match CPU.\n";
            }
            if (cpu_time > 0.0) {
                std::cout << "Basic GPU acceleration: " << cpu_time / (gpu_time_ms / 1000.0f) << "x\n";
            }
        }
    }

    if (mode == "gpu_shared" || mode == "all") {
        const int TILE_SIZE = 32; 
        dim3 threads_shared(TILE_SIZE, TILE_SIZE);
        dim3 blocks_shared((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        CUDA_CHECK(cudaMemset(d_C, 0, sizeC));

        CUDA_CHECK(cudaEventRecord(start));
        gpu_matrix_multiply_shared<<<blocks_shared, threads_shared>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
        std::cout << "GPU shared time: " << gpu_time_ms / 1000.0f << " seconds\n";

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

        if (mode == "all") {
            if (compare_matrices(h_C_cpu, h_C_gpu, M, K)) {
                std::cout << "Shared GPU results match CPU.\n";
            } else {
                std::cout << "Shared GPU results do NOT match CPU.\n";
            }
            if (cpu_time > 0.0) {
                std::cout << "Shared GPU acceleration: " << cpu_time / (gpu_time_ms / 1000.0f) << "x\n";
            }
        }
    }

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    if (mode != "cpu") {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return 0;
}
