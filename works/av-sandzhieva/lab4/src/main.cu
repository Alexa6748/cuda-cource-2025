#include "../include/utils.h"
#include "../include/gpu_radix_sort.h"

template<typename T>
void run_test(size_t n, const char* name) {
    printf("\n %s | %zu элементов \n", name, n);

    std::vector<T> h_in(n);
    srand(42);
    for (size_t i = 0; i < n; i++) {
        h_in[i] = (T)((rand() % 2000000000LL) - 1000000000LL);
    }


    auto h_cpu = h_in;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(h_cpu.begin(), h_cpu.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();


    T *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gpu_radix_sort(d_out, d_in, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float radix_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&radix_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    thrust::sort(thrust::device, d_in, d_in + n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float thrust_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&thrust_ms, start, stop));


    std::vector<T> h_gpu(n);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = std::equal(h_gpu.begin(), h_gpu.end(), h_cpu.begin());

    printf("CPU std::sort     : %.2f ms\n", cpu_ms);
    printf("GPU Radix Sort    : %.2f ms (%.1fx)\n", radix_ms, cpu_ms / radix_ms);
    printf("Thrust::sort      : %.2f ms\n", thrust_ms);
    printf("Correctness       : %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("Lab 4: Radix Sort on CUDA\n");

    size_t sizes[] = {100000, 500000, 1000000, 2000000};
    for (int i = 0; i < 4; i++) {
        run_test<int32_t>(sizes[i], "int32_t");
        run_test<int64_t>(sizes[i], "int64_t");
    }
    return 0;
}

