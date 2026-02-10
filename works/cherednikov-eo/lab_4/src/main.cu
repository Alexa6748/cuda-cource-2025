#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "error_check.cuh"
#include "radix_sort.cuh"

using namespace std;


template<typename T>
bool check_sorting(const T* data, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (data[i] < data[i - 1]) return false;
    }
    return true;
}


template<typename T>
void generate_random_data(T* data, size_t n, bool signed_type) {
    for (size_t i = 0; i < n; i++) {
        if (signed_type) data[i] = (T)((rand() % 2000000) - 1000000);
        else            data[i] = (T)(rand() % 1000000);
    }
}


template <typename T>
void free_all(T* h1, T* h2, T* h3, T* h4, T* d_in, T* d_out) {
    free(h1); free(h2); free(h3); free(h4);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}


template <typename T>
double cpuSort(T* data, size_t n) {
    auto start = chrono::high_resolution_clock::now();
    sort(data, data + n);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}


template <typename T>
using RadixFn = void (*)(T* d_in, T* d_out, int n);


template <typename T>
float gpu_radix_sort(T* d_in, T* d_out, T* h_data, size_t n, RadixFn<T> radix_func) {
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaMemcpy(d_in, h_data, n * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    radix_func(d_in, d_out, (int)n);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data, d_out, n * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}


template <typename T>
float gpu_thrust_sort(T* h_data, size_t n) {
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    thrust::device_vector<T> d_data(n);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_data.data()),
                          h_data, n * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    thrust::sort(d_data.begin(), d_data.end());
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_data, thrust::raw_pointer_cast(d_data.data()),
                          n * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

inline void print_results(const char* type_name, size_t n,
                          double cpu_time, float gpu_radix_time, float gpu_thrust_time) {
    printf("Benchmark: %s[%zu] OK\n", type_name, n);
    printf("Time: CPU=%.5fs, GPU Radix=%.5fs, GPU Thrust=%.5fs\n",
           cpu_time, gpu_radix_time / 1000.0f, gpu_thrust_time / 1000.0f);
    printf("Speedup: Radix vs CPU=%.2fx, Thrust vs CPU=%.2fx, Radix vs Thrust=%.2fx\n\n",
           cpu_time / (gpu_radix_time / 1000.0f),
           cpu_time / (gpu_thrust_time / 1000.0f),
           (gpu_thrust_time / 1000.0f) / (gpu_radix_time / 1000.0f));
}


template<typename T>
void benchmark(size_t n) {
    using method = RadixMethods2<T>;

    T* h_data   = (T*)malloc(n * sizeof(T));
    T* h_cpu    = (T*)malloc(n * sizeof(T));
    T* h_gpu    = (T*)malloc(n * sizeof(T));
    T* h_thrust = (T*)malloc(n * sizeof(T));

    T* d_in  = nullptr;
    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));

    generate_random_data(h_data, n, method::is_signed);

    memcpy(h_cpu,    h_data, n * sizeof(T));
    memcpy(h_gpu,    h_data, n * sizeof(T));
    memcpy(h_thrust, h_data, n * sizeof(T));

    double cpu_time = cpuSort(h_cpu, n);

    float gpu_radix_ms = gpu_radix_sort<T>(d_in, d_out, h_gpu, n, &method::sort);

    bool correct = check_sorting(h_gpu, n);
    if (!correct) {
        printf("Benchmark: %s[%zu] FAILED\n\n", method::name, n);
        free_all(h_data, h_cpu, h_gpu, h_thrust, d_in, d_out);
        return;
    }
    float gpu_thrust_ms = gpu_thrust_sort<T>(h_thrust, n);
    print_results(method::name, n, cpu_time, gpu_radix_ms, gpu_thrust_ms);
    free_all(h_data, h_cpu, h_gpu, h_thrust, d_in, d_out);
}


int main() {
    srand(158);

    size_t sizes[] = {1000, 100000, 5000000, 10000000};

    for (size_t n : sizes) {
        benchmark<uint32_t>(n);
        benchmark<int32_t>(n);
        benchmark<uint64_t>(n);
        benchmark<int64_t>(n);
    }

    return 0;
}