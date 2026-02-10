#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <cassert>



// CUDA kernel: compute predicate (0 or 1) for current bit
__global__ void compute_predicate(
    const int32_t* d_input,
    int* d_predicate,
    int bit,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_predicate[idx] = (d_input[idx] >> bit) & 1;
        if (bit == 31) {
            d_predicate[idx] = 1 - d_predicate[idx];
        }
    }
}

// CUDA kernel: block scan
__global__ void block_scan_kernel(
    const int* d_predicate,
    int* d_scan,
    int* d_block_sums,
    size_t n
) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    temp[tid] = (idx < n) ? d_predicate[idx] : 0;
    __syncthreads();

    // Hillis–Steele scan within block
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write exclusive scan
    if (idx < n) d_scan[idx] = (tid == 0) ? 0 : temp[tid - 1];

    // Save last element as block sum
    if (tid == blockDim.x - 1) d_block_sums[blockIdx.x] = temp[tid];
}

// CUDA kernel: add block offsets kernel
__global__ void add_block_offsets(
    int* d_scan,
    const int* d_block_sums,
    size_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0 || idx >= n) return;
    d_scan[idx] += d_block_sums[blockIdx.x - 1];
}


// CUDA kernel: scatter elements based on prefix sum
__global__ void scatter(
    const int32_t* d_input,
    int32_t* d_output,
    const int* d_predicate,
    const int* d_scan,
    int bit,
    size_t n,
    int total_zeros
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int pred = d_predicate[idx];
        if (pred == 0) {
            d_output[idx - d_scan[idx]] = d_input[idx];
        }
        else {
            d_output[total_zeros + d_scan[idx]] = d_input[idx];
        }
    }
}

// Naive 32-bit Radix Sort GPU
void radix_sort_gpu(
    const std::vector<int32_t>& input,
    std::vector<int32_t>& output,
    float& gpu_time_ms
) {
    size_t n = input.size();
    output.resize(n);

    int32_t* d_input = nullptr, * d_output = nullptr;
    int* d_predicate = nullptr;
    int* d_scan = nullptr;

    cudaMalloc(&d_input, n * sizeof(int32_t));
    cudaMalloc(&d_output, n * sizeof(int32_t));
    cudaMalloc(&d_predicate, n * sizeof(int));
    cudaMalloc(&d_scan, n * sizeof(int));

    cudaMemcpy(d_input, input.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;


    for (int bit = 0; bit < 32; ++bit) {
        // Compute predicate (0/1)
        compute_predicate<<<gridSize, blockSize>>>(d_input, d_predicate, bit, n);
        cudaDeviceSynchronize();

        // Exclusive scan of predicate
        int numBlocks = gridSize;
        int* d_block_sums;
        cudaMalloc(&d_block_sums, numBlocks * sizeof(int));

        block_scan_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(
            d_predicate,
            d_scan,
            d_block_sums,
            n
        );
        cudaDeviceSynchronize();

        // Copy block sums to host
        std::vector<int> h_block_sums(numBlocks);
        cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        // Host scan of block sums
        for (int i = 1; i < numBlocks; i++)
            h_block_sums[i] += h_block_sums[i - 1];

        // Copy back to device
        cudaMemcpy(d_block_sums, h_block_sums.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

        // Add offsets to each block
        add_block_offsets << <gridSize, blockSize >> > (d_scan, d_block_sums, n);
        cudaDeviceSynchronize();

        cudaFree(d_block_sums);

        // Total zeros
        int last_pred, last_scan;
        cudaMemcpy(&last_pred, &d_predicate[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan, thrust::raw_pointer_cast(&d_scan[n - 1]), sizeof(int), cudaMemcpyDeviceToHost);
        int total_ones = last_pred + last_scan;
        int total_zeros = n - total_ones;

        // Scatter
        scatter << <gridSize, blockSize >> > (d_input, d_output, d_predicate,
            d_scan, bit, n, total_zeros);
        cudaDeviceSynchronize();

        // Swap buffers
        std::swap(d_input, d_output);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(output.data(), d_input, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_predicate);
    cudaFree(d_scan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void warmup_radix() {
    std::vector<int32_t> gpu_dummy(1);
    float dummy_time = 0.0f;
    radix_sort_gpu(gpu_dummy, gpu_dummy, dummy_time);
}