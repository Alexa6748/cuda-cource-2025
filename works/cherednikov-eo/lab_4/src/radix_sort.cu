#include <stdio.h>
#include <vector>
#include <chrono>
#include <stdint.h>
#include <cuda_runtime.h>
#include "error_check.cuh"

// Макросы для настройки работы с shared memory и банками памяти
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (2*THREADS_PER_BLOCK)
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS))
static constexpr int RADIX_BITS = 4;
static constexpr int RADIX = 1 << RADIX_BITS;


 // выполнение эксклюзивного префиксного суммирования в рамках одного блока
__global__ void scanSingleBlockExclusive(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint32_t* __restrict__ blockSums, int nTotal) {
    extern __shared__ uint32_t temp[];

    int t = threadIdx.x;
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;

    int ai = t;
    int bi = t + (ELEMENTS_PER_BLOCK / 2);

    // Вычисление смещений для избежания конфликтов банков памяти
    int bankA = CONFLICT_FREE_OFFSET(ai);
    int bankB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankA] = (blockOffset + ai < nTotal) ? in[blockOffset + ai] : 0u;
    temp[bi + bankB] = (blockOffset + bi < nTotal) ? in[blockOffset + bi] : 0u;

    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            int ai2 = offset * (2 * t + 1) - 1;
            int bi2 = offset * (2 * t + 2) - 1;
            ai2 += CONFLICT_FREE_OFFSET(ai2);
            bi2 += CONFLICT_FREE_OFFSET(bi2);
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }
    __syncthreads();


    if (t == 0) {
        int last = ELEMENTS_PER_BLOCK - 1;
        int lastIdx = last + CONFLICT_FREE_OFFSET(last);
        blockSums[blockIdx.x] = temp[lastIdx];
        temp[lastIdx] = 0u;
    }

    // Фаза нисходящего обхода, вычисление префиксных сумм
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai2 = offset * (2 * t + 1) - 1;
            int bi2 = offset * (2 * t + 2) - 1;
            ai2 += CONFLICT_FREE_OFFSET(ai2);
            bi2 += CONFLICT_FREE_OFFSET(bi2);

            uint32_t x = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += x;
        }
    }
    __syncthreads();

    // Записываем всё в глобальгую память
    if (blockOffset + ai < nTotal) out[blockOffset + ai] = temp[ai + bankA];
    if (blockOffset + bi < nTotal) out[blockOffset + bi] = temp[bi + bankB];
}


// добавление префиксной суммы блока к результатам сканирования
__global__ void addBlockPrefixToScanned(uint32_t* __restrict__ scanned, const uint32_t* __restrict__ blockPrefix, int nTotal) {
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;
    uint32_t add = blockPrefix[blockIdx.x];

    int t = threadIdx.x;
    int ai = blockOffset + t;
    int bi = blockOffset + t + (ELEMENTS_PER_BLOCK / 2);

    // Добавление префиксной суммы к каждому элементу блока
    if (ai < nTotal) scanned[ai] += add;
    if (bi < nTotal) scanned[bi] += add;
}


//Убираю malloc/free
struct ScanWorkspaceU32 {
    std::vector<uint32_t*> d_blockSums;
    std::vector<uint32_t*> d_blockPrefix;
    int maxN = 0;

    void release() {
        for (auto p : d_blockSums)   if (p) cudaFree(p);
        for (auto p : d_blockPrefix) if (p) cudaFree(p);
        d_blockSums.clear();
        d_blockPrefix.clear();
        maxN = 0;
    }
};


static void initScanWorkspaceU32(ScanWorkspaceU32& ws, int maxN) {
    if (ws.maxN >= maxN) return;
    ws.release();
    ws.maxN = maxN;

    int n = maxN;
    while (true) {
        int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

        uint32_t* sums = nullptr;
        uint32_t* pref = nullptr;
        CUDA_CHECK(cudaMalloc(&sums, numBlocks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&pref, numBlocks * sizeof(uint32_t)));

        ws.d_blockSums.push_back(sums);
        ws.d_blockPrefix.push_back(pref);

        if (numBlocks <= 1) break;
        n = numBlocks;
    }
}


//Функция для выполнения эксклюзивного префиксного суммирования (uint32_t)
static void scanExclusive_u32(uint32_t* d_in, uint32_t* d_out, int n, ScanWorkspaceU32& ws, int level = 0) {
    if (n <= 0) return;

    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    uint32_t* d_blockSums   = ws.d_blockSums[level];  // Массив для хранения сумм каждого блока
    uint32_t* d_blockPrefix = ws.d_blockPrefix[level];  // Массив для хранения префиксных сумм блоков

    int sharedSize = (ELEMENTS_PER_BLOCK + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK)) * (int)sizeof(uint32_t);

    scanSingleBlockExclusive<<<numBlocks, THREADS_PER_BLOCK, sharedSize>>>(d_in, d_out, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());

    if (numBlocks > 1) {
        scanExclusive_u32(d_blockSums, d_blockPrefix, numBlocks, ws, level + 1);
    } else {
        CUDA_CHECK(cudaMemsetAsync(d_blockPrefix, 0, sizeof(uint32_t)));
    }

    addBlockPrefixToScanned<<<numBlocks, THREADS_PER_BLOCK>>>(d_out, d_blockPrefix, n);
    CUDA_CHECK(cudaGetLastError());
}


template<typename T>
__device__ __forceinline__ uint32_t get_digit(T v, int shift) {
    using U = typename std::make_unsigned<T>::type;
    U x = (U)v;
    return (uint32_t)((x >> shift) & (RADIX - 1));
}


template<typename T>
__global__ void blockHistogramRadix(const T* __restrict__ in, int n, int shift, uint32_t* __restrict__ d_blockHist, int numBlocks) {
    __shared__ uint32_t shist[RADIX];

    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) shist[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t d = get_digit(in[idx], shift);
        atomicAdd(&shist[d], 1u);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < RADIX; d += blockDim.x) {
        d_blockHist[d * numBlocks + blockIdx.x] = shist[d];
    }
}


__global__ void computeGlobalCountsRadix(const uint32_t* __restrict__ d_blockHist, const uint32_t* __restrict__ d_blockPrefix, uint32_t* __restrict__ d_globalCounts, int numBlocks) {
    int d = threadIdx.x;
    if (d < RADIX) {
        uint32_t lastPref = d_blockPrefix[d * numBlocks + (numBlocks - 1)];
        uint32_t lastHist = d_blockHist[d * numBlocks + (numBlocks - 1)];
        d_globalCounts[d] = lastPref + lastHist;
    }
}


__global__ void scanExclusive(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    __shared__ uint32_t temp[RADIX];
    int t = threadIdx.x;
    if (t < RADIX) temp[t] = in[t];
    __syncthreads();

    for (int offset = 1; offset < RADIX; offset <<= 1) {
        uint32_t val = 0;
        if (t < RADIX && t >= offset) val = temp[t - offset];
        __syncthreads();
        if (t < RADIX) temp[t] += val;
        __syncthreads();
    }

    if (t < RADIX) out[t] = (t == 0) ? 0u : temp[t - 1];
}


__global__ void flagEqualsKernel(const uint32_t* __restrict__ digits, uint32_t* __restrict__ flags, int n, uint32_t k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) flags[idx] = (digits[idx] == k) ? 1u : 0u;
}


//распределение элементов по новым позициям на основе битовой маски
template<typename T>
__global__ void scatterRadix(const T* __restrict__ in, T* __restrict__ out, int n, int shift, const uint32_t* __restrict__ d_globalOffsets, const uint32_t* __restrict__ d_blockPrefix, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t digit = get_digit(in[idx], shift);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    constexpr int WARPS = THREADS_PER_BLOCK / 32;

    __shared__ uint16_t warpCount[WARPS][RADIX];
    __shared__ uint16_t warpPrefix[WARPS][RADIX];

    // 1) counts для каждого WARP на цияру
    for (int d = 0; d < RADIX; ++d) {
        unsigned m = __ballot_sync(0xFFFFFFFFu, digit == (uint32_t)d);
        if (lane == 0) warpCount[warp][d] = (uint16_t)__popc(m);
    }
    __syncthreads();

    // 2) префиксы варпов на цифру
    if (lane == 0) {
        for (int d = 0; d < RADIX; ++d) {
            uint16_t pref = 0;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                if (w == warp) break;
                pref += warpCount[w][d];
            }
            warpPrefix[warp][d] = pref;
        }
    }
    __syncthreads();

    // 3) ранжирование внутри WARP
    unsigned maskForMyDigit = 0;
    for (int d = 0; d < RADIX; ++d) {
        unsigned m = __ballot_sync(0xFFFFFFFFu, digit == (uint32_t)d);
        if ((uint32_t)d == digit) maskForMyDigit = m;
    }
    unsigned lower = maskForMyDigit & ((1u << lane) - 1u);
    uint32_t rankInWarp  = (uint32_t)__popc(lower);
    uint32_t rankInBlock = (uint32_t)warpPrefix[warp][digit] + rankInWarp;

    uint32_t base = d_globalOffsets[digit] + d_blockPrefix[digit * numBlocks + blockIdx.x];
    uint32_t pos  = base + rankInBlock;

    out[pos] = in[idx];
}


//инвертирование знакового бита в элементах массива
template<typename U>
__global__ void flipSignBitKernel(U* data, int n, U mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] ^= mask; // Инвертирование знакового бита через XOR
}


template<typename T>
static void radixSortRadix(T* d_in, T* d_out, int n) {
     if (n <= 0) return;

    int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    uint32_t* d_blockHist     = nullptr;
    uint32_t* d_blockPrefix   = nullptr;
    uint32_t* d_globalCounts  = nullptr;
    uint32_t* d_globalOffsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_blockHist,     (size_t)RADIX * numBlocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_blockPrefix,   (size_t)RADIX * numBlocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_globalCounts,  RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_globalOffsets, RADIX * sizeof(uint32_t)));

    ScanWorkspaceU32 scanWS;
    initScanWorkspaceU32(scanWS, numBlocks);

    T* src = d_in;
    T* dst = d_out;

    int numBits = (int)(sizeof(T) * 8);
	//Уменькаем количество проходов
    int passes = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < passes; ++pass) {
        int shift = pass * RADIX_BITS;

        blockHistogramRadix<<<numBlocks, THREADS_PER_BLOCK>>>(src, n, shift, d_blockHist, numBlocks);
        CUDA_CHECK(cudaGetLastError());

        for (int d = 0; d < RADIX; ++d) {
            uint32_t* inCol  = d_blockHist   + (size_t)d * numBlocks;
            uint32_t* outCol = d_blockPrefix + (size_t)d * numBlocks;
            scanExclusive_u32(inCol, outCol, numBlocks, scanWS, 0);
        }

        computeGlobalCountsRadix<<<1, 32>>>(d_blockHist, d_blockPrefix, d_globalCounts, numBlocks);
        CUDA_CHECK(cudaGetLastError());

        scanExclusive<<<1, 32>>>(d_globalCounts, d_globalOffsets);
        CUDA_CHECK(cudaGetLastError());

        scatterRadix<<<numBlocks, THREADS_PER_BLOCK>>>(src, dst, n, shift, d_globalOffsets, d_blockPrefix, numBlocks);
        CUDA_CHECK(cudaGetLastError());

        T* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_out) {
        CUDA_CHECK(cudaMemcpyAsync(d_out, src, (size_t)n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    scanWS.release();
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaFree(d_blockPrefix));
    CUDA_CHECK(cudaFree(d_globalCounts));
    CUDA_CHECK(cudaFree(d_globalOffsets));
}


void radix_sort_uint32(uint32_t* d_in, uint32_t* d_out, int n) {
    radixSortRadix<uint32_t>(d_in, d_out, n);
}


void radix_sort_uint64(uint64_t* d_in, uint64_t* d_out, int n) {
    radixSortRadix<uint64_t>(d_in, d_out, n);
}


void radix_sort_int32(int32_t* d_in, int32_t* d_out, int n) {
    uint32_t* u_in  = reinterpret_cast<uint32_t*>(d_in);
    uint32_t* u_out = reinterpret_cast<uint32_t*>(d_out);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    flipSignBitKernel<uint32_t><<<blocks, THREADS_PER_BLOCK>>>(u_in, n, 0x80000000u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    radixSortRadix<uint32_t>(u_in, u_out, n);

    flipSignBitKernel<uint32_t><<<blocks, THREADS_PER_BLOCK>>>(u_out, n, 0x80000000u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void radix_sort_int64(int64_t* d_in, int64_t* d_out, int n) {
    uint64_t* u_in  = reinterpret_cast<uint64_t*>(d_in);
    uint64_t* u_out = reinterpret_cast<uint64_t*>(d_out);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    flipSignBitKernel<uint64_t><<<blocks, THREADS_PER_BLOCK>>>(u_in, n, 0x8000000000000000ull);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    radixSortRadix<uint64_t>(u_in, u_out, n);

    flipSignBitKernel<uint64_t><<<blocks, THREADS_PER_BLOCK>>>(u_out, n, 0x8000000000000000ull);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}