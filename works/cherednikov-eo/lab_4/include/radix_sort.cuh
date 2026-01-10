#ifndef CUDA_COURCE_2025_RADIX_SORT_CUH
#define CUDA_COURCE_2025_RADIX_SORT_CUH

#include <cuda_runtime.h>
#include <cstdint>

void radix_sort_uint32(uint32_t* d_in, uint32_t* d_out, int n);
void radix_sort_uint64(uint64_t* d_in, uint64_t* d_out, int n);
void radix_sort_int32(int32_t* d_in, int32_t* d_out, int n);
void radix_sort_int64(int64_t* d_in, int64_t* d_out, int n);


template<typename T>
struct RadixMethods2;

template<>
struct RadixMethods2<uint32_t> {
    static constexpr bool is_signed = false;
    static constexpr const char* name = "uint32_t";
    static void sort(uint32_t* d_in, uint32_t* d_out, int n) { radix_sort_uint32(d_in, d_out, n); }
};

template<>
struct RadixMethods2<uint64_t> {
    static constexpr bool is_signed = false;
    static constexpr const char* name = "uint64_t";
    static void sort(uint64_t* d_in, uint64_t* d_out, int n) { radix_sort_uint64(d_in, d_out, n); }
};

template<>
struct RadixMethods2<int32_t> {
    static constexpr bool is_signed = true;
    static constexpr const char* name = "int32_t";
    static void sort(int32_t* d_in, int32_t* d_out, int n) { radix_sort_int32(d_in, d_out, n); }
};

template<>
struct RadixMethods2<int64_t> {
    static constexpr bool is_signed = true;
    static constexpr const char* name = "int64_t";
    static void sort(int64_t* d_in, int64_t* d_out, int n) { radix_sort_int64(d_in, d_out, n); }
};


#endif //CUDA_COURCE_2025_RADIX_SORT_CUH