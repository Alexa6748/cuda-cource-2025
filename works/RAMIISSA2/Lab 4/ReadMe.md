
# CUDA Laboratory Work №4 — GPU Radix Sort

---

## 1. Task Description

The objective of this laboratory work is to **implement and analyze a GPU-based Radix Sort algorithm using CUDA**. The work focuses on developing a custom parallel Radix Sort for signed integers and comparing its performance and correctness against:

1. CPU-based sorting algorithms:

   * `std::sort`
   * `qsort`
2. A GPU reference implementation:

   * `thrust::sort`

The primary goal is **not only performance comparison**, but also a **practical study of parallel scan-based algorithms**, kernel launch overhead, and memory access behavior on modern GPU architectures.

---

## 2. Radix Sort Algorithm Overview

Radix Sort is a **non-comparative sorting algorithm** that processes integers **digit-by-digit** (or bit-by-bit). In this laboratory work, a **Least Significant Bit (LSB) Radix Sort** is implemented, where the input array is processed over **32 passes**, one per bit.

For each bit position $b \in [0, 31]$:

1. A **predicate array** is computed:
   $$
   p_i =
   \begin{cases}
   1, & \text{if bit } b \text{ of } a_i = 0 \\
   0, & \text{otherwise}
   \end{cases}
   $$

2. An **exclusive prefix sum (scan)** over the predicate array computes output indices.

3. A **scatter operation** writes elements into the correct positions in the output buffer.

4. Input and output buffers are swapped for the next pass.

This approach guarantees **stable sorting**, which is a required property for Radix Sort correctness.

---

## 3. Handling Signed Integers

Radix Sort naturally operates on **unsigned integers**. To correctly sort **signed 32-bit integers**, the following technique is applied:

* During the **most significant bit (MSB) pass**, the sign bit is **inverted**.
* This transforms signed integer ordering into lexicographically sortable unsigned ordering.
* After the final pass, the resulting array reflects correct signed ordering.

This method is widely used in GPU radix sorting and avoids post-processing steps.

---

## 4. Project Structure

The project follows a **modular CUDA project layout**, separating kernels, benchmarks, and utilities.

```
Lab 4/
├── include/
│   ├── utils.h
│   └── cpu_sort.h
├── src/
│   ├── utils.cpp
│   ├── cpu_sort.cpp
│   ├── thrust_sort.cu
│   ├── radix_sort.cu
│   ├── benchmark.cu
│   └── main.cu
├── Makefile
└── README.md
```

### Structure Rationale

* `utils.h / utils.cpp`
  * Random data generation

* `cpu_sort.h / cpu_sort.cpp`
  * CPU reference sorting helpers

* `thrust_sort.cu`
  * GPU baseline using `thrust::sort`

* `radix_sort.cu`
  * Custom CUDA Radix Sort implementation

* `benchmark.cu`
  * Main execution
  * Timing
  * Correctness verification

* `main.cu`
  * Warm-ups
  * Main execution
---

## 5. Build System

The project is built using **CMake with CUDA support**.

### Key configuration details:

* C++17 and CUDA C++17
* Release build with `-O3`
* CUDA architecture targeting enabled
* Separate compilation for `.cu` files

The final executable:

* `radix_sort.exe`

---

## 6. CUDA Implementations

### 6.1 Thrust-Based GPU Baseline

The `thrust::sort` implementation serves as a **reference GPU baseline**.

Characteristics:

* Uses highly optimized internal algorithms
* Automatically selects radix-based or merge-based strategies
* Employs:

  * Multi-bit processing per pass
  * Shared memory
  * Warp-level primitives
* Minimal kernel launches

This implementation is used for both **performance comparison** and **correctness validation**.

---

### 6.2 Custom GPU Radix Sort Implementation

The custom implementation consists of the following GPU stages per bit:

1. **Predicate Kernel**
   
   * One thread per element
   * Extracts the current bit
   * Inverts the most significant bit for signed integers

2. **Two-Level Exclusive Prefix-Sum (Scan)**

   * **Block-level scan** using a Hillis–Steele algorithm in shared memory
   * **Block sums collected** and scanned on the host
   * **Offsets added** back to each block to form a global exclusive scan
   * Entire process avoids `thrust::exclusive_scan` and ensures correctness for any array size

3. **Scatter Kernel**

   * Computes final output positions using the global scan and predicate
   * Writes elements to the output buffer according to bit value
   * Correctly separates zeros and ones for each bit

4. **Buffer Swap**

   * Input and output buffers are swapped after each bit pass
   * Ensures the next bit operates on the partially sorted array

All intermediate data structures (predicate array, scan array) reside in **global memory**.

---

## 7. Experimental Setup

* Data type: `int32`
* Input sizes:

  * $10^5$
  * $10^6$
  * $10^7$
  * $10^8$
* Data distribution:

  * Uniform random integers
  * Includes negative values
* Timing method:

  * `cudaEventElapsedTime` (GPU)
  * `std::chrono` (CPU)
* Build type: **Release**

Each measurement reflects the **average of multiple runs**.

---

## 8. Performance Results

### 8.1 Execution Time Comparison

|  Array Size | CPU std::sort (ms) | CPU qsort (ms) | GPU thrust::sort (ms) | GPU Radix Sort (ms) |
| ----------: | -----------------: | -------------: | --------------------: | ------------------: |
|     100,000 |             5.5121 |         7.6491 |                0.1959 |              2.2974 |
|   1,000,000 |            64.6251 |        90.2845 |                0.7686 |             17.3565 |
|  10,000,000 |           771.8870 |      1045.3500 |                3.5579 |             94.8416 |
| 100,000,000 |          8855.9400 |     11767.4000 |               33.5803 |            806.5850 |

---

## 9. Correctness Verification

Correctness is validated by comparing:

* Custom GPU Radix Sort output
* Against CPU `std::sort`
* And GPU `thrust::sort`

The final implementation **passes correctness checks for all tested input sizes**, including arrays with negative integers.

---

## 10. Analysis and Discussion

### 10.1 Kernel Launch and Scan Overhead

The custom Radix Sort now uses a **manual two-level exclusive scan**, which changes the kernel execution pattern:

* **32 predicate kernel launches** (one per bit)
* **32 block-scan kernel launches** for per-block Hillis–Steele scans
* **32 block-offset addition kernel launches** to compute global scan
* **32 scatter kernel launches** to reorder elements

Although the manual scan ensures correctness for all input sizes, it introduces **additional kernel launches and host-device synchronizations** compared to the previous `thrust::exclusive_scan` approach. This overhead dominates the execution time, especially for small to moderate arrays.

---

### 10.2 Memory Access Behavior

Key observations with the manual scan:

* **Predicate, scan, and block-sum arrays reside in global memory**
* **Shared memory** is used only for per-block scans, not for the scatter phase
* Each element is read and written multiple times per bit:

  * Read by predicate kernel
  * Read/written by block-scan kernel
  * Read/written by scatter kernel

* No multi-bit processing is performed
* Global memory accesses remain largely uncoalesced for small arrays

These factors lead to **high global memory traffic** and increased latency, which explains why the GPU Radix Sort is slower than `thrust::sort` for moderate input sizes despite correct results.

---

### 10.3 Comparison with `thrust::sort`

`thrust::sort` continues to significantly outperform the custom Radix Sort because:

* It processes **multiple bits per pass**, reducing the number of iterations
* Fewer kernel launches and synchronizations are required
* Makes extensive use of **shared memory** and **warp-level primitives**
* Applies architecture-specific optimizations such as coalesced memory access and load balancing

In contrast, the manual two-level scan in the custom implementation:

* Requires **additional kernel launches** per bit (block scan + block offsets)
* Introduces **host-device memory transfers** for block sums
* Maintains all intermediate arrays in **global memory**
* Processes only **one bit per pass**

As a result, although both approaches have theoretical complexity $O(N)$, **the custom implementation has larger constant overheads**, which dominate GPU execution time for arrays of size $10^5$–$10^8$, explaining the performance gap observed in the results.

---

## 11. Conclusion

In this laboratory work, a **fully functional GPU-based Radix Sort** was implemented using CUDA. The implementation:

* Correctly handles signed integers
* Demonstrates GPU acceleration over CPU-based sorting
* Allows comparison with a highly optimized library solution

Although the custom Radix Sort does not outperform `thrust::sort`, it clearly illustrates:

* The structure of scan-based parallel algorithms
* The impact of kernel launch and synchronization overhead
* The role of memory hierarchy and global memory usage in performance

This work emphasizes a key practical insight:

> **A correct parallel algorithm may still be inefficient without careful consideration of GPU architecture and memory behavior.**

---