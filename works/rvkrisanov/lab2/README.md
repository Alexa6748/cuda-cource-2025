# Lab 2: Matrix Multiplication

## Задача
Реализовать и оптимизировать перемножение матриц на CUDA. Сравнить производительность с CPU.

## Реализация
1.  **CPU**: Классический алгоритм `O(N^3)`.
2.  **GPU Basic**: Наивная реализация (глобальная память, один элемент на поток).
3.  **GPU Tiled**: Оптимизация с Shared Memory (Tile 16×16).
    *   Каждый поток вычисляет один элемент результирующей матрицы.
    *   Тайлы загружаются в `__shared__` память, что снижает количество обращений к глобальной памяти.
    *   Использование `#pragma unroll` для разворачивания внутреннего цикла умножения.

## Результаты

| Matrix Size | CPU (us) | CUDA Basic (us) | CUDA Tiled (us) | Speedup (Basic vs CPU) | Speedup (Tiled vs CPU) | Speedup (Tiled vs Basic) |
|---|---|---|---|---|---|---|
| 128x128 | 1,175 | 100 | **9** | 11.79x | **127.50x** | **11.11x** |
| 256x256 | 14,274 | 82 | **41** | 173.36x | **348.49x** | **2.00x** |
| 512x512 | 228,354 | 409 | **280** | 558.64x | **816.86x** | **1.46x** |
| 1024x1024 | 6,842,508 | 3,047 | **2,167** | 2,245.55x | **3,157.91x** | **1.41x** |
| 2048x2048 | 163,040,304 | 22,604 | **17,243** | 7,212.81x | **9,455.37x** | **1.31x** |

**Анализ:**
*   **Tiled версия заметно быстрее naive**: Использование Shared Memory с тайлингом обеспечивает ускорение относительно базовой версии на всех размерах. Эффект растёт с увеличением матриц.
*   **Tile size 16×16**: Классический размер тайла без thread coarsening даёт оптимальный баланс между occupancy и shared memory footprint для данной архитектуры.

## Запуск
```bash
make
./matrix_multiply
```

```bash
$ make
nvcc -O2 -std=c++14 -Iinclude -o matrix_multiply src/main.cu src/matrix_multiply_cpu.cu src/matrix_multiply_cuda.cu src/matrix_multiply_cuda_tiled.cu
./matrix_multiply
Benchmarking Matrix 128x128...
  CPU:            1175 us
  CUDA Basic:      100 us (11.79x speedup) [OK]
  CUDA Tiled:        9 us (127.50x speedup) [OK]

Benchmarking Matrix 256x256...
  CPU:           14274 us
  CUDA Basic:       82 us (173.36x speedup) [OK]
  CUDA Tiled:       41 us (348.49x speedup) [OK]

Benchmarking Matrix 512x512...
  CPU:          228354 us
  CUDA Basic:      409 us (558.64x speedup) [OK]
  CUDA Tiled:      280 us (816.86x speedup) [OK]

Benchmarking Matrix 1024x1024...
  CPU:         6842508 us
  CUDA Basic:     3047 us (2245.55x speedup) [OK]
  CUDA Tiled:     2167 us (3157.91x speedup) [OK]

Benchmarking Matrix 2048x2048...
  CPU:        163040304 us
  CUDA Basic:    22604 us (7212.81x speedup) [OK]
  CUDA Tiled:    17243 us (9455.37x speedup) [OK]

## Анализ бенчмарков других студентов
См. [BENCHMARK_ANALYSIS.md](./BENCHMARK_ANALYSIS.md) — детальный разбор проблем в реализациях коллег и почему их нельзя использовать для сравнения.
