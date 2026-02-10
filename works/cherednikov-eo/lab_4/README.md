# Lab 4: Radix Sort - Поразрядная сортировка

## Описание

Реализация алгоритма поразрядной сортировки (Radix Sort) на CUDA для сортировки целых чисел различных типов (32-bit и 64-bit, signed и unsigned).

## Компиляция

```bash
make
```

## Запуск

```bash
./radix
```

или

```bash
make run
```

## Результаты

```
Benchmark: uint32_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.00151s, GPU Thrust=0.00008s
Speedup: Radix vs CPU=0.02x, Thrust vs CPU=0.34x, Radix vs Thrust=0.06x

Benchmark: int32_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.00154s, GPU Thrust=0.00008s
Speedup: Radix vs CPU=0.02x, Thrust vs CPU=0.30x, Radix vs Thrust=0.06x

Benchmark: uint64_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.00271s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.01x, Thrust vs CPU=0.34x, Radix vs Thrust=0.03x

Benchmark: int64_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.00281s, GPU Thrust=0.00010s
Speedup: Radix vs CPU=0.01x, Thrust vs CPU=0.26x, Radix vs Thrust=0.03x

Benchmark: uint32_t[100000] OK
Time: CPU=0.00437s, GPU Radix=0.00154s, GPU Thrust=0.00026s
Speedup: Radix vs CPU=2.84x, Thrust vs CPU=17.03x, Radix vs Thrust=0.17x

Benchmark: int32_t[100000] OK
Time: CPU=0.00421s, GPU Radix=0.00168s, GPU Thrust=0.00026s
Speedup: Radix vs CPU=2.50x, Thrust vs CPU=16.16x, Radix vs Thrust=0.15x

Benchmark: uint64_t[100000] OK
Time: CPU=0.00446s, GPU Radix=0.00302s, GPU Thrust=0.00032s
Speedup: Radix vs CPU=1.48x, Thrust vs CPU=13.71x, Radix vs Thrust=0.11x

Benchmark: int64_t[100000] OK
Time: CPU=0.00442s, GPU Radix=0.00384s, GPU Thrust=0.00037s
Speedup: Radix vs CPU=1.15x, Thrust vs CPU=12.06x, Radix vs Thrust=0.10x

Benchmark: uint32_t[5000000] OK
Time: CPU=0.18748s, GPU Radix=0.00485s, GPU Thrust=0.00086s
Speedup: Radix vs CPU=38.65x, Thrust vs CPU=217.11x, Radix vs Thrust=0.18x

Benchmark: int32_t[5000000] OK
Time: CPU=0.18697s, GPU Radix=0.00528s, GPU Thrust=0.00090s
Speedup: Radix vs CPU=35.44x, Thrust vs CPU=208.85x, Radix vs Thrust=0.17x

Benchmark: uint64_t[5000000] OK
Time: CPU=0.18895s, GPU Radix=0.00989s, GPU Thrust=0.00215s
Speedup: Radix vs CPU=19.10x, Thrust vs CPU=87.79x, Radix vs Thrust=0.22x

Benchmark: int64_t[5000000] OK
Time: CPU=0.19330s, GPU Radix=0.01024s, GPU Thrust=0.00227s
Speedup: Radix vs CPU=18.88x, Thrust vs CPU=85.25x, Radix vs Thrust=0.22x

Benchmark: uint32_t[10000000] OK
Time: CPU=0.37248s, GPU Radix=0.00699s, GPU Thrust=0.00149s
Speedup: Radix vs CPU=53.32x, Thrust vs CPU=250.65x, Radix vs Thrust=0.21x

Benchmark: int32_t[10000000] OK
Time: CPU=0.37310s, GPU Radix=0.00805s, GPU Thrust=0.00151s
Speedup: Radix vs CPU=46.34x, Thrust vs CPU=246.93x, Radix vs Thrust=0.19x

Benchmark: uint64_t[10000000] OK
Time: CPU=0.36992s, GPU Radix=0.01442s, GPU Thrust=0.00400s
Speedup: Radix vs CPU=25.64x, Thrust vs CPU=92.38x, Radix vs Thrust=0.28x

Benchmark: int64_t[10000000] OK
Time: CPU=0.38137s, GPU Radix=0.01612s, GPU Thrust=0.00383s
Speedup: Radix vs CPU=23.65x, Thrust vs CPU=99.46x, Radix vs Thrust=0.24x
```