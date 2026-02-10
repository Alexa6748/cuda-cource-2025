# Lab 2: Matrix Multiplication - Перемножение матриц

## Описание
Реализована программа для умножения матриц на CUDA.  
Поддерживаются как квадратные, так и прямоугольные матрицы произвольных размеров (от малых до 4096×4096, ограничено только объёмом памяти GPU).  

Реализовано три версии вычислений:
- Наивная CPU-версия (трёхвложенные циклы).
- Базовая GPU-версия (простое ядро без оптимизаций).
- Оптимизированная GPU-версия с использованием shared memory и тайлингом (TILE_SIZE = 32).

Программа измеряет время выполнения каждой версии, сравнивает результаты GPU с эталонными CPU-результатами (с epsilon = 1e-3f) и выводит ускорение относительно CPU.  
Матрицы заполняются случайными значениями в диапазоне [0, 1).

Корректность проверена на разных размерах, включая прямоугольные матрицы. Реализован тайлинг с shared memory (TILE_SIZE=32), коалесцированный доступ.

## Компиляция
```bash
make
```

Для очистки:
```bash
make clean
```

## Запуск
```bash
./program [M N K] [--mode cpu|gpu_basic|gpu_shared|all]
```

- Без аргументов — квадратные матрицы 1024×1024 (по умолчанию).
- Пример прямоугольных: `./program 1000 2000 1500`
- Режим `--mode all` запускает все версии (по умолчанию).

## Результаты
Программа выводит размеры матриц, время выполнения и ускорение.  
Результаты GPU сравниваются с CPU (epsilon = 1e-3f из-за особенностей float-арифметики).

### Примеры работы программы

#### Маленькие матрицы (4×4×4)
```
Matrix dimensions: A(4x4), B(4x4), C(4x4)
Mode: all
CPU time: 0.000001 seconds
GPU basic time: 0.000140 seconds
All elements match! Max. diff(eps 1e-3f): 0.000000
Basic GPU results match CPU.
Basic GPU acceleration: 0.01x
GPU shared time: 0.000028 seconds
All elements match! Max. diff(eps 1e-3f): 0.000000
Shared GPU results match CPU.
Shared GPU acceleration: 0.04x
```

#### Средние матрицы (100×200×300)
```
Matrix dimensions: A(100x200), B(200x300), C(100x300)
Mode: all
CPU time: 0.010305 seconds
GPU basic time: 0.000163 seconds
All elements match! Max. diff(eps 1e-3f): 0.000011
Basic GPU results match CPU.
Basic GPU acceleration: 63.23x
GPU shared time: 0.000041 seconds
All elements match! Max. diff(eps 1e-3f): 0.000011
Shared GPU results match CPU.
Shared GPU acceleration: 251.58x
```

#### Прямоугольные матрицы (64×256×128)
```
Matrix dimensions: A(64x256), B(256x128), C(64x128)
Mode: all
CPU time: 0.006019 seconds
GPU basic time: 0.000164 seconds
All elements match! Max. diff(eps 1e-3f): 0.000015
Basic GPU results match CPU.
Basic GPU acceleration: 36.62x
GPU shared time: 0.000045 seconds
All elements match! Max. diff(eps 1e-3f): 0.000015
Shared GPU results match CPU.
Shared GPU acceleration: 133.98x
```

#### Большие матрицы (1000×2000×1500)
```
Matrix dimensions: A(1000x2000), B(2000x1500), C(1000x1500)
Mode: all
CPU time: 13.2937 seconds
GPU basic time: 0.004912 seconds
All elements match! Max. diff(eps 1e-3f): 0.000183
Basic GPU results match CPU.
Basic GPU acceleration: 2706.24x
GPU shared time: 0.003605 seconds
All elements match! Max. diff(eps 1e-3f): 0.000183
Shared GPU results match CPU.
Shared GPU acceleration: 3687.46x
```

#### Квадратные матрицы по умолчанию (1024×1024)
```
Matrix dimensions: A(1024x1024), B(1024x1024), C(1024x1024)
Mode: all
CPU time: 3.51235 seconds
GPU basic time: 0.001606 seconds
All elements match! Max. diff(eps 1e-3f): 0.000092
Basic GPU results match CPU.
Basic GPU acceleration: 2187.13x
GPU shared time: 0.001288 seconds
All elements match! Max. diff(eps 1e-3f): 0.000092
Shared GPU results match CPU.
Shared GPU acceleration: 2726.57x
```

## Замечания
- Оптимизированная версия (shared memory) даёт заметное ускорение на матрицах:

| Размер матриц          | Коэффициент (shared быстрее basic в N раз) |
|-----------------------|--------------------------------------------|
| 4×4×4                 | ≈4.00x                                     |
| 100×200×300           | ≈3.98x                                     |
| 64×256×128            | ≈3.66x                                     |
| 1000×2000×1500        | ≈1.36x                                     |
| 1024×1024             | ≈1.25x                                     |

- Для малых матриц GPU-версии медленнее CPU из-за накладных расходов на запуск ядра и копирование данных.
- Сравнение результатов выполняется с epsilon = 1e-3f, что учитывает накопленные ошибки округления в арифметике одинарной точности (float). 
- Генерация случайных чисел — в диапазоне [0, 1), чтобы минимизировать ошибки округления.
- Программа корректно освобождает всю выделенную память (cudaFree, free).
