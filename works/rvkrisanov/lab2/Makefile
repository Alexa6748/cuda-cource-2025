CFLAGS = -O2 -std=c++14 
INCLUDES = -Iinclude
SOURCES = src/main.cu src/matrix_multiply_cpu.cu src/matrix_multiply_cuda.cu src/matrix_multiply_cuda_tiled.cu
TARGET = matrix_multiply

# Windows-specific settings
ifeq ($(OS),Windows_NT)
    NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
    TARGET := $(TARGET).exe
    RUN_CMD := $(TARGET)
    RM_CMD := del /Q
    CFLAGS += -allow-unsupported-compiler -arch=sm_75 -ccbin clang-cl
else
    NVCC = nvcc
    RUN_CMD := ./$(TARGET)
    RM_CMD := rm -f
endif

all: $(TARGET) run

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES)

run: $(TARGET)
	$(RUN_CMD)

clean:
	$(RM_CMD) $(TARGET) *.o

.PHONY: all clean run
