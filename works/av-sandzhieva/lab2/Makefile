NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -I../include

TARGET = program

all: $(TARGET)

$(TARGET): src/main.cu src/cpu_matrix_mult.cpp src/gpu_matrix_mult.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) $^

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
