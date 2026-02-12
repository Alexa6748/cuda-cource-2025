NVCC    = nvcc
CFLAGS  = -O2 -arch=sm_70 -I../include
TARGET  = program

all: $(TARGET)

$(TARGET): main.o io.o sobel.o
	$(NVCC) main.o io.o sobel.o -o $(TARGET)

main.o: src/main.cu
	$(NVCC) $(CFLAGS) -c src/main.cu -o main.o

io.o: src/io.cu
	$(NVCC) $(CFLAGS) -c src/io.cu -o io.o

sobel.o: src/sobel.cu
	$(NVCC) $(CFLAGS) -c src/sobel.cu -o sobel.o

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
