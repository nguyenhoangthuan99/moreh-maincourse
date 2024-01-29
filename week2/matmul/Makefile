TARGET=main
OBJECTS=util.o matmul.o main.o

CPPFLAGS=-std=c++11 -O3 -march=znver2 -Wall -mavx2 -mno-avx512f -mfma -fopenmp -funroll-loops
CPPFLAGS+= -I/usr/local/cuda/include/
LDFLAGS=-lm -lcudart -lcublas -lnvToolsExt
LDFLAGS+=-L/usr/local/cuda/lib64
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option))
NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

matmul.o: matmul.cu
	$(NVCC) --gpu-architecture=compute_70 --gpu-code=sm_70 -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
