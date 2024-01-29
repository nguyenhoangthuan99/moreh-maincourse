TARGET=main
OBJECTS=main.o util.o convolution.o

CPPFLAGS=-std=c++14 -O3 -Wall -march=znver2 -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option)  -arch=sm_70) --fmad=true

LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm -lcudnn -lcublas

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
