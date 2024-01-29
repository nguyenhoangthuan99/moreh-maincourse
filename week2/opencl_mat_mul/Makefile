TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-O3 -Wall -march=znver2 -I/usr/local/cuda/include -fopenmp
LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lm -pthread -lOpenCL

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
