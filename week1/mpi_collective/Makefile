TARGET=mpi_collective_benchmark
OBJECTS=
CXX=mpic++

CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2
LDFLAGS=-lm -pthread

all: mpi

# $(TARGET): $(OBJECTS)
mpi:
	mpic++ $(CFLAGS) -o mpi_collective_benchmark mpi_collective_benchmark.c

clean:
	rm -rf $(TARGET) $(OBJECTS)