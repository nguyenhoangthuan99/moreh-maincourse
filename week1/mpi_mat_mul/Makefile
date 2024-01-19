TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2 -mavx2 -mfma -fopenmp -funroll-loops -mfpmath=sse
LDLIBS=-lm -lmpi -lmpi_cxx -lnuma -lpthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
