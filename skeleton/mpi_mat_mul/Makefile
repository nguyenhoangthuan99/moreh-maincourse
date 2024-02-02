TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=native -mavx2 -mfma -fopenmp
LDLIBS=-lm -lmpi -lmpi_cxx

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
