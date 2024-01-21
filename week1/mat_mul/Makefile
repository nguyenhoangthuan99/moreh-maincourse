TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2 -mavx2 -mfma -fopenmp -funroll-loops -mfpmath=sse
# -march=znver2 -fcf-protection=none -fno-stack-protector -fopenmp 
LDFLAGS=-lm -pthread
LDLIBS=-lnuma -lm -lmpi -lmpi_cxx

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
