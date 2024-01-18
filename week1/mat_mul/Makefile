TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -Wall -Ofast -march=znver2  -fopenmp -fcf-protection=none -fno-stack-protector 
# -march=znver2 -fcf-protection=none -fno-stack-protector 
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
