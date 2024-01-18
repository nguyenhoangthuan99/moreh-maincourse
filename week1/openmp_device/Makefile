TARGET=main
OBJECTS=util.o vec_add.o

CPPFLAGS=-std=c++11 -O3 -Wall -fopenmp -fcf-protection=none -fno-stack-protector
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
