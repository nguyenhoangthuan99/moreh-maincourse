TARGET=main
OBJECTS=util.o vec_add.o

CPPFLAGS=-std=c++11 -O2 -Wall -fopenmp -march=znver2
LDFLAGS=-pthread
LDLIBS=-lnuma -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
