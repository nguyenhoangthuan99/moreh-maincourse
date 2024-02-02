TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -Wall -O3
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
