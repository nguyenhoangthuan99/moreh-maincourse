TARGET=main
OBJECTS=util.o vec_add.o

CPPFLAGS=-std=c++11 -O3 -Wall
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
