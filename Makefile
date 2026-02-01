CC = gcc
CFLAGS = -O2 -Wall -Wextra
TARGET = sequential
SRC = lr_sequential.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
