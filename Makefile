CC = gcc
CFLAGS = -O2 -Wall -Wextra
TARGET = sequential
SRC = lr_sequential.c
LM = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LM)

clean:
	rm -f $(TARGET)

.PHONY: all clean