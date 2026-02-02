CC = gcc
CFLAGS = -O2 -Wall -Wextra
OMPFLAGS = -fopenmp
LM = -lm

SEQ = sequential
OMP = omp

SRC_SEQ = lr_sequential.c
SRC_OMP = lr_omp.c

EXECUTABLES = $(SEQ) $(OMP)

all: $(EXECUTABLES) list

$(SEQ): $(SRC_SEQ)
	$(CC) $(CFLAGS) $(SRC_SEQ) -o $(SEQ) $(LM)

$(OMP): $(SRC_OMP)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_OMP) -o $(OMP) $(LM)

list:
	@echo "\nChoose one executable between these:"
	@ls -l $(EXECUTABLES)

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean list
