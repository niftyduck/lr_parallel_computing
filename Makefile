CC = gcc
CFLAGS = -O2 -Wall -Wextra
OMPFLAGS = -fopenmp
LM = -lm

SEQ = sequential
OMP = omp
MPI = mpi

SRC_SEQ = lr_sequential.c
SRC_OMP = lr_omp.c
SRC_MPI = lr_mpi.c

EXECUTABLES = $(SEQ) $(OMP) $(MPI)

all: $(EXECUTABLES) list

$(SEQ): $(SRC_SEQ)
	$(CC) $(CFLAGS) $(SRC_SEQ) -o $(SEQ) $(LM)

$(OMP): $(SRC_OMP)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_OMP) -o $(OMP) $(LM)

$(MPI): $(SRC_MPI)
	mpicc $(CFLAGS) $(SRC_MPI) -o $(MPI) $(LM)

list:
	@echo "\nChoose one executable between these:"
	@ls -l $(EXECUTABLES)

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean list
