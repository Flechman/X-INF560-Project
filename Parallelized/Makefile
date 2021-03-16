SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
CFLAGS= -Wall -O3 -I$(HEADER_DIR)
LDFLAGS=-lm
OMPFLAG=-fopenmp
TYPE=mpi

SRC= dgif_lib.c \
     egif_lib.c \
     gif_err.c \
     gif_font.c \
     gif_hash.c \
     gifalloc.c \
     openbsd-reallocarray.c \
     quantize.c

OBJ= $(OBJ_DIR)/dgif_lib.o \
     $(OBJ_DIR)/egif_lib.o \
     $(OBJ_DIR)/gif_err.o \
     $(OBJ_DIR)/gif_font.o \
     $(OBJ_DIR)/gif_hash.o \
     $(OBJ_DIR)/gifalloc.o \
     $(OBJ_DIR)/openbsd-reallocarray.o \
     $(OBJ_DIR)/quantize.o


all: $(OBJ_DIR) prog sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ $(LDFLAGS)

prog:$(OBJ)

ifeq ($(TYPE),mpi)
	$(CC) $(CFLAGS) $^ -c $(SRC_DIR)/mainparaV2-MPI.c -o $(OBJ_DIR)/mainparaV2.o $(LDFLAGS)
else
	$(CC) $(CFLAGS) $(OMPFLAG) $^ -c $(SRC_DIR)/mainparaV2.c -o $(OBJ_DIR)/mainparaV2.o $(LDFLAGS)
endif

sobelf:$(OBJ)
	$(CC) $(OMPFLAG) $^ $(OBJ_DIR)/mainparaV2.o -o $@ $(LDFLAGS)

clean:
	rm -f sobelf $(OBJ) 

