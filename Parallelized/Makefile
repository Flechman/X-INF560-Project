SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
CFLAGS= -Wall -O3 -I$(HEADER_DIR)
LDFLAGS=-lm
OMPFLAG=-fopenmp
NVCC=nvcc
CUDAFLAGS=-lcudart
CUDALIB=-L/usr/local/cuda/lib64
CUDA_SRC=src/cuda_kernels.cu
CUDA_OBJ=obj/cuda_kernels.o

SRC= dgif_lib.c \
     egif_lib.c \
     gif_err.c \
     gif_font.c \
     gif_hash.c \
     gifalloc.c \
     mainparaV2.c \
     openbsd-reallocarray.c \
     quantize.c

OBJ= $(OBJ_DIR)/dgif_lib.o \
     $(OBJ_DIR)/egif_lib.o \
     $(OBJ_DIR)/gif_err.o \
     $(OBJ_DIR)/gif_font.o \
     $(OBJ_DIR)/gif_hash.o \
     $(OBJ_DIR)/gifalloc.o \
     $(OBJ_DIR)/mainparaV2.o \
     $(OBJ_DIR)/openbsd-reallocarray.o \
     $(OBJ_DIR)/quantize.o

all: cuda $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(OMPFLAG) $(CFLAGS) -c $^ -o $@ $(LDFLAGS)

cuda:
	$(NVCC) -c $(CUDA_SRC) -o $(CUDA_OBJ) 

sobelf:$(OBJ)
	$(CC) $(OMPFLAG) $(CUDALIB) $(CUDAFLAGS) $(CUDA_OBJ) $^ -o $@ $(LDFLAGS)

clean:
	rm -f sobelf $(OBJ) $(CUDA_OBJ)
