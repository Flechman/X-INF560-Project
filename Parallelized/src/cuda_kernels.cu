#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void hello_cuda(int rank){
	printf("Hello from CUDA by process %i\n", rank);
}	
extern "C" void get_cuda_devices(int* nb_gpus){
	cudaGetDeviceCount(nb_gpus);
}
extern "C" void set_cuda_devices(int rank, int nb_gpus)
{
	cudaSetDevice(rank % nb_gpus);
}

/* Kernels */
