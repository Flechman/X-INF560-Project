#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


extern "C" void hello_cuda(){
	printf("Hello from CUDA\n");
}	

extern "C" void get_cuda_device_count(int* nbGPUs) {
	cudaGetDeviceCount(nbGPUs);
}

extern "C" void set_cuda_device(int rank, int nbGPUs) {
	cudaSetDevice(rank % nbGPUs);
}
