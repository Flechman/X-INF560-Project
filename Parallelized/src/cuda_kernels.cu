#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gif_lib.h"

__device__ int TWO_D_TO_ONE_D(int l,int c, int nb_c) 
{
	return ((l) * (nb_c) + (c));
}

__device__ int __min__(int a, int b) { return (a < b) ? a : b; }
__device__ int __max__(int a, int b) { return (a > b) ? a : b; }

typedef struct pixel
{
	int r; /* Red */
	int g; /* Green */
	int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
	int n_images;	  /* Number of images */
	int *heightStart;  /* Index of start of each image (for height) */
	int *heightEnd;	/* Index of end of each image (for height) */
	int *actualWidth;  /* Actual width of each image (INITIAL width before parallelism) */
	int *actualHeight; /* Actual height of each image (INITIAL width before parallelism) */
	pixel **p;		   /* Pixels of each image */
	GifFileType *g;	/* Internal representation.
			   DO NOT MODIFY */
} animated_gif;

__global__ void __apply_blur_filter_CUDA__ (pixel* p_i, pixel* receivedTopPart, pixel* receivedBottomPart, pixel* new_p_i, int *end, int size, int threshold, int rank, int height, int width, int heightStart, int heightEnd, int actualWidth, int actualHeight ) 
{

	int idx, j, k;

	int nbThreads = blockDim.x * gridDim.x;

	int imgHeightEnd = (heightEnd >= (actualHeight - 1)) ? actualHeight - 1 : heightEnd;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (k = idx; k < width - 1; k += nbThreads) {
		for (j = heightStart; j < imgHeightEnd; j++) {
			int j2 = j - heightStart;
			new_p_i[TWO_D_TO_ONE_D(j2, k, width)].r = p_i[TWO_D_TO_ONE_D(j2, k, width)].r;
			new_p_i[TWO_D_TO_ONE_D(j2, k, width)].g = p_i[TWO_D_TO_ONE_D(j2, k, width)].g;
			new_p_i[TWO_D_TO_ONE_D(j2, k, width)].b = p_i[TWO_D_TO_ONE_D(j2, k, width)].b;
		}
	}
	//cudaThreadSynchronize();

	if (heightStart < actualHeight / 10) 
	{
		/* Compute blur first 10% image*/
		int heightStartLocal = __max__(size, heightStart);
		int heightEndLocal = __min__(actualHeight / 10 - size, heightEnd);

		for (k = size + idx; k < width - size; k += nbThreads) {
			for (j = heightStartLocal; j < heightEndLocal; ++j) {
				int stencil_j, stencil_k;
				int t_r = 0;
				int t_g = 0;
				int t_b = 0;
				for (stencil_j = -size; stencil_j <= size; ++stencil_j) {
					if (j + stencil_j < heightStart) {
						int j2 = size - (heightStart - j - stencil_j);
						for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
							t_r += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
							t_g += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
							t_b += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
						}
					} else if (j + stencil_j >= heightEnd) {
						int j2 = j + stencil_j - heightEnd;
						for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
							t_r += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
							t_g += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
							t_b += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
						}
					} else {
						int j2 = j + stencil_j - heightStart;
						for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
							t_r += p_i[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
							t_g += p_i[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
							t_b += p_i[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
						}
					}
				}
				int j2 = j - heightStart;
				new_p_i[TWO_D_TO_ONE_D(j2, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
				new_p_i[TWO_D_TO_ONE_D(j2, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
				new_p_i[TWO_D_TO_ONE_D(j2, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
			}
		}
	}
	//	cudaThreadSynchronize();

	int heightStartLocal = __max__(actualHeight / 10 - size, heightStart);
	int heightEndLocal = __min__(heightEnd, actualHeight * 0.9 + size);
	/* Copy the middle part of the image */

	for (k = size + idx; k < width - size; k += nbThreads) {
		for (j = heightStartLocal; j < heightEndLocal; ++j) {
			int j2 = j - heightStart;
			new_p_i[TWO_D_TO_ONE_D(j2, k, width)].r = p_i[TWO_D_TO_ONE_D(j2, k, width)].r;
			new_p_i[TWO_D_TO_ONE_D(j2, k, width)].g = p_i[TWO_D_TO_ONE_D(j2, k, width)].g;
			new_p_i[TWO_D_TO_ONE_D(j2, k, width)].b = p_i[TWO_D_TO_ONE_D(j2, k, width)].b;
		}
	}

	if (heightEnd > actualHeight * 0.9) 
	{
		/* Compute blur last 10% image*/
		int heightStartLocal = __max__(heightStart, actualHeight * 0.9 + size);
		int heightEndLocal = __min__(heightEnd, actualHeight - size);

		for (k = size + idx; k < width - size; k += nbThreads) {
			for (j = heightStartLocal; j < heightEndLocal; j++) {
				int stencil_j, stencil_k;
				int t_r = 0;
				int t_g = 0;
				int t_b = 0;
				for (stencil_j = -size; stencil_j <= size; ++stencil_j) {
					if (j + stencil_j < heightStart) {
						int j2 = size - (heightStart - j - stencil_j);
						for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
							t_r += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
							t_g += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
							t_b += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
						}
					} else if (j + stencil_j >= heightEnd) {
						int j2 = j + stencil_j - heightEnd;
						for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
							t_r += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
							t_g += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
							t_b += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
						}
					} else {
						int j2 = j + stencil_j - heightStart;
						for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
							t_r += p_i[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
							t_g += p_i[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
							t_b += p_i[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
						}
					}
				}
				int j2 = j - heightStart;
				new_p_i[TWO_D_TO_ONE_D(j2, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
				new_p_i[TWO_D_TO_ONE_D(j2, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
				new_p_i[TWO_D_TO_ONE_D(j2, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
			}
		}
	}
	//cudaThreadSynchronize();

	int jBoundDown = __max__(1, heightStart);
	int jBoundUp = __min__(actualHeight - 1, heightEnd);

	for (k = 1 + idx; k < width - 1; k += nbThreads) {
		for (j = jBoundDown; j < jBoundUp; j++) {
			int j2 = j - heightStart;

			float diff_r;
			float diff_g;
			float diff_b;

			diff_r = (new_p_i[TWO_D_TO_ONE_D(j2, k, width)].r - p_i[TWO_D_TO_ONE_D(j2, k, width)].r);
			diff_g = (new_p_i[TWO_D_TO_ONE_D(j2, k, width)].g - p_i[TWO_D_TO_ONE_D(j2, k, width)].g);
			diff_b = (new_p_i[TWO_D_TO_ONE_D(j2, k, width)].b - p_i[TWO_D_TO_ONE_D(j2, k, width)].b);

			if (diff_r > threshold || -diff_r > threshold || diff_g > threshold ||
					-diff_g > threshold || diff_b > threshold || -diff_b > threshold) {
				atomicExch(end, 0);
			}

			p_i[TWO_D_TO_ONE_D(j2, k, width)].r = new_p_i[TWO_D_TO_ONE_D(j2, k, width)].r;
			p_i[TWO_D_TO_ONE_D(j2, k, width)].g = new_p_i[TWO_D_TO_ONE_D(j2, k, width)].g;
			p_i[TWO_D_TO_ONE_D(j2, k, width)].b = new_p_i[TWO_D_TO_ONE_D(j2, k, width)].b;
		}
	}
	//cudaThreadSynchronize();

}
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

extern "C" void apply_blur_filter_CUDA(pixel* p_i, pixel* receivedTopPart, pixel* receivedBottomPart, int* end, int size, int threshold, int rank, int height, int width, int heightStart, int heightEnd, int actualWidth, int actualHeight) 
{

	cudaError_t err;
	pixel *d_receivedTopPart, *d_receivedBottomPart, *d_p_i, *d_new;
	int *d_end;
	err = cudaMalloc((pixel**) &d_receivedTopPart, size * width * sizeof(pixel));
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMalloc((pixel**) &d_receivedBottomPart, size * width * sizeof(pixel));
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMalloc((pixel**) &d_p_i, width * height * sizeof(pixel));
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMalloc((pixel**) &d_end, sizeof(int));
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMalloc((pixel**) &d_new, width * height * sizeof(pixel));
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}

	err = cudaMemcpy(d_receivedTopPart, receivedTopPart, size * width * sizeof(pixel), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMemcpy(d_receivedBottomPart, receivedBottomPart, size * width * sizeof(pixel), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMemcpy(d_p_i, p_i, height * width * sizeof(pixel), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMemcpy(d_end, &end, sizeof(int), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}

	__apply_blur_filter_CUDA__<<<1,20>>>(d_p_i, d_receivedTopPart, d_receivedBottomPart, d_new, d_end, size, threshold, rank, height, width, heightStart, heightEnd, actualWidth, actualHeight);

	err = cudaMemcpy(p_i, d_p_i, height * width * sizeof(pixel), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}
	err = cudaMemcpy(end, d_end, sizeof(int), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank, cudaGetErrorString(err));
	}

	cudaFree(d_end);
	cudaFree(d_p_i);
	cudaFree(d_new);
	cudaFree(d_receivedTopPart);
	cudaFree(d_receivedBottomPart);
}




