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
__global__ void apply_blur_filter_CUDA(pixel *p_i, pixel *receivedTopPart,
                                       pixel *receivedBottomPart, int *end,
                                       int size, int threshold, int rank,
                                       int height, int width, int heightStart,
                                       int heightEnd, int actualWidth,
                                       int actualHeight) {
  int idx, j, k;

  int nbThreads = blockDim.x * gridDim.x;

  cudaError_t err;

  pixel *new;

  err = cudaMalloc((void **)new, width * height * sizeof(pixel));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank,
            cudaGetErrorString(err));
  }

  int imgHeightEnd =
      (heightEnd >= (actualHeight - 1)) ? actualHeight - 1 : heightEnd;

  {
    idx =
        blockIdx.x * blockDim.x + threadIdx.x

                                  for (k = idx; k < width - 1; k += nbThreads) {
      for (j = heightStart; j < imgHeightEnd; j++) {
        int j2 = j - heightStart;
        new[TWO_D_TO_ONE_D(j2, k, width)].r =
            p_i[TWO_D_TO_ONE_D(j2, k, width)].r;
        new[TWO_D_TO_ONE_D(j2, k, width)].g =
            p_i[TWO_D_TO_ONE_D(j2, k, width)].g;
        new[TWO_D_TO_ONE_D(j2, k, width)].b =
            p_i[TWO_D_TO_ONE_D(j2, k, width)].b;
      }
    }
    err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank,
              cudaGetErrorString(err));
    }

    if (heightStart < actualHeight / 10) {
      /* Compute blur first 10% image*/
      int heightStartLocal = max(size, heightStart);
      int heightEndLocal = min(actualHeight / 10 - size, heightEnd);

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
                t_r +=
                    receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                t_g +=
                    receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                t_b +=
                    receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
              }
            } else if (j + stencil_j >= heightEnd) {
              int j2 = j + stencil_j - heightEnd;
              for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
                t_r +=
                    receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)]
                        .r;
                t_g +=
                    receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)]
                        .g;
                t_b +=
                    receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)]
                        .b;
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
          new[TWO_D_TO_ONE_D(j2, k, width)].r =
              t_r / ((2 * size + 1) * (2 * size + 1));
          new[TWO_D_TO_ONE_D(j2, k, width)].g =
              t_g / ((2 * size + 1) * (2 * size + 1));
          new[TWO_D_TO_ONE_D(j2, k, width)].b =
              t_b / ((2 * size + 1) * (2 * size + 1));
        }
      }
    }
    err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank,
              cudaGetErrorString(err));
    }

    int heightStartLocal = max(actualHeight / 10 - size, heightStart);
    int heightEndLocal = min(heightEnd, actualHeight * 0.9 + size);
    /* Copy the middle part of the image */

    for (k = size + idx; k < width - size; k += nbThreads) {
      for (j = heightStartLocal; j < heightEndLocal; ++j) {
        int j2 = j - heightStart;
        new[TWO_D_TO_ONE_D(j2, k, width)].r =
            p_i[TWO_D_TO_ONE_D(j2, k, width)].r;
        new[TWO_D_TO_ONE_D(j2, k, width)].g =
            p_i[TWO_D_TO_ONE_D(j2, k, width)].g;
        new[TWO_D_TO_ONE_D(j2, k, width)].b =
            p_i[TWO_D_TO_ONE_D(j2, k, width)].b;
      }
    }

    if (heightEnd > actualHeight * 0.9) {
      /* Compute blur last 10% image*/
      int heightStartLocal = max(heightStart, actualHeight * 0.9 + size);
      int heightEndLocal = min(heightEnd, actualHeight - size);

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
                t_r +=
                    receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                t_g +=
                    receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                t_b +=
                    receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
              }
            } else if (j + stencil_j >= heightEnd) {
              int j2 = j + stencil_j - heightEnd;
              for (stencil_k = -size; stencil_k <= size; ++stencil_k) {
                t_r +=
                    receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)]
                        .r;
                t_g +=
                    receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)]
                        .g;
                t_b +=
                    receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)]
                        .b;
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
          new[TWO_D_TO_ONE_D(j2, k, width)].r =
              t_r / ((2 * size + 1) * (2 * size + 1));
          new[TWO_D_TO_ONE_D(j2, k, width)].g =
              t_g / ((2 * size + 1) * (2 * size + 1));
          new[TWO_D_TO_ONE_D(j2, k, width)].b =
              t_b / ((2 * size + 1) * (2 * size + 1));
        }
      }
    }
    err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank,
              cudaGetErrorString(err));
    }

    int jBoundDown = max(1, heightStart);
    int jBoundUp = min(actualHeight - 1, heightEnd);

    for (k = 1 + idx; k < width - 1; k += nbThreads) {
      for (j = jBoundDown; j < jBoundUp; j++) {
        int j2 = j - heightStart;

        float diff_r;
        float diff_g;
        float diff_b;

        diff_r = (new[TWO_D_TO_ONE_D(j2, k, width)].r -
                  p_i[TWO_D_TO_ONE_D(j2, k, width)].r);
        diff_g = (new[TWO_D_TO_ONE_D(j2, k, width)].g -
                  p_i[TWO_D_TO_ONE_D(j2, k, width)].g);
        diff_b = (new[TWO_D_TO_ONE_D(j2, k, width)].b -
                  p_i[TWO_D_TO_ONE_D(j2, k, width)].b);

        if (diff_r > threshold || -diff_r > threshold || diff_g > threshold ||
            -diff_g > threshold || diff_b > threshold || -diff_b > threshold) {
          atomicExch(end, 0);
        }

        p_i[TWO_D_TO_ONE_D(j2, k, width)].r =
            new[TWO_D_TO_ONE_D(j2, k, width)].r;
        p_i[TWO_D_TO_ONE_D(j2, k, width)].g =
            new[TWO_D_TO_ONE_D(j2, k, width)].g;
        p_i[TWO_D_TO_ONE_D(j2, k, width)].b =
            new[TWO_D_TO_ONE_D(j2, k, width)].b;
      }
    }
    err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA ERROR: process %d, <%s>\n", rank,
              cudaGetErrorString(err));
    }
  }
  cudaFree(new);
}