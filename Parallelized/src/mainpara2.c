//Just to save some code snippets that were attempts to parallelize

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n;             /* Original number of images (n >= n_images) */
    int n_images;      /* Number of images */
    int *heightStart;  /* Index of start of each image (for height) */
    int *heightEnd;    /* Index of end of each image (for height) */
    int *actualWidth;  /* Actual width of each image (INITIAL width before parallelism) */
    int *actualHeight; /* Actual height of each image (INITIAL width before parallelism) */
    pixel **p;         /* Pixels of each image */
    GifFileType *g;    /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

/* Method to get the process that possesses the row 'indexHeight', looking at the processes > rank. In theory, can have a lot of iterations, but in the practical case it hase a very low number of iterations */
int get_rank(int indexHeight, int actualHeight, int size, int rank, int n, bool is_asc) {
    int i = rank;
    bool in_bounds = false;
    while(!in_bounds)
    {
        int imgStartIndex = (int)(((double)n / (double)size) * i);
        int imgEndIndex = (int)(((double)n / (double)size) * (i + 1));
        double start = i * ((double)n / (double)size) - imgStartIndex;
        double end = (rank + 1) * ((double)n / (double)size) - imgEndIndex;
        int n_images = imgEndIndex - imgStartIndex + 1;
        int heightStart = round(start * actualHeight);
        int heightEnd = round(end * actualHeight);
        if(n_images > 1 && ((heightStart <= indexHeight && is_asc) || (heightEnd > indexHeight && !is_asc))) {
            in_bounds = true;
        } else {
            if (indexHeight >= heightStart && indexHeight < heightEnd) {
                in_bounds = true;
            }
            else {
                i = is_asc ? i+1 : i-1;
            }
        }
    }
    return i;
}

void apply_blur_filter(animated_gif *image, int size, int threshold, int rank, int nbProc)
{
    int i, j, k;
    int width, height;
    int end = 0;
    int n_iter = 0;

    pixel **p;
    pixel *new;

    /* Get the pixels of all images */
    p = image->p;

    /* Process all images */
    for (i = 0; i < image->n_images; i++)
    {
        n_iter = 0;
        width = image->actualWidth[i];
        height = image->heightEnd[i] - image->heightStart[i];

        /* Allocate array of new pixels */
        new = (pixel *)malloc(width * height * sizeof(pixel));

        /* Perform at least one blur iteration */
        do
        {
            end = 1;
            n_iter++;

            int heightEnd = (image->heightEnd[i] >= (image->actualHeight[i] - 1)) ? image->actualHeight[i] - 1 : image->heightEnd[i];
            for (j = image->heightStart[i]; j < heightEnd; j++)
            {
                for (k = 0; k < width-1; k++)
                {
                    int j2 = j - image->heightStart[i];
                    new[TWO_D_TO_ONE_D(j2, k, width)].r = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                    new[TWO_D_TO_ONE_D(j2, k, width)].g = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                    new[TWO_D_TO_ONE_D(j2, k, width)].b = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                }
            }

            if (image->heighStart[i] < image->actualHeight[i] / 10)
            {
                int heightEnd = min(image->heightEnd[i], image->actualHeight[i] / 10);

                /* Send Data */
                for (j = image->heightStart[i]; j <= min(image->heightStart[i] + size - 1, heightEnd - 1); ++j)
                {
                    //Send to processes having rows max(size, i-size) : heightStart-1
                    int *row = malloc(width * 3 * sizeof(int));
                    int j2 = j - image->heightStart[i];
                    for (k = 0; k < width; ++k)
                    {
                        row[k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        row[k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        row[k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    for (k = max(size, j - size), k <= image->heightStart[i] - 1; ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, image->n, false);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            MPI_Send(row, 3 * width, MPI_INTEGER, to_rank, j, MPI_COMM_WORLD);
                        }
                    }
                    free(row);
                }
                for (j = max(heightEnd - size, image->heightStart[i]); j <= heightEnd - 1; ++j)
                {
                    //Send to processes having rows heightEnd : min(H/10-size-1, i+size)
                    int *row = malloc(width * 3 * sizeof(int));
                    int j2 = j - image->heightStart[i];
                    for (k = 0; k < width; ++k)
                    {
                        row[k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        row[k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        row[k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    for (k = heightEnd, k <= min(image->actualHeight[i] / 10 - size - 1, j + size); ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, image->n, true);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            MPI_Send(row, 3 * width, MPI_INTEGER, to_rank, j, MPI_COMM_WORLD);
                        }
                    }
                    free(row);
                }

                /* Receive Data */
                int subStart = (size - image->heightStart[i] > 0) ? size - image->heightStart[i] : 0;
                int subEnd = (heightEnd - 1 - (image->actualHeight[i] / 10 - size - 1) > 0) ? heightEnd - 1 - (image->actualHeight[i] / 10 - size - 1) : 0;
                int topSize = (heightEnd <= size) ? 0 : size - subStart;
                int bottomSize = (image->heightStart[i] >= image->actualHeight[i] / 10 - size) ? 0 : size - subEnd;
                pixel *receivedTopPart = (pixel *)malloc(topSize * width * sizeof(pixel));
                pixel *receivedBottomPart = (pixel *)malloc(bottomSize * width * sizeof(pixel));
                for (j = 0; j < topSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, image->heightStart[i] - topSize + j, MPI_COMM_WORLD);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedTopPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }
                for (j = 0; j < bottomSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, heightEnd + j, MPI_COMM_WORLD);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedBottomPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }

                /* Compute blur */
                for (j = max(image->heightStart[i], size); j < min(image->heightEnd[i], image->actualHeight[i] / 10 - size); ++j)
                {
                    for (k = size; k < width - size; ++k)
                    {
                        int stencil_j, stencil_k;
                        int t_r = 0;
                        int t_g = 0;
                        int t_b = 0;
                        for (stencil_j = -size; stencil_j <= size; ++stencil_j)
                        {
                            if (j + stencil_j < image->heightStart[i])
                            {
                                int j2 = topSize - (image->heightStart[i] - j - stencil_j);
                                for (stencil_k = -size; stencil_k <= size; ++stencil_k)
                                {
                                    t_r += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                                    t_g += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                                    t_b += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
                                }
                            }
                            else if (j + stencil_j >= heightEnd)
                            {
                                int j2 = j + stencil_j - heightEnd;
                                for (stencil_k = -size; stencil_k <= size; ++stencil_k)
                                {
                                    t_r += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                                    t_g += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                                    t_b += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
                                }
                            }
                            else
                            {
                                int j2 = j + stencil_j - image->heightStart[i];
                                for (stencil_k = -size; stencil_k <= size; ++stencil_k)
                                {
                                    t_r += p[i][TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                                    t_g += p[i][TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                                    t_b += p[i][TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
                                }
                            }
                        }
                        int j2 = j - image->heightStart[i];
                        new[TWO_D_TO_ONE_D(j2, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                        new[TWO_D_TO_ONE_D(j2, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                        new[TWO_D_TO_ONE_D(j2, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                    }
                }
                free(receivedTopPart);
                free(receivedBottomPart);
            }

            /* Copy the middle part of the image */
            for(j = max(image->actualHeight[i] / 10 - size, image->heightStart[i]); j < min(image->heightEnd[i], image->actualHeight[i] * 0.9 + size); ++j) {
                int j2 = j - image->heightStart[i];
                for(k = size; k < width - size; ++k) {
                    new[TWO_D_TO_ONE_D(j2, k, width)].r = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                    new[TWO_D_TO_ONE_D(j2, k, width)].g = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                    new[TWO_D_TO_ONE_D(j2, k, width)].b = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                }
            }


            if (image->heightEnd[i] > image->actualHeight[i] * 0.9)
            {
                int heightStart = max(image->heightStart[i], image->actualHeight[i] * 0.9);

                /* Send Data */
                for (j = heightStart; j <= min(heightStart + size - 1, image->heightEnd[i] - 1); ++j)
                {
                    //Send to processes having rows max(actualHeight * 0.9 + size, j-size) : heightStart-1
                    int *row = malloc(width * 3 * sizeof(int));
                    int j2 = j - image->heightStart[i];
                    for (k = 0; k < width; ++k)
                    {
                        row[k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        row[k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        row[k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    for (k = max(image->actualHeight[i] * 0.9 + size, j - size), k <= heightStart - 1; ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, image->n, false);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            MPI_Send(row, 3 * width, MPI_INTEGER, to_rank, j, MPI_COMM_WORLD);
                        }
                    }
                    free(row);
                }
                for (j = max(heightStart, image->heightEnd[i] - size); j <= image->heightEnd[i] - 1; ++j)
                {
                    //Send to processes having rows heightEnd : min(H/10-size-1, i+size)
                    int *row = malloc(width * 3 * sizeof(int));
                    int j2 = j - image->heightStart[i];
                    for (k = 0; k < width; ++k)
                    {
                        row[k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        row[k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        row[k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    for (k = image->heightEnd[i], k <= min(image->actualHeight[i] - size - 1, j + size); ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, image->n, true);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            MPI_Send(row, 3 * width, MPI_INTEGER, to_rank, j, MPI_COMM_WORLD);
                        }
                    }
                    free(row);
                }

                /* Receive Data */
                int subStart = (image->actualHeight[i] * 0.9 + size - heightStart > 0) ? image->actualHeight[i] * 0.9 + size - heightStart : 0;
                int subEnd = (image->heightEnd[i] - 1 - (image->actualHeight[i] - size - 1) > 0) ? image->heightEnd[i] - 1 - (image->actualHeight[i] - size - 1) : 0;
                int topSize = (image->heightEnd[i] <= image->actualHeight[i] * 0.9 + size) ? 0 : size - subStart;
                int bottomSize = (image->heightStart[i] >= image->actualHeight[i] - size) ? 0 : size - subEnd;
                pixel *receivedTopPart = (pixel *)malloc(topSize * width * sizeof(pixel));
                pixel *receivedBottomPart = (pixel *)malloc(bottomSize * width * sizeof(pixel));
                for (j = 0; j < topSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, heightStart - topSize + j, MPI_COMM_WORLD);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedTopPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }
                for (j = 0; j < bottomSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, image->heightEnd[i] + j, MPI_COMM_WORLD);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedBottomPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }

                /* Compute blur */
                for (j = max(image->heightStart[i], image->actualHeight[i] * 0.9 + size); j < min(image->heightEnd[i], image->actualHeight[i] - size); ++j)
                {
                    for (k = size; k < width - size; ++k)
                    {
                        int stencil_j, stencil_k;
                        int t_r = 0;
                        int t_g = 0;
                        int t_b = 0;
                        for (stencil_j = -size; stencil_j <= size; ++stencil_j)
                        {
                            if (j + stencil_j < heightStart)
                            {
                                int j2 = topSize - (heightStart - j - stencil_j);
                                for (stencil_k = -size; stencil_k <= size; ++stencil_k)
                                {
                                    t_r += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                                    t_g += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                                    t_b += receivedTopPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
                                }
                            }
                            else if (j + stencil_j >= image->heightEnd[i])
                            {
                                int j2 = j + stencil_j - image->heightEnd[i];
                                for (stencil_k = -size; stencil_k <= size; ++stencil_k)
                                {
                                    t_r += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                                    t_g += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                                    t_b += receivedBottomPart[TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
                                }
                            }
                            else
                            {
                                int j2 = j + stencil_j - image->heightStart[i];
                                for (stencil_k = -size; stencil_k <= size; ++stencil_k)
                                {
                                    t_r += p[i][TWO_D_TO_ONE_D(j2, k + stencil_k, width)].r;
                                    t_g += p[i][TWO_D_TO_ONE_D(j2, k + stencil_k, width)].g;
                                    t_b += p[i][TWO_D_TO_ONE_D(j2, k + stencil_k, width)].b;
                                }
                            }
                        }
                        int j2 = j - image->heightStart[i];
                        new[TWO_D_TO_ONE_D(j2, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                        new[TWO_D_TO_ONE_D(j2, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                        new[TWO_D_TO_ONE_D(j2, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                    }
                }
                free(receivedTopPart);
                free(receivedBottomPart);
            }

            int jBound = min(image->actualHeight[i] - 1, image->heightEnd[i]);
            for (j = max(1, image->heightStart[i]); j < jBound; j++)
            {
                int j2 = j-image->heightStart[i];
                for (k = 1; k < width - 1; k++)
                {

                    float diff_r;
                    float diff_g;
                    float diff_b;

                    diff_r = (new[TWO_D_TO_ONE_D(j2, k, width)].r - p[i][TWO_D_TO_ONE_D(j2, k, width)].r);
                    diff_g = (new[TWO_D_TO_ONE_D(j2, k, width)].g - p[i][TWO_D_TO_ONE_D(j2, k, width)].g);
                    diff_b = (new[TWO_D_TO_ONE_D(j2, k, width)].b - p[i][TWO_D_TO_ONE_D(j2, k, width)].b);

                    if (diff_r > threshold || -diff_r > threshold ||
                        diff_g > threshold || -diff_g > threshold ||
                        diff_b > threshold || -diff_b > threshold)
                    {
                        end = 0;
                    }

                    p[i][TWO_D_TO_ONE_D(j2, k, width)].r = new[TWO_D_TO_ONE_D(j2, k, width)].r;
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].g = new[TWO_D_TO_ONE_D(j2, k, width)].g;
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].b = new[TWO_D_TO_ONE_D(j2, k, width)].b;
                }
            }
            //CHECK THAT ALL THE OTHER PROCESSES ON THIS IMAGE HAVE END = 0
            int* received_end = malloc((nbProc-1) * sizeof(int));
            MPI_Allgather(&end, 1, MPI_INTEGER, received_end, (nbProc - 1), MPI_INTEGER, MPI_COMM_WORLD);

        } while (threshold > 0 && !end);

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

        free(new);
    }
}