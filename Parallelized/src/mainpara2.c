#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "gif_lib.h"
#include "filters/gray_filter.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Maps 2D (l, c) coordinates to 1D l*nb_c + c coordinate */
#define TWO_D_TO_ONE_D(l, c, nb_c) \
    ((l) * (nb_c) + (c))

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images;      /* Number of images */
    int *heightStart;  /* Index of start of each image (for height) */
    int *heightEnd;    /* Index of end of each image (for height) */
    int *actualWidth;  /* Actual width of each image (INITIAL width before parallelism) */
    int *actualHeight; /* Actual height of each image (INITIAL width before parallelism) */
    pixel **p;         /* Pixels of each image */
    GifFileType *g;    /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

/* Min and Max functions */
int min(int a, int b) { return (a < b) ? a : b; }
int max(int a, int b) { return (a > b) ? a : b; }

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
                for (k = 0; k < width - 1; k++)
                {
                    int j2 = j - image->heightStart[i];
                    new[TWO_D_TO_ONE_D(j2, k, width)].r = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                    new[TWO_D_TO_ONE_D(j2, k, width)].g = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                    new[TWO_D_TO_ONE_D(j2, k, width)].b = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                }
            }

            /* First 10% of the image */
            if (image->heightStart[i] < image->actualHeight[i] / 10)
            {
                int heightEnd = min(image->heightEnd[i], image->actualHeight[i] / 10);

                int subStart = (size - image->heightStart[i] > 0) ? size - image->heightStart[i] : 0;
                int subEnd = (heightEnd - 1 - (image->actualHeight[i] / 10 - size - 1) > 0) ? heightEnd - 1 - (image->actualHeight[i] / 10 - size - 1) : 0;
                int topSize = (heightEnd <= size) ? 0 : size - subStart;
                int bottomSize = (image->heightStart[i] >= image->actualHeight[i] / 10 - size) ? 0 : size - subEnd;
                pixel *receivedTopPart = (pixel *)malloc(topSize * width * sizeof(pixel));
                pixel *receivedBottomPart = (pixel *)malloc(bottomSize * width * sizeof(pixel));

                /* Send & Receive Data : Receive first from upper, and then Send to upper ; Send to lower, Receive from lower */
                /* Receive from upper */
                for (j = 0; j < topSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, image->heightStart[i] - topSize + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedTopPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }
                /* Send to upper */
                int rowUSize = (min(image->heightStart[i] + size - 1, heightEnd - 1) - image->heightStart[i] + 1);
                int **rowU = malloc(rowUSize * sizeof(int *));
                MPI_Request **requestsU = malloc(rowUSize * sizeof(MPI_Request *));
                int *countU = malloc(rowUSize * sizeof(int));
                for (j = image->heightStart[i]; j <= min(image->heightStart[i] + size - 1, heightEnd - 1); ++j)
                {
                    int j2 = j - image->heightStart[i];
                    //Send to processes having rows max(size, j-size) : heightStart-1
                    rowU[j2] = malloc(width * 3 * sizeof(int));
                    for (k = 0; k < width; ++k)
                    {
                        rowU[j2][k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        rowU[j2][k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        rowU[j2][k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    countU[j2] = 0;
                    for (k = max(size, j - size); k <= image->heightStart[i] - 1; ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, false);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            ++countU[j2];
                        }
                    }
                    requestsU[j2] = malloc(countU[j2] * sizeof(MPI_Request));
                    for (k = 0; k < countU[j2]; ++k)
                    {
                        MPI_Isend(rowU[j2], 3 * width, MPI_INTEGER, rank - k - 1, j, MPI_COMM_WORLD, &requestsU[j2][k]);
                    }
                }
                /* Send to lower */
                int rowLSize = (heightEnd - 1 - max(heightEnd - size, image->heightStart[i]) + 1);
                int **rowL = malloc(rowLSize * sizeof(int *));
                MPI_Request **requestsL = malloc(rowLSize * sizeof(MPI_Request *));
                int *countL = malloc(rowLSize * sizeof(int));
                for (j = max(heightEnd - size, image->heightStart[i]); j <= heightEnd - 1; ++j)
                {
                    int j2 = j - image->heightStart[i];
                    int j3 = j - max(heightEnd - size, image->heightStart[i]);
                    //Send to processes having rows heightEnd : min(H/10-size-1, j+size)
                    rowL[j3] = malloc(width * 3 * sizeof(int));
                    for (k = 0; k < width; ++k)
                    {
                        rowL[j3][k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        rowL[j3][k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        rowL[j3][k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    countL[j3] = 0;
                    for (k = heightEnd; k <= min(image->actualHeight[i] / 10 - size - 1, j + size); ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, true);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            ++countL[j3];
                        }
                    }
                    requestsL[j3] = malloc(countL[j3] * sizeof(MPI_Request));
                    for (k = 0; k < countL[j3]; ++k)
                    {
                        MPI_Isend(rowL[j3], 3 * width, MPI_INTEGER, rank + k + 1, j, MPI_COMM_WORLD, &requestsL[j3][k]);
                    }
                }
                /* Receive from lower */
                for (j = 0; j < bottomSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, heightEnd + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedBottomPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }

                for (j = 0; j < rowUSize; ++j)
                {
                    MPI_Waitall(countU[j], requestsU[j], MPI_STATUSES_IGNORE);
                    free(rowU[j]);
                    free(requestsU[j]);
                }
                for (j = 0; j < rowLSize; ++j)
                {
                    MPI_Waitall(countL[j], requestsL[j], MPI_STATUSES_IGNORE);
                    free(rowL[j]);
                    free(requestsL[j]);
                }
                free(rowU);
                free(rowL);
                free(requestsU);
                free(requestsL);
                free(countU);
                free(countL);

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
            for (j = max(image->actualHeight[i] / 10 - size, image->heightStart[i]); j < min(image->heightEnd[i], image->actualHeight[i] * 0.9 + size); ++j)
            {
                int j2 = j - image->heightStart[i];
                for (k = size; k < width - size; ++k)
                {
                    new[TWO_D_TO_ONE_D(j2, k, width)].r = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                    new[TWO_D_TO_ONE_D(j2, k, width)].g = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                    new[TWO_D_TO_ONE_D(j2, k, width)].b = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                }
            }

            /* Last 10% of the image */
            if (image->heightEnd[i] > image->actualHeight[i] * 0.9)
            {
                int heightStart = max(image->heightStart[i], image->actualHeight[i] * 0.9);

                int subStart = (image->actualHeight[i] * 0.9 + size - heightStart > 0) ? image->actualHeight[i] * 0.9 + size - heightStart : 0;
                int subEnd = (image->heightEnd[i] - 1 - (image->actualHeight[i] - size - 1) > 0) ? image->heightEnd[i] - 1 - (image->actualHeight[i] - size - 1) : 0;
                int topSize = (image->heightEnd[i] <= image->actualHeight[i] * 0.9 + size) ? 0 : size - subStart;
                int bottomSize = (image->heightStart[i] >= image->actualHeight[i] - size) ? 0 : size - subEnd;
                pixel *receivedTopPart = (pixel *)malloc(topSize * width * sizeof(pixel));
                pixel *receivedBottomPart = (pixel *)malloc(bottomSize * width * sizeof(pixel));

                /* Send & Receive Data : Receive first from upper, and then Send to upper ; Send to lower, Receive from lower */
                /* Receive from upper */
                for (j = 0; j < topSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, heightStart - topSize + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedTopPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }
                /* Send to upper */
                int rowUSize = min(heightStart + size - 1, image->heightEnd[i] - 1) - heightStart + 1;
                int **rowU = malloc(rowUSize * sizeof(int *));
                MPI_Request **requestsU = malloc(rowUSize * sizeof(MPI_Request *));
                int *countU = malloc(rowUSize * sizeof(int));
                for (j = heightStart; j <= min(heightStart + size - 1, image->heightEnd[i] - 1); ++j)
                {
                    int j2 = j - image->heightStart[i];
                    int j3 = j - heightStart;
                    //Send to processes having rows max(actualHeight * 0.9 + size, j-size) : heightStart-1
                    rowU[j3] = malloc(width * 3 * sizeof(int));
                    for (k = 0; k < width; ++k)
                    {
                        rowU[j3][k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        rowU[j3][k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        rowU[j3][k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    countU[j3] = 0;
                    for (k = max(image->actualHeight[i] * 0.9 + size, j - size); k <= heightStart - 1; ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, false);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            ++countU[j3];
                        }
                    }
                    requestsU[j3] = malloc(countU[j3] * sizeof(MPI_Request));
                    for (k = 0; k < countU[j3]; ++k)
                    {
                        MPI_Isend(rowU[j3], 3 * width, MPI_INTEGER, rank - k - 1, j, MPI_COMM_WORLD, &requestsU[j3][k]);
                    }
                }
                /* Send to lower */
                int rowLSize = image->heightEnd[i] - 1 - max(heightStart, image->heightEnd[i] - size) + 1;
                int **rowL = malloc(rowLSize * sizeof(int *));
                MPI_Request **requestsL = malloc(rowLSize * sizeof(MPI_Request *));
                int *countL = malloc(rowLSize * sizeof(int));
                for (j = max(heightStart, image->heightEnd[i] - size); j <= image->heightEnd[i] - 1; ++j)
                {
                    int j2 = j - image->heightStart[i];
                    int j3 = j - max(heightStart, image->heightEnd[i] - size);
                    //Send to processes having rows heightEnd : min(H/10-size-1, i+size)
                    rowL[j3] = malloc(width * 3 * sizeof(int));
                    for (k = 0; k < width; ++k)
                    {
                        rowL[j3][k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                        rowL[j3][k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                        rowL[j3][k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                    }
                    int to_rank = rank;
                    countL[j3] = 0;
                    for (k = image->heightEnd[i]; k <= min(image->actualHeight[i] - size - 1, j + size); ++k)
                    {
                        int new_to_rank = get_rank(k, image->actualHeight[i], nbProc, to_rank, true);
                        if (new_to_rank != to_rank)
                        {
                            to_rank = new_to_rank;
                            ++countL[j3];
                        }
                    }
                    requestsL[j3] = malloc(countL[j3] * sizeof(MPI_Request));
                    for (k = 0; k < countL[j3]; ++k)
                    {
                        MPI_Isend(rowL[j3], 3 * width, MPI_INTEGER, rank + k + 1, j, MPI_COMM_WORLD, &requestsL[j3][k]);
                    }
                }
                /* Receive from lower */
                for (j = 0; j < bottomSize; ++j)
                {
                    int *tmp = malloc(3 * width * sizeof(int));
                    MPI_Recv(tmp, 3 * width, MPI_INTEGER, MPI_ANY_SOURCE, image->heightEnd[i] + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (k = 0; k < width; ++k)
                    {
                        pixel new_pixel = {.r = tmp[k], .g = tmp[k + width], .b = tmp[k + 2 * width]};
                        receivedBottomPart[j * width + k] = new_pixel;
                    }
                    free(tmp);
                }

                for (j = 0; j < rowUSize; ++j)
                {
                    MPI_Waitall(countU[j], requestsU[j], MPI_STATUSES_IGNORE);
                    free(rowU[j]);
                    free(requestsU[j]);
                }
                for (j = 0; j < rowLSize; ++j)
                {
                    MPI_Waitall(countL[j], requestsL[j], MPI_STATUSES_IGNORE);
                    free(rowL[j]);
                    free(requestsL[j]);
                }
                free(rowU);
                free(rowL);
                free(requestsU);
                free(requestsL);
                free(countU);
                free(countL);

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
                int j2 = j - image->heightStart[i];
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
            int *received_end = malloc((nbProc - 1) * sizeof(int));
            MPI_Allgather(&end, 1, MPI_INTEGER, received_end, (nbProc - 1), MPI_INTEGER, MPI_COMM_WORLD);
            for (j = 0; j < nbProc - 1; ++j)
            {
                if (received_end[j] == 0)
                {
                    end = 0;
                }
            }
            free(received_end);

        } while (threshold > 0 && !end);

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

        free(new);
    }
}