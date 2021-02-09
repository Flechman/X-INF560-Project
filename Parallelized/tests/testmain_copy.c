/*
 *  INF560
 * 
 *  Test file for image filtering project
 */
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Maps 2D (l, c) coordinates to 1D l*nb_c + c coordinate */
#define TWO_D_TO_ONE_D(l, c, nb_c) \
    ((l) * (nb_c) + (c))

typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images;      /* Number of images */
    int *heightStart;  /* Index of start of each image height */
    int *heightEnd;    /* Index of end of each image height */
    int *actualWidth;  /* Actual width of each image */
    int *actualHeight; /* Actual height of each image */
    pixel **p;         /* Pixels of each image */
} animated_gif;

animated_gif *load_pixels(animated_gif *original, int rank, int size);
animated_gif *createImage(int n_images, int width, int height);

void print_pixels(animated_gif *image) {

    int i,j,k;

    for (k = 0; k < image->n_images; k++)
    {
        fflush(stdout);
        printf("Image %d top:\n", k);
        for(i = 0; i < image->actualHeight[k]; i += image->actualWidth[k]) {
            fflush(stdout);
            printf("[\n\t");
            for(j = i; j < image->actualWidth[k]; j++) {
                fflush(stdout);
                printf(" (%d,%d,%d) ", image->p[k][j].r, image->p[k][j].b, image->p[k][j].g);
            }
            fflush(stdout);
            printf("\t\n]");
        }

        fflush(stdout);
        printf("Image %d bottom \n", k);
    }
}
    void
apply_sobel_filter( animated_gif * image, int rank, int size)
{
    int i, j, k ;
    int width, height ;
    MPI_Status status;
    pixel ** p ;

    p = image->p ;



    for ( i = 0 ; i < image->n_images ; i++ )
    {

        pixel *left_border_pixels;
        pixel *right_border_pixels;
        int left_neighbor = -1;
        int right_neighbor = -1;

        /* sharing_enabled is used to check if there would be communication */
        bool sharing_enabled = false;

        width = image->actualWidth[i];
        height = image->heightEnd[i] - image->heightStart[i] ;
        pixel * sobel ; 

        sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;
        left_border_pixels = (pixel *) malloc(width * sizeof(pixel)); /* This stores the first line from right neighbor */
        right_border_pixels = (pixel *) malloc(width * sizeof(pixel)); /* This stores the first line from left neighbor */




        if (height <= 0 ) {
            /* This case is to erradicate the case where height is 0 */
        }
        else
        {
            if (height != image->actualHeight[i]) {
                sharing_enabled = true;
            }

            if (sharing_enabled) {
                // determine your neighbors

                if(image->heightStart[i] != 0) {
                    left_neighbor = rank - 1;
                }
                if (image->heightEnd[i] != image->actualHeight[i])
                    right_neighbor = rank + 1;
            }


            if (left_neighbor != -1)
            {
                for(j= width*(height - 1); j < width * height; j++){
                    int _j = j % width; /* To convert 0 -> (width * height) to 0 -> width */
                    right_border_pixels[_j].r = p[i][j].r;
                    right_border_pixels[_j].b = p[i][j].b;
                    right_border_pixels[_j].g = p[i][j].g;
                }

                MPI_Recv(left_border_pixels, 3*width, MPI_INTEGER, left_neighbor, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                MPI_Send(right_border_pixels, 3*width,  MPI_INTEGER, left_neighbor, rank, MPI_COMM_WORLD);
            }
            if (right_neighbor != -1) {
                printf("start = %d; end = %d\n", width * (height - 1), width * height);

                for(j= width * (height - 1); j < width * height; j++){

                    int _j = j % width; /* To convert 0 -> (width * height) to 0 -> width */
                    left_border_pixels[_j].r = p[i][j].r;
                    left_border_pixels[_j].b = p[i][j].b;
                    left_border_pixels[_j].g = p[i][j].g;


                }
                MPI_Send(left_border_pixels, 3*width, MPI_INTEGER, right_neighbor, rank, MPI_COMM_WORLD);
                MPI_Recv(right_border_pixels, 3*width, MPI_INTEGER, right_neighbor, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }


            int start = 1;
            int end = height - 1;

            if(left_neighbor != -1)
                start = image->heightStart[i];
            if(right_neighbor != -1)
                end = image->heightEnd[i];

            for(j= start; j<end; j++)
            {
                for(k=1; k<width-1; k++)
                {

                    int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                    int pixel_blue_o , pixel_blue  , pixel_blue_e ;
                    int pixel_blue_so, pixel_blue_s, pixel_blue_se;

                    float deltaX_blue ;
                    float deltaY_blue ;
                    float val_blue;

                    if (j == (image->heightEnd[i] - 1)){
                        pixel_blue_no = right_border_pixels[TWO_D_TO_ONE_D(j-1,k-1,width) % width].b;
                        pixel_blue_n  = right_border_pixels[TWO_D_TO_ONE_D(j-1,k  ,width) % width].b ;
                        pixel_blue_ne = right_border_pixels[TWO_D_TO_ONE_D(j-1,k+1,width) % width].b ;
                    }
                    else {
                        pixel_blue_no = p[i][TWO_D_TO_ONE_D(j-1,k-1,width)].b ;
                        pixel_blue_n  = p[i][TWO_D_TO_ONE_D(j-1,k  ,width)].b ;
                        pixel_blue_ne = p[i][TWO_D_TO_ONE_D(j-1,k+1,width)].b ;
                    }
                    if (j == image->heightStart[i]) {
                        pixel_blue_so = left_border_pixels[TWO_D_TO_ONE_D(j+1,k-1,width) % width].b ;
                        pixel_blue_s  = left_border_pixels[TWO_D_TO_ONE_D(j+1,k  ,width) % width].b ;
                        pixel_blue_se = left_border_pixels[TWO_D_TO_ONE_D(j+1,k+1,width) % width].b ;
                    }
                    else {
                        pixel_blue_so = p[i][TWO_D_TO_ONE_D(j+1,k-1,width)].b ;
                        pixel_blue_s  = p[i][TWO_D_TO_ONE_D(j+1,k  ,width)].b ;
                        pixel_blue_se = p[i][TWO_D_TO_ONE_D(j+1,k+1,width)].b ;
                    }
                    pixel_blue_o  = p[i][TWO_D_TO_ONE_D(j  ,k-1,width)].b ;
                    pixel_blue    = p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].b ;
                    pixel_blue_e  = p[i][TWO_D_TO_ONE_D(j  ,k+1,width)].b ;

                    deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;             

                    deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

                    val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;
                    //val_blue = 50;


                    if ( val_blue > 50 ) 
                    {
                        sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].r = 255 ;
                        sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].g = 255 ;
                        sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].b = 255 ;
                    } else
                    {
                        sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].r = 0 ;
                        sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].g = 0 ;
                        sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].b = 0 ;
                    }
                }
            }

            for(j=1; j<height-1; j++)
            {
                for(k=1; k<width-1; k++)
                {
                    p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].r = sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].r ;
                    p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].g = sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].g ;
                    p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].b = sobel[TWO_D_TO_ONE_D(j  ,k  ,width)].b ;
                }
            }
        }

        free (sobel) ;
    }

    print_pixels(image);

}
int main(int argc, char** argv)
{
    int rank, size;

    /* MPI Initialization */
    MPI_Init(&argc, &argv);

    /* Get the rank of the current task and the number of MPI processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int i;

    /* Image parameters ; CHANGE HERE PARAMETERS */
    printf("--- Params ---\nN_IMAGES=%d WIDTH=%d HEIGHT=%d\n", atoi(argv[1]), atoi(argv[2]), atoi(argv[2]));
    int n_images = atoi(argv[1]);
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);

    animated_gif *image = createImage(n_images, width, height);
    //printf("+========================+\n");
    //fflush(stdout);

    //printf("#IMAGES = %d, #PROCESSES = %d\n+========================+\n", n_images, size);
    //fflush(stdout);

    animated_gif **newImg = malloc(size * sizeof(animated_gif *));

    //printf("Creating subimage(s) for process %d...\n", i);
    //fflush(stdout);

    newImg[rank] = load_pixels(image, rank, size);

    //fflush(stdout);
    //printf("Before sobel:\n");
    //print_pixels(newImg[rank]);
    //fflush(stdout);

    apply_sobel_filter(newImg[rank], rank, size);

    //fflush(stdout);
    //printf("After sobel:\n");
    //print_pixels(newImg[rank]);
    //fflush(stdout);

    if (newImg[rank] == NULL)
    {
        fprintf(stderr, "Error while loading pixels for process with RANK %d\n", rank);
        return 1;
    }
    printf("Subimage(s) for process %d successfully created !\n==========================\n", rank);
    fflush(stdout);

    /* Free part */

    MPI_Finalize();
    return 0;
}

/* ============================================================================================ */

animated_gif *load_pixels(animated_gif *original, int rank, int size)
{
    int error;
    int n;
    int n_images;
    int *heightStart;
    int *heightEnd;
    int *actualWidth;
    int *actualHeight;
    pixel **p;
    int i;
    animated_gif *image;

    /* Grab the number of images */
    n = original->n_images;

    //The index of the image at which this process starts
    int imgStartIndex = (int)(((double)n / (double)size) * rank);
    //The index of the image at which this process ends
    int imgEndIndex = (int)(((double)n / (double)size) * (rank + 1));
    //The fraction of the image 'imgStartIndex' at which the process starts
    double start = rank * ((double)n / (double)size) - imgStartIndex;
    //The fraction of the image 'imgEndIndex' at which the process ends
    double end = (rank + 1) * ((double)n / (double)size) - imgEndIndex;
    //Number of images on which this process works on (contiguous images)
    n_images = imgEndIndex - imgStartIndex + 1;

    printf("    - imgStartIndex =.%d\n", imgStartIndex);
    printf("    - imgEndIndex =...%d\n", imgEndIndex);
    printf("    - start =.........%.5f\n", start);
    printf("    - end =...........%.5f\n", end);
    fflush(stdout);

    /* Allocate width and height */
    heightStart = (int *)malloc(n_images * sizeof(int));
    if (heightStart == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return NULL;
    }
    heightEnd = (int *)malloc(n_images * sizeof(int));
    if (heightEnd == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return NULL;
    }
    actualWidth = (int *)malloc(n_images * sizeof(int));
    if (actualWidth == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    actualHeight = (int *)malloc(n_images * sizeof(int));
    if (actualHeight == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return NULL;
    }

    /* Fill the width and height */
    /* If distribution of the image not perfectly balanced, convention that the last portion of the image will take the few pixels more */
    double tmpStart = start;
    for (i = 0; i < n_images; i++)
    {
        int i2 = imgStartIndex + i;
        actualWidth[i] = (i2 >= n) ? 0 : original->actualWidth[i2];
        actualHeight[i] = (i2 >= n) ? 0 : original->actualHeight[i2];
        printf("  Image #%d for Process %d:\n", i, rank);
        fflush(stdout);
        if (i < n)
        {
            printf("    - ActualWidth = %d | ActualHeight = %d\n", actualWidth[i], actualHeight[i]);
        }
        if (i < n_images - 1)
        {
            double ish = tmpStart * actualHeight[i];
            //int rw = computeRemainder(actualWidth[i], w, size, n, rank, tmpStart, i2);
            //int rh = computeRemainder(actualHeight[i], h, size, n, rank, tmpStart, i2);
            heightStart[i] = round(ish);
            heightEnd[i] = actualHeight[i];
            tmpStart = 0;
            printf("    - Width = %d | Height = %d\n", actualWidth[i], heightEnd[i] - heightStart[i]);
            //printf("    - RemainderWidth = %d | RemainderHeight = %d\n", rw, rh);
        }
        else
        {
            //If end = 0 (possible from its computaiton) then w=0, h=0 and we have an empty image, which is not bothering because further access to that image will do nothing
            double ish = tmpStart * actualHeight[i];
            double ieh = end * actualHeight[i];
            heightStart[i] = round(ish);
            heightEnd[i] = round(ieh);
            if(end == 0) { actualWidth[i] = 0; actualHeight[i] = 0; }
            printf("    - Width = %d | Height = %d\n", actualWidth[i], heightEnd[i] - heightStart[i]);
        }
        fflush(stdout);
    }

    /* Allocate the array of pixels to be returned */
    p = (pixel **)malloc(n_images * sizeof(pixel *));
    if (p == NULL)
    {
        fprintf(stderr, "Unable to allocate array of %d images\n",
                n_images);
        return NULL;
    }
    for (i = 0; i < n_images; i++)
    {
        int width = actualWidth[i];
        int height = heightEnd[i] - heightStart[i];
        p[i] = (pixel *)malloc(width * height * sizeof(pixel));
        if (p[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                    i, width * height);
            return NULL;
        }
    }

    /* Fill pixels */

    /* For each image */
    for (i = 0; i < n_images; i++)
    {
        int j;
        int k;
        int i2 = imgStartIndex + i;
        int width = actualWidth[i];
        int height = heightEnd[i] - heightStart[i];
        printf(" Pixel part %d: startIndexHeight = %d\n", i, heightStart[i]);
        fflush(stdout);

        /* Traverse the image and fill pixels */
        for (j = heightStart[i]; j < heightEnd[i]; ++j)
        {
            for (k = 0; k < width; ++k)
            {
                int j2 = j - heightStart[i];

                p[i][j2 * width + k].r = original->p[i2][j * actualWidth[i] + k].r;
                p[i][j2 * width + k].g = original->p[i2][j * actualWidth[i] + k].g;
                p[i][j2 * width + k].b = original->p[i2][j * actualWidth[i] + k].b;
            }
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc(sizeof(animated_gif));
    if (image == NULL)
    {
        fprintf(stderr, "Unable to allocate memory for animated_gif\n");
        return NULL;
    }

    /* Fill image fields */
    image->n_images = n_images;
    image->heightStart = heightStart;
    image->heightEnd = heightEnd;
    image->actualWidth = actualWidth;
    image->actualHeight = actualHeight;
    image->p = p;

    return image;
}


animated_gif *createImage(int n_images, int width, int height)
{
    int i;

    /* Prepare parameters to give to image */
    int *ish;
    int *ieh;
    int *aw;
    int *ah;
    pixel **p;
    ish = (int *)malloc(n_images * sizeof(int));
    if (ish == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    ieh = (int *)malloc(n_images * sizeof(int));
    if (ieh == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return NULL;
    }
    aw = (int *)malloc(n_images * sizeof(int));
    if (aw == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    ah = (int *)malloc(n_images * sizeof(int));
    if (ah == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return NULL;
    }
    for (i = 0; i < n_images; ++i)
    {
        ish[i] = 0;
        ieh[i] = height;
        aw[i] = width;
        ah[i] = height;
    }
    p = (pixel **)malloc(n_images * sizeof(pixel *));
    if (p == NULL)
    {
        fprintf(stderr, "Unable to allocate array of %d images\n",
                n_images);
        return NULL;
    }
    for (i = 0; i < n_images; i++)
    {
        p[i] = (pixel *)malloc(aw[i] * ah[i] * sizeof(pixel));
        if (p[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                    i, aw[i] * ah[i]);
            return NULL;
        }
    }
    for (i = 0; i < n_images; i++)
    {
        int j;

        /* Traverse the image and fill pixels */
        for (j = 0; j < aw[i] * ah[i]; j++)
        {
            /* w[i] x h[i] chessboard */
            p[i][j].r = j % 2 ? 255 : 0;
            p[i][j].g = j % 2 ? 255 : 0;
            p[i][j].b = j % 2 ? 255 : 0;
        }
    }

    /* Allocate image */
    animated_gif *image = (animated_gif *)malloc(sizeof(animated_gif));
    if (image == NULL)
    {
        fprintf(stderr, "Unable to allocate memory for animated_gif\n");
        return NULL;
    }

    image->n_images = n_images;
    image->actualHeight = ah;
    image->actualWidth = aw;
    image->heightStart = ish;
    image->heightEnd = ieh;
    image->p = p;

    printf("Original Animated Image, with %d images:\n", n_images);
    fflush(stdout);
    for (i = 0; i < n_images; ++i)
    {
        printf("    - Width %d = %d | Height %d = %d\n", i, aw[i], i, ah[i]);
    }
    fflush(stdout);

    return image;
}
