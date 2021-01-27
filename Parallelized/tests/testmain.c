/*
 *  INF560
 * 
 *  Test file for image filtering project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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
    int *width;        /* Width of each image */
    int *height;       /* Height of each image */
    int *actualWidth;  /* Actual width of each image */
    int *actualHeight; /* Actual height of each image */
    pixel **p;         /* Pixels of each image */
} animated_gif;

animated_gif *load_pixels(animated_gif * original, int rank, int size);
int computeRemainder(int length, double newlength, int size, int n, double startInImage, int imgIndex);
animated_gif* createImage(int n_images, int width, int height);

int main(void)
{
    int i;

    /* Image parameters ; CHANGE HERE PARAMETERS */
    int n_images = 5;
    int width = 100;
    int height = 100;
    int size = 10; //Number of processes

    animated_gif *image = createImage(n_images, width, height);
    printf("+========================+\n");
    fflush(stdout);

    printf("#IMAGES = %d, #PROCESSES = %d\n+========================+\n", n_images, size);
    fflush(stdout);

    animated_gif **newImg = malloc(size * sizeof(animated_gif *));
    for (i = 0; i < size; ++i)
    {
        printf("Creating subimage(s) for process %d...\n", i);
        fflush(stdout);
        newImg[i] = load_pixels(image, i, size);
        if (newImg[i] == NULL)
        {
            fprintf(stderr, "Error while loading pixels for process with RANK %d\n", i);
            return 1;
        }
        printf("Subimage(s) for process %d successfully created !\n ==========================\n", i);
        fflush(stdout);
    }

    /* Free part */

    return 0;
}

/* ============================================================================================ */

animated_gif *load_pixels(animated_gif * original, int rank, int size)
{
    int error;
    int n;
    int n_images;
    int *width;
    int *height;
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
    width = (int *)malloc(n_images * sizeof(int));
    if (width == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    height = (int *)malloc(n_images * sizeof(int));
    if (height == NULL)
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
    if (height == NULL)
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
        actualWidth[i] = original->width[i2];
        actualHeight[i] = original->height[i2];
        printf("  Image #%d for Process %d:\n", i, rank);
        fflush(stdout);
        printf("    - ActualWidth = %d | ActualHeight = %d\n", actualWidth[i], actualHeight[i]);
        if (i < n_images - 1)
        {
            double w = (1 - tmpStart) * actualWidth[i];
            double h = (1 - tmpStart) * actualHeight[i];
            int rw = computeRemainder(actualWidth[i], w, size, n, tmpStart, i2);
            int rh = computeRemainder(actualHeight[i], h, size, n, tmpStart, i2);
            width[i] = floor(w) + rw;
            height[i] = floor(h) + rh;
            tmpStart = 0;
            printf("    - Width = %d | Height = %d\n", width[i], height[i]);
            printf("    - RemainderWidth = %d | RemainderHeight = %d\n", rw, rh);
        }
        else
        {
            //If end = 0 (possible from its computaiton) then w=0, h=0 and we have an empty image, which is not bothering because further access to that image will do nothing
            double w = (end - tmpStart) * actualWidth[i];
            double h = (end - tmpStart) * actualHeight[i];
            width[i] = floor(w);
            height[i] = floor(h);
            #if SOBELF_DEBUG
                printf("Image %d: w:%d h:%d\n",
                   i2,
                   original->width[i2],
                   original->height[i2]);
            #endif
            printf("    - Width = %d | Height = %d\n", width[i], height[i]);
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
        p[i] = (pixel *)malloc(width[i] * height[i] * sizeof(pixel));
        if (p[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                    i, width[i] * height[i]);
            return NULL;
        }
    }

    /* Fill pixels */
    int startIndexWidth = floor(start * actualWidth[0]);
    int startIndexHeight = floor(start * actualHeight[0]);
    printf(" Pixel part: startIndexWidth = %d | startIndexHeight = %d\n", startIndexWidth, startIndexHeight);
    fflush(stdout);

    /* For each image */
    for (i = 0; i < n_images; i++)
    {
        int j;
        int k;
        int i2 = imgStartIndex + i;

        /* Traverse the image and fill pixels */
        for (j = startIndexHeight; j < startIndexHeight + height[i]; ++j)
        {
            for (k = startIndexWidth; k < startIndexWidth + width[i]; ++k)
            {
                int j2 = j - startIndexHeight;
                int k2 = k - startIndexWidth;

                p[i][j2 * width[i] + k2].r = original->p[i2][j * actualWidth[i] + k].r;
                p[i][j2 * width[i] + k2].g = original->p[i2][j * actualWidth[i] + k].g;
                p[i][j2 * width[i] + k2].b = original->p[i2][j * actualWidth[i] + k].b;
            }
        }
        if (i < n_images - 1)
        {
            startIndexWidth = 0;
            startIndexHeight = 0;
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
    image->width = width;
    image->height = height;
    image->actualWidth = actualWidth;
    image->actualHeight = actualHeight;
    image->p = p;

    #if SOBELF_DEBUG
        printf("-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0]);
    #endif

    return image;
}

int computeRemainder(int length, double newlength, int size, int n, double startInImage, int imgIndex)
{
    int numProc = (n <= size) ? ceil((double)size / (double)n) : (startInImage == 0 ? 1 : 2);
    double tmp = length * ((double)n / (double)size);
    double remainder = tmp - (int)tmp;
    double tmpFirstRemainder = (n <= size) ? (((double)n / (double)size) * numProc * imgIndex - imgIndex) : startInImage; //ISSUE WHEN N <= SIZE
    double firstRemainder = tmpFirstRemainder * length - (int)(tmpFirstRemainder * length);
    double lastRemainder = newlength - (int)newlength;

    return remainder * (numProc - 2) + firstRemainder + lastRemainder;
}













animated_gif* createImage(int n_images, int width, int height) {
    int i;

    /* Prepare parameters to give to image */
    int *w;
    int *h;
    int *aw;
    int *ah;
    pixel **p;
    w = (int *)malloc(n_images * sizeof(int));
    if (w == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    h = (int *)malloc(n_images * sizeof(int));
    if (h == NULL)
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
        w[i] = width;
        h[i] = height;
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
        p[i] = (pixel *)malloc(w[i] * h[i] * sizeof(pixel));
        if (p[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                    i, w[i] * h[i]);
            return NULL;
        }
    }
    for (i = 0; i < n_images; i++)
    {
        int j;

        /* Traverse the image and fill pixels */
        for (j = 0; j < w[i] * h[i]; j++)
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
    image->height = h;
    image->width = w;
    image->p = p;

    printf("Original Animated Image, with %d images:\n", n_images);
    fflush(stdout);
    for(i = 0; i<n_images; ++i) {
        printf("    - Width %d = %d | Height %d = %d\n", i, w[i], i, h[i]);
    }
    fflush(stdout);

    return image;
}