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

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images;      /* Number of images */
    int *widthStart;   /* Index of start of each image width */
    int *widthEnd;     /* Index of end of each image  width */
    int *heightStart;  /* Index of start of each image height */
    int *heightEnd;    /* Index of end of each image height */
    int *actualWidth;  /* Actual width of each image */
    int *actualHeight; /* Actual height of each image */
    pixel **p;         /* Pixels of each image */
} animated_gif;

animated_gif *load_pixels(animated_gif *original, int rank, int size);
animated_gif *createImage(int n_images, int width, int height);

int main(void)
{
    int i;

    /* Image parameters ; CHANGE HERE PARAMETERS */
    int n_images = 3;
    int width = 100;
    int height = 100;
    int size = 2; //Number of processes

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
        printf("Subimage(s) for process %d successfully created !\n==========================\n", i);
        fflush(stdout);
    }

    /* Free part */

    return 0;
}

/* ============================================================================================ */

animated_gif *load_pixels(animated_gif *original, int rank, int size)
{
    int error;
    int n;
    int n_images;
    int *widthStart;
    int *widthEnd;
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
    widthStart = (int *)malloc(n_images * sizeof(int));
    if (widthStart == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    widthEnd = (int *)malloc(n_images * sizeof(int));
    if (widthEnd == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
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
            double isw = tmpStart * actualWidth[i];
            double ish = tmpStart * actualHeight[i];
            //int rw = computeRemainder(actualWidth[i], w, size, n, rank, tmpStart, i2);
            //int rh = computeRemainder(actualHeight[i], h, size, n, rank, tmpStart, i2);
            widthStart[i] = round(isw);
            heightStart[i] = round(ish);
            widthEnd[i] = actualWidth[i];
            heightEnd[i] = actualHeight[i];
            tmpStart = 0;
            printf("    - Width = %d | Height = %d\n", widthEnd[i] - widthStart[i], heightEnd[i] - heightStart[i]);
            //printf("    - RemainderWidth = %d | RemainderHeight = %d\n", rw, rh);
        }
        else
        {
            //If end = 0 (possible from its computaiton) then w=0, h=0 and we have an empty image, which is not bothering because further access to that image will do nothing
            double isw = tmpStart * actualWidth[i];
            double ish = tmpStart * actualHeight[i];
            double iew = end * actualWidth[i];
            double ieh = end * actualHeight[i];
            widthStart[i] = round(isw);
            heightStart[i] = round(ish);
            widthEnd[i] = round(iew);
            heightEnd[i] = round(ieh);
            printf("    - Width = %d | Height = %d\n", widthEnd[i] - widthStart[i], heightEnd[i] - heightStart[i]);
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
        int width = widthEnd[i] - widthStart[i];
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
        int width = widthEnd[i] - widthStart[i];
        int height = heightEnd[i] - heightStart[i];
        printf(" Pixel part %d: startIndexWidth = %d | startIndexHeight = %d\n", i, widthStart[i], heightStart[i]);
        fflush(stdout);

        /* Traverse the image and fill pixels */
        for (j = heightStart[i]; j < heightEnd[i]; ++j)
        {
            for (k = widthStart[i]; k < widthEnd[i]; ++k)
            {
                int j2 = j - heightStart[i];
                int k2 = k - widthStart[i];

                p[i][j2 * width + k2].r = original->p[i2][j * actualWidth[i] + k].r;
                p[i][j2 * width + k2].g = original->p[i2][j * actualWidth[i] + k].g;
                p[i][j2 * width + k2].b = original->p[i2][j * actualWidth[i] + k].b;
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
    image->widthStart = widthStart;
    image->heightStart = heightStart;
    image->widthEnd = widthEnd;
    image->heightEnd = heightEnd;
    image->actualWidth = actualWidth;
    image->actualHeight = actualHeight;
    image->p = p;

#if SOBELF_DEBUG
    printf("-> GIF w/ %d image(s) with first image of size %d x %d\n",
           image->n_images, image->width[0], image->height[0]);
#endif

    return image;
}


animated_gif *createImage(int n_images, int width, int height)
{
    int i;

    /* Prepare parameters to give to image */
    int *isw;
    int *iew;
    int *ish;
    int *ieh;
    int *aw;
    int *ah;
    pixel **p;
    isw = (int *)malloc(n_images * sizeof(int));
    if (isw == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return NULL;
    }
    iew = (int *)malloc(n_images * sizeof(int));
    if (iew == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return NULL;
    }
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
        isw[i] = 0;
        iew[i] = width;
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
    image->widthStart = isw;
    image->widthEnd = iew;
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