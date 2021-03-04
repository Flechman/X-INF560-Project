/*
 * INF560
 *
 * Image Filtering Project
 */
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
    int n_images;   /* Number of images */
    int *heightStart;  /* Index of start of each image (for height) */
    int *heightEnd;    /* Index of end of each image (for height) */
    int *actualWidth;  /* Actual width of each image (INITIAL width before parallelism) */
    int *actualHeight; /* Actual height of each image (INITIAL width before parallelism) */
    pixel **p;      /* Pixels of each image */
    GifFileType *g; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

/* Min and Max functions */
int min(int a, int b) { return (a < b) ? a : b; }
int max(int a, int b) { return (a > b) ? a : b; }

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *load_pixels(char *filename, int rank, int size)
{
    if(rank == 0) {
        GifFileType *g;
        ColorMapObject *colmap;
        int error;
        int n_images;
        int *heightStart;
        int *heightEnd;
        int *actualWidth;
        int *actualHeight;
        pixel **p;
        int i;
        animated_gif *image;

        /* Open the GIF image (read mode) */
        g = DGifOpenFileName(filename, &error);
        if (g == NULL)
        {
            fprintf(stderr, "Error DGifOpenFileName %s\n", filename);
            return NULL;
        }

        /* Read the GIF image */
        error = DGifSlurp(g);
        if (error != GIF_OK)
        {
            fprintf(stderr,
                    "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
            return NULL;
        }

        /* Grab the number of images and the size of each image */
        n_images = g->ImageCount;

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
        for (i = 0; i < n_images; i++)
        {
            actualWidth[i] = g->SavedImages[i].ImageDesc.Width;
            actualHeight[i] = g->SavedImages[i].ImageDesc.Height;
            heightStart[i] = 0;
            heightEnd[i] = round((double)actualHeight[i]/(double)size);

        #if SOBELF_DEBUG
            printf("Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i,
                g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap);
        #endif
        }

        /* Get the global colormap */
        colmap = g->SColorMap;
        if (colmap == NULL)
        {
            fprintf(stderr, "Error global colormap is NULL\n");
            return NULL;
        }

        #if SOBELF_DEBUG
        printf("Global color map: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag);
        #endif

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
            p[i] = (pixel *)malloc(actualWidth[i] * actualHeight[i] * sizeof(pixel));
            if (p[i] == NULL)
            {
                fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                        i, actualWidth[i] * actualHeight[i]);
                return NULL;
            }
        }

        /* Fill pixels */

        /* For each image */
        for (i = 0; i < n_images; i++)
        {
            int j;

            /* Get the local colormap if needed */
            if (g->SavedImages[i].ImageDesc.ColorMap)
            {

                /* TODO No support for local color map */
                fprintf(stderr, "Error: application does not support local colormap\n");
                return NULL;

                colmap = g->SavedImages[i].ImageDesc.ColorMap;
            }

            /* Traverse the image and fill pixels */
            for (j = 0; j < actualWidth[i] * actualHeight[i]; j++)
            {
                int c;

                c = g->SavedImages[i].RasterBits[j];

                p[i][j].r = colmap->Colors[c].Red;
                p[i][j].g = colmap->Colors[c].Green;
                p[i][j].b = colmap->Colors[c].Blue;
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
        image->actualWidth = actualWidth;
        image->actualHeight = actualHeight;
        image->heightStart = heightStart;
        image->heightEnd = heightEnd;
        image->p = p;
        image->g = g;

        #if SOBELF_DEBUG
        printf("-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0]);
        #endif

        return image;
    }
    return NULL;
}

animated_gif* distribute_image(animated_gif* original, int rank, int size) {
    int n_images = 0;
    int *heightStart;
    int *heightEnd;
    int *actualWidth;
    int *actualHeight;
    pixel **p;
    int i;

    if(rank == 0) {
        n_images = original->n_images;
        heightStart = original->heightStart;
        heightEnd = original->heightEnd;
        actualWidth = original->actualWidth;
        actualHeight = original->actualHeight;
        p = original->p;
    }

    MPI_Bcast(&n_images, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    if(rank != 0) {
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

        /* Allocate the array of pixels to be returned */
        p = (pixel **)malloc(n_images * sizeof(pixel *));
        if (p == NULL)
        {
            fprintf(stderr, "Unable to allocate array of %d images\n",
                    n_images);
            return NULL;
        }
    }

    //Fill in width and height
    MPI_Bcast(actualHeight, n_images, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(actualWidth, n_images, MPI_INTEGER, 0, MPI_COMM_WORLD);

    double fractionImage = 1.0 / (double)size;

    if(rank != 0) {
        double start = (double)rank * fractionImage;
        double end = (double)(rank + 1) * fractionImage;

        for(i = 0; i < n_images; ++i) {
            heightStart[i] = round(start * (double)actualHeight[i]);
            heightEnd[i] = round(end * (double)actualHeight[i]);

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
    }

    //Fill pixels
    for(i = 0; i < n_images; ++i) {
        int j, k, l;
        if(rank == 0) {
            //Send to every process its pixels
            for(j = 1; j < size; ++j) {
                int startIndex = round((double)j * fractionImage * (double)actualHeight[i]);
                int endIndex = round((double)(j+1) * fractionImage * (double)actualHeight[i]);
                int height = endIndex - startIndex;
                int rowLength = actualWidth[i] * 3;
                int *data = malloc(rowLength * height * sizeof(int));
                if(data == NULL) {
                    printf("ERROR : could not allocate %d x %d integers\n", height, rowLength);
                    return 0;
                }
                for(k = 0; k < height; ++k) {
                    int k2 = k + startIndex;
                    for(l = 0; l < actualWidth[i]; ++l) {
                        data[l + k * rowLength] = p[i][TWO_D_TO_ONE_D(k2, l, actualWidth[i])].r;
                        data[l + actualWidth[i] + k * rowLength] = p[i][TWO_D_TO_ONE_D(k2, l, actualWidth[i])].g;
                        data[l + 2 * actualWidth[i] + k * rowLength] = p[i][TWO_D_TO_ONE_D(k2, l, actualWidth[i])].b;
                    }
                }
                MPI_Send(data, rowLength * height, MPI_INTEGER, j, i, MPI_COMM_WORLD);
                free(data);
            }
        } else {
            //Receive every pixel, store them
            int height = heightEnd[i] - heightStart[i];
            int rowLength = actualWidth[i] * 3;
            int *data = malloc(rowLength * height * sizeof(int));
            if(data == NULL) {
                printf("ERROR : could not allocate %d x %d integers\n", height, rowLength);
                return 0;
            }
            MPI_Recv(data, rowLength * height, MPI_INTEGER, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(j = 0; j < height; ++j) {
                for(k = 0; k < actualWidth[i]; ++k) {
                    pixel new_pixel = {.r = data[k + j * rowLength], 
                                       .g = data[k + actualWidth[i] + j * rowLength], 
                                       .b = data[k + 2 * actualWidth[i] + j * rowLength] };
                    p[i][TWO_D_TO_ONE_D(j, k, actualWidth[i])] = new_pixel;
                }
            }
            free(data);
        }
    }

    if(rank == 0) {
        return original;
    } else {
        /* Allocate image info */
        animated_gif *image = (animated_gif *)malloc(sizeof(animated_gif));
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
        image->g = NULL;
        
        return image;
    }
}

int output_modified_read_gif(char *filename, GifFileType *g)
{
    GifFileType *g2;
    int error2;

#if SOBELF_DEBUG
    printf("Starting output to file %s\n", filename);
#endif

    g2 = EGifOpenFileName(filename, false, &error2);
    if (g2 == NULL)
    {
        fprintf(stderr, "Error EGifOpenFileName %s\n",
                filename);
        return 0;
    }

    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;

    error2 = EGifSpew(g2);
    if (error2 != GIF_OK)
    {
        fprintf(stderr, "Error after writing g2: %d <%s>\n",
                error2, GifErrorString(g2->Error));
        return 0;
    }

    return 1;
}

int merge_image(animated_gif* image, int rank, int size) {
    int i, j, k, l;
    double fractionImage = 1.0 / (double)size;
    for (i = 0; i < image->n_images; ++i)
    {
        int width = image->actualWidth[i];
        if (rank == 0)
        {
            //Receive pixels of every process
            for (j = 1; j < size; ++j)
            {
                int startIndex = round(j * fractionImage * image->actualHeight[i]);
                int endIndex = round((j+1) * fractionImage * image->actualHeight[i]);
                int height = endIndex - startIndex;
                int rowLength = width * 3;
                int *data = malloc(rowLength * height * sizeof(int));
                if(data == NULL) {
                    printf("ERROR : could not allocate %d x %d integers\n", height, rowLength);
                    return 0;
                }
                MPI_Recv(data, rowLength * height, MPI_INTEGER, j, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (k = 0; k < height; ++k)
                {
                    int k2 = k+startIndex;
                    for(l = 0; l < width; ++l) {
                        pixel new_pixel = {.r = data[l + k * rowLength], .g = data[l + width + k * rowLength], .b = data[l + 2 * width + k * rowLength]};
                        image->p[i][TWO_D_TO_ONE_D(k2, l, width)] = new_pixel;
                    }
                }
                free(data);
            }
        }
        else
        {
            //Send every pixel to process 0
            int height = image->heightEnd[i] - image->heightStart[i];
            int rowLength = width * 3;
            int *data = malloc(rowLength * height * sizeof(int));
            if(data == NULL) {
                printf("ERROR : could not allocate %d x %d integers\n", height, rowLength);
                return 0;
            }
            for (j = 0; j < height; ++j)
            {
                for (k = 0; k < width; ++k)
                {
                    data[k + j * rowLength] = image->p[i][TWO_D_TO_ONE_D(j, k, width)].r;
                    data[k + width + j * rowLength] = image->p[i][TWO_D_TO_ONE_D(j, k, width)].g;
                    data[k + 2 * width + j * rowLength] = image->p[i][TWO_D_TO_ONE_D(j, k, width)].b;
                }
            }
            MPI_Send(data, rowLength * height, MPI_INTEGER, 0, i, MPI_COMM_WORLD);
            free(data);
        }
    }
    
    return 1;
}

int store_pixels(char *filename, animated_gif *image, int rank, int size)
{
    if(rank == 0) {
        int n_colors = 0;
        pixel **p;
        int i, j, k;
        GifColorType *colormap;

        /* Initialize the new set of colors */
        colormap = (GifColorType *)malloc(256 * sizeof(GifColorType));
        if (colormap == NULL)
        {
            fprintf(stderr,
                    "Unable to allocate 256 colors\n");
            return 0;
        }

        /* Everything is white by default */
        for (i = 0; i < 256; i++)
        {
            colormap[i].Red = 255;
            colormap[i].Green = 255;
            colormap[i].Blue = 255;
        }

        /* Change the background color and store it */
        int moy;
        moy = (image->g->SColorMap->Colors[image->g->SBackGroundColor].Red +
            image->g->SColorMap->Colors[image->g->SBackGroundColor].Green +
            image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue) /
            3;
        if (moy < 0) { moy = 0; }
        if (moy > 255) { moy = 255; }

        #if SOBELF_DEBUG
        printf("[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[image->g->SBackGroundColor].Red,
            image->g->SColorMap->Colors[image->g->SBackGroundColor].Green,
            image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue,
            moy, moy, moy);
        #endif

        colormap[0].Red = moy;
        colormap[0].Green = moy;
        colormap[0].Blue = moy;

        image->g->SBackGroundColor = 0;

        n_colors++;

        /* Process extension blocks in main structure */
        for (j = 0; j < image->g->ExtensionBlockCount; j++)
        {
            int f;

            f = image->g->ExtensionBlocks[j].Function;
            if (f == GRAPHICS_EXT_FUNC_CODE)
            {
                int tr_color = image->g->ExtensionBlocks[j].Bytes[3];

                if (tr_color >= 0 &&
                    tr_color < 255)
                {

                    int found = -1;

                    moy =
                        (image->g->SColorMap->Colors[tr_color].Red +
                        image->g->SColorMap->Colors[tr_color].Green +
                        image->g->SColorMap->Colors[tr_color].Blue) /
                        3;
                    if (moy < 0)
                        moy = 0;
                    if (moy > 255)
                        moy = 255;

                    #if SOBELF_DEBUG
                    printf("[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[tr_color].Red,
                        image->g->SColorMap->Colors[tr_color].Green,
                        image->g->SColorMap->Colors[tr_color].Blue,
                        moy, moy, moy);
                    #endif

                    for (k = 0; k < n_colors; k++)
                    {
                        if (
                            moy == colormap[k].Red &&
                            moy == colormap[k].Green &&
                            moy == colormap[k].Blue)
                        {
                            found = k;
                        }
                    }
                    if (found == -1)
                    {
                        if (n_colors >= 256)
                        {
                            fprintf(stderr,
                                    "Error: Found too many colors inside the image\n");
                            return 0;
                        }

                        #if SOBELF_DEBUG
                        printf("[DEBUG]\tNew color %d\n",
                            n_colors);
                        #endif

                        colormap[n_colors].Red = moy;
                        colormap[n_colors].Green = moy;
                        colormap[n_colors].Blue = moy;

                        image->g->ExtensionBlocks[j].Bytes[3] = n_colors;

                        n_colors++;
                    }
                    else
                    {
                        #if SOBELF_DEBUG
                        printf("[DEBUG]\tFound existing color %d\n",
                            found);
                        #endif
                        image->g->ExtensionBlocks[j].Bytes[3] = found;
                    }
                }
            }
        }

        for (i = 0; i < image->n_images; i++)
        {
            for (j = 0; j < image->g->SavedImages[i].ExtensionBlockCount; j++)
            {
                int f;

                f = image->g->SavedImages[i].ExtensionBlocks[j].Function;
                if (f == GRAPHICS_EXT_FUNC_CODE)
                {
                    int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3];

                    if (tr_color >= 0 &&
                        tr_color < 255)
                    {

                        int found = -1;

                        moy =
                            (image->g->SColorMap->Colors[tr_color].Red +
                            image->g->SColorMap->Colors[tr_color].Green +
                            image->g->SColorMap->Colors[tr_color].Blue) /
                            3;
                        if (moy < 0) { moy = 0; }
                        if (moy > 255) { moy = 255; }

                        #if SOBELF_DEBUG
                        printf("[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[tr_color].Red,
                            image->g->SColorMap->Colors[tr_color].Green,
                            image->g->SColorMap->Colors[tr_color].Blue,
                            moy, moy, moy);
                        #endif

                        for (k = 0; k < n_colors; k++)
                        {
                            if (
                                moy == colormap[k].Red &&
                                moy == colormap[k].Green &&
                                moy == colormap[k].Blue)
                            {
                                found = k;
                            }
                        }
                        if (found == -1)
                        {
                            if (n_colors >= 256)
                            {
                                fprintf(stderr,
                                        "Error: Found too many colors inside the image\n");
                                return 0;
                            }

                            #if SOBELF_DEBUG
                            printf("[DEBUG]\tNew color %d\n",
                                n_colors);
                            #endif

                            colormap[n_colors].Red = moy;
                            colormap[n_colors].Green = moy;
                            colormap[n_colors].Blue = moy;

                            image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors;

                            n_colors++;
                        }
                        else
                        {
                            #if SOBELF_DEBUG
                            printf("[DEBUG]\tFound existing color %d\n",
                                found);
                            #endif
                            image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found;
                        }
                    }
                }
            }
        }

        #if SOBELF_DEBUG
        printf("[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors);
        #endif

        p = image->p;

        /* Find the number of colors inside the image */
        for (i = 0; i < image->n_images; i++)
        {

            #if SOBELF_DEBUG
            printf("OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->actualWidth[i], image->actualHeight[i]);
            #endif

            for (j = 0; j < image->actualWidth[i] * image->actualHeight[i]; j++)
            {
                int found = 0;
                for (k = 0; k < n_colors; k++)
                {
                    if (p[i][j].r == colormap[k].Red &&
                        p[i][j].g == colormap[k].Green &&
                        p[i][j].b == colormap[k].Blue)
                    {
                        found = 1;
                    }
                }

                if (found == 0)
                {
                    if (n_colors >= 256)
                    {
                        fprintf(stderr,
                                "Error: Found too many colors inside the image\n");
                        return 0;
                    }

                    #if SOBELF_DEBUG
                    printf("[DEBUG] Found new %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b);
                    #endif

                    colormap[n_colors].Red = p[i][j].r;
                    colormap[n_colors].Green = p[i][j].g;
                    colormap[n_colors].Blue = p[i][j].b;
                    n_colors++;
                }
            }
        }

        #if SOBELF_DEBUG
        printf("OUTPUT: found %d color(s)\n", n_colors);
        #endif

        /* Round up to a power of 2 */
        if (n_colors != (1 << GifBitSize(n_colors)))
        {
            n_colors = (1 << GifBitSize(n_colors));
        }

        #if SOBELF_DEBUG
        printf("OUTPUT: Rounding up to %d color(s)\n", n_colors);
        #endif

        /* Change the color map inside the animated gif */
        ColorMapObject *cmo;

        cmo = GifMakeMapObject(n_colors, colormap);
        if (cmo == NULL)
        {
            fprintf(stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                    n_colors);
            return 0;
        }

        image->g->SColorMap = cmo;

        /* Update the raster bits according to color map */
        for (i = 0; i < image->n_images; i++)
        {
            for (j = 0; j < image->actualWidth[i] * image->actualHeight[i]; j++)
            {
                int found_index = -1;
                for (k = 0; k < n_colors; k++)
                {
                    if (p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                        p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                        p[i][j].b == image->g->SColorMap->Colors[k].Blue)
                    {
                        found_index = k;
                    }
                }

                if (found_index == -1)
                {
                    fprintf(stderr,
                            "Error: Unable to find a pixel in the color map\n");
                    return 0;
                }

                image->g->SavedImages[i].RasterBits[j] = found_index;
            }
        }

        /* Write the final image */
        if (!output_modified_read_gif(filename, image->g))
        {
            return 0;
        }

        return 1;
        }
    return 1;
}

void apply_gray_filter(animated_gif *image, int rank, int size)
{
    int i, j;
    pixel **p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        int width = image->actualWidth[i];
        int height = image->heightEnd[i] - image->heightStart[i];
        for (j = 0; j < width * height; j++)
        {
            int moy;

            moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
            if (moy < 0) { moy = 0; }
            if (moy > 255) { moy = 255; }

            p[i][j].r = moy;
            p[i][j].g = moy;
            p[i][j].b = moy;
        }
    }
}

void apply_gray_line(animated_gif *image, int rank, int size)
{
    int i, j, k;
    pixel **p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        if (image->heightStart[i] < 10)
        {
            int end = (image->heightEnd[i] <= 10) ? image->heightEnd[i] : 10;
            int width = image->actualWidth[i];
            for (j = image->heightStart[i]; j < end; j++)
            {
                for (k = width / 2; k < width; k++)
                {
                    int j2 = j - image->heightStart[i];
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].r = 0;
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].g = 0;
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].b = 0;
                }
            }
        }
    }
}

int get_rank(int indexHeight, int actualHeight, int size, int rank, bool is_asc) {
    int i = rank;
    bool in_bounds = false;
    while(!in_bounds)
    {
        double fractionImage = 1.0/(double)size;
        int startIndex = round(i * fractionImage * actualHeight);
        int endIndex = round((i+1) * fractionImage * actualHeight);
        if(startIndex >= actualHeight) {
            printf("ERROR: get_rank loops forever\n");
            return -1;
        }
        if (indexHeight >= startIndex && indexHeight < endIndex) {
            in_bounds = true;
        } else {
            i = is_asc ? i+1 : i-1;
        }
    }
    return i;
}

/* Assumes that the height of every division of each image is greater or equal to the radius of the blur 'size' */
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
                pixel *receivedTopPart = (pixel *)malloc(size * width * sizeof(pixel));
                pixel *receivedBottomPart = (pixel *)malloc(size * width * sizeof(pixel));

                if (image->heightStart[i] > 0)
                {
                    /* Recv from previous process */
                    int *dataRecv = malloc(size * width * 3 * sizeof(int));
                    MPI_Recv(dataRecv, 3 * width * size, MPI_INTEGER, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            pixel new_pixel = {.r = dataRecv[j * width * 3 + k], .g = dataRecv[j * width * 3 + k + width], .b = dataRecv[j * width * 3 + k + 2 * width]};
                            receivedTopPart[TWO_D_TO_ONE_D(j, k, width)] = new_pixel;
                        }
                    }
                    free(dataRecv);

                    /* Send to previous process */
                    int *dataSend = malloc(size * width * 3 * sizeof(int));
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            dataSend[j * width * 3 + k] = p[i][TWO_D_TO_ONE_D(j, k, width)].r;
                            dataSend[j * width * 3 + k + width] = p[i][TWO_D_TO_ONE_D(j, k, width)].g;
                            dataSend[j * width * 3 + k + 2 * width] = p[i][TWO_D_TO_ONE_D(j, k, width)].b;
                        }
                    }
                    MPI_Send(dataSend, 3 * width * size, MPI_INTEGER, rank - 1, 0, MPI_COMM_WORLD);
                    free(dataSend);
                }

                if (image->heightEnd[i] < image->actualHeight[i] / 10)
                {
                    /* Send to next process */
                    int *dataSend = malloc(size * width * 3 * sizeof(int));
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            int j2 = j + height - size;
                            dataSend[j * width * 3 + k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                            dataSend[j * width * 3 + k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                            dataSend[j * width * 3 + k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                        }
                    }
                    MPI_Send(dataSend, 3 * width * size, MPI_INTEGER, rank + 1, 0, MPI_COMM_WORLD);
                    free(dataSend);

                    /* Recv from next process */
                    int *dataRecv = malloc(size * width * 3 * sizeof(int));
                    MPI_Recv(dataRecv, 3 * width * size, MPI_INTEGER, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            pixel new_pixel = {.r = dataRecv[j * width * 3 + k], .g = dataRecv[j * width * 3 + k + width], .b = dataRecv[j * width * 3 + k + 2 * width]};
                            receivedBottomPart[TWO_D_TO_ONE_D(j, k, width)] = new_pixel;
                        }
                    }
                    free(dataRecv);
                }

                /* Compute blur */
                int heightStartLocal = max(size, image->heightStart[i]);
                int heightEndLocal = min(image->actualHeight[i] / 10 - size, image->heightEnd[i]);
                for (j = heightStartLocal; j < heightEndLocal; ++j)
                {
                    for (k = size; k < width - size; k++)
                    {
                        int stencil_j, stencil_k;
                        int t_r = 0;
                        int t_g = 0;
                        int t_b = 0;
                        for (stencil_j = -size; stencil_j <= size; ++stencil_j)
                        {
                            if (j + stencil_j < image->heightStart[i])
                            {
                                int j2 = size - (image->heightStart[i] - j - stencil_j);
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
                pixel *receivedTopPart = (pixel *)malloc(size * width * sizeof(pixel));
                pixel *receivedBottomPart = (pixel *)malloc(size * width * sizeof(pixel));

                if (image->heightStart[i] > image->actualHeight[i] * 0.9)
                {
                    /* Recv from previous process */
                    int *dataRecv = malloc(size * width * 3 * sizeof(int));
                    MPI_Recv(dataRecv, 3 * width * size, MPI_INTEGER, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            pixel new_pixel = {.r = dataRecv[j * width * 3 + k], .g = dataRecv[j * width * 3 + k + width], .b = dataRecv[j * width * 3 + k + 2 * width]};
                            receivedTopPart[TWO_D_TO_ONE_D(j, k, width)] = new_pixel;
                        }
                    }
                    free(dataRecv);

                    /* Send to previous process */
                    //BE CAREFUL TO THE image->heightStart[i] + j >= image->actualHeight[i]
                    int *dataSend = malloc(size * width * 3 * sizeof(int));
                    for (j = 0; j < size; ++j)
                    {
                        int j2 = j + image->heightStart[i];
                        for (k = 0; k < width; ++k)
                        {
                            if (j2 >= image->actualHeight[i])
                            {
                                dataSend[j * width * 3 + k] = 0;
                                dataSend[j * width * 3 + k + width] = 0;
                                dataSend[j * width * 3 + k + 2 * width] = 0;
                            }
                            else
                            {
                                dataSend[j * width * 3 + k] = p[i][TWO_D_TO_ONE_D(j, k, width)].r;
                                dataSend[j * width * 3 + k + width] = p[i][TWO_D_TO_ONE_D(j, k, width)].g;
                                dataSend[j * width * 3 + k + 2 * width] = p[i][TWO_D_TO_ONE_D(j, k, width)].b;
                            }
                        }
                    }
                    MPI_Send(dataSend, 3 * width * size, MPI_INTEGER, rank - 1, 0, MPI_COMM_WORLD);
                    free(dataSend);
                }

                if (image->heightEnd[i] < image->actualHeight[i])
                {
                    /* Send to next process */
                    int *dataSend = malloc(size * width * 3 * sizeof(int));
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            int j2 = j + height - size;
                            dataSend[j * width * 3 + k] = p[i][TWO_D_TO_ONE_D(j2, k, width)].r;
                            dataSend[j * width * 3 + k + width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].g;
                            dataSend[j * width * 3 + k + 2 * width] = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                        }
                    }
                    MPI_Send(dataSend, 3 * width * size, MPI_INTEGER, rank + 1, 0, MPI_COMM_WORLD);
                    free(dataSend);

                    /* Recv from next process */
                    int *dataRecv = malloc(size * width * 3 * sizeof(int));
                    MPI_Recv(dataRecv, 3 * width * size, MPI_INTEGER, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (j = 0; j < size; ++j)
                    {
                        for (k = 0; k < width; ++k)
                        {
                            pixel new_pixel = {.r = dataRecv[j * width * 3 + k], .g = dataRecv[j * width * 3 + k + width], .b = dataRecv[j * width * 3 + k + 2 * width]};
                            receivedBottomPart[TWO_D_TO_ONE_D(j, k, width)] = new_pixel;
                        }
                    }
                    free(dataRecv);
                }

                /* Compute blur */
                int heightStartLocal = max(image->heightStart[i], image->actualHeight[i] * 0.9 + size);
                int heightEndLocal = min(image->heightEnd[i], image->actualHeight[i] - size);
                for (j = heightStartLocal; j < heightEndLocal; j++)
                {
                    for (k = size; k < width - size; k++)
                    {
                        int stencil_j, stencil_k;
                        int t_r = 0;
                        int t_g = 0;
                        int t_b = 0;
                        for (stencil_j = -size; stencil_j <= size; ++stencil_j)
                        {
                            if (j + stencil_j < image->heightStart[i])
                            {
                                int j2 = size - (image->heightStart[i] - j - stencil_j);
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
            int *received_end = malloc(nbProc * sizeof(int));
            MPI_Allgather(&end, 1, MPI_INTEGER, received_end, 1, MPI_INTEGER, MPI_COMM_WORLD);
            for (j = 0; j < nbProc; ++j)
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

void apply_sobel_filter(animated_gif *image, int rank, int size)
{
    int i, j, k;
    int width, height;

    pixel **p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        width = image->actualWidth[i];
        height = image->heightEnd[i] - image->heightStart[i];

        pixel *sobel;
        pixel *above = malloc(width * sizeof(pixel));
        pixel *below = malloc(width * sizeof(pixel));
        if(image->heightStart[i] > 0) {
            int *dataSend = malloc(width * 3 * sizeof(int));
            int *dataRecv = malloc(width * 3 * sizeof(int));
            for(j = 0; j < width; ++j) {
                dataSend[j] = image->p[i][TWO_D_TO_ONE_D(0, j, width)].r;
                dataSend[j + width] = image->p[i][TWO_D_TO_ONE_D(0, j, width)].g;
                dataSend[j + 2 * width] = image->p[i][TWO_D_TO_ONE_D(0, j, width)].b;
            }
            MPI_Recv(dataRecv, 3 * width, MPI_INTEGER, rank-1, image->heightStart[i]-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(dataSend, 3 * width, MPI_INTEGER, rank-1, image->heightStart[i], MPI_COMM_WORLD);
            for(j = 0; j < width; ++j) {
                pixel new_pixel = {.r = dataRecv[j], .g = dataRecv[j + width], .b = dataRecv[j + 2 * width]};
                below[TWO_D_TO_ONE_D(0, j, width)] = new_pixel;
            }
            free(dataSend);
            free(dataRecv);
        }
        if(image->heightEnd[i] < image->actualHeight[i]) {
            int *dataSend = malloc(width * 3 * sizeof(int));
            int *dataRecv = malloc(width * 3 * sizeof(int));
            int height = image->heightEnd[i] - image->heightStart[i];
            for(j = 0; j < width; ++j) {
                dataSend[j] = image->p[i][TWO_D_TO_ONE_D(height-1, j, width)].r;
                dataSend[j + width] = image->p[i][TWO_D_TO_ONE_D(height-1, j, width)].g;
                dataSend[j + 2 * width] = image->p[i][TWO_D_TO_ONE_D(height-1, j, width)].b;
            }
            MPI_Send(dataSend, 3 * width, MPI_INTEGER, rank+1, image->heightEnd[i]-1, MPI_COMM_WORLD);
            MPI_Recv(dataRecv, 3 * width, MPI_INTEGER, rank+1, image->heightEnd[i], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(j = 0; j < width; ++j) {
                pixel new_pixel = {.r = dataRecv[j], .g = dataRecv[j + width], .b = dataRecv[j + 2 * width]};
                above[TWO_D_TO_ONE_D(0, j, width)] = new_pixel;
            }
            free(dataSend);
            free(dataRecv);
        }

        sobel = (pixel *)malloc(width * height * sizeof(pixel));

        for (j = max(1, image->heightStart[i]); j < min(image->actualHeight[i] - 1, image->heightEnd[i]); j++)
        {
            int j2 = j - image->heightStart[i];
            for (k = 1; k < width - 1; k++)
            {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_o , pixel_blue  , pixel_blue_e ;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;

                float deltaX_blue;
                float deltaY_blue;
                float val_blue;

                pixel_blue_no = (j - 1 < image->heightStart[i]) ? below[TWO_D_TO_ONE_D(0, k - 1, width)].b : p[i][TWO_D_TO_ONE_D(j2 - 1, k - 1, width)].b;
                pixel_blue_n = (j - 1 < image->heightStart[i]) ? below[TWO_D_TO_ONE_D(0, k, width)].b : p[i][TWO_D_TO_ONE_D(j2 - 1, k, width)].b;
                pixel_blue_ne = (j - 1 < image->heightStart[i]) ? below[TWO_D_TO_ONE_D(0, k + 1, width)].b : p[i][TWO_D_TO_ONE_D(j2 - 1, k + 1, width)].b;
                pixel_blue_so = (j + 1 >= image->heightEnd[i]) ? above[TWO_D_TO_ONE_D(0, k - 1, width)].b : p[i][TWO_D_TO_ONE_D(j2 + 1, k - 1, width)].b;
                pixel_blue_s = (j + 1 >= image->heightEnd[i]) ? above[TWO_D_TO_ONE_D(0, k, width)].b : p[i][TWO_D_TO_ONE_D(j2 + 1, k, width)].b;
                pixel_blue_se = (j + 1 >= image->heightEnd[i]) ? above[TWO_D_TO_ONE_D(0, k + 1, width)].b : p[i][TWO_D_TO_ONE_D(j2 + 1, k + 1, width)].b;
                pixel_blue_o = p[i][TWO_D_TO_ONE_D(j2, k - 1, width)].b;
                pixel_blue = p[i][TWO_D_TO_ONE_D(j2, k, width)].b;
                pixel_blue_e = p[i][TWO_D_TO_ONE_D(j2, k + 1, width)].b;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

                deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

                if (val_blue > 50)
                {
                    sobel[TWO_D_TO_ONE_D(j2, k, width)].r = 255;
                    sobel[TWO_D_TO_ONE_D(j2, k, width)].g = 255;
                    sobel[TWO_D_TO_ONE_D(j2, k, width)].b = 255;
                }
                else
                {
                    sobel[TWO_D_TO_ONE_D(j2, k, width)].r = 0;
                    sobel[TWO_D_TO_ONE_D(j2, k, width)].g = 0;
                    sobel[TWO_D_TO_ONE_D(j2, k, width)].b = 0;
                }
            }
        }

        for (j = max(1, image->heightStart[i]); j < min(image->actualHeight[i] - 1, image->heightEnd[i]); j++)
        {
            int j2 = j - image->heightStart[i];
            for (k = 1; k < width - 1; k++)
            {
                p[i][TWO_D_TO_ONE_D(j2, k, width)].r = sobel[TWO_D_TO_ONE_D(j2, k, width)].r;
                p[i][TWO_D_TO_ONE_D(j2, k, width)].g = sobel[TWO_D_TO_ONE_D(j2, k, width)].g;
                p[i][TWO_D_TO_ONE_D(j2, k, width)].b = sobel[TWO_D_TO_ONE_D(j2, k, width)].b;
            }
        }

        free(sobel);
        free(above);
        free(below);
    }
}

/*
 * Main entry point
 */
int main(int argc, char **argv)
{

    int rank, size;

    /* MPI Initialization */
    MPI_Init(&argc, &argv);

    /* Get the rank of the current task and the number of MPI processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /*================ IMPORT ======================*/

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    if(rank == 0) {
        image = load_pixels(input_filename, rank, size);
        if (image == NULL)
        {
            return 1;
        }
    }
    image = distribute_image(image, rank, size);
    if (image == NULL)
    {
        return 1;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /*==============================================*/
    /*================= FILTER =====================*/

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale */
    apply_gray_filter(image, rank, size);

    /* Apply blur filter with convergence value */
    apply_blur_filter(image, 5, 20, rank, size);

    /* Apply sobel filter on pixels */
    apply_sobel_filter(image, rank, size);

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

    /*==============================================*/

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if(!merge_image(image, rank, size)) {
        return 1;
    }
    if(rank == 0) {
        if (!store_pixels(output_filename, image, rank, size))
        {
            return 1;
        }
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);

    MPI_Finalize();
    return 0;
}
