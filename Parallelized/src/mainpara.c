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

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Maps 2D (l, c) coordinates to 1D l*nb_c + c coordinate */
#define TWO_D_TO_ONE_D(l, c, nb_c) \
    ((l) * (nb_c) + (c))

/* Represent one pixel from the image */
typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images;      /* Number of images */
    int *width;
    int *height;
    int *heightStart;  /* Index of start of each image (for height) */
    int *heightEnd;    /* Index of end of each image (for height) */
    int *actualWidth;  /* Actual width of each image (INITIAL width before parallelism) */
    int *actualHeight; /* Actual height of each image (INITIAL width before parallelism) */
    pixel **p;         /* Pixels of each image */
    GifFileType *g;    /* Internal representation.
                          DO NOT MODIFY */
} animated_gif;

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *load_pixels(char *filename, int rank, int size)
{
    GifFileType * g ;
    ColorMapObject * colmap ;
    int error ;
    int n ;
    int n_images ;
    int *heightStart;
    int *heightEnd;
    int *actualWidth;
    int *actualHeight;
    pixel ** p ;
    int i ;
    animated_gif * image ;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName( filename, &error ) ;
    if ( g == NULL ) 
    {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename ) ;
        return NULL ;
    }

    /* Read the GIF image */
    error = DGifSlurp( g ) ;
    if ( error != GIF_OK )
    {
        fprintf( stderr, 
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) ) ;
        return NULL ;
    }

    /* Grab the number of images */
    n = g->ImageCount ;

    //The index of the image at which this process starts to work on
    int imgStartIndex = (int)(((double)n/(double)size) * rank);
    //The index of the image at which this process ends to work on
    int imgEndIndex = (int)(((double)n / (double)size) * (rank + 1));
    //The fraction of the image 'imgStartIndex' at which the process starts (invariant : start of 'rank' is end of 'rank-1')
    double start = rank * ((double)n / (double)size) - imgStartIndex;
    //The fraction of the image 'imgEndIndex' at which the process ends (invariant : end of 'rank' is the start of 'rank+1')
    double end = (rank + 1) * ((double)n / (double)size) - imgEndIndex;
    //Number of images on which this process works on (contiguous images)
    n_images = imgEndIndex - imgStartIndex + 1;

    /* Allocate heightStart, heightEnd, actualWidth, actualHeight */
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

    //Temp fraction of an image, updated at each iteration to dermine new start index (always 0 when i>0)
    double tmpStart = start;

    for (i = 0; i < n_images; i++)
    {
        int i2 = imgStartIndex + i;
        actualWidth[i] = (i2 >= n) ? 0 : g->SavedImages[i2].ImageDesc.Width;
        actualHeight[i] = (i2 >= n) ? 0 : g->SavedImages[i2].ImageDesc.Height;
        if (i < n_images - 1)
        {
            double ish = tmpStart * actualHeight[i];
            heightStart[i] = round(ish);
            heightEnd[i] = actualHeight[i];
            tmpStart = 0;
        }
        else
        {
            //If end = 0 (possible from its computaiton) then w=0, h=0 and we have an empty image, which is not bothering because further access to that image will do nothing
            double ish = tmpStart * actualHeight[i];
            double ieh = end * actualHeight[i];
            heightStart[i] = round(ish);
            heightEnd[i] = round(ieh);
            if (end == 0) { actualWidth[i] = 0; actualHeight[i] = 0; }
        }
#if SOBELF_DEBUG
        if (i2 < n)
        {
            printf("Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                    i2,
                    g->SavedImages[i2].ImageDesc.Left,
                    g->SavedImages[i2].ImageDesc.Top,
                    g->SavedImages[i2].ImageDesc.Width,
                    g->SavedImages[i2].ImageDesc.Height,
                    g->SavedImages[i2].ImageDesc.Interlace,
                    g->SavedImages[i2].ImageDesc.ColorMap);
        }
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
        int i2 = imgStartIndex + i;
        if(i2 < n) {
            int j;
            int k;

            /* Get the local colormap if needed */
            if (g->SavedImages[i2].ImageDesc.ColorMap)
            {

                /* TODO No support for local color map */
                fprintf(stderr, "Error: application does not support local colormap\n");
                return NULL;

                colmap = g->SavedImages[i2].ImageDesc.ColorMap;
            }

            /* Traverse the image and fill pixels */
            for (j = heightStart[i]; j < heightEnd[i]; ++j)
            {
                for (k = 0; k < actualWidth[i]; ++k)
                {
                    int c = g->SavedImages[i2].RasterBits[TWO_D_TO_ONE_D(j, k, actualWidth[i])];
                    int j2 = j-heightStart[i];

                    p[i][TWO_D_TO_ONE_D(j2, k, actualWidth[i])].r = colmap->Colors[c].Red;
                    p[i][TWO_D_TO_ONE_D(j2, k, actualWidth[i])].g = colmap->Colors[c].Green;
                    p[i][TWO_D_TO_ONE_D(j2, k, actualWidth[i])].b = colmap->Colors[c].Blue;
                }
            }
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc( sizeof(animated_gif) ) ;
    if ( image == NULL ) 
    {
        fprintf( stderr, "Unable to allocate memory for animated_gif\n" ) ;
        return NULL ;
    }

    /* Fill image fields */
    image->n_images = n_images ;
    image->heightStart = heightStart ;
    image->heightEnd = heightEnd ;
    image->actualWidth = actualWidth ;
    image->actualHeight = actualHeight ;
    image->height = actualHeight;
    image->width = actualWidth;
    image->p = p ;
    image->g = g ;

#if SOBELF_DEBUG
    printf("-> rank %d w/ %d image(s) with first (sub)image of size %d x %d\n", rank,
            image->n_images, image->widthEnd[0] - image->widthStart[0], image->heightEnd[0] - image->heightStart[0]);
#endif

    return image ;
}

int output_modified_read_gif( char * filename, GifFileType * g ) 
{
    GifFileType * g2 ;
    int error2 ;

#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename ) ;
#endif

    g2 = EGifOpenFileName( filename, false, &error2 ) ;
    if ( g2 == NULL )
    {
        fprintf( stderr, "Error EGifOpenFileName %s\n",
                filename ) ;
        return 0 ;
    }

    g2->SWidth = g->SWidth ;
    g2->SHeight = g->SHeight ;
    g2->SColorResolution = g->SColorResolution ;
    g2->SBackGroundColor = g->SBackGroundColor ;
    g2->AspectByte = g->AspectByte ;
    g2->SColorMap = g->SColorMap ;
    g2->ImageCount = g->ImageCount ;
    g2->SavedImages = g->SavedImages ;
    g2->ExtensionBlockCount = g->ExtensionBlockCount ;
    g2->ExtensionBlocks = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK ) 
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", 
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}


int store_pixels( char * filename, animated_gif * image )
{
    int n_colors = 0 ;
    pixel ** p ;
    int i, j, k ;
    GifColorType * colormap ;

    /* Initialize the new set of colors */
    colormap = (GifColorType *)malloc( 256 * sizeof( GifColorType ) ) ;
    if ( colormap == NULL ) 
    {
        fprintf( stderr,
                "Unable to allocate 256 colors\n" ) ;
        return 0 ;
    }

    /* Everything is white by default */
    for ( i = 0 ; i < 256 ; i++ ) 
    {
        colormap[i].Red = 255 ;
        colormap[i].Green = 255 ;
        colormap[i].Blue = 255 ;
    }

    /* Change the background color and store it */
    int moy ;
    moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
          )/3 ;
    if ( moy < 0 ) moy = 0 ;
    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
    printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
            moy, moy, moy ) ;
#endif

    colormap[0].Red = moy ;
    colormap[0].Green = moy ;
    colormap[0].Blue = moy ;

    image->g->SBackGroundColor = 0 ;

    n_colors++ ;

    /* Process extension blocks in main structure */
    for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ )
    {
        int f ;

        f = image->g->ExtensionBlocks[j].Function ;
        if ( f == GRAPHICS_EXT_FUNC_CODE )
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3] ;

            if ( tr_color >= 0 &&
                    tr_color < 255 )
            {

                int found = -1 ;

                moy = 
                    (
                     image->g->SColorMap->Colors[ tr_color ].Red
                     +
                     image->g->SColorMap->Colors[ tr_color ].Green
                     +
                     image->g->SColorMap->Colors[ tr_color ].Blue
                    ) / 3 ;
                if ( moy < 0 ) moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[ tr_color ].Red,
                        image->g->SColorMap->Colors[ tr_color ].Green,
                        image->g->SColorMap->Colors[ tr_color ].Blue,
                        moy, moy, moy ) ;
#endif

                for ( k = 0 ; k < n_colors ; k++ )
                {
                    if ( 
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                       )
                    {
                        found = k ;
                    }
                }
                if ( found == -1  ) 
                {
                    if ( n_colors >= 256 ) 
                    {
                        fprintf( stderr, 
                                "Error: Found too many colors inside the image\n"
                               ) ;
                        return 0 ;
                    }

#if SOBELF_DEBUG
                    printf( "[DEBUG]\tNew color %d\n",
                            n_colors ) ;
#endif

                    colormap[n_colors].Red = moy ;
                    colormap[n_colors].Green = moy ;
                    colormap[n_colors].Blue = moy ;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors ;

                    n_colors++ ;
                } else
                {
#if SOBELF_DEBUG
                    printf( "[DEBUG]\tFound existing color %d\n",
                            found ) ;
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found ;
                }
            }
        }
    }

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ )
        {
            int f ;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function ;
            if ( f == GRAPHICS_EXT_FUNC_CODE )
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] ;

                if ( tr_color >= 0 &&
                        tr_color < 255 )
                {

                    int found = -1 ;

                    moy = 
                        (
                         image->g->SColorMap->Colors[ tr_color ].Red
                         +
                         image->g->SColorMap->Colors[ tr_color ].Green
                         +
                         image->g->SColorMap->Colors[ tr_color ].Blue
                        ) / 3 ;
                    if ( moy < 0 ) moy = 0 ;
                    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                    printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[ tr_color ].Red,
                            image->g->SColorMap->Colors[ tr_color ].Green,
                            image->g->SColorMap->Colors[ tr_color ].Blue,
                            moy, moy, moy ) ;
#endif

                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if ( 
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                           )
                        {
                            found = k ;
                        }
                    }
                    if ( found == -1  ) 
                    {
                        if ( n_colors >= 256 ) 
                        {
                            fprintf( stderr, 
                                    "Error: Found too many colors inside the image\n"
                                   ) ;
                            return 0 ;
                        }

#if SOBELF_DEBUG
                        printf( "[DEBUG]\tNew color %d\n",
                                n_colors ) ;
#endif

                        colormap[n_colors].Red = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue = moy ;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors ;

                        n_colors++ ;
                    } else
                    {
#if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p ;

    /* Find the number of colors inside the image */
    for ( i = 0 ; i < image->n_images ; i++ )
    {

#if SOBELF_DEBUG
        printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->width[i], image->height[i] ) ;
#endif

        for ( j = 0 ; j < image->actualWidth[i] * image->actualHeight[i] ; j++ ) 
        {
            int found = 0 ;
            for ( k = 0 ; k < n_colors ; k++ )
            {
                if ( p[i][j].r == colormap[k].Red &&
                        p[i][j].g == colormap[k].Green &&
                        p[i][j].b == colormap[k].Blue )
                {
                    found = 1 ;
                }
            }

            if ( found == 0 ) 
            {
                if ( n_colors >= 256 ) 
                {
                    fprintf( stderr, 
                            "Error: Found too many colors inside the image\n"
                           ) ;
                    return 0 ;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red = p[i][j].r ;
                colormap[n_colors].Green = p[i][j].g ;
                colormap[n_colors].Blue = p[i][j].b ;
                n_colors++ ;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif


    /* Round up to a power of 2 */
    if ( n_colors != (1 << GifBitSize(n_colors) ) )
    {
        n_colors = (1 << GifBitSize(n_colors) ) ;
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject * cmo ;

    cmo = GifMakeMapObject( n_colors, colormap ) ;
    if ( cmo == NULL )
    {
        fprintf( stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                n_colors ) ;
        return 0 ;
    }

    image->g->SColorMap = cmo ;

    /* Update the raster bits according to color map */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->actualWidth[i] * image->actualHeight[i] ; j++ ) 
        {
            int found_index = -1 ;
            for ( k = 0 ; k < n_colors ; k++ ) 
            {
                if ( p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                        p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                        p[i][j].b == image->g->SColorMap->Colors[k].Blue )
                {
                    found_index = k ;
                }
            }

            if ( found_index == -1 ) 
            {
                fprintf( stderr,
                        "Error: Unable to find a pixel in the color map\n" ) ;
                return 0 ;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index ;
        }
    }


    /* Write the final image */
    if ( !output_modified_read_gif( filename, image->g ) ) { return 0 ; }

    return 1 ;
}

    void
apply_gray_filter( animated_gif * image, int rank, int size)
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        int width = image->actualWidth[i];
        int height = image->heightEnd[i] - image->heightStart[i];
        for ( j = 0 ; j < width * height ; j++ )
        {
            int moy ;

            moy = (p[i][j].r + p[i][j].g + p[i][j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[i][j].r = moy ;
            p[i][j].g = moy ;
            p[i][j].b = moy ;
        }
    }
}

void apply_gray_line( animated_gif * image, int rank, int size) 
{
    int i, j, k ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        if(image->heightStart[i] < 10) {
            int end = (image->heightEnd[i] <= 10) ? image->heightEnd[i] : 10;
            for (j = image->heightStart[i]; j < end; j++)
            {
                int width = image->actualWidth[i];
                for (k = width/2; k < width; k++)
                {
                    int j2 = j-image->heightStart[i];
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].r = 0;
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].g = 0;
                    p[i][TWO_D_TO_ONE_D(j2, k, width)].b = 0;
                }
            }
        }
    }
}

//void
//apply_blur_filter( animated_gif * image, int size, int threshold, int rank, int nbProc )
//{
//int i, j, k ;
//int width, height ;
//int end = 0 ;
//int n_iter = 0 ;

//pixel ** p ;
//pixel * new ;

///* Get the pixels of all images */
//p = image->p ;


///* Process all images */
//for ( i = 0 ; i < image->n_images ; i++ )
//{
//n_iter = 0 ;
//width = image->widthEnd[i] - image->widthStart[i];
//height = image->heightEnd[i] - image->heightStart[i];

///* Allocate array of new pixels */
//new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;


///* Perform at least one blur iteration */
//do
//{
//end = 1 ;
//n_iter++ ;

//int heightEnd = (image->heightEnd[i] >= (image->actualHeight[i]-1)) ? image->actualHeight[i]-1 : image->heightEnd[i];
//int widthEnd = (image->widthEnd[i] >= (image->actualWidth[i] - 1)) ? image->actualWidth[i] - 1 : image->widthEnd[i];
//for(j=image->heightStart[i]; j<heightEnd; j++)
//{
//for(k=image->widthStart[i]; k<widthEnd; k++)
//{
//int j2 = j - image->heightStart[i];
//int k2 = k - image->widthStart[i];
//new[TWO_D_TO_ONE_D(j2,k2,width)].r = p[i][TWO_D_TO_ONE_D(j2,k2,width)].r ;
//new[TWO_D_TO_ONE_D(j2,k2,width)].g = p[i][TWO_D_TO_ONE_D(j2,k2,width)].g ;
//new[TWO_D_TO_ONE_D(j2,k2,width)].b = p[i][TWO_D_TO_ONE_D(j2,k2,width)].b ;
//}
//}

///* Apply blur on top part of image (10%) DEPENDS ON OTHER PARTS OF THE IMAGE */
//for(j=size; j<height/10-size; j++)
//{
//for(k=size; k<width-size; k++)
//{
//int stencil_j, stencil_k ;
//int t_r = 0 ;
//int t_g = 0 ;
//int t_b = 0 ;

//for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
//{
//for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
//{
//t_r += p[i][TWO_D_TO_ONE_D(j+stencil_j,k+stencil_k,width)].r ;
//t_g += p[i][TWO_D_TO_ONE_D(j+stencil_j,k+stencil_k,width)].g ;
//t_b += p[i][TWO_D_TO_ONE_D(j+stencil_j,k+stencil_k,width)].b ;
//}
//}

//new[TWO_D_TO_ONE_D(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
//new[TWO_D_TO_ONE_D(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
//new[TWO_D_TO_ONE_D(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
//}
//}

///* Copy the middle part of the image */
//for(j=height/10-size; j<height*0.9+size; j++)
//{
//for(k=size; k<width-size; k++)
//{
//new[TWO_D_TO_ONE_D(j,k,width)].r = p[i][TWO_D_TO_ONE_D(j,k,width)].r ; 
//new[TWO_D_TO_ONE_D(j,k,width)].g = p[i][TWO_D_TO_ONE_D(j,k,width)].g ; 
//new[TWO_D_TO_ONE_D(j,k,width)].b = p[i][TWO_D_TO_ONE_D(j,k,width)].b ; 
//}
//}

///* Apply blur on the bottom part of the image (10%) */
//for(j=height*0.9+size; j<height-size; j++)
//{
//for(k=size; k<width-size; k++)
//{
//int stencil_j, stencil_k ;
//int t_r = 0 ;
//int t_g = 0 ;
//int t_b = 0 ;

//for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
//{
//for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
//{
//t_r += p[i][TWO_D_TO_ONE_D(j+stencil_j,k+stencil_k,width)].r ;
//t_g += p[i][TWO_D_TO_ONE_D(j+stencil_j,k+stencil_k,width)].g ;
//t_b += p[i][TWO_D_TO_ONE_D(j+stencil_j,k+stencil_k,width)].b ;
//}
//}

//new[TWO_D_TO_ONE_D(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
//new[TWO_D_TO_ONE_D(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
//new[TWO_D_TO_ONE_D(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
//}
//}

//for(j=1; j<height-1; j++)
//{
//for(k=1; k<width-1; k++)
//{

//float diff_r ;
//float diff_g ;
//float diff_b ;

//diff_r = (new[TWO_D_TO_ONE_D(j  ,k  ,width)].r - p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].r) ;
//diff_g = (new[TWO_D_TO_ONE_D(j  ,k  ,width)].g - p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].g) ;
//diff_b = (new[TWO_D_TO_ONE_D(j  ,k  ,width)].b - p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].b) ;

//if ( diff_r > threshold || -diff_r > threshold 
//||
//diff_g > threshold || -diff_g > threshold
//||
//diff_b > threshold || -diff_b > threshold
//) {
//end = 0 ;
//}

//p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].r = new[TWO_D_TO_ONE_D(j  ,k  ,width)].r ;
//p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].g = new[TWO_D_TO_ONE_D(j  ,k  ,width)].g ;
//p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].b = new[TWO_D_TO_ONE_D(j  ,k  ,width)].b ;
//}
//}

//}
//while ( threshold > 0 && !end ) ;

//#if SOBELF_DEBUG
//printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
//#endif

//free (new) ;
//}

//}

    void
apply_sobel_filter( animated_gif * image, int rank, int size)
{
    fflush(stdout);
    printf("Process %d:-> Sobel started! Image with %d images\n",rank,image->n_images);
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

        if (left_border_pixels == NULL)
        {
            fprintf(stderr, "Unable to allocate %d elements", width);
            exit(1);
        }
        right_border_pixels = (pixel *) malloc(width * sizeof(pixel)); /* This stores the first line from left neighbor */

        if (right_border_pixels == NULL)
        {
            fprintf(stderr, "Unable to allocate %d elements", width);
            exit(1);
        }



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



            for(j= 1; j<height - 1; j++)
            {
                for(k=1; k<width-1; k++)
                {

                    int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                    int pixel_blue_o , pixel_blue  , pixel_blue_e ;
                    int pixel_blue_so, pixel_blue_s, pixel_blue_se;

                    float deltaX_blue ;
                    float deltaY_blue ;
                    float val_blue;

                    pixel_blue_no = left_neighbor != -1 ? left_border_pixels[TWO_D_TO_ONE_D(j-1, k-1, width) % width].b : p[i][TWO_D_TO_ONE_D(j-1,k-1,width)].b ;
                    pixel_blue_n  = left_neighbor != -1 ? left_border_pixels[TWO_D_TO_ONE_D(j-1, k, width) % width].b : p[i][TWO_D_TO_ONE_D(j-1,k  ,width)].b ;
                    pixel_blue_ne = left_neighbor != -1 ? left_border_pixels[TWO_D_TO_ONE_D(j-1, k+1, width) % width].b : p[i][TWO_D_TO_ONE_D(j-1,k+1,width)].b ;
                    pixel_blue_so = right_neighbor != -1 ? right_border_pixels[TWO_D_TO_ONE_D(j+1, k-1, width) % width].b : p[i][TWO_D_TO_ONE_D(j+1,k-1,width)].b ;
                    pixel_blue_s  = right_neighbor != -1 ? right_border_pixels[TWO_D_TO_ONE_D(j+1, k, width) % width].b : p[i][TWO_D_TO_ONE_D(j+1,k  ,width)].b ;
                    pixel_blue_se = right_neighbor != -1 ? right_border_pixels[TWO_D_TO_ONE_D(j+1, k+1, width) % width].b : p[i][TWO_D_TO_ONE_D(j+1,k+1,width)].b ;
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

    fflush(stdout);
    printf("Process %d-> Sobel end! Image with %d images\n",rank, image->n_images);

}

/*
 * Main entry point
 */
int main( int argc, char ** argv )
{

    int rank, size;

    /* MPI Initialization */
    MPI_Init(&argc, &argv);

    /* Get the rank of the current task and the number of MPI processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char * input_filename ; 
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    double duration ;

    /* Check command-line arguments */
    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }

    input_filename = argv[1] ;
    output_filename = argv[2] ;

    /*================ IMPORT ======================*/

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename , rank, size ) ;
    if ( image == NULL ) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration ) ;

    //MPI_IBarrier(MPI_Comm comm, MPI_Request * req);

    /*==============================================*/
    /*================= FILTER =====================*/

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale */
    //apply_gray_filter( image, rank, size) ;

    /* Apply blur filter with convergence value */
    //apply_blur_filter( image, 5, 20, rank, size) ;

    /* Apply sobel filter on pixels */
    apply_sobel_filter( image, rank, size ) ;

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "SOBEL done in %lf s\n", duration ) ;

    /*==============================================*/

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;

    MPI_Finalize();
    return 0 ;
}
