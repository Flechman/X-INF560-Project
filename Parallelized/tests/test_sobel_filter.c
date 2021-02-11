#include <stdio.h>
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
    int *heightStart;  /* Index of start of each image (for height) */
    int *heightEnd;    /* Index of end of each image (for height) */
    int *actualWidth;  /* Actual width of each image (INITIAL width before parallelism) */
    int *actualHeight; /* Actual height of each image (INITIAL width before parallelism) */
    pixel **p;         /* Pixels of each image */
    GifFileType *g;    /* Internal representation.
                          DO NOT MODIFY */
} animated_gif;

    void
apply_sobel_filter( animated_gif * image )
{
    int i, j, k ;
    int width, height ;

    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;

        pixel * sobel ;

        sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_o , pixel_blue  , pixel_blue_e ;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;

                float deltaX_blue ;
                float deltaY_blue ;
                float val_blue;

                pixel_blue_no = p[i][TWO_D_TO_ONE_D(j-1,k-1,width)].b ;
                pixel_blue_n  = p[i][TWO_D_TO_ONE_D(j-1,k  ,width)].b ;
                pixel_blue_ne = p[i][TWO_D_TO_ONE_D(j-1,k+1,width)].b ;
                pixel_blue_so = p[i][TWO_D_TO_ONE_D(j+1,k-1,width)].b ;
                pixel_blue_s  = p[i][TWO_D_TO_ONE_D(j+1,k  ,width)].b ;
                pixel_blue_se = p[i][TWO_D_TO_ONE_D(j+1,k+1,width)].b ;
                pixel_blue_o  = p[i][TWO_D_TO_ONE_D(j  ,k-1,width)].b ;
                pixel_blue    = p[i][TWO_D_TO_ONE_D(j  ,k  ,width)].b ;
                pixel_blue_e  = p[i][TWO_D_TO_ONE_D(j  ,k+1,width)].b ;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;             

                deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


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

        free (sobel) ;
    }

}

int main(int argc, char** argv){

}
