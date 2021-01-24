#ifndef __EXTRA_FUNCTIONS__
#define __EXTRA_FUNCTIONS__

typedef struct
{
    int start;
    int finish;
} image_limits;

int get_start_index(int rank, int size, int n);

int get_finish_index(int start_index,  int parts);

#endif
#ifndef __IMAGE_LIMITS__
#define __IMAGE_LIMITS__

image_limits* set_rank_limits( int processes, int images, int rank);

#endif 
