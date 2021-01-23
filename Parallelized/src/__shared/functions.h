#ifndef __EXTRA_FUNCTIONS__
#define __EXTRA_FUNCTIONS__

int get_start_index(int rank, int size, int n){
    int start_index;

    if(rank == 0) {
        start_index = 0;
    }
    else {
        start_index = (((float)rank /size) * n  ) + 1;

    }

    return start_index;
}

int get_finish_index(int start_index,  int parts){
    int finish_index;

    finish_index = (start_index + parts) - 1;


    return finish_index; 
}

#endif
#ifndef __IMAGE_LIMITS__
#define __IMAGE_LIMITS__
typedef struct {
    int start;
    int finish;
}image_limits;

image_limits* set_rank_limits( int processes, int images, int rank) {

    if ( images < processes && images != 0 ){
        processes = images;
    }

    int parts = images / processes;
    int size = images;
    int remainder = images % processes;

    image_limits* limits = calloc(processes, sizeof(image_limits));

    if ( images >= processes){
        int rank_start = rank * parts;
        int rank_finish = (rank_start + parts);

        (limits + rank)->start = rank_start;
        (limits +rank)->finish = rank_finish;

        (limits + (processes - 1))->finish += remainder;
        return limits;
    }
    else {
        return NULL;
    }
}

#endif 
