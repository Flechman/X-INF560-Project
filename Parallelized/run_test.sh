#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

ARGC=$#

for i in $INPUT_DIR/*gif ; do
    echo "==========================="
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    if [[ $ARGC -eq 1 ]]
    then
        salloc -n $1 mpirun ./sobelf $i $DEST
    elif [[ $ARGC -eq 2 ]]
    then
        salloc -n $1 -N $2  mpirun ./sobelf $i $DEST
    else
        echo "You have to provide one or more arguments for the number of processes and/or number of nodes"
    fi
done
