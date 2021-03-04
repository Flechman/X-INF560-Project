#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed

# Counter for helping profiling
COUNTER=1

mkdir $OUTPUT_DIR 2>/dev/null

for i in $INPUT_DIR/*gif ; do
	echo "==========================="
	DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
	echo "Running test on $i -> $DEST"

	./sobelf $i $DEST $COUNTER
	COUNTER=$((COUNTER+1))
done
