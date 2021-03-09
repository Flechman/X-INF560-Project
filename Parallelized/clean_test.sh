#!/bin/bash

echo "Cleaning directory images/processes/..."
echo "Cleaning directory eval"

EVAL="eval"

if [ -d "$EVAL" ];
then
	rm -rf $EVAL
fi
rm -f images/processed/*.gif

