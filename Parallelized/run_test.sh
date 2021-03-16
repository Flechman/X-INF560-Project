#!/bin/bash


source ~/.bashrc
source ~/set_env.sh

if [[ $# == 1 ]];
then
	TYPE=mpi
elif [[ $# == 2 ]];
then
	TYPE=mpi_omp
else
	TYPE=mpi_omp
fi



INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

ARGC=$#

ARGS=($@)

# Defaults
PROCS=1
NODES=1
THREADS=1
TYPE=mpi

for ((index=0; index < $#-1; index++));
do

	arg=${ARGS[index]}
	next=$((index+1))
	case $arg in
		*"-n"*)
			PROCS="${ARGS[next]}"
			;;
		*"-N"*)
			NODES="${ARGS[next]}"
			;;
		*"-o"*)
			THREADS="${ARGS[next]}"
			;;
		*"-t"*)
			TYPE="${ARGS[next]}";;
	esac
done


echo "-n: ${PROCS}"
echo "-N: ${NODES}"
echo "-o: ${THREADS}"
echo "-t: ${TYPE}"

make clean
make TYPE=$TYPE

# Send a counter to the program; this helps profiling
COUNTER=1
for i in $INPUT_DIR/*gif ; do
	echo "==========================="
	DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
	echo "Running test on $i -> $DEST"

	#if [[ $ARGC -eq 1 ]];
	#then
	#salloc -n $1 mpirun ./sobelf $i $DEST $COUNTER
	##elif [[ $ARGC -eq 2 ]];
	##then
	##salloc -n $1 -N $2  mpirun ./sobelf $i $DEST $COUNTER 
	#elif [[ $ARGC -eq 2 ]];
	#then
	#OMP_NUM_THREADS=$THREADS
	#MPI_EXPORT=$OMP_NUM_THREADS
	#salloc -n $1 mpirun ./sobelf $i $DEST $COUNTER $OMP_NUM_THREADS
	#elif [[ $ARGC -eq 3 ]];
	#then
	#OMP_NUM_THREADS=$2
	#MPI_EXPORT=$OMP_NUM_THREADS
	#salloc -n $1 -N 3 mpirun ./sobelf $i $DEST $COUNTER $OMP_NUM_THREADS
	#else
	#echo "You have to provide one or more arguments for the number of processes and/or number of nodes"
	#fi
	OMP_NUM_THREADS=$THREADS
	MPI_EXPORT=$OMP_NUM_THREADS
	salloc -n $PROCS -N $NODES mpirun ./sobelf $i $DEST $COUNTER

	COUNTER=$((COUNTER+1))
done
