#!/bin/bash


PROFILER_FILEPATH=$(dirname $(realpath $HOME/INF560/X-INF560-Project/profiling/scripts/run_evaluation.sh))
source $PROFILER_FILEPATH/run_evaluation.sh
export PATH=$PROFILER_FILEPATH:$PATH

