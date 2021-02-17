#!/bin/bash

# Exit if not script file is provided
if [[ $# == 0 ]];
then
    echo "Please provide the script you would want to run"
    echo "run script with -h or --help for details on how to run"
    exit
elif [[ $# == 1 ]];
then
    case $1 in
        ("--help") printf "arg[1]-> Your script\narg[2]-> Number of process (leave out if its a sequential program)\narg[3]-> Search keyword (eg. SOBEL)\narg[4]-> Type of program (user-specific)\narg[5]-> Gnuplot file\narg[6]-> Title of graph\n";;
    ("-h") printf "arg[1]-> Your script\narg[2]-> Number of process (leave out if its a sequential program)\narg[3]->Search keyword (eg. SOBEL)\narg[4]->Type of program (user-specific)\narg[5]-> Gnuplot file\narg[6]-> Title of graph\n";;
esac
exit
fi

# Variables
CMD=$1
CMD_DIR=$(dirname $(realpath $CMD))
CMD_BASENAME=./$(basename $CMD)
PLOTTER=$(realpath $4)
TITLE_TEXT=$5
EVAL="eval"

if [[ $# > 5 ]];
then
    CMD_ARG=$2
    KEYWORD=$3
    TYPE=$4
    RESULTS=RESULTS_"$TYPE"_"$CMD_ARG".dat
    PLOTTER=$(realpath $5)
    TITLE_TEXT=$6
else
    KEYWORD=$2
    TYPE=$3
    RESULTS=RESULTS_"$TYPE".dat
fi

# Generate output file for graph
OUTPUT=$(basename $RESULTS | cut --delimiter='.' -f 1).eps

# Change to CMD_DIR
cd $CMD_DIR

# Remove data file
if [ -d "$CMD_DIR\/$EVAL" ];
then
    rm -rf $CMD_DIR/$EVAL
else
    mkdir -p $CMD_DIR/$EVAL
    touch $CMD_DIR/$EVAL/$RESULTS
    touch $CMD_DIR/$EVAL/$OUTPUT
fi

# Add eval folder to variables
RESULTS=$CMD_DIR/$EVAL/$RESULTS
OUTPUT=$CMD_DIR/$EVAL/$OUTPUT

echo "-->"$RESULTS
if [ -f "$RESULTS" ];
then
    echo -e "#IMAGE\tN_IMAGES\tSOBEL_TIME\tIMAGE_FILE" >> $RESULTS 
fi


# Check command string
echo "COMMAND:" $CMD_BASENAME
echo "COMMAND ARG:" $CMD_ARG
echo "KEYWORD:" $KEYWORD
echo "TYPE:" $TYPE
echo "PLOTTER:" $PLOTTER
echo "RESULTS:" $RESULTS
echo "OUTPUT:" $OUTPUT
echo "TITLE:" $TITLE_TEXT

# Run command
if [ -f "$RESULTS" ];
then
    $CMD_BASENAME $CMD_ARG | grep "$KEYWORD" | awk '{printf("%s\t%s\t%s\t", $2, $11, $6); system("echo "$9"| cut --delimiter='/' -f 3")}' >> $RESULTS
fi


# Plot graph

if [ -f "$RESULTS" ];
then
    gnuplot -e "input_file='${RESULTS}'; output_file='${OUTPUT}'; title_text='${TITLE_TEXT}'" $PLOTTER 
fi

# Change to previous directory
cd "$OLDPWD"


