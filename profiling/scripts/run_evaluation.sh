#!/bin/bash

# functions 
help_display() {
    printf "arg[1]-> Your script\n"
    printf "arg[2]-> Number of process (leave out if its a sequential program)\n"
    printf "arg[3]-> Search keyword (eg. SOBEL)\n" 
    printf "arg[4]-> Type of program (user-specific)\n"
    printf "arg[5]-> Title of graph\n"
}

# Exit if not script file is provided
if [[ $# == 0 ]];
then
    echo "Oops! No script"
    echo "Try './$(basename $0) -h' or './$(basename $0) --help' for more details"
    exit
elif [[ $# == 1 ]];
then
    case $1 in
        ("--help") 
            help_display;;
        ("-h") 
            help_display;;
    esac
    exit
fi

# Variables
CMD=$1
CMD_DIR=$(dirname $(realpath $CMD))
CMD_BASENAME=./$(basename $CMD)
PLOTTER=$(realpath ./evaluation.plg)
TITLE_TEXT=$4
EVAL="eval"
KEYWORD=$2
TYPE=$3
RESULTS=RESULTS_"$TYPE".dat

if [[ $# > 4 ]];
then
    CMD_ARG=$2
    KEYWORD=$3
    TYPE=$4
    RESULTS=RESULTS_"$TYPE"_"$CMD_ARG".dat
    TITLE_TEXT=$5
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

if [ -f "$RESULTS" ];
then
    echo -e "#IMAGE\tN_IMAGES\tSOBEL_TIME\tIMAGE_FILE" >> $RESULTS 
fi


# Display script state
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

