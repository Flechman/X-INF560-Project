#!/bin/bash


profiler() 
{
	local CMD
	local CMD_DIR
	local CMD_ARG
	local TITLE_TEXT
	local GRAPH_TYPE
	local PLOTTER_FILE
	local PLOTTER_DIR
	local EVAL


	if [[ $# == 0 ]];
	then
		echo "Oops! No script"
		echo "Try './$(basename $0) -h' or './$(basename $0) --help' for more details"
		return $(())
	elif [[ $# == 1 ]];
	then
		case $1 in
			("--help") 
				printf "\t[-s|--script] -> Your script\n"
				printf "\t[-a|--arg] -> Arguments to your script. Replace spaces with '_' and hyphen with '.'\n"
				printf "\t[-t|--title] -> Title of graph\n"
				printf "\t[-g|--graph_type] -> Graph type\n"
				return $(())
				;;
			("-h") 
				printf "\t[-s|--script] -> Your script\n"
				printf "\t[-a|--arg] -> Arguments to your script. Replace spaces with '_' and hyphen with '.'\n"
				printf "\t[-t|--title] -> Title of graph\n"
				printf "\t[-g|--graph_type] -> Graph type\n"
				return $(())
				;;
		esac
	fi

	local array=($@)
	for (( index=0; index<${#array[@]}; index++ ));do

		local next_item=$((index+1))
		local item=${array[index]}
		case $item in
			(*"-s"*)
				CMD="${array[next_item]}";;
			(*"--script"*)
				CMD="${array[next_item]}";;
			(*"-a"*)
				next_arg=${array[next_item]}
				CMD_ARG=${next_arg//./-}
				CMD_ARG=${CMD_ARG//_/ }	
				;;
			(*"--arg"*)
				next_arg=${array[next_item]}
				CMD_ARG=${next_arg//./-}
				CMD_ARG=${CMD_ARG//_/ }	
				;;
			(*"-t"*)
				TITLE_TEXT="${array[next_item]}";;
			(*"--title"*)
				TITLE_TEXT="${array[next_item]}";;
			(*"-g"*)
				GRAPH_TYPE="${array[next_item]}";;
			(*"--graph_type"*)
				GRAPH_TYPE="${array[next_item]}";;

		esac
	done

	case $GRAPH_TYPE in
		"regular")
			PLOTTER_FILE=evaluation.plg;;
		"reg")
			PLOTTER_FILE=evaluation.plg;;
		"histogram")
			PLOTTER_FILE=evaluation-histogram.plg;;
		*)
			GRAPH_TYPE="regular"
			PLOTTER_FILE=evaluation.plg;;
	esac

	# Variables
	CMD_DIR=$(dirname $(realpath $CMD))
	CMD_BASENAME=./$(basename $CMD)
	PLOTTER_DIR=$HOME/X-INF560-Project/profiling/scripts
	PLOTTER=$(realpath $PLOTTER_DIR)/$PLOTTER_FILE
	EVAL="eval"
	local RESULTS="RESULTS_"${CMD_ARG// /,}"_"$TITLE_TEXT"_"$GRAPH_TYPE".dat"

	# Generate output file for graph
	local OUTPUT=$(basename $RESULTS | cut --delimiter='.' -f 1).eps

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

	#if [ -f "$RESULTS" ];
	#then
	#echo -e "#IMAGE\t#N_IMAGES\t#SOBEL_TIME\t#LOAD_TIME\t#EXPORT_TIME\t#IMAGE_FILE" >> $RESULTS 
	#fi


	# Display script state
	echo "COMMAND:" $CMD_BASENAME
	echo "COMMAND ARG:" $CMD_ARG
	echo "PLOTTER:" $PLOTTER
	echo "RESULTS:" $RESULTS
	echo "OUTPUT:" $OUTPUT
	echo "TITLE:" $TITLE_TEXT
	echo "GRAPH TYPE:" $GRAPH_TYPE

	# Run command
	if [ -f "$RESULTS" ];
	then
		$CMD_BASENAME $CMD_ARG | grep "OUTPUT" | awk '{printf("%s\t%s\t%s\t%s\t%s\t", $3, $12, $7, $17, $21); system("echo "$10"| cut --delimiter='/' -f 3")}' >> $RESULTS

		# Sort file on filter time
		sort --key=3 --numeric-sort "${RESULTS}" --output="${RESULTS}"
	fi


	# Plot graph
	if [ -f "$RESULTS" ];
	then
		gnuplot -e "input_file='${RESULTS}'; output_file='${OUTPUT}'; title_text='${TITLE_TEXT}'" $PLOTTER 
	fi


	# Change to previous directory
	cd "$OLDPWD"
}



multiprofiler()
{
	if [[ $# == 0 ]];
	then
		echo "Oops! No script"
		echo "Try './$(basename $0) -h' or './$(basename $0) --help' for more details"
		return $(())
	elif [[ $# == 1 ]];
	then
		case $1 in
			("--help") 
				printf "\t[-s] -> Your script\n"
				printf "\t[-p] -> Number of processes. Three (3) arguments separated by ','. First is the start of the range, second is the increment, and last is the end of the range\n"
				printf "\t[-n] -> Number of threads. Three (3) arguments separated by ','. First is the start of the range, second is the increment, and last is the end of the range\n"
				return $(())
				;;
			("-h") 
				printf "\t[-s] -> Your script\n"
				printf "\t[-p] -> Number of processes. Three (3) arguments separated by ','. First is the start of the range, second is the increment, and last is the end of the range\n"
				printf "\t[-n] -> Number of threads. Three (3) arguments separated by ','. First is the start of the range, second is the increment, and last is the end of the range\n"
				return $(())
				;;
		esac
	fi

	local array=($@)

	for ((index=0; index < $#-1; index++));
	do

		arg=${array[index]}
		next=$((index+1))
		case $arg in
			*"-s"*)
				local CMD="${array[next]}"
				;;
			*"-p"*)
				local PROCESSES="${array[next]//,/ }"
				;;
			*"-n"*)
				local THREADS="${array[next]//,/ }"
				;;
			*"-t"*)
				local TITLE_TEXT="${array[next]}";;
			*"-g"*)
				local TYPE="${array[next]}";;
			*"-N"*)
				local NODES="${array[next]}";;
		esac
	done

	local CMD_DIR=$(dirname $(realpath $CMD))
	local CMD_BASENAME=./$(basename $CMD)
	local EVAL="eval"
	local MULTI="multi"
	local RESULTS_FILES_LIST=()
	local TITLES=()
	local PLOTTER_DIR=$HOME/X-INF560-Project/profiling/scripts
	local PLOTTER_FILE=evaluation-multi.plg
	local PLOTTER=$(realpath $PLOTTER_DIR)/$PLOTTER_FILE

	echo "COMMAND:" $CMD_BASENAME
	echo "PROCESSES:" $PROCESSES
	echo "THREADS:" $THREADS


	local RESULTS_FOLDER=$CMD_DIR/$EVAL/$MULTI
	local OUTPUT_FILE="$RESULTS_FOLDER/RESULTS_MULTI_${PROCESSES// /,}_procs_${THREADS// /,}_threads.eps"
	local THREADS_PROCESSES_OUTPUT_FILE="$(dirname "${OUTPUT_FILE}")/THREADS_PROCESSES_OUTPUT_FILE.eps"
	local THREADS_PROCESSES_INPUT_FILE="$(dirname $THREADS_PROCESSES_OUTPUT_FILE)/THREADS_PROCESSES_INPUT_FILE.dat"

	# Change to CMD_DIR
	cd $CMD_DIR

	if [ -d "$RESULTS_FOLDER" ];
	then
		rm -rf $RESULTS_FOLDER
	else
		mkdir -p $RESULTS_FOLDER
		touch $OUTPUT_FILE
		touch $THREADS_PROCESSES_OUTPUT_FILE
	fi

	# Loop over processes and threads to run them and build result files
	for i in `seq ${PROCESSES}`;
	do
		for j in `seq ${THREADS}`;
		do 

			local RESULTS_FILE="$RESULTS_FOLDER/RESULTS_MULTI_${i}procs_${j}threads.dat"

			# Remove data file

			if [ -d "$RESULTS_FOLDER" ];
			then
				touch $RESULTS_FILE
			fi


			#if [ -f "$RESULTS_FILE" ];
			#then
			#echo -e  "#IMAGE\t#N_IMAGES\t#SOBEL_TIME\t#LOAD_TIME\t#EXPORT_TIME\t#IMAGE_FILE" >> $RESULTS_FILE
			#fi


			if [ -f "$RESULTS_FILE" ];
			then
				local CMD_ARG="-n ${i} -o ${j} -t ${TYPE} -N ${NODES}"
				echo "ARGS: ${CMD_ARG}"
				$CMD_BASENAME $CMD_ARG | grep "OUTPUT" | awk '{printf("%s\t%s\t%s\t%s\t%s\t", $3, $12, $7, $17, $21); system("echo "$10"| cut --delimiter='/' -f 3")}' >> $RESULTS_FILE

				# Sort file on filter time
				sort --key=3,3nr "${RESULTS_FILE}" --output="${RESULTS_FILE}"
			fi

			mapfile -t -c 1 lines < $RESULTS_FILE
			highest=$(echo ${lines[0]}| awk '{print $3}')
			echo -e "p${i}t${j}\t${highest}" >> $THREADS_PROCESSES_INPUT_FILE
			RESULTS_FILES_LIST+=($RESULTS_FILE)
			TITLES+=("${i} Processes ${j} Threads")
		done
	done


	# Plot the graph from the data
	if [ -f "$OUTPUT_FILE" ];
	then
		gnuplot -e "files='${RESULTS_FILES_LIST[*]}'; output_file='${OUTPUT_FILE}'; title_text='${TITLE_TEXT}'; titles='${TITLES[*]}'" $PLOTTER 
	fi

	if [ -f "$THREADS_PROCESSES_OUTPUT_FILE" ];
	then
		PLOTTER="${PLOTTER_DIR}/evaluation-procs-threads.plg"
		gnuplot -e "input_file='${THREADS_PROCESSES_INPUT_FILE}'; output_file='${THREADS_PROCESSES_OUTPUT_FILE}'; title_text='${TITLE_TEXT}'" $PLOTTER 
	fi

	cd "$OLDPWD"

}
