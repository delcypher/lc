#!/bin/bash 
#script to show the output of the nDump() in Lattice class.

function usage()
{
	echo "Usage: $0 [--index INDEX] [--title TITLE] [--png FILE] [--xlabel \"x label\"] [--ylabel \"y label\"] LATTICE_FILE

	This program plots the state of a lattice using GNUplot where LATTICE_FILE should be a text file containing
	the output of nDump().

	--help
	Show this message.

	--index INDEX
	Pick index INDEX from file.

	--title TITLE
	set graph title

	--png FILE
	Output graph to png instead of displaying in a terminal

	--xlabel "x label"
	Set the x label

	--ylabel "y label"

";
	exit;
}

#declare variables
declare -x TITLE_CMD;
declare -x TITLE;
declare -x PNG_CMD;
declare -x PNG_OUTPUT;
declare -x X_LABEL_CMD;
declare -x X_LABEL;
declare -x Y_LABEL_CMD;
declare -x Y_LABEL;
declare -x LATTICE_FILE;
declare -x INDEX_CMD;

#start of program

if [ $# -eq 0 ]; then
	usage;
	exit 1;
fi

#loop through command line arguments
for((i=1; i<=$#; i++)); do
	
	if [ "${!i}" == "--help" ]; then
		usage;
	fi

	if [ "${!i}" == "--index" ]; then
		i=$((i+1));
		echo "Pick index ${!i}"
		INDEX_CMD="index ${!i}"
		continue;
	fi

	if [ "${!i}" == "--title" ]; then
		i=$((i +1));
		TITLE="${!i}"

		if [ -z "${TITLE}" ]; then
			echo "Title cannot be blank";
			exit 1;
		fi

		echo "Setting title to ${TITLE}"
		TITLE_CMD="set title \"${TITLE}\""
		continue;
	fi

	if [ "${!i}" == "--png" ]; then
		i=$((i + 1));
		PNG_OUTPUT="${!i}"
		if [ "$(echo "${PNG_OUTPUT}" | grep -Ec '.png$')" -eq 0 ]; then
			echo "Cannot output to file ${PNG_OUTPUT}";
			exit 1;
		fi
		
		echo "Setting output to png instead of default terminal"
		PNG_CMD="set terminal png ; set output \"${PNG_OUTPUT}\"";
		continue;
	fi
		
	if [ "${!i}" == "--xlabel" ]; then
		i=$((i + 1));
		X_LABEL="${!i}"
		if [ -z "${X_LABEL}" ]; then
			echo "x label cannot be blank";
			exit 1;
		fi
		
		echo "Setting x label to ${X_LABEL}";
		X_LABEL_CMD="set xlabel \"${X_LABEL}\"";
		continue;
	fi

	if [ "${!i}" == "--ylabel" ]; then
		i=$((i + 1));
		Y_LABEL="${!i}"
		if [ -z "${Y_LABEL}" ]; then
			echo "y label cannot be blank";
			exit 1;
		fi
		
		echo "Setting y label to ${Y_LABEL}";
		Y_LABEL_CMD="set ylabel \"${Y_LABEL}\"";
		continue;
	fi

	#we should of handled all arguments so we should be looking for LATTICE_FILE

	LATTICE_FILE="${!i}"

	if [ ! -r "$LATTICE_FILE" ]; then
		echo "Cannot open LATTICE_FILE ${LATTICE_FILE}"
		exit 1;
	fi
done;


file="$1";
PROG="gnuplot -persist"



#show information about the lattice (assume lines prepended w/ # have info)
cat "$LATTICE_FILE" | grep '^#'

#do plot
$PROG - <<COOL
$X_LABEL_CMD
$Y_LABEL_CMD
$TITLE_CMD
$PNG_CMD

#make sure the key isn't displayed
set key off

#set the increment of x & y major axis tics
set xtics 1 ; set ytics 1

#set the number of intervals between the major axis tics to make the minor axis tics
set mxtics 2 ; set mytics 2

#setup grid
set grid mxtics mytics noxtics noytics

#Make sure the borders can't draw over the vectors
set border back

plot "${LATTICE_FILE}" ${INDEX_CMD} with vectors
COOL
 
