#!/bin/bash
# Script to compile and run all test harnesses automatically. This is intended to be used for checking for
# regressions. If you're making a major change to the code you should run this script after you have done
# so to help you find bugs you have introduced!
#
# This relies on a specially configured makefile where the test harness targets are between
# two special comments in the make file (#TEST HARNESSES START & #TESTHARNESSES END).
# Under the recipe for each rule one or more comment(s) of the form "#ARGS <argument1> <argument2> ..."
# are placed under the recipe for each rule. These comments tell this script how to call each test harness
# once is has been compiled.

#path to makefile
MAKEFILE="makefile"
#Options to pass to make (e.g. -j4 , debug=1, profile=1, etc..)
MAKEFILE_OPTS="";


#check makefile exists
if [ ! -r "$MAKEFILE" ]; then
	echo "Can't find makefile : $MAKEFILE" 1>&2
	exit 1;
fi

declare -xi EXIT_CODE
declare -xi TH_SUCCESS=0
declare -xi TH_BAD_ARGUMENT=1
declare -xi TH_FAIL=2


function redmessage()
{

	echo -en "\033[31m${1}\033[0m";
}

function greenmessage()
{

	echo -en "\033[32m${1}\033[0m";
}

function cyanmessage()
{
	echo -en "\033[35m${1}\033[0m";
}


echo "Getting test harness list from makefile: $MAKEFILE";

#Get line numbers to search through for test harnesses
STARTLINE=$( grep -En --max-count=1 '^#TEST HARNESSES START' "$MAKEFILE" | grep -Eo '[0-9]+'  )
ENDLINE=$( grep -En --max-count=1 '^#TEST HARNESSES END' "$MAKEFILE" | grep -Eo '[0-9]+'  )

#GET list of test harnesses
TESTS=$( sed -n "${STARTLINE},${ENDLINE}p" "$MAKEFILE" | grep -Eo '^.+:' | sed -e 's/:$//'  )

#build array of test harness targets
declare -ix counter
for target in $TESTS; do
	TARGETS[counter]="$target"
	counter=$((counter +1));
done;

total="$counter"
cyanmessage "Found $total test harnesses.\n"


#loop over the different targets
for ((counter=0; counter < $total; counter++)) do
	cyanmessage "Building ${TARGETS[$counter]}..."
	make "${TARGETS[$counter]}" ${MAKEFILE_OPTS} 1> /dev/null

	if [ $? -ne 0 ]; then
		redmessage "Compilation failed. Trying next target.\n"
		//skip to next target
		continue
	else
		greenmessage "Done.\n"
	fi

	#get default arguments to try test harness with
	sline=$( grep -Eno "^${TARGETS[$counter]}:" "$MAKEFILE" | grep -Eo '^[0-9]+:' | sed 's/:$//' )
	
	if [ $counter -eq $(( total -1 )) ]; then
		eline=$ENDLINE
	else
		eline=$( grep -Eno "^${TARGETS[$counter +1]}:" "$MAKEFILE" | grep -Eo '^[0-9]+:' | sed 's/:$//' );
	fi

	#Get the line numbers that the relevent arguments are on
	args=$( sed -n "${sline},${eline}p" "$MAKEFILE" | grep -E '^#ARGS' | sed 's/^#ARGS //' )

	#loop over the arguments on each line of the makefile
	IFS=$'\n'
	for arguments in $args; do
		IFS=$'\n '
		cyanmessage "Running ./${TARGETS[$counter]} $arguments :"
		./${TARGETS[$counter]} $arguments 1> /dev/null

		EXIT_CODE=$?
		case $EXIT_CODE in
		
		$TH_SUCCESS)
			greenmessage "Test successful.\n"
		;;
		$TH_BAD_ARGUMENT)
			redmessage "Bad argument(s) passed.\n"
		;;
		$TH_FAIL)
			redmessage "Test failed.\n"
		;;
		*)
			redmessage "Exit code $EXIT_CODE returned not supported.\n"
		;;
		esac

		IFS=$'\n'
	done;
done;


