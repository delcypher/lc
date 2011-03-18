#!/bin/bash
#Script to build a set binary state files to simulate on a single machine and then do those simulations
# one after another.

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <target_directory> <log_file>"
	echo "<target_directory> - Directory to build simulation directories in"
	echo "<log_file> - A filename to log the jobs started by this script and their associated directories"
	exit 0;
fi

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

#Check bc is installed so we can do floating point math in this shell script
which bc > /dev/null 2>&1
if [ "$?" -ne 0 ]; then
	redmessage "bc is either not in PATH variable or is not installed!"
	exit 1
fi

NUM_REGEX="^[0-9]+"


#Interactive continue function asking to confirm file overwrite
function ask()
{
	FILE="$1"
	redmessage "Warning this will overwrite file ${FILE}\n";
	read -p "continue(y/n)?" -n 1 responce
	#make new line
	echo ""
	
	if [ -z "$responce" ]; then
	#the user pressed enter, make them suffer for it!
		redmessage "Invalid answer. Try again!\n"
		rcode=$( ask "$FILE")
		
		case "$rcode" in
		
		"0")
			responce="y"
		;;

		"1")
			responce="n"
		;;		
		esac
	fi
	
	case "$responce" in
	
	"y")
		return 0
	;;

	"n")
		return 1
	;;
	
	*)
		#bad data entered call again recursively (this could get bad if user keeps being stupid!)
		redmessage "Invalid answer, try again!\n"
		return $(ask "$FILE" > /dev/null  ; echo $?)
	esac

	#user said yes
	return 0
}


#get target directory
TARGET_DIR="$1"
LOG_FILE="$2"

if [ -z "$TARGET_DIR" ]; then
	redmessage "<target_directory> must be specified!\n";
	exit 1;
fi

if [ ! -d "$TARGET_DIR" ]; then
	redmessage "$TARGET_DIR is not a directory!\n";
	exit 1;
fi

if [ ! -w "$TARGET_DIR" ]; then
	redmessage "$TARGET_DIR is not writable!\n";
	exit 1;
fi

if [ -z "$LOG_FILE" ]; then
	redmessage "<log_file> must be specified!\n";
	exit 1;
fi

#Get the directory log file is in
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
	redmessage "$LOG_FILE is not in a directory!\n"
	exit 1;
fi

if [ ! -w "$LOG_DIR" ]; then
	redmessage "$LOG_DIR is not writable!\n"
	exit 1;
fi

#make logfile path absolute
LOG_FILE="$(basename "$LOG_FILE")"
LOG_FILE="$(cd "$LOG_DIR"; echo "$PWD" )/${LOG_FILE}"

#check if log file already exists
if [ -a "$LOG_FILE" ]; then
	ask "$LOG_FILE" || exit 1;
fi

#try to make log file
touch "$LOG_FILE"
if [ "$?" -ne 0 ]; then
	redmessage "Error: Couldn't write logfile to ${LOG_FILE}\n"
	exit 1;
fi
cyanmessage "LOG FILE: ${LOG_FILE}\n"

#Write logfile header
echo $(date) >> ${LOG_FILE}
echo "#[JOB] [BUILD_PATH]" >> ${LOG_FILE}

#get scripts path
SCRIPTS_PATH=$( cd ${BASH_SOURCE[0]%/*} ; echo "$PWD" )

#add *-state tools to workpath
source ${SCRIPTS_PATH}/path.sh

#get absolute path to target directory
TARGET_DIR=$(cd "$TARGET_DIR"; echo "$PWD" )
cyanmessage "Target directory: ${TARGET_DIR}\n"


#note n = (1 + m*0.5) where n is scale factor
LOOP_MAX=10
for ((m=0; m<=LOOP_MAX ;m++))
do
	#Scale the width & height by m
	width=$( echo "scale=1; 30*(1 + ${m}*0.5) " | bc | grep -Eo "$NUM_REGEX" )
	height=$( echo "scale=1; 30*(1 + ${m}*0.5) " | bc | grep -Eo "$NUM_REGEX" )
	beta=1
	
	if [ -z "${width}" ]; then
		exit
	fi

	#lattice boundaries
	top=0
	bottom=0
	left=0
	right=0

	latticeInitialState=0

	#Number of Monte Carlo steps to run through
	mcs=100000

	#nanoparticle configuartion (force a:b ratio 3:1)
	x=$( echo "scale=1; 14*(1 + ${m}*0.5) " | bc | grep -Eo "$NUM_REGEX" )
	y=$( echo "scale=1; 14*(1 + ${m}*0.5) " | bc | grep -Eo "$NUM_REGEX" )
	a=$( echo "scale=1; 12*(1 + ${m}*0.5) " | bc | grep -Eo "$NUM_REGEX" )
	b=$((a/3))
	theta=0
	particleBoundary=0

	#set build directory (make sure slash is appended!)
	BUILD_DIR="${TARGET_DIR}/${width}-${height}/"

	#try to make directory
	mkdir -p "$BUILD_DIR"
	if [ "$?" -ne 0 ]; then
		redmessage "Building directory $BUILD_DIR failed!\n";
		exit 1;
	fi

	#make binary statefile
	STATE_FILENAME="state.bin"

	#check if statefile already exists
	if [ -a "${BUILD_DIR}${STATE_FILENAME}" ]; then
		ask "${BUILD_DIR}${STATE_FILENAME}" || exit 1;
	fi

	ARGS="${BUILD_DIR}${STATE_FILENAME} ${width} ${height} ${beta} ${top} ${bottom} ${left} ${right} ${latticeInitialState} ${x} ${y} ${a} ${b} ${theta} ${particleBoundary}"
	echo "create-state $ARGS"
	create-state $ARGS > /dev/null
	if [ "$?" -ne 0 ]; then
		redmessage "Building state file $STATE_FILENAME failed!\n";
		exit 1;
	fi


	greenmessage "Running job ${m} of ${LOOP_MAX}"
	cd "${BUILD_DIR}"
	#Start simulation 
	sim-state "${STATE_FILENAME}" ${mcs} 

	if [ "$?" -ne 0 ]; then
		redmessage "sim-state failed!"
		exit
	fi
	
	echo "${m} ${BUILD_DIR}" >> ${LOG_FILE}
done
