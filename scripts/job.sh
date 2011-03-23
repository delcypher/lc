#!/bin/bash
#Script to build a set binary state files to simulated on a local machine (local) or on a PBS/torque cluster (pbs)

#SET prefix for PBS/torque jobs
JOB_PREFIX="lc"
#SET value for PI (20dp from Wolfram Alpha)
PI="3.14159265358979323846"

if [ "$#" -ne 3 ]; then
	echo "Usage: $0 <mode> <target_directory> <log_file>"
	echo "<mode> - Job run mode. hould be local or pbs"
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

function calc-int()
{
	echo "scale=1; ${1}" | bc | grep -Eo "^[0-9]+"
}

function calc-float()
{
	echo "scale=20; ${1}" | bc
}


#Check bc is installed so we can do floating point math in this shell script
which bc > /dev/null 2>&1
if [ "$?" -ne 0 ]; then
	redmessage "bc is either not in PATH variable or is not installed!"
	exit 1
fi


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
MODE="$1"
TARGET_DIR="$2"
LOG_FILE="$3"

if [ -z "$MODE" ]; then
	redmessage "<mode> must be specified\n"
	exit 1;
fi

if [ ! \( "$MODE" = "local" -o $MODE = "pbs" \) ]; then
	redmessage "<mode> must be local or pbs\n"
	exit 1;
fi

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

#Check we're running on correct machine if in pbs mode
if [ "$(hostname)" != "calgary.phy.bris.ac.uk" -a "$MODE" = "pbs" ]; then
	redmessage "You should run this script on calgary!\n"
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

#truncate log file
echo -n "" > "${LOG_FILE}"

cyanmessage "LOG FILE: ${LOG_FILE}\n"
cyanmessage "Mode : ${MODE}\n"

#Write logfile header
echo $(date) >> "${LOG_FILE}"
echo "#[JOB_ID] [BUILD_PATH]" >> "${LOG_FILE}"

#get scripts path
SCRIPTS_PATH=$( cd ${BASH_SOURCE[0]%/*} ; echo "$PWD" )

#add *-state tools to workpath
source ${SCRIPTS_PATH}/path.sh

#get absolute path to target directory
TARGET_DIR=$(cd "$TARGET_DIR"; echo "$PWD" )
cyanmessage "Target directory: ${TARGET_DIR}\n"


#note n = (1 + m*0.5) where n is scale factor
#Add loop here
for ((m=0; m<=10 ;m++))
do
	#Scale the width & height by m
	width=$( calc-int " 30*(1 + ${m}*0.5) " )
	height=$( calc-int " 30*(1 + ${m}*0.5) " )
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
	mcs=700000

	#nanoparticle configuartion (force a:b ratio 3:1)
	x=$( calc-int " 15*(1 + ${m}*0.5) -1 " )
	y=$( calc-int " 15*(1 + ${m}*0.5) -1 " )
	a=$( calc-int " 9*(1 + ${m}*0.5) " )
	#enforce 3:1 ratio
	b=$((a/3))
	theta=$( calc-float "${PI}*0" )
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
	
	if [ "$MODE" = "pbs" ]; then
		#Build PBS/Torque script (we shouldn't indent the HEREDOC)
		cat > "${BUILD_DIR}run.sh" <<HEREDOC
#PBS -N ${JOB_PREFIX}-${width}-${height}
#PBS -l cput=40:00:00

cd "${BUILD_DIR}"
#Add tools to work path
source ${SCRIPTS_PATH}/path.sh
#Start simulation putting stdout & stderr to a file so we can view it as we go
sim-state "${STATE_FILENAME}" ${mcs} > std.log 2>&1
HEREDOC

		#Submit job (27626.calgary.phy.bris.ac.uk)
		JOB_ID=$(qsub "${BUILD_DIR}run.sh" || (redmessage "Failed to start job!\n"; exit 1) )

		#write to log
		if [ -z "$JOB_ID" ]; then
			redmessage "Something went wrong... I didn't get a JOB_ID !\n"
			echo "Failed to start job number ${m} in ${BUILD_DIR}\n" >> "${LOG_FILE}"
		else
			greenmessage "Running ${JOB_ID} in ${BUILD_DIR}\n"
			echo "${JOB_ID} ${BUILD_DIR}" >> "${LOG_FILE}"
		fi

	elif [ "$MODE" = "local" ]; then
	
		#Run job locally
		cd "${BUILD_DIR}"

		sim-state "${STATE_FILENAME}" ${mcs}

		if [ $? -ne 0 ]; then
			redmessage "Job ${m} failed!\n"
			echo "Job ${m} failed in ${BUILD_DIR}\n" >> "${LOG_FILE}"
		else
			greenmessage "Job ${m} finished"
			echo "${m} ${BUILD_DIR}" >> "${LOG_FILE}"
		fi

		#change back to original directory
		cd -

	fi

	
done
