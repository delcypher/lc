#!/bin/bash
#Script to build a set binary state files to simulated on a local machine (local) or on a PBS/torque cluster (pbs)

#SET prefix for PBS/torque jobs
JOB_PREFIX="lc"
TIME="/usr/bin/time -p"
#SET value for PI (20dp from Wolfram Alpha)
PI="3.14159265358979323846"
TMP_FILE="/tmp/tmp.$$"

if [ "$#" -lt 3 ]; then
	echo "Usage: $0 <mode> <target_directory> <log_file> [OPTION...]"
	echo "<mode> - Job run mode. Should be local or pbs"
	echo "<target_directory> - Directory to build simulation directories in"
	echo "<log_file> - A filename to log the jobs started by this script and their associated directories"
	echo " "
	echo "[OPTIONS]"
	echo "-d , --dryrun"
	echo "   Do Dry run. All files and directories will be created but jobs will NOT be executed"
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

declare -ix DRY_RUN=0
#call shift 3 times so $1 is next cmd arg
shift 3
while [ -n "$1" ]; do
	case "$1" in
		-d | --dryrun )
		echo "Doing dry run!"
		DRY_RUN=1
		;;

		*)
		echo "Option $1 not recognised!"
		exit 1;
	esac
	shift;
done;

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

#make temporary file
touch "${TMP_FILE}"
if [ $? -ne 0 ]; then
	echo "failed to write temporary file to ${TMP_FILE}"
	exit 1;
fi

#note n = (1 + m*0.5) where n is scale factor
#Add loop here
for ((m=10; m<=210 ;m+=5))
do
	#Scale the width & height by m
	width=${m}
	height=${width}
	
	if [ -z "${width}" ]; then
		exit
	fi
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

	ARGS="${BUILD_DIR}${STATE_FILENAME} ${width} ${height}"
	echo "create-state $ARGS"
	create-state $ARGS > /dev/null
	if [ "$?" -ne 0 ]; then
		redmessage "Building state file $STATE_FILENAME failed!\n";
		exit 1;
	fi

	#Create time log 
	TIME_LOG="${TARGET_DIR}/${width}-${height}-times.log"
	touch "${TIME_LOG}"

	echo "#[Steps] [time(seconds)]" >> "${TIME_LOG}"

	for ((mcs=1500; mcs<=10000; mcs+=500))
	do
		#Number of Monte Carlo steps to run through
		
		for ((run=1; run<=3; run++))
		do
		
		#Write header to logfile entry
		echo "# ${width}x${height} : ${mcs} steps , run ${run}" >> "${LOG_FILE}"
				
		if [ "$MODE" = "pbs" ]; then
			echo "pbs mode disabled in script!"
			exit 1;
		elif [ "$MODE" = "local" ]; then
		
			#Run job locally
			cd "${BUILD_DIR}"

			if [ "${DRY_RUN}" -eq 0 ]; then
				${TIME} sim-state "${STATE_FILENAME}" ${mcs} 2> "${TMP_FILE}"
			else
				echo "Doing dry run. Not running sim-state ${STATE_FILENAME} ${mcs}"	
				#run true so that $? =0
				true
			fi

			if [ $? -ne 0 ]; then
				redmessage "Job ${m} failed!\n"
			else
				greenmessage "Job ${m} finished"

				#if job succeeded we will process time measurements
				#sed command strips out number next to real time and shows only first line
				timetorun=$( cat "${TMP_FILE}" | sed -n 's/real \([0-9.]\+\)/\1/g ; 1p')
				if [ -z "$timetorun" ]; then
					echo "Failed to get timing measurements!"
					exit 1;
				fi
				echo "${mcs} ${timetorun}" >> "${TIME_LOG}"
			fi

			#change back to original directory
			cd -

		fi
		done
	done
	
done
echo "Removing temporary file ${TMP_FILE}"
rm ${TMP_FILE}
