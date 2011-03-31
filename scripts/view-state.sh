#!/bin/bash -x
#Bash script to simplify calling GNUplot script (ildump.gnu) to view a binary state file.

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <binary_state>";
	exit 1;
fi

#Temporary file path (make sure it is an absolute path)
TEMP_FILE="/tmp/state-dump-temp.dump.$$"

#Get filepath
FP=$(basename "$1" )
FP_DIR=$(dirname "$1")
#construct absolute filepath
FP="$( cd ${FP_DIR} ; echo $PWD )/${FP}"

echo "Opening...${FP}"

#get scripts path
SCRIPTS_PATH=$( cd ${BASH_SOURCE[0]%/*} ; echo "$PWD" )

#add *-state tools to workpath
source ${SCRIPTS_PATH}/path.sh

#Check if temporary file already exists
if [ -a "$TEMP_FILE" ]; then
	echo "Temporary file already exists at ${TEMP_FILE}, please remove it !"
	exit 1;
fi

#Make temporary file
echo "Saving temporary dump to ${TEMP_FILE}"
dump-state "$FP" > "$TEMP_FILE"
if [ $? -ne 0 ]; then
	echo "Running dump-state failed"
	exit 1;
fi

#check we can open temporary file
if [ ! -r "${TEMP_FILE}" ]; then
	echo "Can't read temporary file!"
	exit 1;
fi

#Get width & height of lattice
WIDTH=$(probe-state "$FP" | grep -E '^#Lattice Width:[0-9]+$' | grep -Eo '[0-9]+$')
HEIGHT=$(probe-state "$FP" | grep -E '^#Lattice Height:[0-9]+$' | grep -Eo '[0-9]+$')
echo "Determined lattice dimensions (${WIDTH}x${HEIGHT})"

if [ -z "${WIDTH}" ]; then
	echo "Failed to determine width of lattice!"
	exit 1;
fi

if [ -z "${HEIGHT}" ]; then
	echo "Failed to determine height of lattice!"
	exit 1;
fi
#add 1 to width and height which is needed to view boundary properly
WIDTH=$((WIDTH + 1))
HEIGHT=$((HEIGHT + 1))

#Instruct gnuplot to call ildump.gnu to open it and open - (std output so were are left in interactive mode)
#We have to call gnuplot this horrible way (bash process substitution)  as the GNUplot 4.0 doesn't support -e option!
gnuplot <(echo "set term x11; call \"${SCRIPTS_PATH}/ildump.gnu\" \"${TEMP_FILE}\" ${WIDTH} ${HEIGHT}") -

#Remove temporary file
echo "Removing temporary file ${TEMP_FILE}..."
rm "$TEMP_FILE"
