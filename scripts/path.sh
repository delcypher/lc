#!/bin/bash
#Script that adds the executables to the work path
# run as ``source path.sh'' or ``. path.sh'' from the bash shell

addpath=$( echo "$(pwd)" | sed -e  's/\/'$(basename $(pwd))'//' )

declare -i INPATH=0

#check if path already present
tempIFS=$IFS
IFS=$':'
#Extract each path seperated by a ':' character and compare
for singlepath in $PATH ; do
	if [ "$singlepath" = "$addpath" ]; then
		INPATH=1;
	fi
done

if [ $INPATH -eq 0 ]; then
	export PATH=$PATH:$addpath
	echo "Added $addpath to PATH environmental variable"
else
	echo "$addpath is already in PATH environmental variable"
fi

#Set IFS back...
IFS=$tempIFS

