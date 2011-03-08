#!/bin/bash 
#Small helper script to time program run time.

#Assume $1 is no of monte carlo steps
MCS=$1


for ((i=0; i<10; i++)); do
	echo "## $(($i +1))"
	time -p ./sim-state test.bin $MCS 1> /dev/null
	echo "##"
done;
