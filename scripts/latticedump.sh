#!/bin/bash 
#script to show the output of the latticeDump function in lattice.c

file="$1";
PROG="gnuplot -persist"

if [ $# -eq 0 ]; then
	echo "Usage $0 file"
	exit 1;
fi 

if [ ! -r "$file" ]; then
	echo "Can't open $file" 1>&2
	exit 1;
fi

#show information about the lattice (assume lines prepended w/ # have info)
cat "$file" | grep '^#'

#do plot
$PROG - <<COOL
set xlabel "x"
set ylabel "y"
set title "$title"
set key off
plot "$1" with vectors
COOL
 
