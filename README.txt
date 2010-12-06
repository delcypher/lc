2D Liquid Crystal Lattice model by Alex Allen & Dan Liew

This program provides a means to model a Liquid crystal in 2D by minimising the free energy per unit area (frank equation). As this is in 2D there is no twist term in the the frank equation.

HOW TO COMPILE AND RUN:
1. run the following command
make
2. An executable probably named 2dlc will be created (check the makefile to see what it will be called). To execute it run
./2dlc

HOW TO CLEAN UP THE BUILD:
Compiling the program will generate lots of .o files (object files). Run the following command to clean this up.
make clean

SCRIPTS:
In the script/ directory are bash shell scripts for doing various useful things.

latticedump.sh - This is used to show the state of a LatticeObject from a latticeDump() using gnuplot. Run it for usage options.
