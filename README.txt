2D Liquid Crystal Lattice model by Alex Allen & Dan Liew

This program provides a means to model a Liquid crystal in 2D by minimising the free energy per unit area (frank equation). As this is in 2D there is no twist term in the the frank equation. This model uses Nvidia's CUDA so you will need the CUDA SDK installed as well as a Nvidia graphics card that supports CUDA.

HOW TO COMPILE AND RUN:
0. Make sure the CUDA SDK is installed.
1. run the following command
make
2. An executable probably named 2dlc will be created (check the makefile to see what it will be called). To execute it run
./2dlc

HOW TO CLEAN UP THE BUILD:
Compiling the program will generate lots of .o files (object files) & .dep (dependency files for make). Run the following command to clean this up.
make clean

WARNINGS:
1. If you rename any files you should run ``make clean'' to remove old dependency & object files. You should also be using ``git mv'' NOT ``mv''.

SCRIPTS:
In the script/ directory are bash shell scripts for doing various useful things.

latticedump.sh - This is used to show the state of a LatticeObject from a latticeDump() using gnuplot. Run it for usage options.
vpn.sh - This is used to setup a connection to the UoB VPN if pptpclient has already been configured on a GNU/Linux system with ppsetup.

CUDA TOOLS
1. Device probe - This will list all available CUDA cards on the machine it is run on display their compute capability. To build it run
``make device-probe''
