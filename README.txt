2D Liquid Crystal Lattice model by Alex Allen & Dan Liew

This program provides a means to model a Liquid crystal in 2D by minimising the free energy per unit area (frank equation). As this is in 2D there is no twist term in the the frank equation. This model uses Nvidia's CUDA so you will need the CUDA SDK installed as well as a Nvidia graphics card that supports CUDA.

HOW TO COMPILE AND RUN:
0. Make sure the CUDA SDK is installed.
1. run the following command
make
2. An executable probably named 2dlc will be created (check the makefile to see what it will be called). To execute it run
./2dlc

HOW TO ADD YOUR OWN NANOPARTICLES:
1. Declare your own class deriving from the Nanoparticle class (nanoparticle.h) in it's own header (e.g. mynewnanoparticle.h) in
   the "nanoparticles" folder. See nanoparticles/circle.h as an example.
2. Implement your own class in it's own implementation file (e.g. mynewnanoparticle.cpp) . You must implement the processCell() 
   function. See nanoparticles/circle.h as an example.
3. Include your new nanoparticle header in main.cu. (i.e. #include "nanoparticles/mynewnanoparticle.h" )
4. Add your particle to the lattice using latticeAdd() in main.cu
5. Add your nanoparticle to the variable OBJECTS with a ".o" extension in the makefile. (e.g. mynewnanoparticle.o )
6. Recompile by running "make".

HOW TO CLEAN UP THE BUILD:
Compiling the program will generate lots of .o files (object files) & .dep (dependency files for make). Run the following command to clean this up.
"make clean"

WARNINGS:
1. If you rename any files you should run "make clean" to remove old dependency & object files. You should also be using "git mv" NOT "mv".

SCRIPTS:
In the script/ directory are bash shell and gnuplot scripts for doing various useful things.

vpn.sh - This is a bash shell script used to setup a connection to the UoB VPN if pptpclient has already been configured on a GNU/Linux system with ppsetup.
ldump.gnu - This is a GNUplot script to show the output of Lattice::nDump() in GNUplot's interactive mode.
ildump.gnu - This is a GNUplot script to show the output of Lattice::indexedNDump() in GNUplot's interactive mode.


TEST HARNESSES:
Test harnesses are in the test/ directory. See the makefile for the target name to build. For example to build the mod-test harness run
"make mod-test"

See the source code for a particular test harnesses to understand how to use it.

CUDA TOOLS
1. Device probe - This will list all available CUDA cards on the machine it is run on display their compute capability. To build it run
"make device-probe"
