2D Liquid Crystal Lattice model by Alex Allen & Dan Liew

This program provides a means to model a Liquid crystal in 2D by minimising the free energy per unit area (frank equation). As this is in 2D there is no twist term in the the frank equation. 
This is our attempt to get some code that is actually useful as our previous attempts with CUDA were getting no where.

CONVENTIONS:
This README file has the following conventions:
* Shell commands on their own line are prefixed with a $ for example
  $ ls -l 

* Shell commands in sentences are in quotes like this ``ls -l".

* In shell commands an option in square brackets is optional. You are expected to substitue this for a relevant option. E.g.
  $ ls -l [path]

Now to the important stuff...

HOW TO COMPILE AND RUN:
1. run the following command
  $ make
2. An executable probably named 2dlc will be created (check the makefile to see what it will be called). To execute it run
  $ ./2dlc

DEBUGGING AND CODE OPTIMISATION:
By default the makefile is set to build code for debugging with gdb and does no optimisation. To explicitly set this run
  $ make [target] debug=1

To disable debugging and aggressively optimise the code then run
  $ make [target] debug=0

PROFILING:
By default the makefile is set to not build code for profiling with gprof. To explicity set this run
  $ make [target] profile=0

To build code for profiling run
  $ make [target] profile=1

HOW TO ADD YOUR OWN NANOPARTICLES:
1. Declare your own class deriving from the Nanoparticle class (nanoparticle.h) in it's own header (e.g. mynewnanoparticle.h) in
   the "nanoparticles" folder. See nanoparticles/circle.h as an example.
2. Implement your own class in it's own implementation file (e.g. mynewnanoparticle.cpp) . You must implement the processCell() 
   function. See nanoparticles/circle.h as an example.
3. Include your new nanoparticle header in main.cu. (i.e. #include "nanoparticles/mynewnanoparticle.h" )
4. Add your particle to the lattice using latticeAdd() in main.cu
5. Add your nanoparticle to the variable OBJECTS with a ".o" extension in the makefile. (e.g. mynewnanoparticle.o )
6. Recompile by running ``make''

HOW TO CLEAN UP THE BUILD:
Compiling the program will generate lots of .o files (object files) & .dep (dependency files for make). Run the following command to clean this up.
  $ make clean

WARNINGS:
1. If you rename any files you should run ``make clean'' to remove old dependency & object files. You should also be using ``git mv'' NOT ``mv''.

SCRIPTS:
In the scripts/ directory are bash shell and gnuplot scripts for doing various useful things.

ildump.gnu - This is a GNUplot script to show the output of Lattice::indexedNDump() in GNUplot's interactive mode.
ldump.gnu - This is a GNUplot script to show the output of Lattice::nDump() in GNUplot's interactive mode.
tests.sh - This is a script to automatically build and execute test harnesses in the make file.
vpn.sh - This is a bash shell script used to setup a connection to the UoB VPN if pptpclient has already been configured on a GNU/Linux system with ppsetup.

TEST HARNESSES:
The test harnesses are a collection of small programs meant to test various things (e.g. Comparing analytical solutions to computer calculated values) that
should succeed as changes are made to the code.

Test harnesses are in the test/ directory. See the makefile for the target name to build. For example to build the mod-test harness run
$ make mod-test

To build all test-harnesses and execute the ones with defined parameters in the makefile then run
$ scripts/tests.sh

This is useful for testing whether or not you've broken things when you have make changes to the code. It is not a comprehensive test but
if compilation or the test itself fails then you've broken something... SO FIX IT!

See the source code for a particular test harnesses to understand how to use it.

