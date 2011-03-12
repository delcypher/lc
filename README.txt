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

BINARY STATE FILES:
The *-state tools work with "binary state files". These files contain the binary data for the lattice, nanoparticles within the lattice and the monte carlo parameters.
The purpose of these files is to allow simulations to be stopped and resumed as required. It also means the simulator "sim-state" does not need to be recompiled to work
with different situations.

Note that as these "binary state files" are tied to architecture of the CPU that they are made on. So for example a "binary state file" made on a x86_64 CPU will probably
not work on a machine with a x86 CPU.

HOW TO COMPILE AND RUN:
1. run the following command
  $ make
   This will build the following tools:

   create-state : Creates a binary state file.
   dump-state   : Reads a binary state file and sends to standard output data for use with the "ildump.gnu" GNUplot script.
   probe-state  : Displays information about a binary state file.
   sim-state    : Simulates the lattice specified by a binary state file with Monte Carlo parameters specified by the binary state file in
                  a free energy minimisation Monte Carlo simulation.
   
   To add these tools to your work path run
   $ source scripts/path.sh

2. add more later....


SIGNAL HANDLING:
The 2dlc program is designed to handle UNIX kill signals whilst running to do some useful things. You send a signal to the application
by finding out the PID of the running program by running
$ pgrep 2dlc

You can then send the signal by running
 $ kill -<signal> <pid>

For example to send the signal SIGUSR1 to PID 4556 run
$ kill -SIGUSR1 4556

Here are the supported kill signals and what they cause the main program to do:

* SIGINT (pressing CTRL + C) or SIGTERM - Cause the application to complete the currently running Monte Carlo step then a binary
                                          state is saved to the file BACKUP_LATTICE_STATE_FILE (defined in main.cpp) and a viewable
					  state is saved to the file REQUEST_LATTICE_STATE_FILE. Then the program will exit.

* SIGUSR1 - Causes the application to a pause execution, output a viewable state to the file REQUEST_LATTICE_STATE_FILE and then
            resume execution. This is useful for seeing how the lattice looks during a simulation.

DEBUGGING AND CODE OPTIMISATION:
By default the makefile is set to build optimised code. To build for debugging through gdb run the following.
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
2. Implement your own class in it's own implementation file (e.g. mynewnanoparticle.cpp) . You must implement the following methods:

getDescription()
processCell()
saveState()

A constructor that takes regular arguments. e.g. mynanoparticle::mynanoparticle(int width, int height ...);
A constructor that takes an ifstream as input. e.g. mynanoparticle::mynanoparticle(std::ifstream & input);

   See nanoparticles/circle.h and nanoparticles/circle.cpp as an example.
3. Include your new nanoparticle header in the relevant files (e.g. sim-state). (i.e. #include "nanoparticles/mynewnanoparticle.h" )
4. Add an ID for your nanoparticle to the "types" enum in the Nanoparticle class definition.
5. Add the ifstream argument constructor to your nanoparticle to the switch statement in Lattice::saveState() so that your nanoparticle
   can be reconstructed from binary data saved to a file.

5. Add your particle to the lattice using the Lattice::add() method in the relevant programs (e.g. create-state)
6. Add your nanoparticle to the variable OBJECTS with a ".o" extension in the makefile. (e.g. mynewnanoparticle.o )
7. Recompile by running ``make''

HOW TO CLEAN UP THE BUILD:
Compiling the program will generate lots of .o files (object files) & .dep (dependency files for make). Run the following command to clean this up.
  $ make clean

WARNINGS:
1. If you rename any files you should run ``make clean'' to remove old dependency & object files. You should also be using ``git mv'' NOT ``mv''.

SCRIPTS:
In the scripts/ directory are bash shell and gnuplot scripts for doing various useful things.

aangle.gnu - This is a GNUplot script to show the output of the sim-state program on the file defined by the variable ANNEALING_FILE. It plots "Acceptance angle" aginst monte carlo step.
energy.gnu - This is a GNUplot script to show the output of the sim-state program on the file defined by the variable ENERGY_FILE. It plots "Free Energy" against monte carlo step.
ildump.gnu - This is a GNUplot script to show the output of Lattice::indexedNDump() in GNUplot's interactive mode.
itk.gnu - This is a GNUplot script to show the output of the sim-state program on the file defined by the variable ANNEALING_FILE. It plots "iTk" against monte carlo step.
ldump.gnu - This is a GNUplot script to show the output of Lattice::nDump() in GNUplot's interactive mode.
path.sh - This is a bash shell script to add the build directory to the PATH variable so you can run the executables from any directory. To use it run ``source path.sh'' .
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

