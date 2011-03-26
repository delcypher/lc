2D Liquid Crystal Lattice model by Alex Allen & Dan Liew

This program provides a means to model a Liquid crystal in 2D by minimising the free energy per unit area (frank equation). As this is in 2D there is no twist term in the the frank equation. 
This is our attempt to get some code that is actually useful as our previous attempts with CUDA were getting no where.

This code has been built and tested on a GNU/Linux system using g++ . It has not been tested on OSX (although that will probably will work) or on Windows (Probably won't work due to signal.h in sim-state)

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
not work on a machine with a x86 CPU. It also important to note if you change the structs LatticeConfig, DirectorElement then older "binary state files" may no longer
be compatible!

HOW TO COMPILE AND RUN:
1. run the following command
  $ make
   This will build the following tools:

   comp-energy-state : Compares a lattice described by a binary state file against the energy of a known analytical solution.
   comp-angle-state  : Compares a lattice described by a binary state file against the angular distribution  of a known 
   		       analytical solution.
   create-state      : Creates a binary state file.
   dump-state        : Reads a binary state file and sends to standard output data for use with the "ildump.gnu" GNUplot script.
   dvid-state        : Director Variation In Direction. Loads a binary state file and outputs data for plotting angle of Director
   		       in a particular direction.
   probe-state       : Displays information about a binary state file.
   sim-state         : Simulates the lattice specified by a binary state file with Monte Carlo parameters specified by the binary 
   		       state file in a free energy minimisation Monte Carlo simulation.
   
   To add these tools to your work path (BASH shell only) run
   
   $ source scripts/path.sh

   This guide will assume you have.

   Note if you're not sure how to run the tools just run it without arguments and a usage message will be shown. For example

   $ dvid-state
   Usage: ./dvid-state <filename> <x> <y> <angle> <max_r> <angular_restriction> [ -n ]
   <filename> - Binary state file to read
   <x> - The x co-ordinate of the origin of r (int)
   <y> - The y co-ordinate of the origin of r (int)
   <angle> - The angle r makes with x-axis in radians
   <max_r> - The maximum magnitude of r (float)
   <angular_restriction> - The angular restriction to apply on the lattice (enum Lattice::angularRegion)
   
   OPTIONAL:
   -n 
   Do not provide plot data for cells that are marked as being a nanoparticle. This is NOT the default behaviour!

   You can see that dvid-state requires 6 command line arguments and takes one optional argument. Here are a few examples of valid command line parameters.

   $ dvid-state mystate.bin 7 7 1.87 100.0 0
   $ dvid-state mystate.bin 7 7 1.87 100.0 0 -n

2. Now modifiy create-state.cpp as you wish so that you have the lattice you want with nanoparticles etc... If you're not sure how to
   do this you should look at the "onee" and "twoe" branches version of create-state.cpp as an example.

   Now that you're done recompile create-state

   $ make create-state

   Now run it

   $ create-state mystate.bin

   You have now made a binary state file called mystate.bin . You can find out information about it at anytime by running

   $ probe-state mystate.bin

   To visually view the state (2D Director field plot) you run the "view-state.sh" script

   $ scripts/view-state.sh mystate.bin

   or run it via the alias added by scripts/path.sh

   $ view-state mystate.bin

   To produce the output required by GNUplot script "ildump.gnu" (essentially what view-state does for you) run

   $ dump-state mystate.bin > outputfile

   In GNUplot you can then run

   gnuplot> call "scripts/ildump.gnu" "outputfile" <width+1> <height+1>

   where <width+1> & <height+1> are the width and height of the lattice +1 respectively.

3. To now simulate the "cooling" of a lattice state to the minimum energy situation you use the sim-state tool. To run it run

   $ sim-state mystate.bin <steps>

   where <steps> is the number of monte carlo steps to run the simulation for.

   The sim-state program creates various output files. These are created in the working directory that sim-state was called from. For
   example.

   $ pwd
   /home/dan/adir/
   $ /home/dan/lc/sim-state.bin mystate.bin
   $ ls
   annealing.dump  coning.dump  energy.dump  final-lattice-state.bin  final-lattice-state.dump

   The sim-state program is designed to handle UNIX kill signals whilst running to do some useful things. You send a signal to the application
   by finding out the PID of the running program by running
   $ pgrep 2dlc
   
   You can then send the signal by running
    $ kill -<signal> <pid>
   
   For example to send the signal SIGUSR1 to PID 4556 run
   $ kill -SIGUSR1 4556
   
   Here are the supported kill signals and what they cause the main program to do:
   
   * SIGINT (pressing CTRL + C) or SIGTERM - Cause the application to complete the currently running Monte Carlo step then a binary
                                             state file is saved to the file BACKUP_LATTICE_STATE_FILE (defined in sim-state.cpp) and a viewable
   					  state is saved to the file REQUEST_LATTICE_STATE_FILE. Then the program will exit.
   
   * SIGUSR1 - Causes the application to a pause execution, output a viewable state to the file REQUEST_LATTICE_STATE_FILE and then
               resume execution. This is useful for seeing how the lattice looks during a simulation.

   If the simulation is halted by a kill signal before it is completed then it will produce the BACKUP_LATTICE_STATE_FILE which can then be used
   to resume the simulation from. Here's an example

   $ sim-state mystate.bin #We send SIGINT (CTRL+C) during simulation and so sim-state exits.
   $ sim-state backup-lattice-state.bin #We resume the simulation.

   When the simulation completes the following files are outputted (defined in sim-state.cpp)

  ANNEALING_FILE
  CONING_FILE
  ENERGY_FILE
  FINAL_LATTICE_STATE_FILE
  FINAL_LATTICE_BINARY_STATE_FILE

  See sim-state.cpp to see what each of these are.

4. The most important file created by the completion of running sim-state is FINAL_LATTICE_BINARY_FILE . At the time of writting
   this was defined as "final-lattice-state.bin". As before you can run probe-state, dump-state & view-state on it.

   Several other tools are provided for analysis:
   comp-angle-state
   comp-energy-state
   dvid-state

   which were described earlier.

Hopefully that all makes sense!

RUNNING BATCH JOBS
The modular design of the programs is useful for running batch jobs because sim-state can simply be called on many different
binary state files. It should be noted however that sim-state always outputs the same filenames so each job should be run in 
a different directory.

The challenge though is to generate different binary state files first. You should look at the "onee" and "twoee" branches which
provide a version of "create-state.cpp" that takes command line arguments controlling the binary state files it produces. 
In addition these branches provide a bash shell script ``scripts/job.sh'' that automates the creation of binary state files and 
then execution of the batch jobs which can be run locally or on a pbs/torque cluster.

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
3. Include your new nanoparticle header in create-state.cpp & lattice.h . (i.e. #include "nanoparticles/mynewnanoparticle.h" )
4. Add an ID for your nanoparticle to the "types" enum in the Nanoparticle class definition.
5. Add the ifstream argument constructor to your nanoparticle to the switch statement in Lattice::saveState() so that your nanoparticle
   can be reconstructed from binary data saved to a file.
6. Add support for your nanoparticle in the switch statement in the Lattice::Lattice(const char* filepath) constructor so that
   your nanoparticle can be recreated from a binary state file.
7. Add your particle to the lattice using the Lattice::add() method in the relevant programs (e.g. create-state)
8. Add your nanoparticle to the variable OBJECTS with a ".o" extension in the makefile. (e.g. mynewnanoparticle.o )
9. Recompile by running ``make''

HOW TO CLEAN UP THE BUILD:
Compiling the program will generate lots of .o files (object files) & .dep (dependency files for make). Run the following command to clean this up.
  $ make clean

WARNINGS:
1. If you rename any files you should run ``make clean'' to remove old dependency & object files. You should also be using ``git mv'' NOT ``mv''.
2. When switching between git branches remember to recompile the source code using ``make''.

SCRIPTS:
In the scripts/ directory are bash shell and gnuplot scripts for doing various useful things.

annealing.gnu - This is a GNUplot script to show the output of the sim-state program on the file defined by the variable ANNEALING_FILE. It plots "iTk" against monte carlo step.
coning.gnu - This is a GNUplot script to show the output of the sim-state program on the file defined by the variable CONING_FILE. It plots "Acceptance angle" aginst monte carlo step.
energy.gnu - This is a GNUplot script to show the output of the sim-state program on the file defined by the variable ENERGY_FILE. It plots "Free Energy" against monte carlo step.
ildump.gnu - This is a GNUplot script to show the output of Lattice::indexedNDump() in GNUplot's interactive mode.
ldump.gnu - This is a GNUplot script to show the output of Lattice::nDump() in GNUplot's interactive mode.
path.sh - This is a bash shell script to add the build directory to the PATH variable so you can run the executables from any directory. To use it run ``source path.sh'' .
single.sh - This is a bash shell script to build and run a set of binary state and then run them on a single machine.
tests.sh - This is a script to automatically build and execute test harnesses in the make file.
job.sh - This is a script to build a set of binary state files in different folders and either run them locally or submit them to a PBS/torque queing system.
view-state.sh - This script allows a binary state file to quickly viewed by automating the calling of ildump.gnu

TEST HARNESSES:
The test harnesses are a collection of small programs meant to test various things (e.g. Comparing analytical solutions to computer calculated values) that
should succeed as changes are made to the code.

Test harnesses are in the test/ directory. See the makefile for the target name to build. For example to build the ea-test harness run
$ make ea-test

To build all test-harnesses and execute the ones with defined parameters in the makefile then run
$ scripts/tests.sh

This is useful for testing whether or not you've broken things when you have make changes to the code. It is not a comprehensive test but
if compilation or the test itself fails then you've broken something... SO FIX IT!

See the source code for a particular test harnesses to understand how to use it.

