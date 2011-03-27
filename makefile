#make file for 2D LC model by Dan Liew (2010)

#Compiler
CXX = g++

#paths to search
VPATH=nanoparticles tests

#Set defaults that can be overriden by command line passing
debug=0
profile=0
runth=1

#decide whether to build for debugging or optimize the code
ifeq (${debug},1)
  DEBUG_OR_OPT= -g
else
  #Agressively optimize
  DEBUG_OR_OPT= -O3
endif

#decice whether or not to produce code for profiling with gprof
ifeq (${profile},1)
  PROFILE_OPT= -pg
else
  PROFILE_OPT=
endif


#Compiler flags for .cpp files
CPPFLAGS = ${DEBUG_OR_OPT} ${PROFILE_OPT} -Wall -I$(shell pwd)

#Project object files
OBJECTS =  directorelement.o lattice.o mt19937ar.o circle.o ellipse.o 

#Project libraries to use (space seperated)
LIBRARIES = m 


#overwrite implicit rules so we can generate dependency (.dep) files
%.o : %.cpp
	${CXX} -c ${CPPFLAGS} $< -o $@
	${CXX} -M ${CPPFLAGS} $< > $*.dep

#Default target that makes various useful tools
tools : sim-state create-state probe-state dump-state comp-energy-state comp-angle-state dvid-state
ifeq (${runth},1)
	 scripts/tests.sh
endif


#The Monte Carlo annealing simulator
sim-state: sim-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} sim-state.o ${OBJECTS} $(foreach library,$(LIBRARIES),-l$(library))  -o $@ 
	$(info IF YOU RENAME ANY SOURCE FILES RUN ``make clean'' to clean up dependencies)

#include prerequesite files in make file
-include $(OBJECTS:.o=.dep) 

#Helper tools
create-state: create-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@

probe-state: probe-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@

dump-state: dump-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@

comp-energy-state: comp-energy-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@

comp-angle-state: comp-angle-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@

dvid-state: dvid-state.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@

#Test Harnesses must be defined between the "#TEST HARNESSES START" and "#TEST HARNESSES END" lines
#After the rule definition you must supply the arguments to call the test harnesses with on the next line(s).
#
#A line starting with "#ARGS <parameter1> <parameter2> ..." indicates to run that test harness with those parameters.
#If "#ARGS" is specified more than once then the test harness will be run for each set of arguments.

#TEST HARNESSES START

# Analytical energy test.
ea-test: ea-test.o ${OBJECTS}
	${CXX} ${CPPFLAGS} $^ -o $@
#ARGS 50 50 1 1e-5
#ARGS 50 50 2 1e-5
#ARGS 50 50 3 1e-5
#ARGS 50 50 4 1e-5
#Can't really get next test correct as program design doesn't allow k_1=0 and making k_3 very large leads to rounding errors!
#ARGS 50 50 5 1e-1 

initialise-test: initialise-test.o ${OBJECTS}
	${CXX} $^ ${CPPFLAGS} -o $@
#ARGS 30 30 0
#ARGS 30 30 1
#ARGS 30 30 2
#ARGS 30 30 3

uniform-rnd: uniform-rnd.o mt19937ar.o
	${CXX} $^ ${CPPFLAGS} -o $@

rotate-director-test: rotate-director-test.o ${OBJECTS}
	${CXX} $^ ${CPPFLAGS} -o $@

save-load-test: save-load-test.o ${OBJECTS}
	${CXX} $^ ${CPPFLAGS} -o $@
#ARGS 30 30 0
#ARGS 30 30 1
#ARGS 30 30 2
#ARGS 30 30 3

flip-test: flip-test.o ${OBJECTS}
	${CXX} $^ ${CPPFLAGS} -o $@
#ARGS tests/random.bin

#TEST HARNESSES END

#Phont target used to remove generated objects and dependency files
.PHONY: clean
clean: 
	rm *.dep *.o  $(shell grep -Eo '^[a-z0-9_-]+:' makefile | sed 's/://' | sed 's/^clean//' )

