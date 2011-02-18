#make file for 2D LC model by Dan Liew (2010)

#Compiler
CXX = g++

#paths to search
VPATH=nanoparticles tests

#Set defaults that can be overriden by command line passing
debug=1
profile=0

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
OBJECTS =  main.o directorelement.o lattice.o randgen.o differentiate.o circle.o ellipse.o 

#Project libraries to use (space seperated)
LIBRARIES = m 

#Executable filename
EXEC_NAME=2dlc

#overwrite implicit rules so we can generate dependency (.dep) files
%.o : %.cpp
	${CXX} -c ${CPPFLAGS} $< -o $@
	${CXX} -M ${CPPFLAGS} $< > $*.dep

#default target (link)
${EXEC_NAME} : ${OBJECTS}
	${CXX} ${CPPFLAGS} ${OBJECTS} $(foreach library,$(LIBRARIES),-l$(library))  -o $@ 
	$(info IF YOU RENAME ANY SOURCE FILES RUN ``make clean'' to clean up dependencies)

#include prerequesite files in make file
-include $(OBJECTS:.o=.dep) 

#Test Harnesses must be defined between the "#TEST HARNESSES START" and "#TEST HARNESSES END" lines
#After the rule definition you must supply the arguments to call the test harnesses with on the next line(s).
#
#A line starting with "#ARGS <parameter1> <parameter2> ..." indicates to run that test harness with those parameters.
#If "#ARGS" is specified more than once then the test harness will be run for each set of arguments.

#TEST HARNESSES START

k1_equal_k3_ea: k1_equal_k3_ea.o lattice.o directorelement.o differentiate.o randgen.o
	${CXX} ${CPPFLAGS} $^ -o $@
#ARGS 50 50 1e-14 1e-4

k1_dominant_ea: k1_dominant_ea.o lattice.o directorelement.o differentiate.o randgen.o
	${CXX} ${CPPFLAGS} $^ -o $@
#ARGS 50 50 1e-13 1e-13

k3_dominant_ea: k3_dominant_ea.o lattice.o directorelement.o differentiate.o randgen.o
	${CXX} ${CPPFLAGS} $^ -o $@
#ARGS 50 50 1e-13 1e-13

initialise-test: initialise-test.o lattice.o directorelement.o differentiate.o randgen.o circle.o ellipse.o
	${CXX} $^ ${CPPFLAGS} -o $@
#ARGS 30 30 0
#ARGS 30 30 1
#ARGS 30 30 2
#ARGS 30 30 3

uniform-rnd: uniform-rnd.o randgen.o
	${CXX} $^ ${CPPFLAGS} -o $@

rotate-director-test: rotate-director-test.o lattice.o directorelement.o differentiate.o randgen.o circle.o ellipse.o
	${CXX} $^ ${CPPFLAGS} -o $@

calc-cosine-between-test: calc-cosine-between-test.o directorelement.o
	${CXX} $^ ${CPPFLAGS} -o $@

#TEST HARNESSES END

#Phont target used to remove generated objects and dependency files
.PHONY: clean
clean: 
	rm *.dep *.o ${EXEC_NAME} $(shell grep -Eo '^[a-z0-9_-]+:' makefile | sed 's/://' | sed 's/^clean//' )

