#make file for 2D LC model by Dan Liew (2010)

#Compiler
CXX = nvcc

#paths to search
VPATH=nanoparticles tests

#Compiler flags for .cpp files
CPPFLAGS = -g --compiler-options -Wall

#Compiler options for .cu files
NVCCFLAGS = -arch=compute_13 -code=compute_13 -g -G --compiler-options -Wall

#Project object files
OBJECTS =  main.o lattice.o randgen.o differentiate.o circle.o devicemanager.o

#Project libraries to use (space seperated)
LIBRARIES = m 

#Executable filename
EXEC_NAME=2dlc

#overwrite implicit rules so we can generate dependency (.dep) files
%.o : %.cpp
	${CXX} -c ${CPPFLAGS} $< -o $@
	${CXX} -M $< > $*.dep

%.o : %.cu
	${CXX} -c ${NVCCFLAGS} $< -o $@
	${CXX} -M $< > $*.dep

#default target (link)
${EXEC_NAME} : ${OBJECTS}
	${CXX} ${CPPFLAGS} ${OBJECTS} $(foreach library,$(LIBRARIES),-l$(library)) -o $@ 
	$(info IF YOU RENAME ANY SOURCE FILES RUN ``make clean'' to clean up dependencies)

#include prerequesite files in make file
-include $(OBJECTS:.o=.dep) 

#Small tool targets
device-probe: cuda-tools/device-probe.cu devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o $@	

#Test Harnesses must be defined between the "#TEST HARNESSES START" and "#TEST HARNESSES END" lines
#After the rule definition you must supply the arguments to call the test harnesses with on the next line(s).
#
#A line starting with "#ARGS <parameter1> <parameter2> ..." indicates to run that test harness with those parameters.
#If "#ARGS" is specified more than once then the test harness will be run for each set of arguments.

#TEST HARNESSES START

host-energy-analytical: host-energy-analytical.o lattice.o differentiate.o randgen.o devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o $@
#ARGS 50 50 1e-14 1e-4
#ARGS 50 50 1e-20 1e-22

mod-test: mod-test.o lattice.o differentiate.o devicemanager.o randgen.o
	${CXX} ${NVCCFLAGS} $^ -o $@
#ARGS -12 12 3
#ARGS 0 12 5

host-initialise-test: host-initialise-test.o lattice.o differentiate.o randgen.o circle.o devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o $@
#ARGS 30 30 0
#ARGS 30 30 1
#ARGS 30 30 2
#ARGS 30 30 3

copy-to-device-test: copy-to-device-test.o lattice.o differentiate.o randgen.o circle.o devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o $@
#ARGS 30 30 0
#ARGS 30 30 1
#ARGS 30 30 2
#ARGS 30 30 3

getNtest: getNtest.o lattice.o differentiate.o randgen.o circle.o devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o $@
#ARGS 0 0
#ARGS 1 0
#ARGS 0 1
#ARGS 1 1
#ARGS -1 0
#ARGS 0 -1
#ARGS -1 -1

diftest: diftest.o lattice.o differentiate.o randgen.o circle.o devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o $@
#ARGS IGNORED_ARGUMENT


#TEST HARNESSES END

#Phont target used to remove generated objects and dependency files
.PHONY: clean
clean: 
	rm $(OBJECTS:.o=.dep) $(OBJECTS) ${EXEC_NAME}

