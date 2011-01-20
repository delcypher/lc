#make file for 2D LC model by Dan Liew (2010)

#Compiler
CXX = nvcc

#path to search
VPATH=nanoparticles

#Compiler flags
CPPFLAGS = --compiler-options -Wall -g

#Project object files
OBJECTS =  main.o lattice.o randgen.o differentiate.o circle.o devicemanager.o

#Project libraries to use (space seperated)
LIBRARIES = m 

#Executable filename
EXEC_NAME=2dlc

#overwrite implicit rules so we can generate dependency (.dep) files
%.o : %.cpp
	${CXX} -c ${CPPFLAGS} $< -o $@
	${CXX} -M ${CPPFLAGS} $< > $*.dep

%.o : %.cu
	${CXX} -c ${CPPFLAGS} $< -o $@
	${CXX} -M ${CPPFLAGS} $< > $*.dep

#default target (link)
${EXEC_NAME} : ${OBJECTS}
	${CXX} ${CPPFLAGS} ${OBJECTS} $(foreach library,$(LIBRARIES),-l$(library)) -o $@ 
	$(info IF YOU RENAME ANY SOURCE FILES RUN ``make clean'' to clean up dependencies)

#include prerequesite files in make file
-include $(OBJECTS:.o=.d) 

#Small tool targets
device-probe: cuda-tools/device_probe.cu
	${CXX} ${CPPFLAGS} $< -o $@	

#Phont target used to remove generated objects and dependency files
.PHONY: clean
clean: 
	rm $(OBJECTS:.o=.dep) $(OBJECTS) ${EXEC_NAME}

