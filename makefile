#make file for 2D LC model by Dan Liew (2010)

#Compiler
CXX = nvcc

#Path for intermediate build files (objects and dependency files)
BUILDPATH = build

#path to search
VPATH=nanoparticles tests ${BUILDPATH}

#Path for binaries
BINPATH = bin

#Compiler flags for .cpp files
CPPFLAGS = -g --host-compilation c++ --compiler-options -Wall

#Compiler options for .cu files
NVCCFLAGS = -arch=compute_13 -code=compute_13 --host-compilation c++ -g -G --compiler-options -Wall

#Project object files
OBJECTS =  main.o lattice.o randgen.o differentiate.o circle.o devicemanager.o

#Test harness object files
THOBJECTS = mod-test.o

#Project libraries to use (space seperated)
LIBRARIES = m 

#Project Executable filename
EXEC_NAME=2dlc

#overwrite implicit rules so we can generate dependency (.dep) files
%.o : %.cpp
	${CXX} -c ${CPPFLAGS} $< -o ${BUILDPATH}/$@
	${CXX} -M $< > ${BUILDPATH}/$*.dep

%.o : %.cu
	${CXX} -c ${NVCCFLAGS} $< -o ${BUILDPATH}/$@
	${CXX} -M $< > ${BUILDPATH}/$*.dep

#default target (link)
${EXEC_NAME} : ${OBJECTS}
	${CXX} ${CPPFLAGS} $(addprefix ${BUILDPATH}/,${OBJECTS}) $(foreach library,$(LIBRARIES),-l$(library)) -o ${BINPATH}/$@ 
	$(info IF YOU RENAME ANY SOURCE FILES RUN ``make clean'' to clean up dependencies)

#include prerequesite files in make file
-include $(OBJECTS:.o=.dep) 
-include $(THOBJECTS:.o=.dep)

#Small tool targets
device-probe: cuda-tools/device-probe.cu devicemanager.o
	${CXX} ${NVCCFLAGS} $^ -o ${BINPATH}/$@	

#TEST HARNESSES

mod-test: mod-test.o lattice.o differentiate.o devicemanager.o randgen.o
	${CXX} ${NVCCFLAGS} $^ -o ${BINPATH}/$@


#Phont target used to remove generated objects and dependency files
.PHONY: clean
clean: 
	rm $(addprefix ${BUILDPATH}/,$(OBJECTS:.o=.dep)) $(addprefix ${BUILDPATH}/,$(OBJECTS)) ${BINPATH}/${EXEC_NAME}

