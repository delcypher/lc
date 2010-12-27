#make file for 2D LC model by Dan Liew (2010)

#Compiler
CXX = gcc

#Compiler flags
CPPFLAGS = -Wall -g

#Project object files
OBJECTS =  main.o lattice.o randgen.o differentiate.o

#Project libraries to use (space seperated)
LIBRARIES = m 

#default target
2dlc : ${OBJECTS}
	${CXX} ${CPPFLAGS} ${OBJECTS} $(foreach library,$(LIBRARIES),-l$(library)) -o $@ 

# target : dependencies...
lattice.o : lattice.c lattice.h randgen.h lattice.h
	${CXX} ${CPPFLAGS} -c lattice.c
randgen.o : randgen.c randgen.h
	${CXX} ${CPPFLAGS} -c randgen.c
differentiate.o : differentiate.c differentiate.h randgen.h lattice.h
	${CXX} ${CPPFLAGS} -c differentiate.c
main.o : main.c lattice.h randgen.h differentiate.h
	${CXX} ${CPPFLAGS} -c main.c

#Phont target used to remove generated objects
.PHONY: clean
clean: 
	rm ${OBJECTS} 

