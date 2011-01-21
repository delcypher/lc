/* Header file for LatticeObject functions, structs & enums
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef TWO_D_CUDA_LATTICE
	
	#include "common.h"
	#include "nanoparticle.h"

	/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
	*
	*/
	__device__ double dev_calculateCosineBetween(DirectorElement* a, DirectorElement* b);

	/* Flips a DirectorElement (vector in physics sense) in the opposite direction
	*
	*/
	__device__ void dev_flipDirector(DirectorElement* a);


	/*
	This function returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
	boundary conditions of a LatticeObject (theLattice). You need to pass a pointer to the LatticeObject.
	
	*/
	__device__ DirectorElement* latticeGetN(LatticeObject* theLattice, int xPos, int yPos);
	
	

	/* Calculate the "free energy per unit area" for a cell at (xPos, yPos) using the frank equation in 2D
	*
	*/
	__device__ double latticeCalculateEnergyOfCell(const LatticeObject* l, int xPos, int yPos);


	/* This function returns the correct modulo for dealing with negative a. Note % does not!
	 *
	 * dev_mod(a,b) = a mod b
	*/
	__device__ inline int dev_mod(int a, int b);

	#define TWO_D_CUDA_LATTICE 1	
#endif
