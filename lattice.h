/* Header file for LatticeObject functions, structs & enums
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef TWO_D_LATTICE

	//The approximate value of PI
	extern const float PI;

	/* DirectorElement is a 2D vector (in physics sense)
	*  expressed in cartesian co-ordinates. Arrays of 
	*  these are used to build a vector field.
	*/
	typedef struct 
	{
		float x,y;
	} DirectorElement;


	/* LatticeConfig is used to hold initial configuration parameters
	* for a lattice and should be passed to latticeInitialise();
	*
	*/
	typedef struct
	{
		unsigned int width;
		unsigned int height;

		/*
		* assume k_1 = 1
		* k_3 = beta* k_1
		*/
		float beta;

		enum latticeBoundary
		{
			BOUNDARY_PARALLEL,
			BOUNDARY_PERPENDICULAR,
			BOUNDARY_PERIODIC
		} topBoundary, bottomBoundary, leftBoundary, rightBoundary;

		enum latticeState
		{
			RANDOM,
			PARALLEL_X,
			PARALLEL_Y,
			BOT_PAR_TOP_NORM
		} initialState;

	} LatticeConfig;

	

	typedef struct
	{
				
		//Lattice Parameters
		LatticeConfig param;

	
		//Define the 2D lattice array
		DirectorElement** lattice;

	} LatticeObject;	

	
	/* See above:
	 * nSystem struct is defined with {width,height, beta}
	*/

	/*
	This function returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
	boundary conditions of a LatticeObject (theLattice). You need to pass a pointer to the LatticeObject.
	
	*/
	DirectorElement* latticeGetN(LatticeObject* theLattice, int xPos, int yPos);
	
	
	/* This function is used free memory allocated by the latticeInitialise function
	*  You should pass it a pointer to a LatticeObject type. 
	*
	*/
	void latticeFree(LatticeObject* theLattice);
	

	/* This function is used in initialise a LatticeObject from the freestore and returns a pointer to 
	*  the newly made object. Use latticeFree() to remove the object from the freestore
	*
	*/
	LatticeObject* latticeInitialise(LatticeConfig configuration);
	
	/*
	* This function outputs the current state of the lattice "theLattice" to standard output in a format
	* compatible with GNUplot. A simple plot command is `set key off; plot 'file' with vectors`
	*/
	void latticeDump(LatticeObject* theLattice);

	#define TWO_D_LATTICE 1	
#endif
