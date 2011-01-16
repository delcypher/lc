/* Header file for LatticeObject functions, structs & enums
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef TWO_D_LATTICE

	//The approximate value of PI
	extern const double PI;

	/* DirectorElement is a 2D vector (in physics sense)
	*  expressed in cartesian co-ordinates. Arrays of 
	*  these are used to build a vector field.
	*/
	typedef struct 
	{
		double x,y;
	} DirectorElement;


	/* LatticeConfig is used to hold initial configuration parameters
	* for a lattice and should be passed to latticeInitialise();
	*
	*/
	typedef struct
	{
		/*
		* Although width & height should not be < 0
		* using unsigned int causes problems when comparing to signed ints
		* e.g. 
		* signed int x=5;
		* if(-4 < x) this is false if x is unsigned
		* need to do if( -4 < (signed int) x) to get true
		* 
		* We do not want to be doing typecasts everytime we do a comparision so just use
		* signed ints instead!
		*/
		signed int width;
		signed int height;

		/*
		* assume k_1 = 1
		* k_3 = beta* k_1
		*/
		double beta;

		/* These define the different type of boundary conditions on the edges of the lattice
		 * The BOUNDARY_PARALLEL & BOUNDARY_PERPENDICULAR conditions are relative to the x-axis
		 * and not the edge itself.
		 *
		 * For example if leftBoundary = BOUNDARY_PERPENDICULAR
		 * The left boundary will be perpendilcar to the x-axis and not to the left boundary edge
		*/
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

		
	/* This struct is the container for everything to do with the lattice
	 *
	*/
	typedef struct
	{
				
		//Lattice Parameters
		LatticeConfig param;

	
		//Define the 2D lattice array
		DirectorElement** lattice;

	} LatticeObject;	

	/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
	*
	*/
	double calculateCosineBetween(DirectorElement* a, DirectorElement* b);

	/* Flips a DirectorElement (vector in physics sense) in the opposite direction
	*
	*/
	inline void flipDirector(DirectorElement* a);

	/*
	This function returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
	boundary conditions of a LatticeObject (theLattice). You need to pass a pointer to the LatticeObject.
	
	*/
	DirectorElement* latticeGetN(const LatticeObject* theLattice, int xPos, int yPos);
	
	
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
	* compatible with the shell script latticedump.sh which uses GNUplot . The director field is plotted as 
	* 1/2 unit vectors rather than unit vectors so that neighbouring vectors when plotted do not overlap.
	*/
	void latticeHalfUnitVectorDump(LatticeObject* theLattice);

	/* Calculate the "free energy per unit area" for a cell at (xPos, yPos) using the frank equation in 2D
	*
	*/
	double latticeCalculateEnergyOfCell(const LatticeObject* l, int xPos, int yPos);

	/* Calculate the "free energy" of entire lattice. Note this calculation may not be very efficient!
	*
	*/
	double latticeCalculateTotalEnergy(const LatticeObject* l);

	/*
	* This function outputs the current state of the lattice "theLattice" to standard output in a format
	* compatible with shell script latticedump.sh which uses GNUplot. The director field is plotted as
	* unit vectors that are translated so that the centre of the vector rather than the end of the vector
	* is plotted at point (xPos,yPos)
	*/
	void latticeTranslatedUnitVectorDump(LatticeObject* theLattice);
	
	/* This function returns the correct modulo for dealing with negative a. Note % does not!
	 *
	 * mod(a,b) = a mod b
	*/
	inline int mod(int a, int b);

	#define TWO_D_LATTICE 1	
#endif
