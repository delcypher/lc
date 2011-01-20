/* Header file for LatticeObject functions, structs & enums
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef TWO_D_LATTICE
	
	#include "common.h"
	#include "nanoparticle.h"

	//The approximate value of PI
	extern const double PI;



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

	
		//Define the 2D lattice array (we use a linear memory block however)
		DirectorElement* lattice;
		
		DirectorElement PERPENDICULAR_DIRECTOR;
		DirectorElement PARALLEL_DIRECTOR;

	} LatticeObject;	


	/* The Lattice class is used as a "wrapper" class for LatticeObject that handles
	*  the device pointers and has associated methods
	*/
	class Lattice
	{
		private:
			DirectorElement* devLatticeArray; //pointer to device's lattice ary
			
			/* This frees allocated memory of the CUDA device.
			*  This is called internally by the destructor so you shouldn't to call it yourself.
			*/
			void freeCuda();
		
			/* This initialises memory on the CUDA device and
			*  copies the host's LatticeObject to the CUDA device.
			*/
			void initialiseCuda();

			public:
			LatticeObject* hostLatticeObject; //pointer to host's LatticeObject.
			LatticeObject* devLatticeObject; //pointer to device's LatticeObject.
			Lattice(LatticeConfig configuration);
			~Lattice(); //destructor

			//Copies the Host LatticeObject to the device
			void copyHostToDevice();

			//Copies the Device LatticeObject to the host.
			void copyDeviceToHost();

			/* Adds a nanoparticle (np) (of type that should be derived from class Nanoparticle) to the lattice.
			*  The method will return true if successful or false if something goes wrong!
			*/
			bool add(Nanoparticle* np);

			/*
			This method returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
			boundary conditions of a LatticeObject (theLattice).
			*/
			DirectorElement* getN(int xPos, int yPos);
			
			void reInitialise(enum LatticeConfig::latticeState initialState);

			/* These are the different dumping modes used by translatedUnitVectorDump()
			*  EVERYTHING - NANOPARTICLES AND NORMAL LATTICE POINTS
			*  PARTICLES - NANOPARTICLES ONLY
			*  NOT_PARTICLES - NORMAL LATTICE POINTS ONLY
			*/
			enum dumpMode
			{
				EVERYTHING,
				PARTICLES,
				NOT_PARTICLES,
			};

			/*
			* This method outputs the current state of the lattice to standard output in a format
			* compatible with shell script latticedump.sh which uses GNUplot. The director field is plotted as
			* unit vectors that are translated so that the centre of the vector rather than the end of the vector
			* is plotted at point (xPos,yPos).
			*/
			void translatedUnitVectorDump(enum Lattice::dumpMode mode) const;

			/*
			* This method outputs the current state of the lattice to standard output in a format
			* compatible with the shell script latticedump.sh which uses GNUplot . The director field is plotted as 
			* 1/2 unit vectors rather than unit vectors so that neighbouring vectors when plotted do not overlap.
			*/
			void HalfUnitVectorDump() const;

			double calculateEnergyOfCell(int xPos, int yPos);
			double calculateTotalEnergy();

	};


	/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
	*
	*/
	double calculateCosineBetween(DirectorElement* a, DirectorElement* b);

	/* Flips a DirectorElement (vector in physics sense) in the opposite direction
	*
	*/
	void flipDirector(DirectorElement* a);

	/* Adds a nanoparticle (np) (of type that should be derived from class Nanoparticle) to lattice (lat).
	*  The function will return true if successful or false if something goes wrong!
	*/
	//bool latticeAdd(LatticeObject* lat, Nanoparticle* np);
	
	//DirectorElement* latticeGetN(const LatticeObject* theLattice, int xPos, int yPos);
	
	
	/* This function is used free memory allocated by the latticeInitialise function
	*  You should pass it a pointer to a LatticeObject type. 
	*
	*/
	//void latticeFree(LatticeObject* theLattice);
	

	/* This function is used in initialise a LatticeObject from the freestore and returns a pointer to 
	*  the newly made object. Use latticeFree() to remove the object from the freestore
	*
	*/
	//LatticeObject* latticeInitialise(LatticeConfig configuration);
	
	/* This function is used to reinitialise the state of an existing lattice (how the directors are pointing)
	*  by passing in an initialState.
	*  The return value is 0 for failure, 1 for success.
	*/
	
	//int latticeReinitialise(LatticeObject* theLattice, enum LatticeConfig::latticeState initialState);

	/*
	* This function outputs the current state of the lattice "theLattice" to standard output in a format
	* compatible with the shell script latticedump.sh which uses GNUplot . The director field is plotted as 
	* 1/2 unit vectors rather than unit vectors so that neighbouring vectors when plotted do not overlap.
	*/
	//void latticeHalfUnitVectorDump(LatticeObject* theLattice);

	/* Calculate the "free energy per unit area" for a cell at (xPos, yPos) using the frank equation in 2D
	*
	*/
	//double latticeCalculateEnergyOfCell(const LatticeObject* l, int xPos, int yPos);

	/* Calculate the "free energy" of entire lattice. Note this calculation may not be very efficient!
	*
	*/
	//double latticeCalculateTotalEnergy(const LatticeObject* l);

	
	/*
	* This function outputs the current state of the lattice "theLattice" to standard output in a format
	* compatible with shell script latticedump.sh which uses GNUplot. The director field is plotted as
	* unit vectors that are translated so that the centre of the vector rather than the end of the vector
	* is plotted at point (xPos,yPos).
	*/
	//void latticeTranslatedUnitVectorDump(LatticeObject* theLattice, enum dumpMode mode);
	
	/* This function returns the correct modulo for dealing with negative a. Note % does not!
	 *
	 * mod(a,b) = a mod b
	*/
	inline int mod(int a, int b);

	#define TWO_D_LATTICE 1	
#endif
