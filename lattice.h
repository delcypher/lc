/* Header file for Lattice Class & LatticeObject functions, structs & enums
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef TWO_D_LATTICE
	
	#include "common.h"
	#include "nanoparticle.h"



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
			const int DUMP_PRECISION; //the precision for output used by translatedUnitVectorDump()

		public:
			LatticeObject* hostLatticeObject; //pointer to host's LatticeObject.
			LatticeObject* devLatticeObject; //pointer to device's LatticeObject.

			/* This frees allocated memory of the CUDA device.
			* 
			*/
			void freeCuda();
		
			/* This initialises memory on the CUDA device and
			*  copies the host's LatticeObject to the CUDA device.
			*/
			void initialiseCuda();

			/* This initialises memory on the host. No memory is allocated
			*  on the CUDA device until initialiseCuda() is called.
			*/
			Lattice(LatticeConfig configuration, int precision);
			
			~Lattice(); //destructor

			//Copies the Host LatticeObject to the device
			void copyHostToDevice();

			//Copies the Device LatticeObject to the host.
			void copyDeviceToHost();

			/* Adds a nanoparticle (np) (of type that should be derived from class Nanoparticle) to the lattice.
			*  The method will return true if successful or false if something goes wrong!
			*/
			bool add(Nanoparticle* np);

			/* This method returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
			 * boundary conditions of a LatticeObject (theLattice).
			*/
			DirectorElement* getN(int xPos, int yPos);
			
			/* This sets the state of the lattice to one of the initialState presets.
			*  This only affects the lattice on the host. To pass this change to the lattice 
			*  on the device you should call copyHostToDevice() afterwards.
			*/
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
			void translatedUnitVectorDump(enum Lattice::dumpMode mode, FILE* stream) const;

			
			/* Calculate the Free energy per unit area in the cell located at (xPos,yPos) in the lattice on the HOST!
			*/
			double calculateEnergyOfCell(int xPos, int yPos);
			
			/* Calculate the Free energy of the lattice on the HOST!
			*/
			double calculateTotalEnergy();

			int getDumpPrecision() { return DUMP_PRECISION;}

	};


	/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
	*
	*/
	double calculateCosineBetween(DirectorElement* a, DirectorElement* b);

	/* Flips a DirectorElement (vector in physics sense) in the opposite direction
	*
	*/
	void flipDirector(DirectorElement* a);

		
	/* This function returns the correct modulo for dealing with negative a. Note % does not!
	 *
	 * mod(a,b) = a mod b
	*/
	inline int mod(int a, int b);

	#define TWO_D_LATTICE 1	
#endif
