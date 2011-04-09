/* Header file for Lattice Class & LatticeObject functions, structs & enums
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef TWO_D_LATTICE
	
	#include "common.h"
	#include "directorelement.h"
	#include "nanoparticle.h"
	


	/* LatticeConfig is used to hold initial configuration parameters
	* for a lattice and should be passed to Lattice::Lattice();
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

			/* K1_EQUAL_K3, K1_DOMINANT & K3_DOMINANT should be used in conjunction with
			*  param.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL
			*  param.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR
			*
			*  These are the minimum free energy configurations for the analytical
			*  solutions assuming the director is only a function of y
			*  using the above boundary conditions and assuming the behaviour of K_1 and K_3.
			*/
			K1_EQUAL_K3,
			K1_DOMINANT,
			K3_DOMINANT,
			
			/* LAPLACE_BOX_RIGHT & LAPLACE_BOX_LEFT should be used with
			* param.topBoundary = LatticeConfig::BOUNDARY_PARALLEL
			* param.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL
			* param.leftBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR
			* param.rightBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR
			*
			* In the report this is the "2D box with homogenous surface alignment"
			*
			* These are the configurations that correspond to the minimum free energy configuration
			* of the director assuming K_1 = K_3 and that the director is a function of x & y with the 
			* above boundary conditions.
			*
			*/
			LAPLACE_BOX_RIGHT,
			LAPLACE_BOX_LEFT,

			/* LAPLACE_BOX2_RIGHT & LAPLACE_BOX2_LEFT should be used with
			* param.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR
			* param.bottomBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR
			* param.leftBoundary = LatticeConfig::BOUNDARY_PARALLEL
			* param.rightBoundary = LatticeConfig::BOUNDARY_PARALLEL
			*
			* In the report this is the "2D box with homeotropic surface alignment"
			*
			* These are the configurations that correspond to the minimum free energy configuration
			* of the director assuming K_1 = K_3 and that the director is a function of x & y with the 
			* above boundary conditions.
			*
			*/
			LAPLACE_BOX2_LEFT,
			LAPLACE_BOX2_RIGHT


		} initialState;

		/* Monte Carlo and coning algorithm parameters */

		double iTk;//Inverse "Temperature" of lattice for Monte Carlo algorithm
		unsigned long mStep; //The current Monte Carlo step number
		int acceptCounter, rejectCounter; //Counters used for coning algorithm
		double aAngle; //The current acceptance angle
		double desAcceptRatio; //The Desired acceptance ratio
		
		/* This is the seed used for the random number generator. If running Lattice::Lattice(LatticeConfig state)
		*  and state.initialState == LatticeConfig::RANDOM then it will used as the random initialisation seed.
		*  The sim-state tool later overwrites this value with its own used for the simulation.
		*
		*/
		unsigned long randSeed; 

	} LatticeConfig;


	class Lattice
	{
		private:
			int mNumNano; //The number of nanoparticles associated with this lattice
			Nanoparticle** mNanoparticles; //An array of pointers to the nanoparticles associated with this lattice
			bool badState;

			//Define the 2D lattice array (we use a linear memory block however)
			DirectorElement* lattice;
			
			const bool constructedFromFile;// Used to indicate which constructor was called
			const DirectorElement PARALLEL_DIRECTOR;
			const DirectorElement PERPENDICULAR_DIRECTOR;
			
			/* This DUMMY_DIRECTOR exists so that a pointer to something that isn't
			*  NULL is returned by getN() and setN() if an invalid point in the lattice
			*  is requested.
			*
			*/
			const DirectorElement DUMMY_DIRECTOR; 

			/* The CORNER_DIRECTOR defines the director in the undefined corner of the lattice of size wxh.
			*  It is used for points in the lattice (-1,-1), (-1,w), (h,-1) & (w,h) when NO periodic boundary conditions are present.
			*
			*  It can be set to any unit vector that won't cause a discontinuity in derivitives between vectors (1,0) or (0,1). That is
			*  to say it should NOT be set to (1,0), (0,1) or (0,0). A value of (1/sqrt(2),1/sqrt(2)) is usually chosen.
			*/
			const DirectorElement CORNER_DIRECTOR;


			//Helper function of calculateEnergyOfCell()
			double calculateCosineBetween(const DirectorElement* C, const DirectorElement* O, const double& flipSign) const;


			/* Calculates the nth (int n) term of the fourier series that gives the angle the director makes
			*  with the x-axis for a 2D box with 
			*
			* homeotropic (LAPLACE_BOX2_LEFT, LAPLACE_BOX2_RIGHT)
			* or 
			* homogeneous (LAPLACE_BOX_LEFT, LAPLACE_BOX_RIGHT) 
			* surface alignment on all sides:
			* 
			* THIS IS A HELPER FUNCTION FOR reInitialise() and should NOT call it yourself, that's why its private damn it!
			*/
			double calculateLaplaceAngleTerm(int xPos,int yPos,int n,enum LatticeConfig::latticeState solutionType);
		public:
			//Lattice Parameters
			LatticeConfig param;
			
			/* This enum defines different angular restriction regions for the restrictAngularRange() method.
			*    y
			* TL | TR
			* ___|___x    The four quardrants : TL (Top left), BL (Bottom Left), TR (Top Right) & BR (Bottom Right)
			*    |    
			* BL | BR
			*
			* These quadrants are used to define two different angular restriction regions
			*
			* REGION_RIGHT : Restrict DirectorElements to be in TR & BR . Angular restriction (-PI/2,PI/2] w.r.t to x-axis
			* REGION_TOP : Restrict DirectorElements to be in TL & TR . Angular restriction [0,PI) w.r.t to x-aix
			*
			*/
			enum angularRegion
			{
				REGION_RIGHT,
				REGION_TOP

			};

			/* This initialises the lattice from a LatticeConfig struct
			* 
			*/
			Lattice(LatticeConfig configuration);
		
			/* Initialise lattice from a saved state file generated by
			*  the saveState() method.
			*/
			Lattice(const char* filename);

			~Lattice(); //destructor


			/* Adds a nanoparticle (np) (of type that should be derived from class Nanoparticle) to the lattice.
			*  The method will return true if successful or false if something goes wrong!
			*/
			bool add(Nanoparticle& np);

			/* This method returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
			 * boundary conditions of a LatticeObject (theLattice). Note that if you wish to change the director value at point (x,y)
			 * you should use getN().
			*/
			const DirectorElement* getN(int xPos, int yPos) const;
		
			/* This is like getN() accept it allows you to change the director value at point (x,y).
			*  Use with CAUTION!
			*/
			DirectorElement * setN(int xPos, int yPos);

			/* This sets the state of the lattice to one of the initialState presets.
			*  Note: This only affects lattice cells that aren't marked as Nanoparticles.
			*/
			void reInitialise(enum LatticeConfig::latticeState initialState);

			/* These are the different dumping modes used by translatedUnitVectorDump()
			*  EVERYTHING - NANOPARTICLES AND NORMAL LATTICE POINTS
			*  PARTICLES - NANOPARTICLES ONLY
			*  NOT_PARTICLES - NORMAL LATTICE POINTS ONLY
			*/
			enum dumpMode
			{
				EVERYTHING, //prints entire lattice but not the boundary
				PARTICLES, //prints just the particles
				NOT_PARTICLES, //prints everything but the particles and boundary
				BOUNDARY //prints the boundary	
			};

			/*
			* This method outputs the current state of the lattice to filestream stream (e.g. stdout) in a format
			* compatible with the GNUplot script "ldump.gnu". The director field is plotted as
			* unit vectors that are translated so that the centre of the vector rather than the end of the vector
			* is plotted at point (xPos,yPos).
			*/
			void nDump(enum Lattice::dumpMode mode, std::ostream& stream) const;
			
			/* This method outputs the current state of the lattice to a filestream stream in a format compatible
			*  with the GNUplot script "ildump.gnu". It outputs three indexes.
			*  index 0 - BOUNDARY
			*  index 1 - NOT_PARTICLES 
			*  index 1 - PARTICLES
			*/
			void indexedNDump(std::ostream& stream) const;
			
			/* Dump a description of lattice parameters to out Output stream. E.g. std::cout
			*
			*/
			void dumpDescription(std::ostream& stream) const;

			/* Calculate the Free energy per unit volume in the cell located at (xPos,yPos) in the lattice
			*/
			double calculateEnergyOfCell(int xPos, int yPos) const;
			
			/* Calculate the Free energy of the lattice.
			*/
			double calculateTotalEnergy() const;
			
			/* Calculate the Free energy of the lattice excluding contributions
			*  from nanoparticle cells.
			*/
			double calculateTotalNotNPEnergy() const;

			/* Calculate the average Free energy of a lattice cell.
			*
			*/
			double calculateAverageEnergy() const
			{
				return calculateTotalEnergy() / getArea();
			}

			/* Calculates the average free energy per unit volume of non Nanoparticle cells.
			*
			*/
			double calculateNotNPAverageEnergy() const
			{
				return calculateTotalNotNPEnergy() / (getArea() - getNanoparticleCellCount()  ) ;
			}

			//returns true if Lattice is in a bad state (usually from initialisation or add() )
			bool inBadState() const { return badState;}

			/* Saves binary state file to filename
			*/
			bool saveState(const char* filename) const;

			//Calculate the number of Nanoparticle cells in lattice.
			int getNanoparticleCellCount() const;

			//Calculate the area of the lattice (excluding boundaries)
			int getArea() const;
			
			/* Modify DirectorElements so that they are in an angular range defined by one of the enums in the 
			 * Lattice::angularRegion enum.
			 *
			 * This is done by flipping DirectorElements by PI radians (preserving uniaxial properties).
			 * This is need for calculateAverageAngle() , calculateAngularStdDev() and angleCompareWith()
			 */
			void restrictAngularRange(enum Lattice::angularRegion region);

			/* Calculate average orientation of the DirectorElements in the lattice as an angle
			*  in radians w.r.t to x-axis (anti-clockwise).
			*
			*  It is highly recommended restrictAngularRange() is called first!
			*/
			double calculateAverageAngle() const;

			/* Calculate the standard deviation of the orientation of the
			*  DirectorElements in the lattice in radians w.r.t to the x-axis (anti-clockwise)
			*
			*  It is highly recommended restrictAngularRange() is called first!
			*/
			double calculateAngularStdDev() const;

			//overloaded comparison operator
			bool operator==(const Lattice & rhs) const;
			
			//overloaded != operator
			bool operator!=(const Lattice & rhs) const;
			
			/* Output energy comparision data between this lattice and a lattice in an analytical minimum
			*  energy state specified by "state". Note supported states are the following:
			*
			*  PARALLEL_X, (T,B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC OR BOUNDARY_PARALLEL)
			*  PARALLEL_Y, (T,B BOUNDARY_PERPENDICULAR ; L,R BOUNDARY_PERIODIC OR BOUNDARY_PERPENDICULAR)
			*  K1_EQUAL_K3, (T BOUNDARY_PERPENDICULAR ; B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC)
			*  K1_DOMINANT,(T BOUNDARY_PERPENDICULAR ; B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC)
			*  K3_DOMINANT,(T BOUNDARY_PERPENDICULAR ; B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC)
			*
			*  Note: The boundary conditions expected for a state are shown in brackets with 
			*  T = Top Boundary condition
			*  B = Bottom Boundary condition
			*  L = Left Boundary condition
			*  R = Right Boundary condition
			*
			*  For these states the following energy per unit volume (E) is expected to the same in all cells (uniform)
			*  and is expected to have the following values for different states ( note height is the lattice height)
			*  
			*  PARALLEL_X & PARALLEL_Y : E=0
			*  K1_EQUAL_K3 : E = (PI^2)/(8*(height +1)^2);
			*  K1_DOMINANT : E = 1/( 2*(height +1)^2 )  (assumes k_1 = 1)
			*  K3_DOMINANT : E = (k_3)/( 2*(height +1)^2  )
			*
			*  This information is outputted to an std::ostream "stream".
			*
			*  acceptibleError is the acceptible absolute error used for comparisions.
			*
			*  The return value is true if ALL cells of this lattice match the analytical situation specified by "state"
			*  within "acceptibleError".
			* 
			*  The return value is false if one or more cells don't match within "acceptibleErroe" or if 
			*  there is a problem with the passed arguments.
			*
			*  Two comparisions are done
			*  1. Each cell energy value is compared to the analytical value (for "state") for that cell
			*  2. Each cell energy value is compared with the energy of the first cell. This is done because
			      for the analytical solutions energy per unit volume is uniform.
			*/
			bool energyCompareWith(enum LatticeConfig::latticeState state, std::ostream& stream, double acceptibleError) const;

			/* NOTE: This method is NOT const because it calls restrictAngularRange(REGION_RIGHT) in order to sort out 
			*        DirectorElement orientation so we can do an meaningful comparision.
			*
			* Output angular comparision data between this lattice and a lattice in an analytical minimum
			*  energy state specified by "state". Note supported states are the following:
			*
			*  PARALLEL_X, (T,B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC OR BOUNDARY_PARALLEL)
			*  PARALLEL_Y, (T,B BOUNDARY_PERPENDICULAR ; L,R BOUNDARY_PERIODIC OR BOUNDARY_PERPENDICULAR)
			*  K1_EQUAL_K3, (T BOUNDARY_PERPENDICULAR ; B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC)
			*  K1_DOMINANT,(T BOUNDARY_PERPENDICULAR ; B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC)
			*  K3_DOMINANT,(T BOUNDARY_PERPENDICULAR ; B BOUNDARY_PARALLEL ; L,R BOUNDARY_PERIODIC)
			*
			*  Note: The boundary conditions expected for a state are shown in brackets with 
			*  T = Top Boundary condition
			*  B = Bottom Boundary condition
			*  L = Left Boundary condition
			*  R = Right Boundary condition
			*
			*  For these states the following orientation is expected.
			*  The configuration is n = (cos(theta), sin(theta), 0) = ( x , y , z )
			*
			*  PARALLEL_X : theta = 0
			*  PARALLEL_Y : theta = PI/2
			*  K1_EQUAL_K3: theta = (PI/2)(y + 1)/(height +1)
			*  K1_DOMINANT: theta = arcsin( (y + 1)/(height +1) )
			*  K3_DOMINANT: theta = arccos( 1 - (y + 1)/(height +1) )
			*
			*
			*  This information is outputted to an std::ostream "stream".
			*
			*  acceptibleError is the acceptible absolute error used for comparisions.
			*
			*  The return value is true if ALL cells of this lattice match the expected angular orientation of a
			*  lattice (within "acceptibleError") in the analytical minimum energy state (specified by "state").
			* 
			*  The return value is false if one or more cells don't match within "acceptibleErroe" or if 
			*  there is a problem with the passed arguments.
			*
			*  One type of comparision is done:
			*  1. The angular orientation of each of cell is compared to the expected angular orientation for the
			*     specified ( "state" ) analytical minimum energy state. 
			*/
			bool angleCompareWith(enum LatticeConfig::latticeState state, std::ostream& stream, double acceptibleError);

	};

	#define TWO_D_LATTICE 1	
#endif
