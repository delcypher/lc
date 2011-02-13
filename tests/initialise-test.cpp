/* This is a test for initialising the lattice on the host by visual expection.
*  The standard output of this program can be given to gnuplot script "ildump.gnu"
*
*  ./test-host-initialise <width> <height> <initial_state_enum>
*
*  It is not actually much of a test harness as the program "should" never return
*  TH_FAIL
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "exitcodes.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"


int main(int n, char* argv[])
{

	if(n!=4)
	{
		cerr << "Usage: " << argv[0] << " <width> <height> <initial_state_enum>\n";
		exit(TH_BAD_ARGUMENT);
	}

	LatticeConfig configuration;

	configuration.width = atoi(argv[1]);
	configuration.height= atoi(argv[2]);

	//set initial director alignment
	configuration.initialState = (LatticeConfig::latticeState) atoi(argv[3]);

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);


	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(7,7,5,CircularNanoparticle::PARALLEL);

	//create elliptical nanoparticle
	//(xCentre,yCentre, a, b, theta, boundary)
	EllipticalNanoparticle particle2 = EllipticalNanoparticle(20,20,10,5,PI/4,EllipticalNanoparticle::PARALLEL);

	//add nanoparticles to lattice
	nSystem.add(&particle1);
	nSystem.add(&particle2);

	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	nSystem.indexedNDump(stdout);


	return TH_SUCCESS;
}


