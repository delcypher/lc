/* This is a test for initialising the lattice on the host by visual expection.
*  The standard output of this program can be given to gnuplot script "ildump.gnu"
*
*  ./test-host-initialise <width> <height> 
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



int main(int n, char* argv[])
{

	if(n!=3)
	{
		cerr << "Usage: " << argv[0] << " <width> <height> \n";
		exit(TH_BAD_ARGUMENT);
	}

	LatticeConfig configuration;

	configuration.width = atoi(argv[1]);
	configuration.height= atoi(argv[2]);

	//set initial director alignment
	configuration.initialState = LatticeConfig::TOP_PERP_BOT_PAR;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	cout << "# Creating nanoparticle" << endl; 

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	cout << "# Adding nanoparticle" << endl;

	//add nanoparticle to lattice
	nSystem.add(&particle1);

	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	nSystem.indexedNDump(stdout);


	return TH_SUCCESS;
}


