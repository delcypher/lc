/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdio.h>
#include <stdlib.h>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"


int main()
{
	LatticeConfig configuration;
	
	//setup lattice parameters
	configuration.width =10;
	configuration.height=10;
	//set initial director alignment
	configuration.initialState = RANDOM;

	//set boundary conditions
	configuration.topBoundary = BOUNDARY_PARALLEL;
	configuration.bottomBoundary = BOUNDARY_PERPENDICULAR;
	configuration.leftBoundary = BOUNDARY_PERIODIC;
	configuration.rightBoundary = BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice
	LatticeObject* nSystem = latticeInitialise(configuration);
	
	if(nSystem == NULL)
	{
		fprintf(stderr,"Error: Couldn't construct lattice.");
		return 1;
	}

	//do stuff with lattice
	latticeDump(nSystem);

	//remove lattice
	latticeFree(nSystem);

	return 0;
}
