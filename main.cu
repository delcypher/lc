/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdio.h>
#include <stdlib.h>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"

int main()
{
	LatticeConfig configuration;
	
	//setup lattice parameters
	configuration.width =20;
	configuration.height=20;
	//set initial director alignment
	configuration.initialState = LatticeConfig::RANDOM;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice
	LatticeObject* nSystem = latticeInitialise(configuration);
	
	if(nSystem == NULL)
	{
		fprintf(stderr,"Error: Couldn't construct lattice.\n");
		return 1;
	}

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	//add nanoparticle to lattice
	latticeAdd(nSystem,&particle1);

	//Dump the current state of the lattice to standard output.
	latticeTranslatedUnitVectorDump(nSystem, EVERYTHING);
	//remove lattice

	double energy = latticeCalculateTotalEnergy(nSystem);
	printf("Energy of lattice:%.20f \n",energy);

	latticeFree(nSystem);

	return 0;
}
