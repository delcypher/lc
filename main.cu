/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdio.h>
#include <stdlib.h>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "devicemanager.h"

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

	//pick a GPU to use
	int deviceSelected = pickGPU(1,3);
	if(deviceSelected==-1)
	{
		exit(1);
	}

	printf("#Selecting CUDA device:%d \n",deviceSelected);

	//create lattice object
	Lattice nSystem = Lattice(configuration);

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	//add nanoparticle to lattice
	nSystem.add(&particle1);

	//Initialise the lattice on the device
	nSystem.initialiseCuda();

	//Dump the current state of the lattice to standard output.
	nSystem.translatedUnitVectorDump(Lattice::EVERYTHING);

	double energy = nSystem.calculateTotalEnergy();
	printf("Energy of lattice:%.20f \n",energy);


	return 0;
}
