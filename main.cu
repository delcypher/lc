/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "dev_lattice.cuh"
#include "devicemanager.h"

//a bit of a hack, removed later if possible
#include "dev_lattice.cu"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"

const int threadDim = 16;

__global__ void kernel(LatticeObject *baconlatticetomato);

int main()
{
	LatticeConfig configuration;
	
	//setup lattice parameters
	configuration.width = threadDim*10;
	configuration.height= threadDim*10;

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

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	//add nanoparticle to lattice
	nSystem.add(&particle1);

	//Initialise the lattice on the device
	//nSystem.initialiseCuda();
	
	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	nSystem.indexedNDump(stdout);

	//Alex's wizardry
	//dim3 blocks(configuration.width/threadDim, configuration.height/threadDim);
	//dim3 threads(threadDim, threadDim);
	//kernel<<<blocks, threads>>>(nSystem.devLatticeObject);
	//nSystem.copyDeviceToHost();
	cout << "\n\n\n";

	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::EVERYTHING,stdout);

	double energy = nSystem.calculateTotalEnergy();
	printf("#Energy of lattice:%.20f \n",energy);


	return 0;
}

__global__ void kernel(LatticeObject *baconlatticetomato)
{
	int x = threadIdx.x + blockIdx.x * gridDim.x;
	int y = threadIdx.y + blockIdx.y * gridDim.y;

	DirectorElement* direl = latticeGetN(baconlatticetomato,x,y);

	if(direl->isNanoparticle ==0)
	{
		direl->x = 1;
		direl->y = 0;
	}


}
