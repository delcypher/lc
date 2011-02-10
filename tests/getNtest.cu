#include "exitcodes.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "dev_lattice.cuh"
#include "dev_differentiate.cuh"
#include "devicemanager.h"

/* CUDA does not support external (i.e. to other objects used in iterative compilation) function calls.
*  So we must include the implementations of the device functions in the same file (and hence object) as
*  our kernel function(s). This slows down compilation but at least it works!
*/
#include "dev_lattice.cu"
#include "dev_differentiate.cu"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"

const int threadDim = 16;

__global__ void kernel(LatticeObject *baconlatticetomato, double *dataOne, double *dataTwo, int xoffset, int yoffset);

/*   This program checks for each thread (and threadspace maps perfectly onto liquid crystal grid), that getN and 
     latticeGetN return the same cell by comparing the x and y values they output. If nothing goes wrong then
     the program will appear to not do anything. Else it will print out some blurb which you can either ask Alex
     about, or you can read the code and it'll be obvious. You're reading the code anyway so that's probably the
     way forward.
*/
int main(int argc, char *argv[])
{
	if(argc<3)
	{
		cout << "I want two command line arguments! offset in x first, then offset in y." << endl;
		exit(TH_BAD_ARGUMENT);
	}
	if(argc>3)
	{
		cout << "You suck, I only wanted two arguements." << endl;
		exit(TH_BAD_ARGUMENT);
	}

	int xoffset = atoi(argv[1]);
	int yoffset = atoi(argv[2]);

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

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	//Initialise the lattice on the device
	nSystem.initialiseCuda();
	
        //Alex's wizardry
        int xblocks = configuration.width/threadDim, yblocks = configuration.height/threadDim;
	int arraySize = xblocks*threadDim * yblocks*threadDim;
        dim3 blocks(xblocks, yblocks);
        dim3 threads(threadDim, threadDim);

        double dataOne[arraySize], *dev_dataOne, dataTwo[arraySize], *dev_dataTwo;
        deviceErrorHandle(cudaMalloc((void**) &dev_dataOne, arraySize*sizeof(double)));
        deviceErrorHandle(cudaMalloc((void**) &dev_dataTwo, arraySize*sizeof(double)));

        kernel<<<blocks, threads>>>(nSystem.devLatticeObject, dev_dataOne, dev_dataTwo, xoffset, yoffset);
   
	cudaMemcpy(dataOne, dev_dataOne, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dataTwo, dev_dataTwo, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	
	double cpux[arraySize], cpuy[arraySize];
	int x,y;
	for(y=0; y<yblocks*threadDim; y++)
	{
		for(x=0; x<xblocks*threadDim; x++)
		{
			cpux[x+y*xblocks*threadDim] = nSystem.getN(x+xoffset,y+yoffset)->x;
			cpuy[x+y*xblocks*threadDim] = nSystem.getN(x+xoffset,y+yoffset)->y;
		}
	}

	bool makedanhappy = false;
	for(int i=0; i<arraySize; i++)
	{
		if( dataOne[i] - cpux[i]  || dataTwo[i] - cpuy[i])
		{
			cout << i << " " << dataOne[i] - cpux[i] << " " << dataTwo[i] - cpuy[i] << endl;
			makedanhappy = true;
		}
	}

	if(makedanhappy)
	{
		return TH_FAIL;
	}
	else
	{
		return TH_SUCCESS;
	}
}

__global__ void kernel(LatticeObject *baconlatticetomato, double *dataOne, double *dataTwo, int xoffset, int yoffset)
{
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        dataOne[x + y*blockDim.x*gridDim.x] = latticeGetN(baconlatticetomato, x+xoffset, y+yoffset)->x;
        dataTwo[x + y*blockDim.x*gridDim.x] = latticeGetN(baconlatticetomato, x+xoffset, y+yoffset)->y;
}
