/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

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

__global__ void kernel(LatticeObject *baconlatticetomato, double *blockEnergies);

int main()
{
	LatticeConfig configuration;
	FILE* fout = fopen("dump.txt", "w");

	cout << "# Setting lattice config parameters" << endl;	
	//setup lattice parameters
	configuration.width = threadDim*2;
	configuration.height= threadDim*2;

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

	printf("# Selecting CUDA device:%d \n",deviceSelected);

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	cout << "# Creating nanoparticle" << endl; 

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	cout << "# Adding nanoparticle" << endl;

	//add nanoparticle to lattice
	nSystem.add(&particle1);

	cout << "# Initialise lattice on device" << endl;

	//Initialise the lattice on the device
	nSystem.initialiseCuda();
	
	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	nSystem.indexedNDump(fout);

        //Alex's wizardry
        int xblocks = configuration.width/threadDim, yblocks = configuration.height/threadDim;
        dim3 blocks(xblocks, yblocks);
        dim3 threads(threadDim, threadDim);

	cout << "# Create variables and allocate device memory" << endl;

        double totalEnergy=0, blockEnergies[xblocks * yblocks], *dev_blockEnergies;
        deviceErrorHandle(cudaMalloc((void**) &dev_blockEnergies, sizeof(double) * xblocks * yblocks));

	cout << "# Run kernel" << endl;
        kernel<<<blocks, threads>>>(nSystem.devLatticeObject, dev_blockEnergies);
   
	cout << "# Copy energy from device to host" << endl;
	cudaMemcpy(blockEnergies, dev_blockEnergies, xblocks * yblocks * sizeof(double), cudaMemcpyDeviceToHost);
	
	cout << "# Copy nSystem from device to host" << endl; 
	nSystem.copyDeviceToHost();

	cout << "# Sum block energies" << endl;
        int i;
        for(i=0; i<xblocks*yblocks; i++)
        {
                totalEnergy+=blockEnergies[i];
        }

	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::EVERYTHING,stdout);

	double energy = nSystem.calculateTotalEnergy();
	cout << "# Energy calculated on CPU: " << energy << endl;
	cout << "# Energy calculated on GPU: " << totalEnergy << endl;
	cout << "# Block energies were: ";

	for(i=0; i<xblocks*yblocks; i++)
	{
		cout << blockEnergies[i] << " ";
	}
	cout << endl;

	fclose(fout);

	return 0;
}

__global__ void kernel(LatticeObject *baconlatticetomato, double *blockEnergies)
{
        __shared__ double energy[threadDim*threadDim];
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int blockID = blockIdx.x + blockIdx.y * gridDim.x;
        int threadID = threadIdx.x + threadIdx.y * blockDim.x;
        int j = blockDim.x * blockDim.y / 2;

        energy[threadID] = latticeCalculateEnergyOfCell(baconlatticetomato, x,y);

        // sum energy in a block
        __syncthreads();

        while(j!=0)
        {
                if(threadID<j) energy[threadID] += energy[threadID+j];
                __syncthreads();
                j/=2;
        }
        if(threadID==0) blockEnergies[blockID] = energy[0];
}
