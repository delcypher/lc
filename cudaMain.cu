#include <iostream>
using namespace std;

const int width = 160;
const int height= 160;
const int threadDim = 16;

__global__ void kernel(LatticeObject *nSystem, double *energy);

int main()
{
	// Create lattice configuration, CPU and GPU energy variables
	latticeConfig configuration;
	int noBlocks = width/threadDim * height/threadDim;
	double energy[noBlocks], *dev_energy, totalEnergy=0;
	cudaMalloc((void**) dev_energy, noBlocks*sizeof(double));

	// Configure Lattice
	configuration.width = width;
	configuration.height= width;
	configuration.initialState = LatticeConfig::RANDOM;
	configuration.topBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.beta = 3.5;

	// Create lattice
	LatticeObject *dev_nSystem, *nSystem = latticeInitialise(configuration);
	if(nSystem == NULL)
	{
		cerr << "Error: Couldn't construct lattice" << endl;
		return 1;
	}

	// Copy lattice to GPU memory 
	latticeCopy(dev_nSystem, nSystem, cudaMemcpyHostToDevice);

	// Run kernel to calculate system energy
	dim3 threads(threadDim, threadDim);
	dim3 blocks(width/threadDim, height/threadDim);
	kernel<<<blocks, threads>>>(dev_nSystem, dev_energy);

	// Copy energy from GPU to CPU
	cudaMemcpy(&energy, dev_energy, noBlocks*sizeof(double), cudaMemcpyDeviceToHost);

	// Sum remaining energy
	for(int i=0;i<noBlocks;i++)
	{
		totalEnergy += energy[i];
	}

	// Output things of interest
	latticeHalfUnitVectorDump(nSystem);
	cout << "Energy of lattice: " << totalEnergy << endl;

	// It just needs doing really!
	latticeFree(nSystem);
	cudaLatticeFree(dev_nSystem);
	cudaFree(dev_energy);

	return 0;
}

__global__ void kernel(LatticeObject *nSystem, double *energy)
{
	int xpos = threadIdx.x + blockIdx.x*blockDim.x;
	int ypos = threadIdx.y + blockIdx.y*blockDim.y;
	int threadID = threadIdx.x + blockDim.x*threadIdx.y;
	__shared__ double cache[blockDim.x*blockDim.y];

	// Calculate energy
	cache[threadID] = latticeCalculateEnergyOfCell(nSystem, xpos, ypos);

	// All code from here is data reduction as in CUDA by Example
	// Wait until all threads have finished for thread reduction
	__syncThreads();

	// Add all the data in the block is summed in parallel
	int i = 0.5*(blockDim.x * blockDim.y);
	while(i != 0 )
	{
		if(threadID < i) cache[threadId] += cache[threadID + i];
		__syncthreads();
		i /= 2;
	}

	// Only the energy for the whole block is output
	if(threadID == 0) energy[blockIdx.x + gridDim.x*blockIdx.y] = cache[0];
}
