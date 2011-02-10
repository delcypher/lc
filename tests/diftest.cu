#include <iomanip>
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

const int threadDim = 8;

__global__ void kernel(LatticeObject *blt, double *dNxdxF, 
                                           double *dNxdxB, 
                                           double *dNxdyF, 
                                           double *dNxdyB,
					   double *dNydxF,
					   double *dNydxB,
					   double *dNydyF,
					   double *dNydyB);

main(int argc, char *argv[])
{
	// boring set up bit
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

	// interesting test bit
	int xblocks = configuration.width/threadDim, yblocks = configuration.height/threadDim;
        int arraySize = xblocks*threadDim * yblocks*threadDim;
        dim3 blocks(xblocks, yblocks);
        dim3 threads(threadDim, threadDim);

	double dNxdxF[arraySize], dNxdxB[arraySize], dNxdyF[arraySize], dNxdyB[arraySize], dNydxF[arraySize], dNydxB[arraySize], dNydyF[arraySize], dNydyB[arraySize]; 
	double cpudNxdxF[arraySize], cpudNxdxB[arraySize], cpudNxdyF[arraySize], cpudNxdyB[arraySize], cpudNydxF[arraySize], cpudNydxB[arraySize], cpudNydyF[arraySize], cpudNydyB[arraySize]; 
	double *dev_dNxdxF, *dev_dNxdxB, *dev_dNxdyF, *dev_dNxdyB, *dev_dNydxF, *dev_dNydxB, *dev_dNydyF, *dev_dNydyB;

	cudaMalloc((void**) &dev_dNxdxF , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNxdxB , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNxdyF , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNxdyB , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNydxF , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNydxB , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNydyF , arraySize*sizeof(double));
	cudaMalloc((void**) &dev_dNydyB , arraySize*sizeof(double));

	kernel<<<blocks,threads>>>(nSystem.devLatticeObject, dev_dNxdxF, dev_dNxdxB, dev_dNxdyF, dev_dNxdyB, dev_dNydxF, dev_dNydxB, dev_dNydyF, dev_dNydyB);

	cudaMemcpy(dNxdxF, dev_dNxdxF, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNxdxB, dev_dNxdxB, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNxdyF, dev_dNxdyF, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNxdyB, dev_dNxdyB, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNydxF, dev_dNydxF, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNydxB, dev_dNydxB, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNydyF, dev_dNydyF, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dNydyB, dev_dNydyB, arraySize*sizeof(double), cudaMemcpyDeviceToHost);

	for(int y=0; y<yblocks*threadDim; y++)
	{
		for(int x=0; x<xblocks*threadDim; x++)
		{
			cpudNxdxF[x+y*xblocks*threadDim] = dNxdx_F(&nSystem, x, y);
			cpudNxdxB[x+y*xblocks*threadDim] = dNxdx_B(&nSystem, x, y);
			cpudNxdyF[x+y*xblocks*threadDim] = dNxdy_F(&nSystem, x, y);
			cpudNxdyB[x+y*xblocks*threadDim] = dNxdy_B(&nSystem, x, y);
			cpudNydxF[x+y*xblocks*threadDim] = dNydx_F(&nSystem, x, y);
			cpudNydxB[x+y*xblocks*threadDim] = dNydx_B(&nSystem, x, y);
			cpudNydyF[x+y*xblocks*threadDim] = dNydy_F(&nSystem, x, y);
			cpudNydyB[x+y*xblocks*threadDim] = dNydy_B(&nSystem, x, y);
		}
	}

	cout.precision(3);
	double a, b, c, d, e, f, g, h;
	bool happydan = true;
	for(int i=0; i<arraySize; i++)
	{
		a = cpudNxdxF[i] - dNxdxF[i];
		b = cpudNxdxB[i] - dNxdxB[i];
		c = cpudNxdyF[i] - dNxdyF[i];
		d = cpudNxdyB[i] - dNxdyB[i];
		e = cpudNydxF[i] - dNydxF[i];
		f = cpudNydxB[i] - dNydxB[i];
		g = cpudNydyF[i] - dNydyF[i];
		h = cpudNydyB[i] - dNydyB[i];

		if(a || b || c || d || e || f || g || h)
		{
			happydan = false;
			cout << setw(4) << i << " ";
			cout << setw(4) << a << " ";
			cout << setw(4) << b << " ";
			cout << setw(4) << c << " ";
			cout << setw(4) << d << " ";
			cout << setw(4) << e << " ";
			cout << setw(4) << f << " ";
			cout << setw(4) << g << " ";
			cout << setw(4) << h << endl;
		}
	}

	if(happydan)
	{
		return TH_SUCCESS;
	}
	else
	{
		return TH_FAIL;
	}

}

__global__ void kernel(LatticeObject *blt, double *dNxdxF, 
                                           double *dNxdxB, 
                                           double *dNxdyF, 
                                           double *dNxdyB,
					   double *dNydxF,
					   double *dNydxB,
					   double *dNydyF,
					   double *dNydyB)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int ID = x + y*blockDim.x*gridDim.x;

	dNxdxF[ID] = dev_dNxdx_F(blt, x, y);
	dNxdxB[ID] = dev_dNxdx_B(blt, x, y);
	dNxdyF[ID] = dev_dNxdy_F(blt, x, y);
	dNxdyB[ID] = dev_dNxdy_B(blt, x, y);
	dNydxF[ID] = dev_dNydx_F(blt, x, y);
	dNydxB[ID] = dev_dNydx_B(blt, x, y);
	dNydyF[ID] = dev_dNydy_F(blt, x, y);
	dNydyB[ID] = dev_dNydy_B(blt, x, y);
}
