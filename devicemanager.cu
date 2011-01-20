/* CUDA device manager by Dan Liew & Alex Allen */
#include <stdio.h>
#include "devicemanager.h"

bool deviceErrorHandle(cudaError_t error)
{
	switch(error)
	{
		case cudaSuccess:
			//fprintf(stderr,"CUDA Success.\n");
			return true;

		
		default:
			fprintf(stderr,"CUDA Error: %s.\n",cudaGetErrorString(error));
			return false;
	}

}

int pickGPU(int maj, int min)
{
        int dev;
	int deviceCount;

        // Create cudaDeviceProp with the specifications we want
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(cudaDeviceProp));
        prop.major = maj;
        prop.minor = min;

	//get the number of CUDA devices.	
	deviceErrorHandle( cudaGetDeviceCount(&deviceCount) );

	if(deviceCount==0)
	{
		fprintf(stderr,"Error: No CUDA devices available.\n");
		exit(1);
	}

        // Use built in functions to pick a device with those specs
        deviceErrorHandle( cudaChooseDevice(&dev, &prop) );
        deviceErrorHandle( cudaSetDevice(dev) );

        return dev;
}

