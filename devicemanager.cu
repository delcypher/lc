/* CUDA device manager by Dan Liew & Alex Allen */
#include <stdio.h>
#include "devicemanager.h"

bool deviceErrorHandle(cudaError_t error)
{
	switch(error)
	{
		case cudaSuccess:
			//fprintf(stderr,"CUDA Success.");
			return true;

		
		default:
			fprintf(stderr,"CUDA Error: CUDA %s.",cudaGetErrorString(error));
			return false;
	}

}
