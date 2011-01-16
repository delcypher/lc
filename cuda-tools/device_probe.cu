#include <stdio.h>
#include <cuda_runtime.h>

/* This small program is used to probe currently available CUDA devices
*  and lists their computer capability.
*
*/

int main()
{
	int deviceCount=0;
	int counter=0;
	cudaDeviceProp deviceProperties;

	cudaGetDeviceCount(&deviceCount);
	
	printf("Found %d CUDA devices.\n", deviceCount);

	for(counter=0; counter < deviceCount; counter++)
	{
		cudaGetDeviceProperties(&deviceProperties, counter);
		printf("[%d]Name: %s\n",counter, deviceProperties.name);
		printf("[%d]Compute capability: %d.%d\n", counter, deviceProperties.major, deviceProperties.minor);
		printf("\n\n");

	}
	


}
