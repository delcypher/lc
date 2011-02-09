#include <cstdio>
#include <cstdlib>
#include "../devicemanager.h"
#include "../lattice.h"
#include "../dev_lattice.cuh"
#include "../dev_differentiate.cuh"

#include "exitcodes.h"

/* Test harness to test the mod() & dev_mod()  functions
*  ./host-mod-test <lower_range> <upper_range> <modulo>
*/

//CUDA kernels can't call external object functions so include them here
#include "../dev_lattice.cu"
#include "../dev_differentiate.cu"

//kernel to use dev_mod()
__global__ void kernel(int* input, int* output,int modulo);

//global arrays and device pointers
int* host_input;
int* host_output;
int* devices_output;
int* dev_input;
int* dev_output;

//function to free memory on host and device
void cleanup()
{
	//free memory
	deviceErrorHandle( cudaFree(dev_input) );
	deviceErrorHandle( cudaFree(dev_output) );
	free(host_input);
	free(host_output);
	free(devices_output);
}


int main(int n, char* argv[])
{
	int lowRange,highRange, modulo;
	//get arguments
	if (n!=4)
	{
		fprintf(stderr,"Need arguments <lower_range> <upper_range> <modulo>\n");
		exit(TH_BAD_ARGUMENT);
	}

	lowRange = atoi(argv[1]);
	highRange = atoi(argv[2]);
	modulo = atoi(argv[3]);

	if (lowRange >= highRange)
	{
		fprintf(stderr,"<upper_range> must be greater than <lower_range>\n");
		exit(TH_BAD_ARGUMENT);
	}
	
	int numberOfItems = highRange - lowRange +1;

	printf("Low:%d , High:%d, Modulo:%d\n",lowRange,highRange,modulo);

	//allocate memory  and initialise for array of numbers
	host_input = (int*) calloc(numberOfItems,sizeof(int));
	host_output = (int*) calloc(numberOfItems,sizeof(int));
	devices_output = (int*) calloc(numberOfItems,sizeof(int));

	//fill host input array
	for(int counter=0; counter < numberOfItems; counter++)
	{
		host_input[counter] = lowRange + counter;
	}

	//allocate space on device for input & output arrays
	
	//let CUDA run time automatically pick device instead of using cudaSetDevice()
	//deviceErrorHandle( cudaSetDevice(1) );

	deviceErrorHandle( cudaMalloc( (void**) &dev_input ,sizeof(int)*numberOfItems) );
	deviceErrorHandle( cudaMalloc( (void**) &dev_output ,sizeof(int)*numberOfItems) );

	//copy arrays to device
	deviceErrorHandle ( cudaMemcpy(dev_input,host_input,sizeof(int)*numberOfItems,cudaMemcpyHostToDevice) );
	deviceErrorHandle ( cudaMemcpy(dev_output,host_output,sizeof(int)*numberOfItems,cudaMemcpyHostToDevice) );

	//use host mod() function
	for(int counter=0; counter < numberOfItems; counter++)
	{
		host_output[counter] = mod(host_input[counter],modulo);
	}


	//use dev_mod() on device()
	kernel<<<numberOfItems,1>>>(dev_input,dev_output,modulo);

	//copy device's output (dev_output) on to devices_output so we can do comparison
	deviceErrorHandle( cudaMemcpy(devices_output,dev_output,sizeof(int)*numberOfItems,cudaMemcpyDeviceToHost) );

	//Display input to output
	for(int counter=0; counter < numberOfItems; counter++)
	{
		
		printf("mod(%d,%d) = %d ",host_input[counter],modulo,host_output[counter]);
		printf(", dev_mod(%d,%d) = %d \n",host_input[counter],modulo,devices_output[counter]);

		if(host_output[counter] != devices_output[counter])
		{
			printf("FAIL!");
			cleanup();
			exit(TH_FAIL);
		}
	}
	
	cleanup();
	
	return TH_SUCCESS;
}

__global__ void kernel(int* input, int* output, int modulo)
{
	output[blockIdx.x] = dev_mod(input[blockIdx.x],modulo);
}
