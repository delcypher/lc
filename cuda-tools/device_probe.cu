#include <iostream>
#include <cuda_runtime.h>
using namespace std;

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
	
	cout << "\nFound " << deviceCount << " CUDA devices.\n\n";

	for(counter=0; counter < deviceCount; counter++)
	{
		cudaGetDeviceProperties(&deviceProperties, counter);
		cout << "Device " << counter << ":\n";
		cout << "Name: " << deviceProperties.name << endl;
		cout << "Compute capability: " << deviceProperties.major << "." << deviceProperties.minor << endl;
		cout << "Multiprocessor count: " << deviceProperties.multiProcessorCount << endl;
		cout << "Maximum threads per block: " << deviceProperties.maxThreadsPerBlock << endl;
		cout << "Total device memory (MiB): " << (double) deviceProperties.totalGlobalMem / (1024*1024) << endl;
		cout << "Maximum shared memory per block (KiB): " << (double) deviceProperties.sharedMemPerBlock / 1024 << endl;
		cout << "Maximum constant memory (KiB): " << (double) deviceProperties.totalConstMem / 1024 << endl;
		cout << "\n\n";

	}
	


}
