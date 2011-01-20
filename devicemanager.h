/* Header file for CUDA device manager by Dan Liew & Alex Allen */

#ifndef DEVICE_ERROR_HANDLE

	/* Handle CUDA function errors.
	*  If a success will return true with no output.
	*  If something went wrong will return false and print an error message on standard error.
	*/
	bool deviceErrorHandle(cudaError_t error);

	/* Picks the CUDA device that matches the compute capability maj.min (e.g. 1.3).
	*  It returns the device number picked.
	*  If no device is available it will cause the application to exit and print an error message.
	*/
	int pickGPU(int maj, int min);

	#define DEVICE_ERROR_HANDLE
#endif
