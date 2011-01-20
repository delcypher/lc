/* Header file for CUDA device manager by Dan Liew & Alex Allen */

#ifdef DEVICE_ERROR_HANDLE

	/* Handle CUDA function errors.
	*  If a success will return true with no output.
	*  If something went wrong will return false and print an error message on standard error.
	*/
	bool deviceErrorHandle(cudaError_t error);

	#define DEVICE_ERROR_HANDLE
#endif
