/* Implementation of the random number generator functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdlib.h>

float cpuRnd()
{
	

	return (float) rand()/RAND_MAX;
}

float gpuRnd()
{


	return 0;
}

