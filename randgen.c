/* Implementation of the random number generator functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdlib.h>
#include <time.h>

float cpuRnd()
{
	return (float) rand()/RAND_MAX;
}

void cpuSetRandomSeed()
{
	srand( time(NULL));
}

float gpuRnd()
{


	return 0;
}

