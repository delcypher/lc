/* Implementation of the random number generator functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <cstdlib>
#include <ctime>

double cpuRnd()
{
	return (double) rand()/RAND_MAX;
}

void cpuSetRandomSeed()
{
	srand( time(NULL));
}

double gpuRnd()
{


	return 0;
}

