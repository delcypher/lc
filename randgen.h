/* Header file for the random number generator functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef AWESOME_RND
	/* Header file for Heterogeneous random number generators
	*  
	*/

	/*
	* This generates a random number in the range [0,1].
	* This will be ran on the cpu.
	*/
	double cpuRnd();

	/*
	* This will initialise the random seed for the cpuRnd() function
	* in a way that should hopefully be different every time
	*/
	void cpuSetRandomSeed();

	#define AWESOME_RND

#endif
