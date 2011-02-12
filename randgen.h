/* Header file for the random number generator functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef AWESOME_RND
	// Header file for Heterogeneous random number generators
	
	// This sets the seed for the random number generator
	void sgenrand(unsigned long seed);

	// This generates a random number in the range [0,1]
	double rnd();

	// Set seed = unix time
	void setSeed();

	#define AWESOME_RND

#endif
