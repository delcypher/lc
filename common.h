/* common.h contains things needed by multiple files
   By Dan Liew & Alex Allen
*/

#ifndef COMMON_2D

	//PI to 58d.p taken from Wolfram alpha ( http://www.wolframalpha.com/input/?i=pi )
	const double PI=3.1415926535897932384626433832795028841971693993751058209749;

      	/* This function returns the correct modulo for dealing with negative a. Note % does not!
         *
         * mod(a,b) = a mod b
        */
        inline int mod(int a, int b)
        {
                return (a%b + b)%b;
        }
	
	//Note the iostream dependency comes from need for the ios_base::fmtflags type
	#include <iostream>
	
	/* Set the output format:
	*  0 - significant figures
	*  std::ios::fixed - Fixed decimal places
	*  std::ios::scientific - Scientific notation
	*/
	const std::ios_base::fmtflags STREAM_FLOAT_FORMAT = std::ios::fixed;
	
	const int FILE_PRECISION=50; //Precision (d.p) used for saving state to files.
	const int STDOE_PRECISION=10; //Precision (d.p) used for outputing to std::cout and std::cerr
	#define COMMON_2D 1

#endif
