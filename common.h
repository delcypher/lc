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
	

	#define COMMON_2D 1

#endif
