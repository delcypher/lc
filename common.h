/* common.h contains things needed by multiple files
   By Dan Liew & Alex Allen
*/

#ifndef COMMON_2D

	/* DirectorElement is a 2D vector (in physics sense)
        *  expressed in cartesian co-ordinates. Arrays of
        *  these are used to build a vector field.
        *  Included with this 2D vector is a variable identifying if the
        *  DirectorElement is a nanoparticle.
        */
        typedef struct
        {
                double x,y;
                int isNanoparticle;
        } DirectorElement;

	//PI to 58d.p taken from Wolfram alpha ( http://www.wolframalpha.com/input/?i=pi )
	const double PI=3.1415926535897932384626433832795028841971693993751058209749;

	#define COMMON_2D

#endif
