/* Header file for DirectorElement and functions that act on it */

#ifndef DIRECTOR_ELEMENT
	
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

	/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
        *
        */
        double calculateCosineBetween(const DirectorElement* a, const DirectorElement* b);

        /* Flips a DirectorElement (vector in physics sense) in the opposite direction
        *
        */
        void flipDirector(DirectorElement* a);


        /* This function returns the correct modulo for dealing with negative a. Note % does not!
         *
         * mod(a,b) = a mod b
        */
        inline int mod(int a, int b)
        {
                return (a%b + b)%b;
        }


	#define DIRECTOR_ELEMENT 1
#endif
