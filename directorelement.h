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

	/* Set the Angle(in radians) the DirectorElement makes with the x-axis
	*
	*/
	void setDirectorAngle(DirectorElement* a, double angle);
	
	/* Rotate the DirectorElement by an angle "angle" in radians.
	*
	*/
	void rotateDirector(DirectorElement* a, double angle);

	#define DIRECTOR_ELEMENT 1
#endif
