/* Header file for DirectorElement and functions that act on it */

#ifndef DIRECTOR_ELEMENT
	
	/* DirectorElement is a 2D vector (in physics sense)
        *  expressed in cartesian co-ordinates. Arrays of
        *  these are used to build a vector field.
        *  Included with this 2D vector is a variable identifying if the
        *  DirectorElement is a nanoparticle.
        */
        struct DirectorElement
        {
                double x,y;
                bool isNanoparticle;
		DirectorElement(double xValue, double yValue, bool inp) : x(xValue) , y(yValue) , isNanoparticle(inp)
		{
			//do nothing
		}
        };

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
