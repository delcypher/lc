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
		
		/* Set DirectorElement so that it makes angle of "angle" in radians w.r.t
		*  the x-axis. Note this is done so that the DirectorElement is a unit vector.
		*/
		void setAngle(double angle);
		
		/* Get the angle that the DirectorElement makes with the x-axis (anti-clockwise rotation)
		*
		*/
		double calculateAngle() const;

		/* Rotate the DirectorElement by an angle "angle"
		*  in radians (anti-clockwise rotation).
		*/
		void rotate(double angle);

		/* Calculate the modulus of the DirectorElement
		*/
		double calculateModulus() const;

		/* Modify componenets of DirectorElement so that it is a unit vector
		*/
		bool makeUnitVector();
        };

	#define DIRECTOR_ELEMENT 1
#endif
