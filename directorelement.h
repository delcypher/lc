/* Header file for DirectorElement class that act on it 

   Copyright (C) 2010 Dan Liew & Alex Allen
   
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

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
