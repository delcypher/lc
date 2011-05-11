/* DirectorElement class implementation

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

#include "directorelement.h"
#include <cmath>

void DirectorElement::setAngle(double angle)
{
	x = cos(angle);
	y = sin(angle);
}

double DirectorElement::calculateAngle() const
{
	return atan2(y,x);
}

void DirectorElement::rotate(double angle)
{
	//Work out the angle the DirectorElement currently makes
	double currentAngle=atan2(y,x);

	x=cos(currentAngle + angle);
	y=sin(currentAngle + angle);
}

double DirectorElement::calculateModulus() const
{
	return sqrt(x*x + y*y);
}

bool DirectorElement::makeUnitVector()
{
	double modulus = calculateModulus();
	if(modulus ==0)
	{
		//can't do divide by zero
		return false;
	}
	else
	{
		/* R vector
		*  |R| modulus of vector R
		* r unit vector of vector R
		* R=|R|r
		*
		* so r = R/|r|
		*/
		x /= modulus;
		y /= modulus;
		
		//Try to sort rounding errors for vector aligned with axes
		if(x==1 || x==-1)
			y=0;

		if(y==1 || y==-1)
			x=0;

		return true;
	}
}
