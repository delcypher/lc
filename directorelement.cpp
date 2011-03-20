/* DirectorElement functions By Dan Liew and Alex Allen.
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
