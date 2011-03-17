/* DirectorElement functions By Dan Liew and Alex Allen.
*/

#include "directorelement.h"
#include <cmath>


void setDirectorAngle(DirectorElement* a, double angle)
{
	a->x = cos(angle);
	a->y = sin(angle);
}

void rotateDirector(DirectorElement* a, double angle)
{
	double tempX = a->x;
	double tempY = a->y;

	a->x = cos(angle)*tempX -sin(angle)*tempY;
	a->y = sin(angle)*tempX + cos(angle)*tempY;
}
