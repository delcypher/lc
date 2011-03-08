/* DirectorElement functions By Dan Liew and Alex Allen.
*/

#include "directorelement.h"
#include <cmath>


/* Flips a DirectorElement (vector in physics sense) in the opposite direction
*
*/
void flipDirector(DirectorElement* a)
{
        //flip component directions
        a->x *= -1;
        a->y *= -1;
}

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
