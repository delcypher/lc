/* DirectorElement functions By Dan Liew and Alex Allen.
*/

#include "directorelement.h"
#include <cmath>

/*
* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
*
*/
double calculateCosineBetween(const DirectorElement* a, const DirectorElement* b)
{
        double cosine;

        /*
        * Calculate cosine using formula for dot product between vectors cos(theta) = a.b/|a||b|
        * Note if using unit vectors then |a||b| =1, could use this shortcut later.
        */
        cosine = ( (a->x)*(b->x) + (a->y)*(b->y) )/
                ( sqrt( (a->x)*(a->x) + (a->y)*(a->y) )*sqrt( (b->x)*(b->x) + (b->y)*(b->y) ) ) ;

        return cosine;
}

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
