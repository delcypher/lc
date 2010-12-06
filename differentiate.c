/* Implementation of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <math.h>
#include <stdio.h>
#include "lattice.h"
#include "randgen.h"
#include "differentiate.h"

//select the differencing scheme to use
enum differencingSchemes diffScheme = FORWARD_DIFF;

/* Calculate partial derivative the "dirComp" component of the director 
*  w.r.t to x then evaluated at point (xPos,yPos)
* 
*/
float dndx(enum directorComponent dirComp, int xPos, int yPos, LatticeObject* theLattice)
{

	return 0;

}


/* Calculate partial derivative w.r.t to y then evaluated at point (xPos,yPos)
* 
*/
float dndy(enum directorComponent dirComp, int xPos, int yPos, LatticeObject* theLattice )
{

	return 0;
}

