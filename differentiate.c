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
float dndx(enum directorComponent dirComp, LatticeObject* l, int xPos, int yPos)
{

	switch(diffScheme)
	{
		case FORWARD_DIFF:
		{
			/*see if we need to flip directorElement to make sure angle between
			* vectors is < 90deg so cos(theta) < 0.
			*/
			if(calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos +1, yPos)) < 0)
			{
				/*flip vector, We could do (xPos,yPos) or (xPos +1,yPos) but (xPos,yPos) should be in lattice
				* and (xPos +1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
				* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
				* is always used.
				*/
				flipDirector(latticeGetN(l,xPos,yPos));
			}

			switch(dirComp)
			{
				case N_X:
					return latticeGetN(l,xPos +1,yPos)->x - latticeGetN(l, xPos, yPos)->x;
				break;

				case N_Y:
					return latticeGetN(l,xPos +1,yPos)->y - latticeGetN(l, xPos, yPos)->y;
				break;

				default:
					return 0;
			}

		}
		break;

		case BACKWARD_DIFF:
		{
			/*see if we need to flip directorElement to make sure angle between
			* vectors is < 90deg so cos(theta) < 0.
			*/
			if(calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos -1, yPos)) < 0)
			{
				/*flip vector, We could do (xPos,yPos) or (xPos -1,yPos) but (xPos,yPos) should be in lattice
				* and (xPos -1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
				* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
				* is always used.
				*/
				flipDirector(latticeGetN(l,xPos,yPos));
			}

			switch(dirComp)
			{
				case N_X:
					return latticeGetN(l,xPos,yPos)->x - latticeGetN(l, xPos -1, yPos)->x;
				break;

				case N_Y:
					return latticeGetN(l,xPos,yPos)->y - latticeGetN(l, xPos -1, yPos)->y;
				break;

				default:
					return 0;
			}
		}
		break;

		case CENTRAL_DIFF:
		{
			//We are not sure how to handle director flipping so it's not implemented yet
			switch(dirComp)
			{
				case N_X:
					return (latticeGetN(l,xPos +1,yPos)->x - latticeGetN(l, xPos -1, yPos)->x)/2;
				break;

				case N_Y:
					return (latticeGetN(l,xPos +1,yPos)->y - latticeGetN(l, xPos -1, yPos)->y)/2;
				break;

				default:
					return 0;
			}


		}
		break;

		default:
			return 0;

	}
}


/* Calculate partial derivative w.r.t to y then evaluated at point (xPos,yPos)
* 
*/
float dndy(enum directorComponent dirComp, LatticeObject* l, int xPos, int yPos)
{

	switch(diffScheme)
	{
		case FORWARD_DIFF:
		{
			/*see if we need to flip directorElement to make sure angle between
			* vectors is < 90deg so cos(theta) < 0.
			*/
			if(calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos +1)) < 0)
			{
				/*flip vector, We could do (xPos,yPos) or (xPos,yPos +1) but (xPos,yPos) should be in lattice
				* and (xPos, yPos +1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
				* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
				* is always used.
				*/
				flipDirector(latticeGetN(l,xPos,yPos));
			}

			switch(dirComp)
			{
				case N_X:
					return latticeGetN(l,xPos,yPos +1)->x - latticeGetN(l, xPos, yPos)->x;
				break;

				case N_Y:
					return latticeGetN(l,xPos,yPos +1)->y - latticeGetN(l, xPos, yPos)->y;
				break;

				default:
					return 0;
			}

		}
		break;

		case BACKWARD_DIFF:
		{
			/*see if we need to flip directorElement to make sure angle between
			* vectors is < 90deg so cos(theta) < 0.
			*/
			if(calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos -1)) < 0)
			{
				/*flip vector, We could do (xPos,yPos) or (xPos,yPos -1) but (xPos,yPos) should be in lattice
				* and (xPos, yPos -1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
				* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
				* is always used.
				*/
				flipDirector(latticeGetN(l,xPos,yPos));
			}

			switch(dirComp)
			{
				case N_X:
					return latticeGetN(l,xPos,yPos)->x - latticeGetN(l, xPos, yPos -1)->x;
				break;

				case N_Y:
					return latticeGetN(l,xPos,yPos)->y - latticeGetN(l, xPos, yPos -1)->y;
				break;

				default:
					return 0;
			}
		}
		break;

		case CENTRAL_DIFF:
		{
			//We are not sure how to handle director flipping so it's not implemented yet
			switch(dirComp)
			{
				case N_X:
					return (latticeGetN(l,xPos,yPos +1)->x - latticeGetN(l, xPos, yPos -1)->x)/2;
				break;

				case N_Y:
					return (latticeGetN(l,xPos,yPos +1)->y - latticeGetN(l, xPos, yPos -1)->y)/2;
				break;

				default:
					return 0;
			}


		}
		break;

		default:
			return 0;

	}

}

