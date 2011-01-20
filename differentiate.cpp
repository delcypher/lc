/* Implementation of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <math.h>
#include <stdio.h>
#include "lattice.h"
#include "randgen.h"
#include "differentiate.h"

	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdx_F(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos +1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos +1 ,yPos) but (xPos,yPos) should be in lattice
			* and (xPos +1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos + 1,yPos)->x - l->getN(xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdx_B(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos -1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos -1 ,yPos) but (xPos,yPos) should be in lattice
			* and (xPos -1 , yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->x - l->getN(xPos -1, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_F(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos +1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos,yPos +1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos +1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos,yPos +1)->x - l->getN(xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_B(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos -1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos,yPos -1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos -1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->x - l->getN(xPos, yPos -1)->x;

	}

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_F(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos +1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos +1,yPos) but (xPos,yPos) should be in lattice
			* and (xPos +1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos +1 ,yPos)->y - l->getN(xPos, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_B(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos -1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos -1,yPos) but (xPos,yPos) should be in lattice
			* and (xPos -1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->y - l->getN(xPos -1, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_F(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos +1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPo,yPos +1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos +1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos,yPos +1)->y - l->getN(xPos,yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_B(Lattice* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos -1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPo,yPos -1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos -1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			flipDirector(l->getN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->y - l->getN(xPos,yPos -1)->y;

	}

