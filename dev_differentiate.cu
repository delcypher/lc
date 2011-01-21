/* Implementation of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <math.h>
#include <stdio.h>
#include "lattice.h"
#include "dev_lattice.cuh"
#include "dev_differentiate.cuh"

	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdx_F(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos +1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos +1 ,yPos) but (xPos,yPos) should be in lattice
			* and (xPos +1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos + 1,yPos)->x - latticeGetN(l,xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdx_B(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos -1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos -1 ,yPos) but (xPos,yPos) should be in lattice
			* and (xPos -1 , yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->x - latticeGetN(l,xPos -1, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdy_F(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos +1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos,yPos +1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos +1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos +1)->x - latticeGetN(l,xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdy_B(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos -1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos,yPos -1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos -1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->x - latticeGetN(l,xPos, yPos -1)->x;

	}

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydx_F(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos +1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos +1,yPos) but (xPos,yPos) should be in lattice
			* and (xPos +1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos +1 ,yPos)->y - latticeGetN(l,xPos, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydx_B(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos -1, yPos)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPos -1,yPos) but (xPos,yPos) should be in lattice
			* and (xPos -1, yPos) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->y - latticeGetN(l,xPos -1, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydy_F(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos +1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPo,yPos +1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos +1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos +1)->y - latticeGetN(l,xPos,yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydy_B(LatticeObject* l, int xPos, int yPos)
	{
		/*see if we need to flip directorElement to make sure angle between
		* vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degress
		*/
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos -1)) < 0)
		{
			/*flip vector, We could flip (xPos,yPos) or (xPo,yPos -1) but (xPos,yPos) should be in lattice
			* and (xPos, yPos -1) could be PERPENDICULAR_DIRECTOR or PARALLEL_DIRECTOR which it would be
			* nice if we didn't flip, although it shouldn't cause a problem provided the flipping algorithm
			* is always used.
			*/
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->y - latticeGetN(l,xPos,yPos -1)->y;

	}

