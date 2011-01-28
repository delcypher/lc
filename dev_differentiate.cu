/* Implementation of differentiate functions for the device
*  By Alex Allen & Daniel Liew (2010)
*/

#include <math.h>
#include <stdio.h>
#include "lattice.h"
#include "dev_lattice.cuh"
#include "dev_differentiate.cuh"

	/* The functions here represent the calculation of different partial deriviative evaluated at a point (xPos,yPos)
	*  using different differencing schemes.
	*  
	*  In each method we see if we need to flip directorElement to make sure angle between
	*  vectors is < 90deg so if cos(theta) < 0 then we should flip directorElement by 180degrees.
        *        
	*  When we do flipping we always try to flip (xPos,yPos) rather than e.g. (xPos -1,yPos) so that we always
	*  flip lattice cells and not boundary cells. Note it shouldn't actually matter if we did flip boundary cells.
	*
	*
	*/


	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdx_F(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos +1, yPos)) < 0)
		{
					dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos + 1,yPos)->x - latticeGetN(l,xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdx_B(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos -1, yPos)) < 0)
		{
					dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->x - latticeGetN(l,xPos -1, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdy_F(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos +1)) < 0)
		{
					dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos +1)->x - latticeGetN(l,xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdy_B(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos -1)) < 0)
		{
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->x - latticeGetN(l,xPos, yPos -1)->x;

	}

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydx_F(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos +1, yPos)) < 0)
		{
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos +1 ,yPos)->y - latticeGetN(l,xPos, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydx_B(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos -1, yPos)) < 0)
		{
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->y - latticeGetN(l,xPos -1, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydy_F(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos +1)) < 0)
		{
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos +1)->y - latticeGetN(l,xPos,yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydy_B(LatticeObject* l, int xPos, int yPos)
	{
		if(dev_calculateCosineBetween(latticeGetN(l,xPos,yPos), latticeGetN(l, xPos, yPos -1)) < 0)
		{
			dev_flipDirector(latticeGetN(l,xPos,yPos));
		}

		return latticeGetN(l,xPos,yPos)->y - latticeGetN(l,xPos,yPos -1)->y;

	}

