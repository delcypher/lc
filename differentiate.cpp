/* Implementation of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <cmath>
#include <cstdio>
#include "lattice.h"
#include "randgen.h"
#include "differentiate.h"

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
	double dNxdx_F(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos +1, yPos)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos + 1,yPos)->x - l->getN(xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdx_B(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos -1, yPos)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->x - l->getN(xPos -1, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_F(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos +1)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos,yPos +1)->x - l->getN(xPos, yPos)->x;

	}

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_B(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos -1)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->x - l->getN(xPos, yPos -1)->x;

	}

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_F(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos +1, yPos)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos +1 ,yPos)->y - l->getN(xPos, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_B(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos -1, yPos)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->y - l->getN(xPos -1, yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_F(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos +1)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos,yPos +1)->y - l->getN(xPos,yPos)->y;

	}

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_B(Lattice* l, int xPos, int yPos)
	{
		if(calculateCosineBetween(l->getN(xPos,yPos), l->getN( xPos, yPos -1)) < 0)
		{
			flipDirector(l->setN(xPos,yPos));
		}

		return l->getN(xPos,yPos)->y - l->getN(xPos,yPos -1)->y;

	}

