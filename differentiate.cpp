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
	double dNxdx_F(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement right =  *(l->getN(xPos +1, yPos));

		if(calculateCosineBetween(&centre, &right) < 0)
		{
			flipDirector(&centre);
		}

		return right.x - centre.x;

	}

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdx_B(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement left =  *(l->getN(xPos -1, yPos));

		if(calculateCosineBetween(&centre, &left) < 0)
		{
			flipDirector(&centre);
		}

		return centre.x - left.x;

	}

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_F(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement top =  *(l->getN(xPos, yPos +1));

		if(calculateCosineBetween(&centre, &top) < 0)
		{
			flipDirector(&centre);
		}

		return top.x - centre.x;

	}

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_B(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement bottom =  *(l->getN(xPos, yPos -1));

		if(calculateCosineBetween(&centre, &bottom) < 0)
		{
			flipDirector(&centre);
		}

		return centre.x - bottom.x;

	}

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_F(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement right =  *(l->getN(xPos +1, yPos));

		if(calculateCosineBetween(&centre, &right) < 0)
		{
			flipDirector(&centre);
		}

		return right.y - centre.y;

	}

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_B(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement left =  *(l->getN(xPos -1, yPos));

		if(calculateCosineBetween(&centre, &left) < 0)
		{
			flipDirector(&centre);
		}

		return centre.y - left.y;

	}

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_F(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement top =  *(l->getN(xPos, yPos +1));

		if(calculateCosineBetween(&centre, &top) < 0)
		{
			flipDirector(&centre);
		}

		return top.y - centre.y;

	}

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_B(const Lattice* l, int xPos, int yPos)
	{
		//make a local copy of director elements
		DirectorElement centre = *(l->getN(xPos,yPos));
		const DirectorElement bottom =  *(l->getN(xPos, yPos -1));

		if(calculateCosineBetween(&centre, &bottom) < 0)
		{
			flipDirector(&centre);
		}

		return centre.y - bottom.y;

	}

