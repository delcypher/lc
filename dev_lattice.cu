/* Implementation of the LatticeObject functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lattice.h"
#include "dev_differentiate.cuh"

/* This function returns the correct modulo for dealing with negative a. Note % does not!
* 
* dev_mod(a,b) = a mod b
*/
__device__ inline int dev_mod(int a, int b)
{
	return (a%b + b)%b;
}

/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
*
*/
__device__ double dev_calculateCosineBetween(DirectorElement* a, DirectorElement* b)
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
__device__ void dev_flipDirector(DirectorElement* a)
{
	//flip component directions
	a->x *= -1;
	a->y *= -1;
}

/*
This function returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
boundary conditions of a LatticeObject (theLattice). You need to pass a pointer to the LatticeObject.

*/
__device__ DirectorElement* latticeGetN(LatticeObject* theLattice, int xPos, int yPos)
{
	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= theLattice->param.width && theLattice->param.rightBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = dev_mod(xPos, theLattice->param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && theLattice->param.leftBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = dev_mod(xPos, theLattice->param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= theLattice->param.height && theLattice->param.topBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = dev_mod(yPos, theLattice->param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && theLattice->param.bottomBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = dev_mod(yPos, theLattice->param.height);
	}
	
	/* All periodic boundary conditions have now been handled
	*/

	/*
	* If the requested "DirectorElement" is in the lattice array just return it.
	*/
	if(xPos >= 0 && xPos < theLattice->param.width && yPos >= 0 && yPos < theLattice->param.height)
	{
		return &(theLattice->lattice[ xPos + (theLattice->param.width)*yPos ]);
	}

	/*we now know (xPos,yPos) isn't in lattice so either (xPos,yPos) is on the PARALLEL or PERPENDICULAR
	* boundary or an invalid point has been requested
	*/

	//in top boundary and within lattice along x
	if(yPos >= theLattice->param.height && xPos >= 0 && xPos < theLattice->param.width)
	{
		if(theLattice->param.topBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(theLattice->PARALLEL_DIRECTOR);
		} 
		else if(theLattice->param.topBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(theLattice->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in bottom boundary and within lattice along x
	if(yPos <= -1 && xPos >= 0 && xPos < theLattice->param.width)
	{
		if(theLattice->param.bottomBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(theLattice->PARALLEL_DIRECTOR);
		}
		else if(theLattice->param.bottomBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(theLattice->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in left boundary and within lattice along y
	if(xPos <= -1 && yPos >= 0 && yPos < theLattice->param.height)
	{
		if(theLattice->param.leftBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(theLattice->PARALLEL_DIRECTOR);
		}
		else if(theLattice->param.leftBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(theLattice->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in right boundary and within lattice along y
	if(xPos >= theLattice->param.width && yPos >= 0 && yPos < theLattice->param.height)
	{
		if(theLattice->param.rightBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(theLattice->PARALLEL_DIRECTOR);
		}
		else if(theLattice->param.rightBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(theLattice->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//Every case should already of been handled. An invalid point (xPos,yPos) must of been asked for
	return NULL;
}


/* Calculate the "free energy per unit area" for a cell at (xPos, yPos) using the frank equation in 2D
*
*/
__device__ double latticeCalculateEnergyOfCell(LatticeObject* l, int xPos, int yPos)
{
	/*   |T|     y|
	*  |L|X|R|    |
	*    |B|      |_____
	*                  x
	* energy = 0.5*(k_1*firstTerm + k_3*(n_x^2 + n_y^2)*secondTerm)
	* firstTerm= (dn_x/dx + dn_y/dy)^2
	* secondTerm = (dn_y/dx - dn_x/dy)^2
	*
	* firstTerm & secondTerm are estimated by using every possible combination of differencing type for derivative
	* and then taking average.
	*
	* Note we assume k_1 =1 & k_3=beta*k_1
	*/

	double firstTerm=0;
	double secondTerm=0;
	double temp=0;
	double temp2=0;

	//Estimate first term by calculating the 4 different ways of calculating the first term and taking the average
	
	//Using T & R (forward differencing in both directions)
	temp = dev_dNxdx_F(l,xPos,yPos) + dev_dNydy_F(l,xPos,yPos);
	firstTerm = temp*temp;

	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	temp = dev_dNxdx_F(l,xPos,yPos) + dev_dNydy_B(l,xPos,yPos);
	firstTerm += temp*temp;	

	//Using B & L (backward differencing in both directions)
	temp = dev_dNxdx_B(l,xPos,yPos) + dev_dNydy_B(l,xPos,yPos);
	firstTerm += temp*temp;

	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	temp = dev_dNxdx_B(l,xPos,yPos) + dev_dNydy_F(l,xPos,yPos);
	firstTerm += temp*temp;

	//Divide by 4 to get average to estimate first term
	firstTerm /= 4.0;

	//Estimate second term by calculating the 4 different ways of calculating the first term and taking the average
	
	//Using T & R (forward differencing in both directions)
	temp = dev_dNydx_F(l,xPos,yPos) - dev_dNxdy_F(l,xPos,yPos);
	secondTerm = temp*temp;

	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	temp = dev_dNydx_F(l,xPos,yPos) - dev_dNxdy_B(l,xPos,yPos);
	secondTerm += temp*temp;

	//Using B & L (backward differencing in both directions)
	temp = dev_dNydx_B(l,xPos,yPos) - dev_dNxdy_B(l,xPos,yPos);
	secondTerm += temp*temp;

	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	temp = dev_dNydx_B(l,xPos,yPos) - dev_dNxdy_F(l,xPos,yPos);
	secondTerm += temp*temp;

	//Divide by 4 to get average to estimate second term
	secondTerm /= 4.0;
	
	//calculate n_x^2
	temp = latticeGetN(l,xPos,yPos)->x;
	temp *=temp;

	temp2 = latticeGetN(l,xPos,yPos)->y;
	temp2 *=temp2;
	
	return 0.5*(firstTerm + (l->param.beta)*(temp + temp2)*secondTerm );
}


