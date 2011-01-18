/* Implementation of the LatticeObject functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "randgen.h"
#include "lattice.h"
#include "differentiate.h"

const double PI=3.1415926;

/* Define the perpendicular & parallel directors used by
 * BOUNDARY_PERPENDICULAR & BOUNDARY_PARALLEL respectively
*/
DirectorElement PERPENDICULAR_DIRECTOR = {0,1};
DirectorElement PARALLEL_DIRECTOR = {1,0};

/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
*
*/
double calculateCosineBetween(DirectorElement* a, DirectorElement* b)
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

/*
This function returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
boundary conditions of a LatticeObject (theLattice). You need to pass a pointer to the LatticeObject.

*/
DirectorElement* latticeGetN(const LatticeObject* theLattice, int xPos, int yPos)
{
	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= theLattice->param.width && theLattice->param.rightBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, theLattice->param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && theLattice->param.leftBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, theLattice->param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= theLattice->param.height && theLattice->param.topBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, theLattice->param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && theLattice->param.bottomBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, theLattice->param.height);
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
			return &PARALLEL_DIRECTOR;
		} 
		else if(theLattice->param.topBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &PERPENDICULAR_DIRECTOR;
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
			return &PARALLEL_DIRECTOR;
		}
		else if(theLattice->param.bottomBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &PERPENDICULAR_DIRECTOR;
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
			return &PARALLEL_DIRECTOR;
		}
		else if(theLattice->param.leftBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &PERPENDICULAR_DIRECTOR;
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
			return &PARALLEL_DIRECTOR;
		}
		else if(theLattice->param.rightBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &PERPENDICULAR_DIRECTOR;
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

/* This function is used free memory allocated by the latticeInitialise function
*  You should pass it a pointer to a LatticeObject type. 
*
*/
void latticeFree(LatticeObject* theLattice)
{
	free(theLattice->lattice);
	free(theLattice);
}

/* This function is used in initialise a LatticeObject from the freestore and returns a pointer to 
*  the newly made object. Use latticeFree() to remove the object from the freestore
*
*/

LatticeObject* latticeInitialise(LatticeConfig configuration)
{

	//check that the width & height have been specified
	if(configuration.width <= 0 || configuration.height <= 0)
	{
		fprintf(stderr, "Error: The width and/or height have not been set to valid values ( > 0). Can't initialise lattice.\n");
		return NULL;
	}


	//Try and allocate memory from freestore for LatticeObject
	LatticeObject* theLattice = (LatticeObject*) malloc(sizeof(LatticeObject));
	
	if(theLattice==NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for LatticeObject\n");
		return NULL;
	}

	//set lattice parameters
	theLattice->param = configuration;

	//allocate memory for lattice (theLattice[index]) part of array
	theLattice->lattice = (DirectorElement*) malloc(sizeof(DirectorElement) * (theLattice->param.width)*(theLattice->param.height));
	
	if(theLattice->lattice == NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for lattice array in LatticeObject.\n");
		latticeFree(theLattice); //We should free anything we allocated to prevent memory leaks.
		return NULL;
	}

	//initialise the lattice to a particular state
	if(latticeReinitialise(theLattice, theLattice->param.initialState) == 0)
	{
		fprintf(stderr,"Error: Couldn't set initial lattice state!\n");
	}
	

	return theLattice;
}


int latticeReinitialise(LatticeObject* theLattice, enum LatticeConfig::latticeState initialState)
{
	if(theLattice == NULL)
	{
		fprintf(stderr,"Error: Can't reinitialise on LatticeObject with NULL pointer\n");
		return 0;
	}

	theLattice->param.initialState = initialState;

	//we should reset the random seed so we don't generate the set of pseudo random numbers every time	
	cpuSetRandomSeed();
	
	/* Loop through lattice array (theLattice->lattice[index]) and initialise
	*  Note in C we must use RANDOM,... but if using C++ then must use LatticeConfig::RANDOM , ...
	*/
	int xPos,yPos;
	int index=0;
	double randomAngle;

	for (yPos = 0; yPos < theLattice->param.height; yPos++)
	{
		for (xPos = 0; xPos < theLattice->param.width; xPos++)
		{
			index = xPos + (theLattice->param.width)*yPos;
			switch(theLattice->param.initialState)
			{

				case LatticeConfig::RANDOM:
				{
					//generate a random angle between 0 & 2*PI radians
					randomAngle = 2*PI*cpuRnd();
					theLattice->lattice[index].x=cos(randomAngle);
					theLattice->lattice[index].y=sin(randomAngle);
				}

				break;
				
				case LatticeConfig::PARALLEL_X:
					theLattice->lattice[index].x=1;
					theLattice->lattice[index].y=0;
				break;

				case LatticeConfig::PARALLEL_Y:
					theLattice->lattice[index].x=0;
					theLattice->lattice[index].y=1;
				break;

				case LatticeConfig::BOT_PAR_TOP_NORM:
					/*
					* This isn't implemented yet
					*/
					theLattice->lattice[index].x=0;
					theLattice->lattice[index].y=0;
				break;
				
				default:
					//if we aren't told what to do we will set all zero vectors!
					theLattice->lattice[index].x=0;
					theLattice->lattice[index].y=0;

			}
		}
	}

	return 1;
}

/*
* This function outputs the current state of the lattice "theLattice" to standard output in a format
* compatible with the shell script latticedump.sh which uses GNUplot . The director field is plotted as 
* 1/2 unit vectors rather than unit vectors so that neighbouring vectors when plotted do not overlap.
*/
void latticeHalfUnitVectorDump(LatticeObject* theLattice)
{
	if(theLattice ==NULL)
	{
		fprintf(stderr,"Error: Received NULL pointer to LatticeObject.\n");
		return;
	}

	//print lattice information
	printf("#Lattice Width:%d \n",theLattice->param.width);
	printf("#Lattice Height:%d \n", theLattice->param.height);
	printf("#Lattice beta value:%f \n", theLattice->param.beta);
	printf("#Lattice top Boundary: %d (enum) \n",theLattice->param.topBoundary);
	printf("#Lattice bottom Boundary: %d (enum) \n",theLattice->param.bottomBoundary);
	printf("#Lattice left Boundary: %d (enum) \n",theLattice->param.leftBoundary);
	printf("#Lattice right Boundary: %d (enum) \n",theLattice->param.rightBoundary);
	printf("#Lattice initial state: %d (enum) \n",theLattice->param.initialState);

	printf("\n\n # (x) (y) (n_x) (n_y)\n");

	//print lattice state
	int xPos, yPos, index;

	for(yPos=0; yPos < theLattice->param.height; yPos++)
	{
		for(xPos=0; xPos < theLattice->param.width; xPos++)
		{
			index = xPos + (theLattice->param.width)*yPos;
			printf("%d %d %f %f \n",
			xPos, 
			yPos, 
			(theLattice->lattice[index].x)*0.5, 
			(theLattice->lattice[index].y)*0.5);
		}
	}

	printf("\n#End of Lattice Dump");
}

/* Calculate the "free energy per unit area" for a cell at (xPos, yPos) using the frank equation in 2D
*
*/
double latticeCalculateEnergyOfCell(const LatticeObject* l, int xPos, int yPos)
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
	temp = dNxdx_F(l,xPos,yPos) + dNydy_F(l,xPos,yPos);
	firstTerm = temp*temp;

	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	temp = dNxdx_F(l,xPos,yPos) + dNydy_B(l,xPos,yPos);
	firstTerm += temp*temp;	

	//Using B & L (backward differencing in both directions)
	temp = dNxdx_B(l,xPos,yPos) + dNydy_B(l,xPos,yPos);
	firstTerm += temp*temp;

	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	temp = dNxdx_B(l,xPos,yPos) + dNydy_F(l,xPos,yPos);
	firstTerm += temp*temp;

	//Divide by 4 to get average to estimate first term
	firstTerm /= 4.0;

	//Estimate second term by calculating the 4 different ways of calculating the first term and taking the average
	
	//Using T & R (forward differencing in both directions)
	temp = dNydx_F(l,xPos,yPos) - dNxdy_F(l,xPos,yPos);
	secondTerm = temp*temp;

	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	temp = dNydx_F(l,xPos,yPos) - dNxdy_B(l,xPos,yPos);
	secondTerm += temp*temp;

	//Using B & L (backward differencing in both directions)
	temp = dNydx_B(l,xPos,yPos) - dNxdy_B(l,xPos,yPos);
	secondTerm += temp*temp;

	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	temp = dNydx_B(l,xPos,yPos) - dNxdy_F(l,xPos,yPos);
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


/* Calculate the "free energy" of entire lattice. Note this calculation may not be very efficient!
*
*/
double latticeCalculateTotalEnergy(const LatticeObject* l)
{
	/*
	* This calculation isn't very efficient as it uses calculateEngergyOfCell() for everycell
	* which results in various derivatives being calculated more than once.
	*/

	int xPos,yPos;
	double energy=0;

	for(yPos=0; yPos < (l->param.height); yPos++)
	{
		for(xPos=0; xPos < (l->param.width); xPos++)
		{
			energy += latticeCalculateEnergyOfCell(l,xPos,yPos);	
		}
	}

	return energy;
}


/*
* This function outputs the current state of the lattice "theLattice" to standard output in a format
* compatible with shell script latticedump.sh which uses GNUplot. The director field is plotted as
* unit vectors that are translated so that the centre of the vector rather than the end of the vector
* is plotted at point (xPos,yPos)
*/
void latticeTranslatedUnitVectorDump(LatticeObject* theLattice)
{
	if(theLattice ==NULL)
	{
		fprintf(stderr,"Error: Received NULL pointer to LatticeObject.\n");
		return;
	}

	//print lattice information
	printf("#Lattice Width:%d \n",theLattice->param.width);
	printf("#Lattice Height:%d \n", theLattice->param.height);
	printf("#Lattice beta value:%f \n", theLattice->param.beta);
	printf("#Lattice top Boundary: %d (enum) \n",theLattice->param.topBoundary);
	printf("#Lattice bottom Boundary: %d (enum) \n",theLattice->param.bottomBoundary);
	printf("#Lattice left Boundary: %d (enum) \n",theLattice->param.leftBoundary);
	printf("#Lattice right Boundary: %d (enum) \n",theLattice->param.rightBoundary);
	printf("#Lattice initial state: %d (enum) \n",theLattice->param.initialState);

	printf("\n\n # (x) (y) (n_x) (n_y)\n");

	//print lattice state
	int xPos, yPos, index;

	for(yPos=0; yPos < theLattice->param.height; yPos++)
	{
		for(xPos=0; xPos < theLattice->param.width; xPos++)
		{
			index = xPos + (theLattice->param.width)*yPos;
			printf("%f %f %f %f \n",
			( (double) xPos) - 0.5*(theLattice->lattice[index].x), 
			( (double) yPos) - 0.5*(theLattice->lattice[index].y), 
			(theLattice->lattice[index].x), 
			(theLattice->lattice[index].y));
		}
	}

	printf("\n#End of Lattice Dump");


}

/* This function returns the correct modulo for dealing with negative a. Note % does not!
* 
* mod(a,b) = a mod b
*/
inline int mod(int a, int b)
{
	return (a%b + b)%b;
}
