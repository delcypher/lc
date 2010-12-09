/* Implementation of the LatticeObject functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "randgen.h"
#include "lattice.h"

const float PI=3.1415926;

/* Define the perpendicular & parallel directors used by
 * BOUNDARY_PERPENDICULAR & BOUNDARY_PARALLEL respectively
*/
const DirectorElement PERPENDICULAR_DIRECTOR = {0,1};
const DirectorElement PARALLEL_DIRECTOR = {1,0};

/*
This function returns a pointer to the "element" of the director field at (xPos, yPos) with the constraints of the 
boundary conditions of a LatticeObject (theLattice). You need to pass a pointer to the LatticeObject.

*/
const DirectorElement* latticeGetN(const LatticeObject* theLattice, int xPos, int yPos)
{
	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= theLattice->param.width && theLattice->param.rightBoundary == BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, theLattice->param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && theLattice->param.leftBoundary == BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, theLattice->param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= theLattice->param.height && theLattice->param.topBoundary == BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, theLattice->param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && theLattice->param.bottomBoundary == BOUNDARY_PERIODIC)
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
		return &(theLattice->lattice[xPos][yPos]);
	}

	/*we now know (xPos,yPos) isn't in lattice so either (xPos,yPos) is on the PARALLEL or PERPENDICULAR
	* boundary or an invalid point has been requested
	*/

	//in top boundary and within lattice along x
	if(yPos == theLattice->param.height && xPos >= 0 && xPos < theLattice->param.width)
	{
		if(theLattice->param.topBoundary == BOUNDARY_PARALLEL)
		{
			return &PARALLEL_DIRECTOR;
		} 
		else if(theLattice->param.topBoundary == BOUNDARY_PERPENDICULAR)
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
	if(yPos == -1 && xPos >= 0 && xPos < theLattice->param.width)
	{
		if(theLattice->param.bottomBoundary == BOUNDARY_PARALLEL)
		{
			return &PARALLEL_DIRECTOR;
		}
		else if(theLattice->param.bottomBoundary == BOUNDARY_PERPENDICULAR)
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
	if(xPos == -1 && yPos >= 0 && yPos < theLattice->param.height)
	{
		if(theLattice->param.leftBoundary == BOUNDARY_PARALLEL)
		{
			return &PARALLEL_DIRECTOR;
		}
		else if(theLattice->param.leftBoundary == BOUNDARY_PERPENDICULAR)
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
	if(xPos == theLattice->param.width && yPos >= 0 && yPos < theLattice->param.height)
	{
		if(theLattice->param.rightBoundary == BOUNDARY_PARALLEL)
		{
			return &PARALLEL_DIRECTOR;
		}
		else if(theLattice->param.rightBoundary == BOUNDARY_PERPENDICULAR)
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
	int xPos;
	for (xPos=0; xPos < theLattice->param.width; xPos++)
	{
		free(theLattice->lattice[xPos]); 
	}
	
	free(theLattice->lattice);
	free(theLattice);
}

/* This function is used in initialise a LatticeObject from the freestore and returns a pointer to 
*  the newly made object. Use latticeFree() to remove the object from the freestore
*
*/

LatticeObject* latticeInitialise(LatticeConfig configuration)
{
	int xPos, yPos;
	float randomAngle;

	//check that the width & height have been specified
	if(configuration.width <= 0 || configuration.height <= 0)
	{
		fprintf(stderr, "Error: The width and/or height have not been set. Can't initialise lattice.");
		return NULL;
	}


	//Try and allocate memory from freestore for LatticeObject
	LatticeObject* theLattice = malloc(sizeof(LatticeObject));
	
	if(theLattice==NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for LatticeObject");
		return NULL;
	}

	//set lattice parameters
	theLattice->param = configuration;

	//allocate memory for the first index (theLattice[index]) part of array
	theLattice->lattice = (DirectorElement**) malloc(sizeof(DirectorElement*) * theLattice->param.width);
	
	if(theLattice->lattice == NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for lattice array first dimension in LatticeObject.");
	}

	//alocate memory for the second index (thelattice[x][index]) part of the array
	for (xPos=0; xPos < theLattice->param.width; xPos++)
	{
		theLattice->lattice[xPos] = (DirectorElement*) malloc(sizeof(DirectorElement) * theLattice->param.height);
		if(theLattice->lattice[xPos] == NULL)
		{
			fprintf(stderr,"Error: Couldn't allocate memory for lattice array second dimension in LatticeObject.");
		}
	}

	
	
	/* Loop through lattice array (theLattice->lattice[x][y]) and initialise
	*  Note in C we must use RANDOM,... but if using C++ then must use LatticeConfig::RANDOM , ...
	*/
	for (yPos = 0; yPos < theLattice->param.height; yPos++)
	{
		for (xPos = 0; xPos < theLattice->param.width; xPos++)
		{
			switch(theLattice->param.initialState)
			{

				case RANDOM :
				{
					//generate a random angle between 0 & 2*PI radians
					randomAngle = 2*PI*cpuRnd();
					theLattice->lattice[xPos][yPos].x=cos(randomAngle);
					theLattice->lattice[xPos][yPos].y=sin(randomAngle);
				}

				break;
				
				case PARALLEL_X:
					theLattice->lattice[xPos][yPos].x=1;
					theLattice->lattice[xPos][yPos].y=0;
				break;

				case PARALLEL_Y:
				
					theLattice->lattice[xPos][yPos].x=0;
					theLattice->lattice[xPos][yPos].y=1;
				break;

				case BOT_PAR_TOP_NORM:
					/*
					* This isn't implemented yet
					*/
					theLattice->lattice[xPos][yPos].x=0;
					theLattice->lattice[xPos][yPos].y=0;
				break;
				
				default:
					//if we aren't told what to do we will set all zero vectors!
					theLattice->lattice[xPos][yPos].x=0;
					theLattice->lattice[xPos][yPos].y=0;

			}
		}
	}
	

	return theLattice;
}

/*
* This function outputs the current state of the lattice "theLattice" to standard output in a format
* compatible with GNUplot. A simple plot command is `set key off; plot 'file' with vectors`
*/
void latticeDump(LatticeObject* theLattice)
{
	if(theLattice ==NULL)
	{
		fprintf(stderr,"Error: Received NULL pointer to LatticeObject");
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
	int xPos, yPos;

	for(yPos=0; yPos < theLattice->param.height; yPos++)
	{
		for(xPos=0; xPos < theLattice->param.width; xPos++)
		{
			printf("%d %d %f %f \n",xPos, yPos, theLattice->lattice[xPos][yPos].x, theLattice->lattice[xPos][yPos].y);
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
