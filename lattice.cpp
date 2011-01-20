/* Implementation of the LatticeObject functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "randgen.h"
#include "lattice.h"
#include "differentiate.h"

//initialisation constructor
Lattice::Lattice(LatticeConfig configuration) 
{
	//check that the width & height have been specified
	if(configuration.width <= 0 || configuration.height <= 0)
	{
		fprintf(stderr, "Error: The width and/or height have not been set to valid values ( > 0). Can't initialise lattice.\n");
	}
	
	//Try and allocate memory from freestore for LatticeObject
	hostLatticeObject = (LatticeObject*) malloc(sizeof(LatticeObject));
	
	if(hostLatticeObject==NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for LatticeObject on host!\n");
	}

	//set lattice parameters
	hostLatticeObject->param = configuration;
	
	/* Initialise PARALLEL_DIRECTOR. This is a DirectorElement that is
	* parallel with the x-axis of the lattice.
	*/
	hostLatticeObject->PARALLEL_DIRECTOR.x=1;
	hostLatticeObject->PARALLEL_DIRECTOR.y=0;
	hostLatticeObject->PARALLEL_DIRECTOR.isNanoparticle=0;

	/* Initialise PERPENDICULAR_DIRECTOR. This is a DirectorElement that is
	* perpendicular with the x-axis of the lattice.
	*/
	hostLatticeObject->PERPENDICULAR_DIRECTOR.x=0;
	hostLatticeObject->PERPENDICULAR_DIRECTOR.y=1;
	hostLatticeObject->PERPENDICULAR_DIRECTOR.isNanoparticle=0;

	//allocate memory for lattice (hostLatticeObject[index]) part of array
	hostLatticeObject->lattice = (DirectorElement*) malloc(sizeof(DirectorElement) * (hostLatticeObject->param.width)*(hostLatticeObject->param.height));
	
	if(hostLatticeObject->lattice == NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for lattice array in LatticeObject.\n");
		free(hostLatticeObject); //We should free anything we allocated to prevent memory leaks.
	}

	//initialise the lattice to a particular state
	reInitialise(hostLatticeObject->param.initialState);
	
	//initialiseCuda();

}

//destructor
Lattice::~Lattice()
{
	//destroyCuda();
	free(hostLatticeObject->lattice);
	free(hostLatticeObject);
}


void Lattice::destoryCuda()
{


}

void Lattice::initialiseCuda()
{


}

void Lattice::copyHostToDevice()
{

}

void Lattice::copyDeviceToHost()
{

}

bool Lattice::add(Nanoparticle* np)
{
	//check nanoparticle location is inside the lattice.
	if( np->getX() >= hostLatticeObject->param.width || np->getX() < 0 || np->getY() >= hostLatticeObject->param.height || np->getX() < 0)
	{
		fprintf(stderr,"Error: Can't add nanoparticle that is not in the lattice.\n");
		return false;
	}

	//Do a dry run adding the nanoparticle. If it fails we know that there is an overlap with an existing nanoparticle
	for(int y=0; y < hostLatticeObject->param.height; y++)
	{
		for(int x=0; x < hostLatticeObject->param.width; x++)
		{
			if(! np->processCell(x,y,Nanoparticle::DRY_ADD, getN(x,y)) )
			{
				fprintf(stderr,"Error: Adding nanoparticle on dry run failed.\n");
				return false;
			}
		}
	}

	//Do actuall run adding the nanoparticle. If it fails we know that there is an overlap with itself
	for(int y=0; y < hostLatticeObject->param.height; y++)
	{
		for(int x=0; x < hostLatticeObject->param.width; x++)
		{
			if(! np->processCell(x,y,Nanoparticle::ADD, getN(x,y)) )
			{
				fprintf(stderr,"Error: Adding nanoparticle on actuall run failed.\n");
				return false;
			}
		}
	}

	return true;

}

DirectorElement* Lattice::getN(int xPos, int yPos)
{
	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= hostLatticeObject->param.width && hostLatticeObject->param.rightBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, hostLatticeObject->param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && hostLatticeObject->param.leftBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, hostLatticeObject->param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= hostLatticeObject->param.height && hostLatticeObject->param.topBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, hostLatticeObject->param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && hostLatticeObject->param.bottomBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, hostLatticeObject->param.height);
	}
	
	/* All periodic boundary conditions have now been handled
	*/

	/*
	* If the requested "DirectorElement" is in the lattice array just return it.
	*/
	if(xPos >= 0 && xPos < hostLatticeObject->param.width && yPos >= 0 && yPos < hostLatticeObject->param.height)
	{
		return &(hostLatticeObject->lattice[ xPos + (hostLatticeObject->param.width)*yPos ]);
	}

	/*we now know (xPos,yPos) isn't in lattice so either (xPos,yPos) is on the PARALLEL or PERPENDICULAR
	* boundary or an invalid point has been requested
	*/

	//in top boundary and within lattice along x
	if(yPos >= hostLatticeObject->param.height && xPos >= 0 && xPos < hostLatticeObject->param.width)
	{
		if(hostLatticeObject->param.topBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject->PARALLEL_DIRECTOR);
		} 
		else if(hostLatticeObject->param.topBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in bottom boundary and within lattice along x
	if(yPos <= -1 && xPos >= 0 && xPos < hostLatticeObject->param.width)
	{
		if(hostLatticeObject->param.bottomBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject->PARALLEL_DIRECTOR);
		}
		else if(hostLatticeObject->param.bottomBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in left boundary and within lattice along y
	if(xPos <= -1 && yPos >= 0 && yPos < hostLatticeObject->param.height)
	{
		if(hostLatticeObject->param.leftBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject->PARALLEL_DIRECTOR);
		}
		else if(hostLatticeObject->param.leftBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject->PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in right boundary and within lattice along y
	if(xPos >= hostLatticeObject->param.width && yPos >= 0 && yPos < hostLatticeObject->param.height)
	{
		if(hostLatticeObject->param.rightBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject->PARALLEL_DIRECTOR);
		}
		else if(hostLatticeObject->param.rightBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject->PERPENDICULAR_DIRECTOR);
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

void Lattice::reInitialise(enum LatticeConfig::latticeState initialState)
{
	hostLatticeObject->param.initialState = initialState;

	//we should reset the random seed so we don't generate the set of pseudo random numbers every time	
	cpuSetRandomSeed();
	
	/* Loop through lattice array (hostLatticeObject->lattice[index]) and initialise
	*  Note in C we must use RANDOM,... but if using C++ then must use LatticeConfig::RANDOM , ...
	*/
	int xPos,yPos;
	int index=0;
	double randomAngle;

	for (yPos = 0; yPos < hostLatticeObject->param.height; yPos++)
	{
		for (xPos = 0; xPos < hostLatticeObject->param.width; xPos++)
		{
			index = xPos + (hostLatticeObject->param.width)*yPos;
			switch(hostLatticeObject->param.initialState)
			{

				case LatticeConfig::RANDOM:
				{
					//generate a random angle between 0 & 2*PI radians
					randomAngle = 2*PI*cpuRnd();
					hostLatticeObject->lattice[index].x=cos(randomAngle);
					hostLatticeObject->lattice[index].y=sin(randomAngle);
				}

				break;
				
				case LatticeConfig::PARALLEL_X:
					hostLatticeObject->lattice[index].x=1;
					hostLatticeObject->lattice[index].y=0;
				break;

				case LatticeConfig::PARALLEL_Y:
					hostLatticeObject->lattice[index].x=0;
					hostLatticeObject->lattice[index].y=1;
				break;

				case LatticeConfig::BOT_PAR_TOP_NORM:
					/*
					* This isn't implemented yet
					*/
					hostLatticeObject->lattice[index].x=0;
					hostLatticeObject->lattice[index].y=0;
				break;
				
				default:
					//if we aren't told what to do we will set all zero vectors!
					hostLatticeObject->lattice[index].x=0;
					hostLatticeObject->lattice[index].y=0;

			}
		}
	}

}

void Lattice::translatedUnitVectorDump(enum Lattice::dumpMode mode) const
{

	//print lattice information
	printf("#Lattice Width:%d \n",hostLatticeObject->param.width);
	printf("#Lattice Height:%d \n", hostLatticeObject->param.height);
	printf("#Lattice beta value:%f \n", hostLatticeObject->param.beta);
	printf("#Lattice top Boundary: %d (enum) \n",hostLatticeObject->param.topBoundary);
	printf("#Lattice bottom Boundary: %d (enum) \n",hostLatticeObject->param.bottomBoundary);
	printf("#Lattice left Boundary: %d (enum) \n",hostLatticeObject->param.leftBoundary);
	printf("#Lattice right Boundary: %d (enum) \n",hostLatticeObject->param.rightBoundary);
	printf("#Lattice initial state: %d (enum) \n",hostLatticeObject->param.initialState);

	printf("\n\n # (x) (y) (n_x) (n_y)\n");

	//print lattice state
	int xPos, yPos, index;

	for(yPos=0; yPos < hostLatticeObject->param.height; yPos++)
	{
		for(xPos=0; xPos < hostLatticeObject->param.width; xPos++)
		{
			index = xPos + (hostLatticeObject->param.width)*yPos;
			
			switch(mode)
			{
				case EVERYTHING:	
					printf("%f %f %f %f \n",
					( (double) xPos) - 0.5*(hostLatticeObject->lattice[index].x), 
					( (double) yPos) - 0.5*(hostLatticeObject->lattice[index].y), 
					(hostLatticeObject->lattice[index].x), 
					(hostLatticeObject->lattice[index].y));
				break;

				case PARTICLES:
					printf("%f %f %f %f \n",
					( (double) xPos) - 0.5*(hostLatticeObject->lattice[index].x), 
					( (double) yPos) - 0.5*(hostLatticeObject->lattice[index].y), 
					( (hostLatticeObject->lattice[index].isNanoparticle==1)?(hostLatticeObject->lattice[index].x):0 ), 
					( (hostLatticeObject->lattice[index].isNanoparticle==1)?(hostLatticeObject->lattice[index].y):0 ) 
					);
				break;

				case NOT_PARTICLES:

					printf("%f %f %f %f \n",
					( (double) xPos) - 0.5*(hostLatticeObject->lattice[index].x), 
					( (double) yPos) - 0.5*(hostLatticeObject->lattice[index].y), 
					( (hostLatticeObject->lattice[index].isNanoparticle==0)?(hostLatticeObject->lattice[index].x):0 ), 
					( (hostLatticeObject->lattice[index].isNanoparticle==0)?(hostLatticeObject->lattice[index].y):0 ) 
					);

				break;

				default:
					fprintf(stderr,"Error: drawMode not supported");
			}
		}
	}

	printf("\n#End of Lattice Dump");

}

void Lattice::HalfUnitVectorDump() const
{
	//print lattice information
	printf("#Lattice Width:%d \n",hostLatticeObject->param.width);
	printf("#Lattice Height:%d \n", hostLatticeObject->param.height);
	printf("#Lattice beta value:%f \n", hostLatticeObject->param.beta);
	printf("#Lattice top Boundary: %d (enum) \n",hostLatticeObject->param.topBoundary);
	printf("#Lattice bottom Boundary: %d (enum) \n",hostLatticeObject->param.bottomBoundary);
	printf("#Lattice left Boundary: %d (enum) \n",hostLatticeObject->param.leftBoundary);
	printf("#Lattice right Boundary: %d (enum) \n",hostLatticeObject->param.rightBoundary);
	printf("#Lattice initial state: %d (enum) \n",hostLatticeObject->param.initialState);

	printf("\n\n # (x) (y) (n_x) (n_y)\n");

	//print lattice state
	int xPos, yPos, index;

	for(yPos=0; yPos < hostLatticeObject->param.height; yPos++)
	{
		for(xPos=0; xPos < hostLatticeObject->param.width; xPos++)
		{
			index = xPos + (hostLatticeObject->param.width)*yPos;
			printf("%d %d %f %f \n",
			xPos, 
			yPos, 
			(hostLatticeObject->lattice[index].x)*0.5, 
			(hostLatticeObject->lattice[index].y)*0.5);
		}
	}

	printf("\n#End of Lattice Dump");

}

double Lattice::calculateEnergyOfCell(int xPos, int yPos)
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
	temp = dNxdx_F(this,xPos,yPos) + dNydy_F(this,xPos,yPos);
	firstTerm = temp*temp;

	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	temp = dNxdx_F(this,xPos,yPos) + dNydy_B(this,xPos,yPos);
	firstTerm += temp*temp;	

	//Using B & L (backward differencing in both directions)
	temp = dNxdx_B(this,xPos,yPos) + dNydy_B(this,xPos,yPos);
	firstTerm += temp*temp;

	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	temp = dNxdx_B(this,xPos,yPos) + dNydy_F(this,xPos,yPos);
	firstTerm += temp*temp;

	//Divide by 4 to get average to estimate first term
	firstTerm /= 4.0;

	//Estimate second term by calculating the 4 different ways of calculating the first term and taking the average
	
	//Using T & R (forward differencing in both directions)
	temp = dNydx_F(this,xPos,yPos) - dNxdy_F(this,xPos,yPos);
	secondTerm = temp*temp;

	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	temp = dNydx_F(this,xPos,yPos) - dNxdy_B(this,xPos,yPos);
	secondTerm += temp*temp;

	//Using B & L (backward differencing in both directions)
	temp = dNydx_B(this,xPos,yPos) - dNxdy_B(this,xPos,yPos);
	secondTerm += temp*temp;

	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	temp = dNydx_B(this,xPos,yPos) - dNxdy_F(this,xPos,yPos);
	secondTerm += temp*temp;

	//Divide by 4 to get average to estimate second term
	secondTerm /= 4.0;
	
	//calculate n_x^2
	temp = getN(xPos,yPos)->x;
	temp *=temp;

	temp2 = getN(xPos,yPos)->y;
	temp2 *=temp2;
	
	return 0.5*(firstTerm + (hostLatticeObject->param.beta)*(temp + temp2)*secondTerm );

}

double Lattice::calculateTotalEnergy()
{
	/*
	* This calculation isn't very efficient as it uses calculateEngergyOfCell() for everycell
	* which results in various derivatives being calculated more than once.
	*/

	int xPos,yPos;
	double energy=0;

	for(yPos=0; yPos < (hostLatticeObject->param.height); yPos++)
	{
		for(xPos=0; xPos < (hostLatticeObject->param.width); xPos++)
		{
			energy += calculateEnergyOfCell(xPos,yPos);	
		}
	}

	return energy;

}


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


/* This function returns the correct modulo for dealing with negative a. Note % does not!
* 
* mod(a,b) = a mod b
*/
inline int mod(int a, int b)
{
	return (a%b + b)%b;
}
