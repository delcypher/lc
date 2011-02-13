/* Implementation of the LatticeObject functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "randgen.h"
#include "lattice.h"
#include "differentiate.h"
#include <cstring>

//initialisation constructor
Lattice::Lattice(LatticeConfig configuration, int precision) : DUMP_PRECISION(precision)
{
	//check that the width & height have been specified
	if(configuration.width <= 0 || configuration.height <= 0)
	{
		fprintf(stderr, "Error: The width and/or height have not been set to valid values ( > 0). Can't initialise lattice.\n");
	}
	

	//set lattice parameters
	hostLatticeObject.param = configuration;
	
	//Really BAD way to intialise PARALLEL_DIRECTOR
	const DirectorElement tempParallel = {1,0,0};
	memcpy( (DirectorElement*) &(hostLatticeObject.PARALLEL_DIRECTOR),&tempParallel,sizeof(DirectorElement));

	//Really BAD way to intialise PERPENDICULAR_DIRECTOR
	const DirectorElement tempPerpendicular = {0,1,0};
	memcpy( (DirectorElement*) &(hostLatticeObject.PERPENDICULAR_DIRECTOR),&tempPerpendicular,sizeof(DirectorElement));
	
	//allocate memory for lattice (hostLatticeObject[index]) part of array
	hostLatticeObject.lattice = (DirectorElement*) malloc(sizeof(DirectorElement) * (hostLatticeObject.param.width)*(hostLatticeObject.param.height));
	
	if(hostLatticeObject.lattice == NULL)
	{
		fprintf(stderr,"Error: Couldn't allocate memory for lattice array in LatticeObject.\n");
		exit(1);
	}

	//initialise the lattice to a particular state
	reInitialise(hostLatticeObject.param.initialState);


}

//destructor
Lattice::~Lattice()
{
	free(hostLatticeObject.lattice);
}


bool Lattice::add(Nanoparticle* np)
{
	//check nanoparticle location is inside the lattice.
	if( np->getX() >= hostLatticeObject.param.width || np->getX() < 0 || np->getY() >= hostLatticeObject.param.height || np->getX() < 0)
	{
		fprintf(stderr,"Error: Can't add nanoparticle that is not in the lattice.\n");
		return false;
	}

	//Do a dry run adding the nanoparticle. If it fails we know that there is an overlap with an existing nanoparticle
	for(int y=0; y < hostLatticeObject.param.height; y++)
	{
		for(int x=0; x < hostLatticeObject.param.width; x++)
		{
			if(! np->processCell(x,y,Nanoparticle::DRY_ADD, setN(x,y)) )
			{
				fprintf(stderr,"Error: Adding nanoparticle on dry run failed.\n");
				return false;
			}
		}
	}

	//Do actuall run adding the nanoparticle. If it fails we know that there is an overlap with itself
	for(int y=0; y < hostLatticeObject.param.height; y++)
	{
		for(int x=0; x < hostLatticeObject.param.width; x++)
		{
			if(! np->processCell(x,y,Nanoparticle::ADD, setN(x,y)) )
			{
				fprintf(stderr,"Error: Adding nanoparticle on actuall run failed.\n");
				return false;
			}
		}
	}

	return true;

}

const DirectorElement* Lattice::getN(int xPos, int yPos)
{
	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= hostLatticeObject.param.width && hostLatticeObject.param.rightBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, hostLatticeObject.param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && hostLatticeObject.param.leftBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, hostLatticeObject.param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= hostLatticeObject.param.height && hostLatticeObject.param.topBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, hostLatticeObject.param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && hostLatticeObject.param.bottomBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, hostLatticeObject.param.height);
	}
	
	/* All periodic boundary conditions have now been handled
	*/

	/*
	* If the requested "DirectorElement" is in the lattice array just return it.
	*/
	if(xPos >= 0 && xPos < hostLatticeObject.param.width && yPos >= 0 && yPos < hostLatticeObject.param.height)
	{
		return &(hostLatticeObject.lattice[ xPos + (hostLatticeObject.param.width)*yPos ]);
	}

	/*we now know (xPos,yPos) isn't in lattice so either (xPos,yPos) is on the PARALLEL or PERPENDICULAR
	* boundary or an invalid point has been requested
	*/

	//in top boundary and within lattice along x
	if(yPos >= hostLatticeObject.param.height && xPos >= 0 && xPos < hostLatticeObject.param.width)
	{
		if(hostLatticeObject.param.topBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject.PARALLEL_DIRECTOR);
		} 
		else if(hostLatticeObject.param.topBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject.PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in bottom boundary and within lattice along x
	if(yPos <= -1 && xPos >= 0 && xPos < hostLatticeObject.param.width)
	{
		if(hostLatticeObject.param.bottomBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject.PARALLEL_DIRECTOR);
		}
		else if(hostLatticeObject.param.bottomBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject.PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in left boundary and within lattice along y
	if(xPos <= -1 && yPos >= 0 && yPos < hostLatticeObject.param.height)
	{
		if(hostLatticeObject.param.leftBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject.PARALLEL_DIRECTOR);
		}
		else if(hostLatticeObject.param.leftBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject.PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			return NULL;
		}
	}

	//in right boundary and within lattice along y
	if(xPos >= hostLatticeObject.param.width && yPos >= 0 && yPos < hostLatticeObject.param.height)
	{
		if(hostLatticeObject.param.rightBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(hostLatticeObject.PARALLEL_DIRECTOR);
		}
		else if(hostLatticeObject.param.rightBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(hostLatticeObject.PERPENDICULAR_DIRECTOR);
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


DirectorElement* Lattice::setN(int xPos, int yPos)
{

	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= hostLatticeObject.param.width && hostLatticeObject.param.rightBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, hostLatticeObject.param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && hostLatticeObject.param.leftBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, hostLatticeObject.param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= hostLatticeObject.param.height && hostLatticeObject.param.topBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, hostLatticeObject.param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && hostLatticeObject.param.bottomBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, hostLatticeObject.param.height);
	}
	
	/* All periodic boundary conditions have now been handled
	*/

	/*
	* If the requested "DirectorElement" is in the lattice array just return it.
	*/
	if(xPos >= 0 && xPos < hostLatticeObject.param.width && yPos >= 0 && yPos < hostLatticeObject.param.height)
	{
		return &(hostLatticeObject.lattice[ xPos + (hostLatticeObject.param.width)*yPos ]);
	}

	/*we now know (xPos,yPos) isn't in lattice so either (xPos,yPos) is on the PARALLEL or PERPENDICULAR
	* boundary or an invalid point has been requested
	*/

	//in top boundary and within lattice along x OR
	//in bottom boundary and within lattice along x OR
	//in left boundary and within lattice along y OR
	//in right boundary and within lattice along y OR
	if
	( 
		(yPos >= hostLatticeObject.param.height && xPos >= 0 && xPos < hostLatticeObject.param.width) ||
		(yPos <= -1 && xPos >= 0 && xPos < hostLatticeObject.param.width) ||
		(xPos <= -1 && yPos >= 0 && yPos < hostLatticeObject.param.height) ||
		(xPos >= hostLatticeObject.param.width && yPos >= 0 && yPos < hostLatticeObject.param.height)
	)
	{
		//We shouldn't be trying to change the boundary!
		fprintf(stderr,"Error: setN() tried to access boundary at (%d,%d).\n",xPos,yPos);
		return NULL;
	}

	//Every case should already of been handled. An invalid point (xPos,yPos) must of been asked for
	fprintf(stderr,"Error: setN() point (%d,%d) which does NOT exist in lattice.\n",xPos,yPos);
	return NULL;

}

void Lattice::reInitialise(enum LatticeConfig::latticeState initialState)
{
	hostLatticeObject.param.initialState = initialState;

	//we should reset the random seed so we don't generate the set of pseudo random numbers every time	
	setSeed();
	
	/* Loop through lattice array (hostLatticeObject.lattice[index]) and initialise
	*  Note in C we must use RANDOM,... but if using C++ then must use LatticeConfig::RANDOM , ...
	*/
	int xPos,yPos;
	int index=0;
	double angle;
	bool badState=false;

	for (yPos = 0; yPos < hostLatticeObject.param.height; yPos++)
	{
		for (xPos = 0; xPos < hostLatticeObject.param.width; xPos++)
		{
			index = xPos + (hostLatticeObject.param.width)*yPos;
			switch(hostLatticeObject.param.initialState)
			{

				case LatticeConfig::RANDOM:
				{
					//generate a random angle between 0 & 2*PI radians
					angle = 2*PI*rnd();
					hostLatticeObject.lattice[index].x=cos(angle);
					hostLatticeObject.lattice[index].y=sin(angle);
				}

				break;
				
				case LatticeConfig::PARALLEL_X:
					hostLatticeObject.lattice[index].x=1;
					hostLatticeObject.lattice[index].y=0;
				break;

				case LatticeConfig::PARALLEL_Y:
					hostLatticeObject.lattice[index].x=0;
					hostLatticeObject.lattice[index].y=1;
				break;

				case LatticeConfig::K1_EQUAL_K3:
				{
					angle = PI*( (double) (yPos + 1)/(2*(hostLatticeObject.param.height +1)) );
					hostLatticeObject.lattice[index].x=cos(angle);
					hostLatticeObject.lattice[index].y=sin(angle);
				}

				break;

				case LatticeConfig::K1_DOMINANT:
				{
					//the cast to double is important else we will do division with ints and discard remainder
					angle = PI/2 - acos( (double) (yPos + 1)/(hostLatticeObject.param.height + 1));
					hostLatticeObject.lattice[index].x=cos(angle);
					hostLatticeObject.lattice[index].y=sin(angle);
				}

				break;

				case LatticeConfig::K3_DOMINANT:
				{
					//the cast to double is important else we will do division with ints and discard remainder
					angle = PI/2 -asin(1 - (double) (yPos +1)/(hostLatticeObject.param.height +1)   );
					hostLatticeObject.lattice[index].x=cos(angle);
					hostLatticeObject.lattice[index].y=sin(angle);
				}
				break;

				default:
					//if we aren't told what to do we will set all zero vectors!
					hostLatticeObject.lattice[index].x=0;
					hostLatticeObject.lattice[index].y=0;
					badState=true;

			}
		}
	}

	if(badState)
	{
		fprintf(stderr,"Error: Lattice has been but in bad state as supplied initial state %d is not supported.\n",hostLatticeObject.param.initialState);
	}

}

void Lattice::nDump(enum Lattice::dumpMode mode, FILE* stream)
{

	//print lattice information
	fprintf(stream,"#Lattice Width:%d \n",hostLatticeObject.param.width);
	fprintf(stream,"#Lattice Height:%d \n", hostLatticeObject.param.height);
	fprintf(stream,"#Lattice beta value:%f \n", hostLatticeObject.param.beta);
	fprintf(stream,"#Lattice top Boundary: %d (enum) \n",hostLatticeObject.param.topBoundary);
	fprintf(stream,"#Lattice bottom Boundary: %d (enum) \n",hostLatticeObject.param.bottomBoundary);
	fprintf(stream,"#Lattice left Boundary: %d (enum) \n",hostLatticeObject.param.leftBoundary);
	fprintf(stream,"#Lattice right Boundary: %d (enum) \n",hostLatticeObject.param.rightBoundary);
	fprintf(stream,"#Lattice initial state: %d (enum) \n",hostLatticeObject.param.initialState);

	fprintf(stream,"\n\n # (x) (y) (n_x) (n_y)\n");

	//print lattice state
	int xPos, yPos, xInitial, yInitial, xFinal, yFinal;
	
	if(mode==BOUNDARY)
	{
		//in BOUNDARY mode we go +/- 1 outside of lattice.
		xInitial=-1;
		yInitial=-1;
		xFinal= hostLatticeObject.param.width;
		yFinal= hostLatticeObject.param.height;
	}
	else
	{
		//not in boundary mode so we will dump just in lattice
		xInitial=0;
		yInitial=0;
		xFinal = hostLatticeObject.param.width -1;
		yFinal = hostLatticeObject.param.height -1;
	}

	for(yPos=yInitial; yPos <= yFinal ; yPos++)
	{
		for(xPos=xInitial; xPos <= xFinal; xPos++)
		{
			
			switch(mode)
			{
				case EVERYTHING:	
					fprintf(stream,"%f %f %.*f %.*f \n",
					( (double) xPos) - 0.5*(getN(xPos,yPos)->x), 
					( (double) yPos) - 0.5*(getN(xPos,yPos)->y), 
					DUMP_PRECISION,(getN(xPos,yPos)->x), 
					DUMP_PRECISION,(getN(xPos,yPos)->y));
				break;

				case PARTICLES:
					fprintf(stream,"%f %f %.*f %.*f \n",
					( (double) xPos) - 0.5*(getN(xPos,yPos)->x), 
					( (double) yPos) - 0.5*(getN(xPos,yPos)->y), 
					DUMP_PRECISION,( (getN(xPos,yPos)->isNanoparticle==1)?(getN(xPos,yPos)->x):0 ), 
					DUMP_PRECISION,( (getN(xPos,yPos)->isNanoparticle==1)?(getN(xPos,yPos)->y):0 ) 
					);
				break;

				case NOT_PARTICLES:

					fprintf(stream,"%f %f %.*f %.*f \n",
					( (double) xPos) - 0.5*(getN(xPos,yPos)->x), 
					( (double) yPos) - 0.5*(getN(xPos,yPos)->y), 
					DUMP_PRECISION,( (getN(xPos,yPos)->isNanoparticle==0)?(getN(xPos,yPos)->x):0 ), 
					DUMP_PRECISION,( (getN(xPos,yPos)->isNanoparticle==0)?(getN(xPos,yPos)->y):0 ) 
					);

				break;
				
				case BOUNDARY:
					if(xPos==xInitial || xPos==xFinal || yPos==yInitial || yPos==yFinal)
					{
						fprintf(stream,"%f %f %.*f %.*f \n",
						( (double) xPos) - 0.5*(getN(xPos,yPos)->x), 
						( (double) yPos) - 0.5*(getN(xPos,yPos)->y), 
						DUMP_PRECISION,(getN(xPos,yPos)->x), 
						DUMP_PRECISION,(getN(xPos,yPos)->y));

					}
				break;

				default:
					fprintf(stderr,"Error: drawMode not supported");
			}
		}
	}

	fprintf(stream,"\n#End of Lattice Dump");

}

void Lattice::indexedNDump(FILE* stream)
{
	fprintf(stream,"\n#BOUNDARY DUMP\n");
	nDump(BOUNDARY,stream);
	fprintf(stream,"\n#NOT_PARTICLES\n\n");
	nDump(NOT_PARTICLES,stream);
	fprintf(stream,"\n#PARTICLES\n\n");
	nDump(PARTICLES,stream);
	fprintf(stream,"\n\n");
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

	//Estimate second term by calculating the 4 different ways of calculating the second term and taking the average
	
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
	
	return 0.5*(firstTerm + (hostLatticeObject.param.beta)*(temp + temp2)*secondTerm );

}

double Lattice::calculateTotalEnergy()
{
	/*
	* This calculation isn't very efficient as it uses calculateEngergyOfCell() for everycell
	* which results in various derivatives being calculated more than once.
	*/

	int xPos,yPos;
	double energy=0;

	for(yPos=0; yPos < (hostLatticeObject.param.height); yPos++)
	{
		for(xPos=0; xPos < (hostLatticeObject.param.width); xPos++)
		{
			energy += calculateEnergyOfCell(xPos,yPos);	
		}
	}

	return energy;

}


/* This function calculates & returns the cosine of the angle between two DirectorElements (must be passed as pointers)
*
*/
double calculateCosineBetween(const DirectorElement* a, const DirectorElement* b)
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


