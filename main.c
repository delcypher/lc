/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <stdio.h>
#include <stdlib.h>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"


int main()
{
	LatticeConfig configuration;
	
	//setup lattice parameters
	configuration.width =10;
	configuration.height=10;
	//set initial director alignment
	configuration.initialState = RANDOM;

	//set boundary conditions
	configuration.topBoundary = BOUNDARY_PARALLEL;
	configuration.bottomBoundary = BOUNDARY_PERPENDICULAR;
	configuration.leftBoundary = BOUNDARY_PERIODIC;
	configuration.rightBoundary = BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice
	LatticeObject* nSystem = latticeInitialise(configuration);
	
	if(nSystem == NULL)
	{
		fprintf(stderr,"Error: Couldn't construct lattice.");
		return 1;
	}


	DirectorElement* element;

	//loop through lattice boundaries on purpose
	signed int x,y;

	for(y = -1; y <= nSystem->param.height; y++)
	{
		
		for(x = -1; x<= nSystem->param.width; x++)
		{
			element = latticeGetN(nSystem,x,y);
			if(element == NULL)
			{
				fprintf(stderr,"Something went wrong");
			}
		
			printf("%d %d %f %f\n",x,y,element->x,element->y);
		}
		
	}

	//remove lattice
	latticeFree(nSystem);

	return 0;
}
