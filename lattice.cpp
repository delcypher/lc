/* Implementation of the LatticeObject functions
*  By Alex Allen & Daniel Liew (2010)
*/

#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cmath>
#include "randgen.h"
#include "lattice.h"
#include <cstring>
#include <fstream>

/* Include the nanoparticles header files that the Lattice class needs
   to handle here
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"

using namespace std;

//initialisation constructor
Lattice::Lattice(LatticeConfig configuration) : 
constructedFromFile(false) , 
PARALLEL_DIRECTOR(1,0,false) , 
PERPENDICULAR_DIRECTOR(0,1,false), 
DUMMY_DIRECTOR(0,0,0), 
CORNER_DIRECTOR(1/sqrt(2),1/sqrt(2),false)
{
	//set initial badState
	badState=false;

	//check that the width & height have been specified
	if(configuration.width <= 0 || configuration.height <= 0)
	{
		cerr << "Error: The width and/or height have not been set to valid values ( > 0). Can't initialise lattice." << endl;
		badState=true;
	}
	

	//set lattice parameters
	param = configuration;
	
	//allocate memory for lattice (index]) part of array
	lattice = (DirectorElement*) malloc(sizeof(DirectorElement) * (param.width)*(param.height));
	
	if(lattice == NULL)
	{
		cerr << "Error: Couldn't allocate memory for lattice array in LatticeObject.\n" << endl;
		badState=true;
		exit(1);
	}

	//set every lattice point to not be a nanoparticle
	for(int point=0; point < (param.width)*(param.height) ; point++)
	{
		lattice[point].isNanoparticle=false;
	}
	
	//set the number of nanoparticles associated with the lattice to 0
	mNumNano=0;
	mNanoparticles=NULL;
	

	//initialise the lattice to a particular state
	reInitialise(param.initialState);


}


//constructor for savedStates
Lattice::Lattice(const char* filepath) : 
constructedFromFile(true) , 
PARALLEL_DIRECTOR(1,0,false) , 
PERPENDICULAR_DIRECTOR(0,1,false), 
DUMMY_DIRECTOR(0,0,false), 
CORNER_DIRECTOR(1/sqrt(2),1/sqrt(2),false)
{
	/* Assume following ordering of binary blocks
	*  <configuration><mNumNano><lattice><Nanoparticle_1_ID><Nanoparticle_1_data><Nanoparticle_2_ID><Nanoparticle_2_data>...
	*
	*/

	badState=false;
	mNumNano=0;
	mNanoparticles=NULL;

	param.width=0;
	param.height=0;

	//allocate memory for lattice (index]) part of array
        lattice = (DirectorElement*) malloc(sizeof(DirectorElement) * (param.width)*(param.height));
        
        if(lattice == NULL)
        {
                cerr << "Error: Couldn't allocate memory for lattice array in LatticeObject.\n" << endl;
                badState=true;
                exit(1);
        }


	ifstream state(filepath, ios::binary | ios::in);
	
	if(!state.is_open())
	{
		cerr << "Error: Couldn't open file " << filepath << " to load state from." << endl;
		state.close();
		badState=true;
		exit(1);
	}	
	
	//read in parameters
	state.read( (char*) &(param), sizeof(LatticeConfig));

	if(!state.good())
	{
		cerr << "Error: Couldn't read lattice parameters from file " << filepath << endl;
		state.close();
		badState=true;
		exit(1);
	}	
	
	//read number of Nanoparticles
	state.read( (char*) &mNumNano, sizeof(int));
	
	if(!state.good())
	{
		cerr << "Error: Couldn't read the number of nanoparticles from file " << filepath << endl;
		state.close();
		badState=true;
		exit(1);
	}

	//allocate memory for the lattice
	lattice = (DirectorElement*) calloc( (param.width)*(param.height),sizeof(DirectorElement));
	
	if(lattice ==NULL)
	{
		cerr << "Error: Couldn't allocate memory for lattice array" << endl;
		badState=true;
		exit(1);
	}

	//read the saved state lattice into the allocated memory for lattice
	state.read( (char*) lattice, sizeof(DirectorElement)*(param.width)*(param.height) );

	if(!state.good())
	{
		cerr << "Error: Couldn't read lattice array from file " << filepath << endl;
		badState=true;
		exit(1);
	}

	//allocate memory for array of pointers to nanoparticles
	if(mNumNano>0)
	{
		mNanoparticles = (Nanoparticle**) calloc(mNumNano,sizeof(Nanoparticle*));

		if(mNanoparticles==NULL)
		{
			cerr << "Error: Couldn't allocate memory for array of pointers to Nanoparticles" << endl;
			badState=true;
			exit(1);
		}
	
		//loop through nanoparticles assuming data format <size><data>
		for(int counter=0; counter < mNumNano; counter++)
		{
			enum Nanoparticle::types id= (Nanoparticle::types) -1;
			
			//get the type of nanoparticle
			state.read( (char*) &(id), sizeof(enum Nanoparticle::types));
			
			//Pick the correct constructor to use based on the id and allocate memory for it.
			switch(id)
			{
				case Nanoparticle::CIRCLE :
					mNanoparticles[counter] = new CircularNanoparticle(state);
					break;

				case Nanoparticle::ELLIPSE :
					mNanoparticles[counter] = new EllipticalNanoparticle(state);
					break;

				default:
					cerr << "Error: Lattice constructor does not support nanoparticle of type " << id << " (enum) " << endl;
					badState=true;
			}	
			
			if(!state.good())
			{
				cerr << "Error: Failed to create Nanoparticle  " << counter << endl;
				badState=true;
				exit(1);
			}
		}	

	}
	
	

	state.close();
}

//destructor
Lattice::~Lattice()
{
	//if this lattice was constructed from a file then memory was allocated for each nanoparticle, we should free it!
	if(constructedFromFile && mNumNano > 0)
	{
		for(int counter=0; counter < mNumNano; counter++)
		{
			delete (mNanoparticles[counter]);
		}
	}

	free(mNanoparticles);
	free(lattice);
}


bool Lattice::add(Nanoparticle& np)
{
	//check nanoparticle location is inside the lattice.
	if( np.getX() >= param.width || np.getX() < 0 || np.getY() >= param.height || np.getX() < 0)
	{
		cerr << "Error: Can't add nanoparticle that is not in the lattice.\n" << endl;
		
		//don't need to set badState as nothing has been changed yet
		
		return false;
	}

	//Do a dry run adding the nanoparticle. If it fails we know that there is an overlap with an existing nanoparticle
	for(int y=0; y < param.height; y++)
	{
		for(int x=0; x < param.width; x++)
		{
			if(! np.processCell(x,y,Nanoparticle::DRY_ADD, setN(x,y)) )
			{
				cerr << "Error: Adding nanoparticle on dry run failed." << endl;
				badState=true;
				return false;
			}
		}
	}

	//Do actuall run adding the nanoparticle. If it fails we know that there is an overlap with itself
	for(int y=0; y < param.height; y++)
	{
		for(int x=0; x < param.width; x++)
		{
			if(! np.processCell(x,y,Nanoparticle::ADD, setN(x,y)) )
			{
				cerr << "Error: Adding nanoparticle on actuall run failed." << endl;
				badState=true;
				return false;
			}
		}
	}
	
	//add pointer to nanoparticle to array so the lattice can keep track of the nanoparticles it has

	//deal with the case that a nanoparticle has never been added.
	if(mNumNano==0)
	{
		//This is the first time a nanoparticle has been added so need to allocate memory for array
		mNanoparticles = (Nanoparticle**) calloc(1,sizeof(Nanoparticle*));
		if(mNanoparticles==NULL)
		{
			cerr << "Error: Couldn't allocated memory for Nanoparticle array in Lattice." << endl;
			badState=true;
			return false;
		}
		mNumNano++;
		mNanoparticles[0] = &np;

	}
	else
	{
		//Need to allocate memory for new array
		Nanoparticle** tempArray = (Nanoparticle**) calloc(mNumNano +1,sizeof(Nanoparticle*));
		if(tempArray==NULL)
		{
			cerr << "Error: Couldn't allocate memory for Nanoparticle array in Lattice when resizing" << endl;
			badState=true;
			return false;

		}

		//copy pointers accross
		for(int counter=0; counter < mNumNano; counter++)
		{
			tempArray[counter] = mNanoparticles[counter];
		}

		//add new nanoparticle
		tempArray[mNumNano] = &np;
		mNumNano++;

		//Delete old array on free store
		free(mNanoparticles);

		//Make temporary array the new Nanoparticle pointer array
		mNanoparticles = tempArray;


	}
	
	return true;

}


const DirectorElement* Lattice::getN(int xPos, int yPos) const
{
	/*
	* If the requested "DirectorElement" is in the lattice array just return it.
	* We do this first (and then again after handling periodic boundary conditions)
	* because the majority of calls to getN() should be in the lattice.
	*/
	if(xPos >= 0 && xPos < param.width && yPos >= 0 && yPos < param.height)
	{
		return &(lattice[ xPos + (param.width)*yPos ]);
	}
	
	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/

	//Handle xPos going off the lattice in the x direction to the right
	if(xPos >= param.width && param.rightBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, param.width);
	}

	//Handle xPos going off the lattice in x direction to the left
	if(xPos < 0 && param.leftBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		xPos = mod(xPos, param.width);
	}

	//Handle yPos going off the lattice in the y directory to the top
	if(yPos >= param.height && param.topBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, param.height);
	}

	//Handle yPos going off the lattice in the y directory to the bottom
	if(yPos < 0  && param.bottomBoundary == LatticeConfig::BOUNDARY_PERIODIC)
	{
		yPos = mod(yPos, param.height);
	}
	
	/* All periodic boundary conditions have now been handled
	*/

	/*
	* If the requested "DirectorElement" is in the lattice array just return it.
	*/
	if(xPos >= 0 && xPos < param.width && yPos >= 0 && yPos < param.height)
	{
		return &(lattice[ xPos + (param.width)*yPos ]);
	}

	/*we now know (xPos,yPos) isn't in lattice so either (xPos,yPos) is on the PARALLEL or PERPENDICULAR
	* boundary or an invalid point has been requested
	*/

	//in top boundary and within lattice along x
	if(yPos >= param.height && xPos >= 0 && xPos < param.width)
	{
		if(param.topBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(PARALLEL_DIRECTOR);
		} 
		else if(param.topBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			cerr << "Error: Attempt to access boundary at (" << xPos << "," << yPos << ") where an unsupported boundary is set." << endl;
			return &(DUMMY_DIRECTOR);
		}
	}

	//in bottom boundary and within lattice along x
	if(yPos <= -1 && xPos >= 0 && xPos < param.width)
	{
		if(param.bottomBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(PARALLEL_DIRECTOR);
		}
		else if(param.bottomBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			cerr << "Error: Attempt to access boundary at (" << xPos << "," << yPos << ") where an unsupported boundary is set." << endl;
			return &(DUMMY_DIRECTOR);
		}
	}

	//in left boundary and within lattice along y
	if(xPos <= -1 && yPos >= 0 && yPos < param.height)
	{
		if(param.leftBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(PARALLEL_DIRECTOR);
		}
		else if(param.leftBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			cerr << "Error: Attempt to access boundary at (" << xPos << "," << yPos << ") where an unsupported boundary is set." << endl;
			return &(DUMMY_DIRECTOR);
		}
	}

	//in right boundary and within lattice along y
	if(xPos >= param.width && yPos >= 0 && yPos < param.height)
	{
		if(param.rightBoundary == LatticeConfig::BOUNDARY_PARALLEL)
		{
			return &(PARALLEL_DIRECTOR);
		}
		else if(param.rightBoundary == LatticeConfig::BOUNDARY_PERPENDICULAR)
		{
			return &(PERPENDICULAR_DIRECTOR);
		}
		else
		{
			//Boundary cases should of already been handled, something went wrong if we get here!
			cerr << "Error: Attempt to access boundary at (" << xPos << "," << yPos << ") where an unsupported boundary is set." << endl;
			return &(DUMMY_DIRECTOR);
		}
	}

	/* Handle case with non-periodic boundary conditions asking for undefined points (-1,-1);(width,-1);(-1,height);(width,height)
	*  We will just return the dummy director. This will occur when asking to calculate the energy of a cell in the boundary
	*/
	if( (xPos==-1 && yPos==-1 && param.bottomBoundary!= LatticeConfig::BOUNDARY_PERIODIC && param.leftBoundary != LatticeConfig::BOUNDARY_PERIODIC) ||
	    (xPos==param.width && yPos==-1 && param.bottomBoundary != LatticeConfig::BOUNDARY_PERIODIC && param.rightBoundary != LatticeConfig::BOUNDARY_PERIODIC) ||
	    (xPos==-1 && yPos==param.height && param.topBoundary != LatticeConfig::BOUNDARY_PERIODIC && param.leftBoundary != LatticeConfig::BOUNDARY_PERIODIC) ||
	    (xPos==param.width && yPos==param.height && param.topBoundary != LatticeConfig::BOUNDARY_PERIODIC && param.rightBoundary != LatticeConfig::BOUNDARY_PERIODIC)
	  )
	{
		return &(CORNER_DIRECTOR);
	}


	//Every case should already of been handled. An invalid point (xPos,yPos) must of been asked for
	cerr << "Error: Attempt to access boundary a point (" << xPos << ","  << yPos << ") which couldn't be handled!" << endl;
	return &(DUMMY_DIRECTOR);

}


DirectorElement* Lattice::setN(int xPos, int yPos)
{

	/* set xPos & yPos in the lattice taking into account periodic boundary conditions
	*  of the 2D lattice
	*/
	return (DirectorElement*) getN(xPos,yPos);
}

void Lattice::reInitialise(enum LatticeConfig::latticeState initialState)
{
	param.initialState = initialState;

	//we should reset the random seed so we don't generate the set of pseudo random numbers every time	
	setSeed();
	
	/* Loop through lattice array (lattice[index]) and initialise
	*  Note in C we must use RANDOM,... but if using C++ then must use LatticeConfig::RANDOM , ...
	*/
	int xPos,yPos;
	int index=0;
	double angle;
	bool badEnum=false;

	for (yPos = 0; yPos < param.height; yPos++)
	{
		for (xPos = 0; xPos < param.width; xPos++)
		{
			index = xPos + (param.width)*yPos;

			//only set if the lattice cell isn't a nanoparticle.
			if(lattice[index].isNanoparticle==false)
			{
				switch(param.initialState)
				{

					case LatticeConfig::RANDOM:
					{
						//generate a random angle between 0 & 2*PI radians
						angle = 2*PI*rnd();
						lattice[index].setAngle(angle);
					}

					break;
					
					case LatticeConfig::PARALLEL_X:
						lattice[index].x=1;
						lattice[index].y=0;
					break;

					case LatticeConfig::PARALLEL_Y:
						lattice[index].x=0;
						lattice[index].y=1;
					break;

					case LatticeConfig::K1_EQUAL_K3:
					{
						angle = PI*( (double) (yPos + 1)/(2*(param.height +1)) );
						lattice[index].setAngle(angle);
					}

					break;

					case LatticeConfig::K1_DOMINANT:
					{
						//the cast to double is important else we will do division with ints and discard remainder
						angle = PI/2 - acos( (double) (yPos + 1)/(param.height + 1));
						lattice[index].setAngle(angle);
					}

					break;

					case LatticeConfig::K3_DOMINANT:
					{
						//the cast to double is important else we will do division with ints and discard remainder
						angle = PI/2 -asin(1 - (double) (yPos +1)/(param.height +1)   );
						lattice[index].setAngle(angle);
					}
					break;

					default:
						//if we aren't told what to do we will set all zero vectors!
						lattice[index].x=0;
						lattice[index].y=0;
						badState=true;
						badEnum=true;

				}
			}
		}
	}

	if(badEnum)
	{
		cerr << "Error: Lattice has been put in bad state as supplied initial state " << 
		param.initialState <<
		" is not supported." << endl;
	}

}

void Lattice::nDump(enum Lattice::dumpMode mode, std::ostream& stream) const
{
	stream << "# (x) (y) (n_x) (n_y)\n";

	//print lattice state
	int xPos, yPos, xInitial, yInitial, xFinal, yFinal;
	
	if(mode==BOUNDARY)
	{
		//in BOUNDARY mode we go +/- 1 outside of lattice.
		xInitial=-1;
		yInitial=-1;
		xFinal= param.width;
		yFinal= param.height;
	}
	else
	{
		//not in boundary mode so we will dump just in lattice
		xInitial=0;
		yInitial=0;
		xFinal = param.width -1;
		yFinal = param.height -1;
	}

	for(yPos=yInitial; yPos <= yFinal ; yPos++)
	{
		for(xPos=xInitial; xPos <= xFinal; xPos++)
		{
			
			switch(mode)
			{
				case EVERYTHING:	
					stream << ( ( (double) xPos) - 0.5*(getN(xPos,yPos)->x) ) << " " <<
						( ( (double) yPos) - 0.5*(getN(xPos,yPos)->y) ) << " " <<
						(getN(xPos,yPos)->x) << " " <<
						(getN(xPos,yPos)->y) << "\n";
				break;

				case PARTICLES:
					stream << ( ( (double) xPos) - 0.5*(getN(xPos,yPos)->x) ) << " " <<
						( ( (double) yPos) - 0.5*(getN(xPos,yPos)->y) ) << " " <<
						( (getN(xPos,yPos)->isNanoparticle==true)?(getN(xPos,yPos)->x):0 ) << " " <<
						( (getN(xPos,yPos)->isNanoparticle==true)?(getN(xPos,yPos)->y):0 ) << "\n";
				break;

				case NOT_PARTICLES:

					stream << ( ( (double) xPos) - 0.5*(getN(xPos,yPos)->x) ) << " " <<
						( ( (double) yPos) - 0.5*(getN(xPos,yPos)->y) ) << " " <<
						( (getN(xPos,yPos)->isNanoparticle==false)?(getN(xPos,yPos)->x):0 ) << " " <<
						( (getN(xPos,yPos)->isNanoparticle==false)?(getN(xPos,yPos)->y):0 ) << "\n";

				break;
				
				case BOUNDARY:
					if(xPos==xInitial || xPos==xFinal || yPos==yInitial || yPos==yFinal)
					{
						stream << ( ( (double) xPos) - 0.5*(getN(xPos,yPos)->x) ) << " " <<
							( ( (double) yPos) - 0.5*(getN(xPos,yPos)->y) ) << " " <<
							(getN(xPos,yPos)->x) << " " <<
							(getN(xPos,yPos)->y) << "\n";

					}
				break;

				default:
					cerr << "Error: dumpMode not supported" << endl;
			}
		}
	}

	stream << "#End of Lattice Dump\n\n\n";
	stream.flush();

}

void Lattice::indexedNDump(std::ostream& stream) const
{
	dumpDescription(stream);

	//Note that indexes must be seperated by two newlines for GNUplot

	stream << "\n#BOUNDARY DUMP\n";
	nDump(BOUNDARY,stream);
	
	stream << "#NOT_PARTICLES\n";
	nDump(NOT_PARTICLES,stream);

	stream << "#PARTICLES\n";
	nDump(PARTICLES,stream);

	stream.flush();
}

void Lattice::dumpDescription(std::ostream& stream) const
{
	stream << "#Lattice Parameters:\n" << 
		"#State is " << (badState?"Bad":"Good") << "\n" <<
		"#Lattice Width:" << param.width << "\n" <<
		"#Lattice Height:" << param.height << "\n" <<
		"#Beta:" << param.beta << "\n" <<
		"#Top Boundary (enum):" << param.topBoundary << "\n" <<
		"#Bottom Boundary (enum):" << param.bottomBoundary << "\n" <<
		"#Left Boundary (enum):" << param.leftBoundary << "\n" <<
		"#Right Boundary (enum):" << param.rightBoundary << "\n" <<
		"#Initial State (enum):" << param.initialState << "\n" <<
		"#Number of Nanoparticles:" << mNumNano << "\n" <<
		"#State:" << (badState?"Bad":"Good") << "\n" <<
		"#Monte Carlo parameters:" << "\n\n" <<
		"#1/TK :" << param.iTk << "\n" <<
		"#Current Monte Carlo step:" << param.mStep << "\n" <<
		"#Accept Counter:" << param.acceptCounter << "\n" <<
		"#Reject Counter:" << param.rejectCounter << "\n" <<
		"#Current Acceptance angle:" << param.aAngle << "\n" <<
		"#Desired Acceptance ratio:" << param.desAcceptRatio << "\n" <<
		"#" << "\n" <<
		"#Nanoparticle cells in lattice:" << getNanoparticleCellCount() << "/" << getArea() << " (" << ( (double) 100*getNanoparticleCellCount()/getArea() ) << " %)" << "\n" <<
		"#\n" <<
		"#Total Free energy of lattice:" << calculateTotalEnergy() << "\n" <<
		"#Average free energy per unit volume of cell:" << calculateAverageEnergy() << "\n";
		if(mNanoparticles!=NULL)
		{
			for(int counter=0; counter < mNumNano; counter++)
			{
				stream << "#Particle[" << counter << "] :";
				if( mNanoparticles[counter] !=NULL)
				{
					stream << mNanoparticles[counter]->getDescription() << "\n";
				}
				else
				{
					stream << "Error: Couldn't get description.";
				}
			}
		}
	
	stream.flush();
}

inline double Lattice::calculateCosineBetween(const DirectorElement* C, const DirectorElement* O, const double& flipSign) const
{
	/* This is calculated using the definition of the dot product cos(theta) = a.b /|a||b|
	*  between two vectors "a" and "b". It also uses the definition of the dot product that is
	*  a.b = (a->x)*(b->x) + (a->y)*(b->y)
	*/


	/* Note we drop the denominator (|a||b|) as the director should be a unit vector.
	*  denominator because the director should be a unit vector, hence the denominator should be 1.
	*/
	return flipSign*( (C->x)*(O->x) + (C->y)*(O->y) );

}
double Lattice::calculateEnergyOfCell(int xPos, int yPos) const
{
	/*   |T|     y|
	*  |L|C|R|    |
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

	//Get pointers to T,B,L,R & C cells
	const DirectorElement* T = getN(xPos, yPos +1);
	const DirectorElement* B = getN(xPos, yPos -1);
	const DirectorElement* L = getN(xPos -1, yPos);
	const DirectorElement* R = getN(xPos +1, yPos);
	const DirectorElement* C = getN(xPos,yPos);
	
	/* set original flip sign to 1 (we assume no initial flipping needed).
	*  This flip sign says what the sign of the components in the C (central) cell are.
	*  If we flip that cell all components of that cell get mulitplied by -1.
	*/
	double flipSign=1;
	
	/* The convention for deriviate variables names is d{component}{w.r.t. variable}_{estimate_method}
	*  For example dNxdy_B => differentiate the x component of N (Nx) w.r.t y using backward differencing.
	* 
	*  We use to different schemes for estimating the partial derivitive.
	*  Forward differencing : df/dx at point (x_0,y_0) = ( f( x_0 + L, y_0) - f(x_0,y_0) )/L
	*  Backward differencing : df/dx at point (x_0,y_0) = ( f(x_0,y_0) - f(x_0 -L, y_0) )/L
	*
	* Note in our code L=1 so we don't do the division.  
	*
	*  For Each derivitive we calculate what the (smallest) angle between DirectorElements are for
	*  the derivitive. If this is > 90 degrees we flip the Centre (C) DirectorElement by 180degrees.
	*  This effectively changes the sign of the components of the Centre DirectorElement (C). We do this
	*  because of unixial property of LCs (i.e. n = -n). Note that derivitive varies under the change of sign
	*  (but the free energy per unit volume does not). Our estimates of the derivitive are also vary on the change
	*  of sign so we need to pick a method to consistently obtain the same derivitive for n or -n. The above method
	*  is one of the ways of doing this.
	* 
	*/
		
	//calculate derivatives for first term and second term

	if(calculateCosineBetween(C,R,flipSign) < 0) flipSign *= -1;
	double dNxdx_F = R->x - flipSign*(C->x);
	double dNydx_F = R->y - flipSign*(C->y);

	if(calculateCosineBetween(C,L,flipSign) < 0) flipSign *= -1;
	double dNxdx_B = flipSign*(C->x) - L->x;
	double dNydx_B = flipSign*(C->y) - L->y; 

	if(calculateCosineBetween(C,T,flipSign) < 0) flipSign *= -1;
	double dNydy_F = T->y - flipSign*(C->y);
	double dNxdy_F = T->x - flipSign*(C->x);

	if(calculateCosineBetween(C,B,flipSign) < 0) flipSign *= -1;
	double dNydy_B = flipSign*(C->y) - B->y;
	double dNxdy_B = flipSign*(C->x) - B->x;

	double firstTerm=0;
	double secondTerm=0;

	//Estimate first term by calculating the 4 different ways of calculating the first term and taking the average
	//Using T & R (forward differencing in both directions)
	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	//Using B & L (backward differencing in both directions)
	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	firstTerm = ( (dNxdx_F + dNydy_F)*(dNxdx_F + dNydy_F) +
		(dNxdx_F + dNydy_B)*(dNxdx_F + dNydy_B) +
		(dNxdx_B + dNydy_B)*(dNxdx_B + dNydy_B) +
		(dNxdx_B + dNydy_F)*(dNxdx_B + dNydy_F) ) /4.0;
	
	//Estimate second term by calculating the 4 different ways of calculating the second term and taking the average
	//Using T & R (forward differencing in both directions)
	//Using B & R (forward differencing in x direction & backward differencing in y direction)
	//Using B & L (backward differencing in both directions)
	//Using T & L (backward differencing in x direction & forward differencing in y direction)
	secondTerm = ( (dNydx_F - dNxdy_F)*(dNydx_F - dNxdy_F) +
		(dNydx_F - dNxdy_B)*(dNydx_F - dNxdy_B) +
		(dNydx_B - dNxdy_B)*(dNydx_B - dNxdy_B) +
		(dNydx_B - dNxdy_F)*(dNydx_B - dNxdy_F) ) /4.0;
		
		return 0.5*(firstTerm + (param.beta)*((C->x)*(C->x) + (C->y)*(C->y))*secondTerm );

}

double Lattice::calculateTotalEnergy() const
{
	/*
	* This calculation isn't very efficient as it uses calculateEngergyOfCell() for everycell
	* which results in various derivatives being calculated more than once.
	*/

	int xPos,yPos;
	double energy=0;

	for(yPos=0; yPos < (param.height); yPos++)
	{
		for(xPos=0; xPos < (param.width); xPos++)
		{
			energy += calculateEnergyOfCell(xPos,yPos);	
		}
	}

	return energy;

}

bool Lattice::saveState(const char* filename) const
{
	/* Assume following ordering of binary blocks
	*  <configuration><mNumNano><lattice><Nanoparticle_1_ID><Nanoparticle_1_data><Nanoparticle_2_ID><Nanoparticle_2_data>...
	*
	*/

	if(badState)
	{
		cerr << "Error: Lattice is in a bad state. Refusing to save state!" << endl;
		return false;
	}

	std::ofstream output(filename, ios::binary | ios::out | ios::trunc);

	if(!output.is_open())
	{
		cerr << "Error: Couldn't save state to " << filename << " . Couldn't open file" <<endl;
		output.close();
		return false;
	}
	
	//Write parameters
	output.write( (char*) &param, sizeof(LatticeConfig));

	if(!output.good())
	{
		cerr << "Error: Couldn't save state to " << filename << " . Write failed during parameter write." <<endl;
		output.close();
		return false;
	}

	//Write number of nanoparticles
	output.write( (char*) &mNumNano,sizeof(int));

	if(!output.good())
	{
		cerr << "Error: Couldn't save state to " << filename << " . Write failed during number of nanoparticles write." <<endl;
		output.close();
		return false;
	}

	//Write Lattice
	output.write( (char*) lattice, sizeof(DirectorElement)*param.width*param.height);

	if(!output.good())
	{
		cerr << "Error: Couldn't save state to " << filename << " . Write failed during lattice array write." <<endl;
		output.close();
		return false;
	}

	//loop through nanoparticles and write <Nanoparticle_ID><Nanoparticle_data>
	for(int counter=0; counter < mNumNano; counter++)
	{
		//write the ID of the nanoparticle.
		enum Nanoparticle::types theType = mNanoparticles[counter]->ID;
		output.write( (char*) &theType,sizeof(enum Nanoparticle::types)); 

		if(!output.good())
		{
			cerr << "Error: Couldn't save state to " << filename << " . Write failed writing ID for nanoparticle " << counter << endl;
			output.close();
			return false;
		}

		//write data
		mNanoparticles[counter]->saveState(output);

		if(!output.good())
		{
			cerr << "Error: Couldn't save state to " << filename << " . Write failed writing data for nanoparticle " << counter << endl;
			output.close();
			return false;
		}
	}

	

	output.close();

	return true;
}

//compare this lattice with rhs and return true if they are identical
bool Lattice::operator==(const Lattice & rhs) const
{
	//Check lattice configuration parameters
	if(this->mNumNano != rhs.mNumNano)
		return false;
	
	if(this->param.width != rhs.param.width)
		return false;

	if(this->param.height != rhs.param.height)
		return false;

	
	if(this->param.beta != rhs.param.beta)
		return false;

	
	if(this->param.topBoundary != rhs.param.topBoundary)
		return false;
	

	if(this->param.bottomBoundary != rhs.param.bottomBoundary)
		return false;

	if(this->param.leftBoundary != rhs.param.leftBoundary)
		return false;

	if(this->param.rightBoundary != rhs.param.rightBoundary)
		return false;

	if(this->param.initialState != rhs.param.initialState)
		return false;

	//check Monte carlo and conning algorithm parameters

	if(this->param.iTk != rhs.param.iTk)
		return false;

	if(this->param.mStep != rhs.param.mStep)
		return false;
	

	if(this->param.acceptCounter != rhs.param.acceptCounter)
		return false;

	
	if(this->param.rejectCounter != rhs.param.rejectCounter)
		return false;

	if(this->param.aAngle != rhs.param.aAngle)
		return false;

	if(this->param.desAcceptRatio != rhs.param.desAcceptRatio)
		return false;

	//compare lattice array elements
	for(int index=0; index < (param.width)*(param.height) ; index++)
	{
		if( (this->lattice[index]).x != (rhs.lattice[index]).x )
			return false;

		if( (this->lattice[index]).y != (rhs.lattice[index]).y )
			return false;

		if( (this->lattice[index]).isNanoparticle != (rhs.lattice[index]).isNanoparticle )
			return false;
	}


	//compare nanoparticles using their getDescription() method.
	for(int counter=0; counter < mNumNano; counter ++)
	{
		if ( (*(this->mNanoparticles[counter])).getDescription() != (*(rhs.mNanoparticles[counter])).getDescription() )
			return false;
	}

	//looks like everything's the same :)
	return true;

}

bool Lattice::operator!=(const Lattice & rhs) const
{
	return !(*this == rhs);
}

int Lattice::getNanoparticleCellCount() const
{
	int numNanoparticleCells=0;

	for(int index=0; ( index < (param.width)*(param.height) ); index++)
	{
		if(lattice[index].isNanoparticle == true)
			numNanoparticleCells++;
	}	
	
	return numNanoparticleCells;
}

int Lattice::getArea() const
{
	return (param.width)*(param.height);
}

void Lattice::restrictAngularRange(enum Lattice::angularRegion region)
{
	switch(region)
	{
		case REGION_RIGHT :
			
			//Restrict angular range to (-PI/2,PI/2]
			for(int index=0; index < (param.width)*(param.height) ; index++)
			{
				/* for lattice[index].y == -1 we implicitly assume lattice[index].x=0 but this
				*  isn't always true due rounding errors where the x component is almost 0 but not quite.
				*/
				if( (lattice[index].x < 0) ||  (lattice[index].y == -1 ) )
				{
					//flip components
					lattice[index].x *= -1;
					lattice[index].y *= -1;
				}

				
			}
		break;

		case REGION_TOP :
			//Restrict angular range to [0,PI)
			for(int index=0; index < (param.width)*(param.height) ; index++)
			{
				/* for lattice[index].x == -1 we implicitly assume lattice[index].y=0 but this
				*  isn't always true due rounding errors where the x component is almost 0 but not quite.
				*/
				if( (lattice[index].y < 0) ||  (lattice[index].x == -1 ) )
				{
					//flip components
					lattice[index].x *= -1;
					lattice[index].y *= -1;
				}

				
			}
		break;

		default :
			cerr << "Error: Angular restriction region " << region << " (enum) not supported!" << endl;

	}
}

double Lattice::calculateAverageAngle() const
{
	double avAngle=0;

	for(int y=0; y < param.height; y++)
	{
		for(int x=0; x < param.width; x++)
		{
			avAngle += getN(x,y)->calculateAngle();
		}
	}

	avAngle /= (param.width*param.height);

	return avAngle;
}

double Lattice::calculateAngularStdDev() const
{
	double average = calculateAverageAngle();
	double stddev=0;

	for(int y=0; y < param.height; y++)
	{
		for(int x=0; x < param.width; x++)
		{
			stddev += pow( (getN(x,y)->calculateAngle() - average) ,2);	
		}
	}

	//Divide by "(N-1)"
	stddev /= ( (param.width*param.height) -1);

	stddev = sqrt(stddev);

	return stddev;
}

bool Lattice::energyCompareWith(enum LatticeConfig::latticeState state, std::ostream& stream, double acceptibleError) const
{
	bool ccMatch=true;//Cell-Cell comparison success
	bool caMatch=true;//Cell-Analytical comparision success
	double expectedEnergy=0;

	if(!stream.good())
	{
		cerr << "Error: Can't use stream in energyCompareWith()" << endl;
		return false;
	}

	if(acceptibleError < 0)
	{
		cerr << "Error: Acceptible absolute error must be >= 0" << endl;
		return false;
	}
		
	stream << "Comparing current state energy to state " << state << " (enum)..." << endl <<
		"Using absolute error : " << acceptibleError << endl;

	switch(state)
	{

		//Handle different states
		case LatticeConfig::PARALLEL_X :
			expectedEnergy=0;
		break;

		case LatticeConfig::PARALLEL_Y :
			expectedEnergy=0;
		break;

		case LatticeConfig::K1_EQUAL_K3 :
			//Assume k_1=1
			expectedEnergy=PI*PI/(8*(param.height +1)*(param.height +1));
		break;

		case LatticeConfig::K1_DOMINANT :
			//Assume k_1=1
			expectedEnergy= (double) 1/( 2*(param.height +1)*(param.height +1) );
		break;

		case LatticeConfig::K3_DOMINANT :
			//Assume k_1=1
			expectedEnergy = (double) (param.beta)/( 2*(param.height +1 )*(param.height +1) );
		break;

		default :	
			stream << "comparision to state " << state << " (enum) not supported!" << endl;
			return false;
	}

	
	/* Do Energy comparision */
	double energyAE=0;
	stream << "Doing energy comparision (absolute error) (should be ~ " << expectedEnergy << ")..." << endl;
	for(int y=0; y < param.height ; y++)
	{
		for(int x=0; x < param.width; x++)
		{
			energyAE = calculateEnergyOfCell(x,y) - expectedEnergy;
			if(fabs(energyAE) > acceptibleError)
			{
				stream << "C-A (" << x << "," << y << ") ABS ERROR:" << energyAE << endl;
				caMatch=false;
			}
		}
	}

	if(caMatch)
		stream << "Cell analytical match within absolute error" << endl;

	/* Now do cell-cell comparision. For the analytical situations each cell
	*  should have the same energy per unit volume.
	*  This test compares all cell energies to the energy of the first cell.
	*/
	double firstCellEnergy = calculateEnergyOfCell(0,0);
	double ccAE=0;// Cell-Cell absolute error
	stream << "Doing cell-cell comparision (analytical situation should have uniform energy)..." << endl;
	for(int y=0; y < param.height; y++)
	{
		for(int x=0; x < param.width; x++)
		{
			ccAE = (calculateEnergyOfCell(x,y) - firstCellEnergy);
			if ( fabs(ccAE) > acceptibleError )
			{
				stream << "C-C (" << x << "," << y << " ABS ERROR:" << ccAE << endl;
				ccMatch=false;
			}
		}
	}

	if(ccMatch)
		stream << "Cell-Cell match within absolute error" << endl;

	return (caMatch && ccMatch);

}

bool Lattice::angleCompareWith(enum LatticeConfig::latticeState state, std::ostream& stream, double acceptibleError)
{
	bool match=true;

	if(!stream.good())
	{
		cerr << "Error: Can't use stream in angleCompareWith()" << endl;
		return false;
	}

	if(acceptibleError < 0)
	{
		cerr << "Error: Acceptible relative error must be >= 0" << endl;
		return false;
	}
		
	stream << "Comparing current state angular distribution to state " << state << " (enum)..." << endl <<
		"Using absolute error : " << acceptibleError << endl <<
		"Average Angle (radians): " << calculateAverageAngle() << endl <<
		"Standard deviation of Angles (radians): " << calculateAngularStdDev() << endl;
	
	double analyticalAngle=0;

	//restrict angular region to (-PI/2,PI/2]
	restrictAngularRange(REGION_RIGHT);

	// Do angular comparsion
	double angularAE=0;
	stream << "Doing angular comparision..." << endl;
	for(int y=0; y < param.height ; y++)
	{
		for(int x=0; x < param.width ; x++)
		{
			//Handle different states
			switch(state)
			{
				case LatticeConfig::PARALLEL_X :
					analyticalAngle=0;	
				break;

				case LatticeConfig::PARALLEL_Y :
					analyticalAngle=PI/2;
				break;

				case LatticeConfig::K1_EQUAL_K3 :
					analyticalAngle= PI*( (double) (y + 1)/(2*(param.height +1)) );
				break;

				case LatticeConfig::K1_DOMINANT :
					analyticalAngle= PI/2 - acos( (double) (y + 1)/(param.height + 1));
				break;

				case LatticeConfig::K3_DOMINANT :
					analyticalAngle= PI/2 -asin(1 - (double) (y +1)/(param.height +1)   );
				break;

				default:
					cerr << "State " << state << " (enum) not supported!" << endl;
					return false;
			}

			angularAE = getN(x,y)->calculateAngle() - analyticalAngle;
			if(fabs(angularAE) > acceptibleError)
			{
				stream << "(" << x << "," << y << ") ABS ERROR:" << angularAE << endl;	
				match=false;
			}
		}
	}
	
	if(match)
		stream << "Angular comparision matched within absolute error" << endl;
			
	return match;
}
