/* This is a test for initialising the to LatticeConfig::PARALLEL_X and then rotating all
   DirectorElements apart from nanoparticles in the lattice by a specified <angle> this 
   can only be checked byvisual expection.
*  The standard output of this program can be given to gnuplot script "ildump.gnu"
*
*  ./rotateDirector <width> <height> <angle>
*
* 
*/

#include <iostream>
#include <cstdlib>
#include "randgen.h"
#include "lattice.h"
#include "exitcodes.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"

int main(int n, char* argv[])
{

	if(n!=4)
	{
		cerr << "Usage: " << argv[0] << " <width> <height> <angle>" << endl <<
			"<angle> is rotation in degrees" << endl;
		exit(TH_BAD_ARGUMENT);
	}
	

	bool badState=false;

	LatticeConfig configuration;

	configuration.width = atoi(argv[1]);
	configuration.height= atoi(argv[2]);

	//get rotation angle
	float rotAngle = (PI/180)*atof(argv[3]);
	cerr << "#Rotate by angle (deg):" << atof(argv[3]) << " , in radians:" << rotAngle << endl;

	//set initial director alignment
	configuration.initialState = LatticeConfig::PARALLEL_X;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice object
	Lattice nSystem = Lattice(configuration);

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(7,7,5,CircularNanoparticle::PARALLEL);
	cout << "#Particle 1: " << particle1.getDescription() << endl;
	
	if(particle1.inBadState())
		badState=true;

	//create elliptical nanoparticle
	//(xCentre,yCentre, a, b, theta, boundary)

	EllipticalNanoparticle particle2 = EllipticalNanoparticle(20,20,10,5,PI/4,EllipticalNanoparticle::PARALLEL);
	cout << "#Particle 2: " << particle2.getDescription() << endl;

	if(particle2.inBadState())
		badState=true;

	//add nanoparticles to lattice
	if(! nSystem.add(particle1) )
		badState=true;

	if(! nSystem.add(particle2) )
		badState=true;

	if(nSystem.inBadState())
		badState=true;

	//loop over every DirectorElement in Lattice and rotate it
	for(int yPos=0; yPos < configuration.height; yPos++)
	{
		for(int xPos=0; xPos < configuration.width; xPos++)
		{
			if ( (nSystem.getN(xPos,yPos))->isNanoparticle ==0 )
			{
				rotateDirector(nSystem.setN(xPos,yPos),rotAngle);
			}
		}
	}

	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	cout.precision(STATE_SAVE_PRECISION);
	nSystem.indexedNDump(std::cout);


	return badState?TH_FAIL:TH_SUCCESS;
}


