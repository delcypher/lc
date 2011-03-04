/* This program is used to create a binary state file from the configuration
*  specified in this program. It is saved to the file specified on the command
*  line.
*
*/

#include <iostream>
#include <cstdlib>
#include "lattice.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"


int main(int n, char* argv[])
{
	if(n !=2)
	{
		cerr << "Usage: " << argv[0] << " <filename>" << endl <<
		"<filename> - Filename to save created binary state file to" << endl;
		exit(1);
	}

	char* savefile = argv[1];
	
	bool badState=false;
	//set cout precision
	cout.precision(STD_PRECISION);
	cout << "#Displaying values to " << STD_PRECISION << " decimal places" << endl;

	LatticeConfig configuration;

	configuration.width = 50;
	configuration.height= 50;

	//set initial director alignment
	configuration.initialState = LatticeConfig::RANDOM;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;
	
	//set the initial Monte Carlo and coning algorithm parameters
	configuration.iTk = 2.5;
	configuration.mStep=0;
	configuration.acceptCounter=0;
	configuration.rejectCounter=0;
	configuration.aAngle=PI/2;
	configuration.desAcceptRatio=0.5;

	//create circular nanoparticle (x,y,radius, boundary)
/*	CircularNanoparticle particle1 = CircularNanoparticle(7,7,5,CircularNanoparticle::PARALLEL);

	if(particle1.inBadState())
		badState=true;

	//create elliptical nanoparticle
	//(xCentre,yCentre, a, b, theta, boundary)
	EllipticalNanoparticle particle2 = EllipticalNanoparticle(20,20,6,2,PI/4,EllipticalNanoparticle::PARALLEL);

	if(particle2.inBadState())
		badState=true;

*/	//create lattice object
	Lattice nSystem = Lattice(configuration);

	//add nanoparticles to lattice
/*	if(! nSystem.add(particle1) )
		badState=true;

	if(! nSystem.add(particle2) )
		badState=true;
*/
	if(nSystem.inBadState())
	{
		cerr << "Lattice in bad state!" << endl;
		badState=true;
		exit(1);
	}

	cout << "#Created Lattice with the following parameters:" << endl;
	nSystem.dumpDescription(std::cout);

	
	//save lattice state to file
	cout << "Saving state to file " << savefile << "..."; cout.flush();
	if(nSystem.saveState(savefile))
	{
		cout << "done" << endl;
	}
	else
	{
		badState=true;
		cerr << "FAILED!" << endl;
		exit(1);
	}

	//build new lattice from saved state to perform verification.
	Lattice revived(savefile);
	cout << "#Verifiying file " << savefile << "..."; cout.flush();

	if(nSystem != revived)
	{
		cerr << "FAIL!" << endl;
		badState=true;
	}
	else
	{
		cout << "Success!" << endl;
	}

	return badState?1:0;
}


