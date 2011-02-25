/* This is a test for initialising the lattice calling saveState() and then
*  building a new lattice by loading that state from the saved file and 
*  comparing the lattices.
*
*  ./test-host-initialise <width> <height> <initial_state_enum>
*
*/

#include <iostream>
#include <cstdlib>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "exitcodes.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"

const char savefile[] = "tempsave.bin";

int main(int n, char* argv[])
{

	if(n!=4)
	{
		cerr << "Usage: " << argv[0] << " <width> <height> <initial_state_enum>\n";
		exit(TH_BAD_ARGUMENT);
	}

	bool badState=false;

	LatticeConfig configuration;

	configuration.width = atoi(argv[1]);
	configuration.height= atoi(argv[2]);

	//set initial director alignment
	configuration.initialState = (LatticeConfig::latticeState) atoi(argv[3]);

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;
	
	//set Monte Carlo and coning algorithm parameters
	configuration.iTk = 2.5;
	configuration.mStep=50;
	configuration.acceptCounter=1;
	configuration.rejectCounter=2;
	configuration.aAngle=PI/2;
	configuration.desAcceptRatio=0.5;

	//create lattice object
	Lattice nSystem = Lattice(configuration);


	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(7,7,5,CircularNanoparticle::PARALLEL);
	cout << "#Particle 1: " << particle1.getDescription() << endl;
	cout << "#Particle 1 data size:" << particle1.getSize() << endl;
	if(particle1.inBadState())
		badState=true;

	//create elliptical nanoparticle
	//(xCentre,yCentre, a, b, theta, boundary)

	EllipticalNanoparticle particle2 = EllipticalNanoparticle(20,20,10,5,PI/4,EllipticalNanoparticle::PARALLEL);
	cout << "#Particle 2: " << particle2.getDescription() << endl;

	cout << "#Particle 2 data size:" << particle2.getSize() << endl;
	if(particle2.inBadState())
		badState=true;

	//add nanoparticles to lattice
	if(! nSystem.add(particle1) )
		badState=true;

	if(! nSystem.add(particle2) )
		badState=true;

	if(nSystem.inBadState())
	{
		cerr << "Lattice in bad state!" << endl;
		badState=true;
	}
	
	
	//save lattice state to file
	nSystem.saveState(savefile);

	//build new lattice from saved state
	Lattice revived(savefile);

	if(nSystem != revived)
	{
		cerr << "Error: Lattice states do not match!" << endl;
		badState=true;
	}
	else
	{
		cout << "Success: Lattice states do match!" << endl;
	}

	return badState?TH_FAIL:TH_SUCCESS;
}


