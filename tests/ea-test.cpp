/* This is a test for checking the Lattice::calculateEnergyOfCell() method gives
*  the same answer as a specified known analytical solution. It is also a test
*  for checking the angular distribution (this really shouldn't fail!)
* 
*  The lattice is configured in one of 5 minimum energy states:
*
*  The configuration is n = (cos(theta), sin(theta), 0)
*
* We perform two test:
* Lattice::energyCompareWith()
* Lattice:angleCompareWith()
*
* See lattice.h for a description of theses tests.
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "randgen.h"
#include "lattice.h"
#include "exitcodes.h"

using namespace std;

int main(int n, char* argv[])
{
	//used to keep track of failure of all tests and so determines program's exit code
	bool test1 =true;
	bool test2=true;

	if(n!=5)
	{
		cerr << "Usage: " << argv[0] << 
		" <width> <height> <state> <absolute_error>\n\n" <<
		"<state> - Enum (LatticeConfig::latticeState) corresponding to state.\n" <<
		"<absolute_error> - The maximum acceptible absolute error." << endl;
		exit(TH_BAD_ARGUMENT);
	}


	LatticeConfig configuration;

	configuration.width = atoi(argv[1]);
	configuration.height= atoi(argv[2]);

	//set lattice state
	LatticeConfig::latticeState state = (LatticeConfig::latticeState) atoi(argv[3]);

	//set maximum acceptible absolute error
	double acceptibleAE = atof(argv[4]);

	//set initial director alignment
	configuration.initialState = state;

	//set boundary conditions (top and bottom will change depending on which analytical situation we wish to compare)
	if(state == LatticeConfig::PARALLEL_X)
	{
		configuration.topBoundary = LatticeConfig::BOUNDARY_PARALLEL;
		configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	}

	if(state == LatticeConfig::PARALLEL_Y)
	{
		configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
		configuration.bottomBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	}

	if(state == LatticeConfig::K1_EQUAL_K3 || state == LatticeConfig::K1_DOMINANT || state == LatticeConfig::K3_DOMINANT)
	{
		configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
		configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	}

	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value (assume k_1*beta = k_3)
	if(state == LatticeConfig::K1_DOMINANT)
	{
		configuration.beta = 0;
	} 
	else if(state == LatticeConfig::K3_DOMINANT)
	{
		//can't set beta to infinity so pick very large number
		configuration.beta=1e10;
	}
	else
	{
		//assume k_1 = k_3
		configuration.beta=1;
	}

	//create lattice object
	Lattice nSystem = Lattice(configuration);
	
	//run tests
	test1 = nSystem.energyCompareWith(state, std::cout, acceptibleAE);
	std::cout << endl;

	//Need to restrict angular range to do next test (it is done internally)
	test2 = nSystem.angleCompareWith(state, std::cout, acceptibleAE);

	return (test1 && test2)?TH_SUCCESS:TH_FAIL;
}

