/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"

const int threadDim = 8;


int main()
{
	LatticeConfig configuration;
	FILE* fout = fopen("dump.txt", "w");

	cout << "# Setting lattice config parameters" << endl;	
	//setup lattice parameters
	configuration.width = threadDim*10;
	configuration.height= threadDim*10;

	//set initial director alignment
	configuration.initialState = LatticeConfig::RANDOM;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value
	configuration.beta = 3.5;


	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	cout << "# Creating nanoparticle" << endl; 

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	cout << "# Adding nanoparticle" << endl;

	//add nanoparticle to lattice
	nSystem.add(&particle1);

	cout << "# Initialise lattice on device" << endl;

	
	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	nSystem.indexedNDump(fout);

        
	double energy = nSystem.calculateTotalEnergy();
	cout << "# Energy calculated on CPU: " << energy << endl;

	fclose(fout);

	return 0;
}

