/* This is a tests copying a lattice array to the device using a particular configuration
*  using Lattice::copyHostToDevice() . The array and the latticeconfiguration is copied back manually and compared.
*  The Lattice configuration is also copied back and manually compared.
* 
*  ./copy-to-device-test <width> <height> <inital_state_enum>
*
*  <initial_state_enum> - Should be a number corresponding to the latticeState enum (lattice.h)
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "exitcodes.h"
#include "../devicemanager.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"

DirectorElement* devicesArray;

//device's lattice object
LatticeObject dlo;

void cleanup()
{
	free(devicesArray);
}

int main(int n, char* argv[])
{

	if(n!=4)
	{
		cerr << "Usage: " << argv[0] << " <width> <height> <initial_state_enum>\n";
		exit(TH_BAD_ARGUMENT);
	}

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

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	cout << "# Creating nanoparticle" << endl; 

	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
	cout << "# Adding nanoparticle" << endl;

	//add nanoparticle to lattice
	nSystem.add(&particle1);

	//initialise lattice on device
	nSystem.initialiseCuda();

	//copy to device
	nSystem.copyHostToDevice();

	//initialise memory to hold the device's copy of the lattice array
	devicesArray = (DirectorElement*) calloc((configuration.width*configuration.height),sizeof(DirectorElement));

	//initialise variable to hold device's copy of lattice configuration to zero.
	memset( &dlo ,0,sizeof(LatticeObject));

	//copy device's lattice configuration to host manually
	deviceErrorHandle( cudaMemcpy(&dlo,nSystem.devLatticeObject,sizeof(LatticeObject),cudaMemcpyDeviceToHost) );

	//copy device's lattice array to host manually
	deviceErrorHandle( 
		cudaMemcpy(devicesArray,
			nSystem.getDeviceLatticeArrayPointer(),
			sizeof(DirectorElement)*configuration.width*configuration.height,
			cudaMemcpyDeviceToHost)
			);
	
	//loop through array on host and devices array and compare
	int index;	
	for(int y=0; y< configuration.height; y++)
	{
		for(int x=0; x< configuration.width; x++)
		{
			index = x + (configuration.width)*y;

			//Check the x element of the DirectorElement between host and device
			if(nSystem.hostLatticeObject->lattice[index].x != devicesArray[index].x)
			{
				cerr << "Error: x DirectorElement does not match in lattice @ (" << x << "," << y << ")" 
				<< "Host:" << nSystem.hostLatticeObject->lattice[index].x <<
				" Device:" << devicesArray[index].x << endl;
				cleanup();
				exit(TH_FAIL);
			}

			//Check the y element of the DirectorElement between host and device
			if(nSystem.hostLatticeObject->lattice[index].y != devicesArray[index].y)
			{
				cerr << "Error: x DirectorElement does not match in lattice @ (" << x << "," << y << ")" 
				<< "Host:" << nSystem.hostLatticeObject->lattice[index].y <<
				" Device:" << devicesArray[index].y << endl;
				cleanup();
				exit(TH_FAIL);
			}
			
			//Check the isNanoparticle element of the DirectorElement between host and device
			if(nSystem.hostLatticeObject->lattice[index].isNanoparticle != devicesArray[index].isNanoparticle)
			{
				cerr << "Error: x DirectorElement does not match in lattice @ (" << x << "," << y << ")" 
				<< "Host:" << nSystem.hostLatticeObject->lattice[index].isNanoparticle <<
				" Device:" << devicesArray[index].isNanoparticle << endl;
				cleanup();
				exit(TH_FAIL);
			}
		}
	}

	//check device's lattice configuration matches host
	if (dlo.param.width != nSystem.hostLatticeObject->param.width)
	{
		cerr << "Error: Lattice configuration (width) does not match between host and device" << endl;
		exit(TH_FAIL);
	}

	if (dlo.param.height != nSystem.hostLatticeObject->param.height)
	{
		cerr << "Error: Lattice configuration (height) does not match between host and device" << endl;
		exit(TH_FAIL);
	}


	if (dlo.param.beta != nSystem.hostLatticeObject->param.beta)
	{
		cerr << "Error: Lattice configuration (beta) does not match between host and device" << endl;
		exit(TH_FAIL);
	}

	if (dlo.param.topBoundary != nSystem.hostLatticeObject->param.topBoundary)
	{
		cerr << "Error: Lattice configuration (topBoundary) does not match between host and device" << endl;
		exit(TH_FAIL);
	}


	if (dlo.param.bottomBoundary != nSystem.hostLatticeObject->param.bottomBoundary)
	{
		cerr << "Error: Lattice configuration (bottomBoundary) does not match between host and device" << endl;
		exit(TH_FAIL);
	}
	if (dlo.param.leftBoundary != nSystem.hostLatticeObject->param.leftBoundary)
	{
		cerr << "Error: Lattice configuration (topBoundary) does not match between host and device" << endl;
		exit(TH_FAIL);
	}
	if (dlo.param.rightBoundary != nSystem.hostLatticeObject->param.rightBoundary)
	{
		cerr << "Error: Lattice configuration (topBoundary) does not match between host and device" << endl;
		exit(TH_FAIL);
	}

	if (dlo.param.initialState != nSystem.hostLatticeObject->param.initialState)
	{
		cerr << "Error: Lattice configuration (topBoundary) does not match between host and device" << endl;
		exit(TH_FAIL);
	}

	cleanup();
	return TH_SUCCESS;
}


