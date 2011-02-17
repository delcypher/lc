/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"

int main()
{
	LatticeConfig configuration;
	FILE* fout = fopen("dump.txt", "w");
	ofstream dump("annealing.dump");
	if(!dump)
	{
		cout << "I hate you so I'm not going to work properly for you." << endl;
		return -180;
	}

	cout << "# Setting lattice config parameters" << endl;	
	//setup lattice parameters
	configuration.width = 100;
	configuration.height= 100;

	//set initial director alignment
	configuration.initialState = LatticeConfig::RANDOM;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.iTk = 1;

	//set lattice beta value
	configuration.beta = 3.5;

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

//	cout << "# Creating nanoparticle" << endl; 

	//create circular nanoparticle (x,y,radius, boundary)
//	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
//	cout << "# Adding nanoparticle" << endl;

	//add nanoparticle to lattice
//	nSystem.add(&particle1);

	//Dump the current state of the lattice to standard output.
	//nSystem.nDump(Lattice::BOUNDARY,stdout);
	nSystem.indexedNDump(fout);

	double energy = nSystem.calculateTotalEnergy();

	setSeed(); // for rng

	DirectorElement *temp;
	int x, y, accept = 0, deny = 0;
	double angle, before, after, oldNx, oldNy, dE, rollOfTheDice;
	double aAngle = PI * 0.5; // acceptiance angle
	double curAccept = 0, desAccept = 0.5;

	cout << "Starting Monte Carlo process\n";
	cout << "\nStep    Acceptance angle    1/Tk" << endl;

	for(unsigned long steps = 0; steps < 100000; steps++)
	{
		for(int i=0; i < configuration.width*configuration.height; i++)
		{
			x = intRnd() % configuration.width;
			y = intRnd() % configuration.height;
			cout << i << " " << x << " " << y << endl;
			temp = nSystem.setN(x,y);
			angle = (2*rnd()-1)*aAngle; // optimize later
			oldNx = temp->x;
			oldNy = temp->y;

			before = nSystem.calculateEnergyOfCell(x,y);
			before += nSystem.calculateEnergyOfCell(x+1,y);
			before += nSystem.calculateEnergyOfCell(x-1,y);
			before += nSystem.calculateEnergyOfCell(x,y+1);
			before += nSystem.calculateEnergyOfCell(x,y-1);
			
			// rotate director
			rotateDirector(temp, angle);
			
			after = nSystem.calculateEnergyOfCell(x,y);
			after += nSystem.calculateEnergyOfCell(x+1,y);
			after += nSystem.calculateEnergyOfCell(x-1,y);
			after += nSystem.calculateEnergyOfCell(x,y+1);
			after += nSystem.calculateEnergyOfCell(x,y-1);

			dE = after-before;

			if(dE>0) // if the energy increases, determine if change is accepted of rejected
			{
				rollOfTheDice = rnd();
				if(rollOfTheDice > exp(-dE*configuration.iTk)) // reject change
				{
					temp->x = oldNx;
					temp->y = oldNy;
					deny++;
				}
				else accept++;
			}
			else accept++;
		}
		
		// coning algorithm
		curAccept = (double) accept / (accept+deny);
		aAngle *= curAccept / desAccept; // acceptance angle *= (current accept. ratio) / (desired accept. ratio = 0.5)
		accept = 0;
		deny = 0;

		// cooling algorithm
		if(!steps%150000) configuration.iTk *= 1.01;

		dump << steps << "           " << aAngle << "             " << configuration.iTk << endl;

//		cout << "\r" << steps/100000 << "%";

		

	}
	cout << endl;

	fclose(fout);

	return 0;
}

