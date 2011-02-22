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

/* Output filenames:
*  ANNEALING_FILE - contains iTK and acceptance angle as the simulation progresses.
*  ENERGY_FILE - Contains energy and monte carlo step as the simulation progresses.
*  FINAL_LATTICE_STATE_FILE - Contains the final state of lattice from Lattice::indexedNDump() for when the simulation end.
*  REQUEST_LATTICE_STATE_FILE - Contains the current state of the lattice from Lattice::indexedNDump() at any point
*                               in the simulation. This is written to when SIGUSR1 is sent to this program.
*  BACKUP_LATTICE_STATE - Contains the lattice state that can be reloaded by the program to resume simulation.
*/

char ANNEALING_FILE[] = "annealing.dump";
char ENERGY_FILE[] = "energy.dump";
char FINAL_LATTICE_STATE_FILE[] = "final-lattice-state.dump";
char REQUEST_LATTICE_STATE_FILE[] = "current-lattice-state.dump";
char BACKUP_LATTICE_STATE_FILE[] = "backup-lattice-state.bak";

int main()
{
	LatticeConfig configuration;
	
	//open and check we have access to necessary files which we truncate
	ofstream annealF(ANNEALING_FILE, ios::trunc);
	if(!annealF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << ANNEALING_FILE  << endl;
		return 1;
	}

	ofstream finalLF (FINAL_LATTICE_STATE_FILE, ios::trunc);
	if(!finalLF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << FINAL_LATTICE_STATE_FILE << endl;
		return 1;
	}
	
	cout << "# Setting lattice config parameters" << endl;	
	//setup lattice parameters
	configuration.width = 50;
	configuration.height= 50;

	//set initial director alignment
	configuration.initialState = LatticeConfig::RANDOM;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
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

	//Dump the initial state of the lattice to standard output
	nSystem.dumpDescription(std::cout);

	double energy = nSystem.calculateTotalEnergy();

	setSeed(); // for rng

	DirectorElement *temp;
	int x, y, accept = 0, deny = 0;
	unsigned long loopMax = 100000;
	double angle, before, after, oldNx, oldNy, dE, rollOfTheDice;
	double aAngle = PI * 0.5; // acceptance angle
	double curAccept = 0, desAccept = 0.5;
	double progress = 0, oldProgress = 0;

	cout << "# Starting Monte Carlo process\n";
	annealF << "\n# Step    Acceptance angle    1/Tk" << endl;

	for(unsigned long steps = 0; steps < loopMax; steps++)
	{
		progress = (double) steps / loopMax * 100;
		if(progress - oldProgress > 1) 
		{
			cout  << "\r" << progress << "%  ";
			oldProgress = progress;
			cout.flush();
		}

		for(int i=0; i < configuration.width*configuration.height; i++)
		{
			x = intRnd() % configuration.width;
			y = intRnd() % configuration.height;
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
		if(!steps%150000 && steps!=0) configuration.iTk *= 1.01;

		// output junk
		if(!steps%10) annealF << steps << "           " << aAngle << "             " << configuration.iTk << endl;
	}
	
	cout << "\r100%  " << endl;	

	//output final lattice state
	nSystem.indexedNDump(finalLF);
	finalLF.close();

	annealF.close();

	return 0;
}

