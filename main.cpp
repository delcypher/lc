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
#include <signal.h>
#include <ctime>

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
*                               in the simulation. This is written to when SIGUSR1 is sent to this program or when it is
* 				terminated by SIGTERM or SIGINT.
*  BACKUP_LATTICE_STATE - Contains the lattice state that can be reloaded by the program to resume simulation.
*/

const char ANNEALING_FILE[] = "annealing.dump";
const char ENERGY_FILE[] = "energy.dump";
const char FINAL_LATTICE_STATE_FILE[] = "final-lattice-state.dump";
const char REQUEST_LATTICE_STATE_FILE[] = "current-lattice-state.dump";
const char BACKUP_LATTICE_STATE_FILE[] = "backup-lattice-state.bak";


void exitHandler();
void setExit(int sig);
void requestStateHandler(int sig);
void dumpViewableLatticeState();
void closeFiles();

bool requestExit=false;
Lattice* nSystemp;
ofstream annealF, finalLF, energyF;
time_t rawTime;



int main()
{
	//add signal handlers
	signal(SIGINT,&setExit);
	signal(SIGTERM,&setExit);
	signal(SIGUSR1,&requestStateHandler);

	LatticeConfig configuration;
	
	//open and check we have access to necessary files

	//append file output as when we resume we'd like to keep the results of previous attempt.
	annealF.open(ANNEALING_FILE, ios::app);
	if(!annealF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << ANNEALING_FILE  << endl;
		return 1;
	}

	//truncate file (erase old contents) as we don't want old file contents
	finalLF.open(FINAL_LATTICE_STATE_FILE, ios::trunc);
	if(!finalLF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << FINAL_LATTICE_STATE_FILE << endl;
		return 1;
	}

	//append file output as when we resume we'd like to keep the results of previous attempt.
	energyF.open(ENERGY_FILE, ios::app);
	if(!energyF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << ENERGY_FILE << endl;
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

	//setup nSystem pointer so other functions can access it.
	nSystemp = &nSystem;

//	cout << "# Creating nanoparticle" << endl; 

	//create circular nanoparticle (x,y,radius, boundary)
//	CircularNanoparticle particle1 = CircularNanoparticle(10,10,5,CircularNanoparticle::PARALLEL);
	
//	cout << "# Adding nanoparticle" << endl;

	//add nanoparticle to lattice
//	nSystem.add(&particle1);

	//Dump the initial state of the lattice to standard output
	nSystem.dumpDescription(std::cout);

	double energy = nSystem.calculateTotalEnergy();

	//set seed for random number generator
	setSeed();

	DirectorElement *temp;
	int x, y; 
	int acceptCounter = 0, rejectCounter = 0;
	unsigned long loopMax = 100000;
	double angle, before, after, oldNx, oldNy, dE, rollOfTheDice;
	double aAngle = PI * 0.5; // acceptance angle
	double oldaAngle;
	double CurAcceptRatio = 0, desAcceptRatio = 0.5;
	double progress = 0, oldProgress = 0;

	//Get the current time to show in files.
	time(&rawTime);

	cout << "#Starting Monte Carlo process:" << ctime(&rawTime) << endl;
	
	//output header for annealing file
	annealF << "#Starting at:" << ctime(&rawTime);
	annealF << "# Step    Acceptance angle    1/Tk" << endl;
	
	//output initial energy
	energyF << "#Starting at:" << ctime(&rawTime);
	energyF << "#Step\tEnergy" << endl;
	energyF << -1 << "\t" << energy << endl;

	
	for(unsigned long step = 0; step < loopMax; step++)
	{
		progress = (double) step / loopMax * 100;
		if(progress - oldProgress > 1) 
		{
			cout  << "\r" << progress << "%  ";
			oldProgress = progress;
			cout.flush();
		}

		for(int i=0; i < configuration.width*configuration.height; i++)
		{
			//pick "random" (x,y) co-ordinate in range ( [0,configuration.width -1] , [0,configuration.height -1] )
			x = intRnd() % configuration.width;
			y = intRnd() % configuration.height;

			temp = nSystem.setN(x,y);
			angle = (2*rnd()-1)*aAngle; 
			oldNx = temp->x;
			oldNy = temp->y;

			before = nSystem.calculateEnergyOfCell(x,y);
			before += nSystem.calculateEnergyOfCell(x+1,y);
			before += nSystem.calculateEnergyOfCell(x-1,y);
			before += nSystem.calculateEnergyOfCell(x,y+1);
			before += nSystem.calculateEnergyOfCell(x,y-1);
			
			// rotate director by random angle
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
				if(rollOfTheDice > exp(-dE*configuration.iTk)) 
				{
					// reject change
					temp->x = oldNx;
					temp->y = oldNy;
					rejectCounter++;
				}
				else acceptCounter++;
			}
			else acceptCounter++;
		}
		
		/* coning algorithm
		*  NOTE: Hobdell & Windle do this every 500 steps and apply additional check (READ THE PAPER)
		*  Previous project students calculate the acceptance angle every 10,000 steps.
		*
		*  This additional check is implemented here and is the one described by Hobdell & Windle
		*
		*  BEWARE: if width*height*500 > MAXIMUM VALUE of data type of acceptCounter 
		*          then there is a risk that wrap around will occur and the algorithm will be broken!
		*
		*          This applies similarly to rejectCounter
		*/
		if((step%500)==0 && step !=0)
		{
			CurAcceptRatio = (double) acceptCounter / (acceptCounter+rejectCounter);
			oldaAngle = aAngle;
			aAngle *= CurAcceptRatio / desAcceptRatio; // acceptance angle *= (current accept. ratio) / (desired accept. ratio = 0.5)
			
			//reject new acceptance angle if it has changed by more than a factor of 10
			if( (aAngle/oldaAngle) > 10 || (aAngle/oldaAngle) < 0.1 )
			{
				cerr << "# Rejected new acceptance angle:" << aAngle << " from old acceptance angle " << oldaAngle << "on step " << step << endl;
				aAngle = oldaAngle;
				
			}

			acceptCounter = 0;
			rejectCounter = 0;
		}

		/* cooling algorithm
		*  After every 150,000 m.c.s we increase iTK i.e. we decrease the "temperature".
		*/
		if((step%150000)==0 && step!=0) configuration.iTk *= 1.01;

		//output annealing information
		annealF << step << "           " << aAngle << "             " << configuration.iTk << endl;
	
		//output energy information
		energy = nSystem.calculateTotalEnergy();
		energyF << step << "\t" << energy << endl;

		//check if a request to exit has occured
		if(requestExit)
		{
			exitHandler();
		}
	}
	
	cout << "\r100%  " << endl;	

	//output final lattice state
	nSystem.indexedNDump(finalLF);

	closeFiles();

	return 0;
}


void setExit(int sig)
{
	cout << "Received signal:" << sig << "\n" <<
		"Will exit when current m.c.s completes." << endl;
	requestExit=true;
}

void requestStateHandler(int sig)
{
	cout << "Received signal:" << sig << endl;
	dumpViewableLatticeState();
}

void exitHandler()
{
	cout << "Last m.c.s complete, saving state to " << BACKUP_LATTICE_STATE_FILE << "...";
	cout.flush();

	//insert backup code here

	cout << "done" << endl;
	
	//dump the lattice state
	dumpViewableLatticeState();
	
	closeFiles();
	exit(0);
}

void closeFiles()
{
	finalLF.close();
	energyF.close();
	annealF.close();
}

void dumpViewableLatticeState()
{
	//dump current lattice state for use will ildump.gnu
	cout << "Dumping viewable lattice State to " << REQUEST_LATTICE_STATE_FILE << "...";
	ofstream requestDumpF (REQUEST_LATTICE_STATE_FILE, ios::trunc);
	if(!requestDumpF.is_open())
	{
		cerr << "Error: Couldn't open " << REQUEST_LATTICE_STATE_FILE << endl;
		return;
	}

	requestDumpF << "#Viewable dump requested for run starting at:" << ctime(&rawTime) << endl;
	time(&rawTime);
	requestDumpF << "#Dump requested at:" << ctime(&rawTime) << endl;

	nSystemp->indexedNDump(requestDumpF);
	requestDumpF.close();
	cout << "done" << endl;

}
