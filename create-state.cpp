/* This program is used to create a binary state file from the configuration
*  specified in this program. It is saved to the file specified on the command
*  line.
*
*/

#include <iostream>
#include <cstdlib>
#include "lattice.h"
#include <cmath>
#include <ctime>

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"

/* This method tries to work out the best value for iTk (related to simulation temperature)
*  such that the largest possible increase in free energy will have a
*  0.5 probability of being accepted or rejected in the Monte carlo simulation from the Boltzmann factor.
*/
double getiTk(const LatticeConfig& realConfig);

unsigned long seedToUse;
char* savefile;
LatticeConfig configuration;

void handleOptionalArgs(int n, char* argv[], int argIndex)
{
	/* argIndex should start at the index of the first
	*  optional argument
	*/
	while(argIndex < n)
	{
		if(strcmp(argv[argIndex],"--rand-seed") ==0)
		{
			argIndex++;
			//check we have an argument in front
			if(n > argIndex)
			{
				if(atoi(argv[argIndex]) < 0)
				{
					cerr << "Error: <seed> must be >=0" << endl;
					exit(1);
				}

				seedToUse=atoi(argv[argIndex]);

				argIndex++;
				continue;
			}
			else
			{
				cerr << "Error: Expected <seed>" << endl;
				exit(1);

			}
		}
		
		//handled all supported arguments, throw error
		cerr << "Error: Argument " << argv[argIndex] << " not supported!" << endl;
		exit(1);

	}
}


int main(int n, char* argv[])
{
	//By default set random initialisation seed to UNIX time
	seedToUse = time(NULL);

	if(n<2)
	{
		cerr << "Usage: create-state  <filename> [options]" << endl <<
		"<filename> - Filename to save created binary state file to.\n\n" <<
		"[Options]\n\n" <<
		"--rand-seed <seed>\n" <<
		"Set the random seed for initialisation (only applies for initialState == LatticeConfig::RANDOM) to <seed> where <seed> is a positive integer" << endl;
		exit(1);
	}

	savefile = argv[1];


	bool badState=false;
	//set cout precision
	cout.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	cout.precision(STDOE_PRECISION);
	cout << "#Displaying values to " << STDOE_PRECISION << " decimal places" << endl;

	//set lattice configuration
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
	configuration.iTk = getiTk(configuration); //Automatically determined
	configuration.mStep=0;
	configuration.acceptCounter=0;
	configuration.rejectCounter=0;
	configuration.aAngle=PI/2;
	configuration.desAcceptRatio=0.5;

	//Handle optional arguments (i.e. what random seed to use)
	handleOptionalArgs(n,argv,2);

	//IF having initial Random state should set random seed!
	configuration.randSeed = seedToUse;
	cout << "#Using random initialisation seed: " << configuration.randSeed << endl;


	//create circular nanoparticle (x,y,radius, boundary)
	CircularNanoparticle particle1 = CircularNanoparticle(7,7,5,CircularNanoparticle::PARALLEL);

	if(particle1.inBadState())
		badState=true;

	//create elliptical nanoparticle
	//(xCentre,yCentre, a, b, theta, boundary)
	EllipticalNanoparticle particle2 = EllipticalNanoparticle(20,20,6,2,PI/4,EllipticalNanoparticle::PARALLEL);

	if(particle2.inBadState())
		badState=true;

	//create lattice object
	Lattice nSystem = Lattice(configuration);

	//add nanoparticles to lattice
	if(! nSystem.add(particle1) )
		badState=true;

	if(! nSystem.add(particle2) )
		badState=true;

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


double getiTk(const LatticeConfig& realConfig)
{
	/* We make a small 3x3 lattice with periodic boundary conditions everywhere that has all DirectorElements 
	*  parallel to the x-axis. We rotate the DirectorElement cell (1,1) by 90degrees (we assume this will give the largest 
	* change in free energy) and workout the change in free energy this introduces.
	*/
	LatticeConfig configuration;

	configuration.width = 3;
	configuration.height= 3;

	//set initial director alignment
	configuration.initialState = LatticeConfig::PARALLEL_X;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value (we must use the real lattice's value)
	configuration.beta = realConfig.beta;

	//make the lattice
	Lattice smallL = Lattice(configuration); 

	double beforeEnergy=0;
	double afterEnergy=0;
	double deltaE=0;

	/*     |T|      y |
	*    |L|C|R|      |
	*      |B|        |________ x
	*
	* [Cell] [co-ordinate]
	* C (1,1)
	* T (1,2)
	* B (1,0)
	* L (0,1)
	* R (2,1)
	*/
	int x=1, y=1;

	beforeEnergy = smallL.calculateEnergyOfCell(x,y); // add energy per unit volumne of Cell C
	beforeEnergy += smallL.calculateEnergyOfCell(x-1,y); // add energy per unit volume of cell L
	beforeEnergy += smallL.calculateEnergyOfCell(x+1,y); // add energy per unit volume of cell R
	beforeEnergy += smallL.calculateEnergyOfCell(x,y-1); // add energy per unit volume of cell B
	beforeEnergy += smallL.calculateEnergyOfCell(x,y+1); // add energy per unit volume of cell T

	//rotate cell C by 90 degrees
	DirectorElement* temp = smallL.setN(x,y);
	temp->x=0;
	temp->y=1;

	//calculate the energy of cells after

	afterEnergy = smallL.calculateEnergyOfCell(x,y); // add energy per unit volumne of Cell C
	afterEnergy += smallL.calculateEnergyOfCell(x-1,y); // add energy per unit volume of cell L
	afterEnergy += smallL.calculateEnergyOfCell(x+1,y); // add energy per unit volume of cell R
	afterEnergy += smallL.calculateEnergyOfCell(x,y-1); // add energy per unit volume of cell B
	afterEnergy += smallL.calculateEnergyOfCell(x,y+1); // add energy per unit volume of cell T

	//calculate the change in energy
	deltaE = afterEnergy - beforeEnergy;

	if(deltaE<=0)
	{
		cerr << "Error: getiTk() failed because deltaE <=0" << endl;
	}

	//this part assumes the probability of acceptance for the maximum deltaE is 0.5
	double iTk= log(2)/deltaE;
	return iTk;
}
