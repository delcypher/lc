/* Main section of code where the LatticeObject is setup and processed
*  By Alex Allen & Daniel Liew (2010)
*/

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "mt19937ar.h"
#include "lattice.h"
#include <signal.h>
#include <ctime>

using namespace std;

/* Output filenames:
*  ANNEALING_FILE - contains iTk (inverse temperature) and monte carlo step as simulation progresses. 
*  CONING_FILE - contaings acceptance angle and monte carlo step as simulation progresses.
*  ENERGY_FILE - Contains energy and monte carlo step as the simulation progresses.
*  FINAL_LATTICE_BINARY_STATE_FILE - Contains the final binary state of the lattice that can be loaded using the Lattice::Lattice(const char* filepath)
					constructor.
*  REQUEST_LATTICE_STATE_FILE - Contains the current state of the lattice from Lattice::indexedNDump() at any point
*                               in the simulation. This is written to when SIGUSR1 is sent to this program or when it is
* 				terminated by SIGTERM or SIGINT.
*  BACKUP_LATTICE_STATE - Contains the binary lattice state that can be reloaded by the program to resume simulation.
*/

const char ANNEALING_FILE[] = "annealing.dump";
const char CONING_FILE[] = "coning.dump";
const char ENERGY_FILE[] = "energy.dump";
const char FINAL_LATTICE_BINARY_STATE_FILE[] = "final-lattice-state.bin";
const char REQUEST_LATTICE_STATE_FILE[] = "current-lattice-state.dump";
const char BACKUP_LATTICE_STATE_FILE[] = "backup-lattice-state.bin";


void exitHandler();
void setExit(int sig);
void requestStateHandler(int sig);
void dumpViewableLatticeState();
void closeFiles();
bool openFiles(bool overwrite);
void handleArgs(int n, char* argv[]);
void usageMessage();

bool requestExit=false;
Lattice* nSystemp;
ofstream annealF, coningF, energyF;
ifstream finalLF;
time_t rawTime;

char* stateFile;
unsigned long loopMax;
unsigned long annealStep;
unsigned long seedToUse;


int main(int n, char* argv[])
{
	//set the random seed we use to UNIX time, user can overwrite this with a command line argument
	seedToUse= time(NULL);

	//set the default anneal step that the user can overwrite with a command line argument
	annealStep=280;

	//handle command line arguments
	handleArgs(n,argv);
	

	//add signal handlers
	signal(SIGINT,&setExit);
	signal(SIGTERM,&setExit);
	signal(SIGUSR1,&requestStateHandler);

	//set precision for std::cout and std::cerr
	cout.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	cout.precision(STDOE_PRECISION);

	cerr.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	cerr.precision(STDOE_PRECISION);
	

	//create lattice object
	Lattice nSystem = Lattice(stateFile);
	
	

	//setup nSystem pointer so other functions can access it.
	nSystemp = &nSystem;

	//check if lattice is in bad state
	if(nSystem.inBadState())
	{
		cerr << "Lattice in bad state!" << endl;
		exit(2);
	}
	//Dump the initial state of the lattice to standard output
	nSystem.dumpDescription(std::cout);

	//open files, if new simulation truncate all files, if not append to some files
	if(nSystem.param.mStep ==0 )
	{
		//truncate (overwrite) all files
		if(! openFiles(true))
			exit(2);
	}
	else
	{
		//Append to some files
		if(! openFiles(false))
			exit(2);
	}	

	//START MONTE CARLO ALGORITHM

	double energy = nSystem.calculateTotalEnergy();

	//set the random number generator seed
	init_genrand(time(NULL));

	//Tell user how often we cool
	cout << "#Annealing every " << annealStep << " monte carlo steps" << endl;

	//Dump the initial state of the lattice to standard output
	nSystem.dumpDescription(std::cout);


	DirectorElement *temp=NULL;
	int x, y; 
	double angle, before, after, oldNx, oldNy, dE, rollOfTheDice;
	double oldaAngle;
	double currentAcceptanceRatio = 0;

	//roughly (because of integer division) the number of steps in 1%
	int percentStep = loopMax /100;
	if(percentStep ==0)
	{
		//We must be doing less than 100 steps, set this way so program doesn't crash.
		percentStep=1;
		cerr << "Warning: Progress information will be very inaccurate!" << endl;
	}


	//Get the current time to show in files.
	time(&rawTime);

	cout << "#Starting Monte Carlo process:" << ctime(&rawTime) << endl;

	//exit if loaded binary state has already completed simulation
	if(nSystem.param.mStep >= loopMax)
	{
		cerr << "Error: Loaded binary state file at monte carlo step " << nSystem.param.mStep << " but simulation only runs for "
			<< loopMax << " monte carlo steps" << endl;
		closeFiles();
		return 1;
		
	}
	
	cout << "#Running simulation from monte carlo step " << nSystem.param.mStep << " of " << loopMax << endl;
	

	//output header for annealing file
	annealF << "#Starting at:" << ctime(&rawTime);
	annealF << "#[Step]    [1/Tk]" << endl;
	//output initial iTk
	if(nSystem.param.mStep==0)
	{
		annealF << -1 << " " <<  nSystem.param.iTk << endl;
	}

	//output header for annealing file
	coningF << "#Starting at:" << ctime(&rawTime);
	coningF << "#[Step] [Acceptance angle]" << endl;
	//output initial acceptance angle
	if(nSystem.param.mStep==0)
	{
		coningF << -1 << " " << nSystem.param.aAngle << endl;
	}

	//output initial energy
	energyF << "#Starting at:" << ctime(&rawTime);
	energyF << "#[Step] [Energy]" << endl;
	if(nSystem.param.mStep==0)
	{
		energyF << -1 << " " << energy << endl;
	}
	
	for(; nSystem.param.mStep < loopMax; nSystem.param.mStep++)
	{
		//output progress as a percentage
		if(nSystem.param.mStep % percentStep ==0) 
		{
			cout  << "\r" << ( (int) (nSystem.param.mStep)/percentStep) << "%  ";
			cout.flush();
		}

		for(int i=0; i < (nSystem.param.width)*(nSystem.param.height); i++)
		{
			/* Keep randomly selecting lattice cells until we find one that isn't a nanoparticle.
			*  This has the potential of becoming very inefficient for a lattice that has lots of nanoparticles!
			*/
			do
			{
				//pick "random" (x,y) co-ordinate in range ( [0, lattice width -1] , [0, lattice height -1] )
				x = genrand_int32() % (nSystem.param.width);
				y = genrand_int32() % (nSystem.param.height);
				
				temp = nSystem.setN(x,y);
			} while (temp->isNanoparticle ==true);
			
			
			//pick a random angle between [- nSystem.param.aAngle , nSystem.param.aAngle]
			angle = (2*genrand_real1()  -1)*nSystem.param.aAngle; 

			//Make a copy of the DirectorElement n_x & n_y values so we can set temp back to its original value if we reject a change
			oldNx = temp->x;
			oldNy = temp->y;

			//calculate the Energy per unit volume of cells that will affected by change to cell (x,y)
			before = nSystem.calculateEnergyOfCell(x,y);
			before += nSystem.calculateEnergyOfCell(x+1,y);
			before += nSystem.calculateEnergyOfCell(x-1,y);
			before += nSystem.calculateEnergyOfCell(x,y+1);
			before += nSystem.calculateEnergyOfCell(x,y-1);
			
			// rotate director by random angle
			temp->rotate(angle);
			
			//calculate the Energy per unit volumes of cells that have been affect by change to cell (x,y)
			after = nSystem.calculateEnergyOfCell(x,y);
			after += nSystem.calculateEnergyOfCell(x+1,y);
			after += nSystem.calculateEnergyOfCell(x-1,y);
			after += nSystem.calculateEnergyOfCell(x,y+1);
			after += nSystem.calculateEnergyOfCell(x,y-1);

			//Work out the change in energy of the lattice
			dE = after-before;

			if(dE>0) // if the energy increases, determine if change is accepted of rejected
			{
				//pick a random number in range [0,1]
				rollOfTheDice = genrand_real1();
				if(rollOfTheDice > exp(-dE*nSystem.param.iTk)) 
				{
					// reject change
					temp->x = oldNx;
					temp->y = oldNy;
					nSystem.param.rejectCounter++;
				}
				else nSystem.param.acceptCounter++;
			}
			else nSystem.param.acceptCounter++;
		}
		
		//output coning information
		if((nSystem.param.mStep%100)==0) 
			coningF << nSystem.param.mStep << " " << nSystem.param.aAngle << endl;

		/* coning algorithm
		*  NOTE: Hobdell & Windle do this every 500 trial flips and apply additional check (READ THE PAPER)
		*  Previous project students (Kimmet & Young) calculate the acceptance angle every 10,000 trial flips.
		*
		* We work in terms of monte carlo flips and NOT trials so our algorithm will need to be a little different.
		*/
		
		//calculate currentAcceptance Ratio and acceptance Angle every m.c.s
		currentAcceptanceRatio = (double) nSystem.param.acceptCounter / (nSystem.param.acceptCounter + nSystem.param.rejectCounter);
		oldaAngle = nSystem.param.aAngle;
		nSystem.param.aAngle *= currentAcceptanceRatio / nSystem.param.desAcceptRatio; // acceptance angle *= (current accept. ratio) / (desired accept. ratio = 0.5)
		
		//Force acceptance angle to stay in range [0.01, PI/2] radians ~ [0.5,90] degrees (Assuming it was in that range to begin with)
		if( (nSystem.param.aAngle > PI/2 ) || (nSystem.param.aAngle < 0.01) )
		{
			nSystem.param.aAngle = oldaAngle;
		}

		nSystem.param.acceptCounter = 0;
		nSystem.param.rejectCounter = 0;
		
		
		/* cooling algorithm
		*  After every 280 m.c.s we increase iTk i.e. we decrease the "temperature".
		*  Note this is NOT equivilant to "Kimmet & Young"'s code.
		*
		* It appears that this value needs to be scaled with the lattice dimensions. 280 appears to be adequate for up to 180x180
		*/
		if(( nSystem.param.mStep % annealStep)==0 && nSystem.param.mStep!=0) 
		{
			nSystem.param.iTk *= 1.01;

			//output annealing information
			annealF << nSystem.param.mStep << " " << nSystem.param.iTk << endl;
		}

		//output energy information
		if( (nSystem.param.mStep%10)==0 )
		{
			energy = nSystem.calculateTotalEnergy();
			energyF << nSystem.param.mStep << " " << energy << endl;
		}

		//check if a request to exit has occured
		if(requestExit)
		{
			exitHandler();
		}
	}
	
	cout << "\r100%  " << endl;	
	cout << "#Finished simulation doing " << loopMax << " monte carlo steps." << endl;
	//output final viewable lattice state

	//output state to binary file which can be used to resume simulation (if loopMax is modified)
	cout << "#Saving binary state to file " << FINAL_LATTICE_BINARY_STATE_FILE << "...";
	cout.flush();
	if( nSystemp->saveState(FINAL_LATTICE_BINARY_STATE_FILE) )
	{
		cout << "done" << endl;
	}
	else
	{
		cout << "failed" << endl;
	}

	closeFiles();

	return 0;
}


void setExit(int sig)
{
	cout << "Received signal:" << sig << "\n" <<
		"Will exit when monte carlo step " << nSystemp->param.mStep << " completes." << endl;
	requestExit=true;
}

void requestStateHandler(int sig)
{
	cout << "Received signal:" << sig << endl;
	dumpViewableLatticeState();
}

void exitHandler()
{
	cout << "Monte carlo step " << nSystemp->param.mStep << " complete, saving binary state file to " << BACKUP_LATTICE_STATE_FILE << "...";
	cout.flush();
	bool saveOk=true;

	saveOk = nSystemp->saveState(BACKUP_LATTICE_STATE_FILE);

	if(saveOk)
		cout << "done" << endl;
	else
		cout << "failed" << endl;
	
	//dump the lattice state
	dumpViewableLatticeState();

	closeFiles();
	exit(0);
}

bool openFiles(bool overwrite)
{
	//open and check we have access to necessary files. Then set precision on them.
	
	//Decide whether to append or truncate all files but FINAL_LATTICE_STATE_FILE
	std::ios_base::openmode mode;
	if(overwrite)
	{
		mode=ios::trunc;
		cout << "#Truncating files..." << ANNEALING_FILE << " , " <<
			CONING_FILE << " , " << ENERGY_FILE << endl;
	}
	else
	{
		mode=ios::app;
		cout << "#Appending to files..." << ANNEALING_FILE << " , " <<
			CONING_FILE << " , " << ENERGY_FILE << endl;
	}

	annealF.open(ANNEALING_FILE, mode);
	if(!annealF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << ANNEALING_FILE  << endl;
		return false;
	}

	annealF.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	annealF.precision(STDOE_PRECISION);

	//append file output as when we resume we'd like to keep the results of previous attempt.
	coningF.open(CONING_FILE, mode);
	if(!coningF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " <<  CONING_FILE  << endl;

		//close fstreams
		annealF.close();

		return false;
	}

	coningF.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	coningF.precision(STDOE_PRECISION);


	//append file output as when we resume we'd like to keep the results of previous attempt.
	energyF.open(ENERGY_FILE, mode);
	if(!energyF.is_open())
	{
		cerr << "Error: couldn't open open ofstream on file " << ENERGY_FILE << endl;

		//close fstreams
		annealF.close();
		coningF.close();
		energyF.close();

		return false;
	}

	energyF.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	energyF.precision(FILE_PRECISION);

	return true;
}

void closeFiles()
{
	coningF.close();
	energyF.close();
	annealF.close();
}

void dumpViewableLatticeState()
{
	//dump current lattice state for use will ildump.gnu
	cout << "Dumping viewable lattice State to " << REQUEST_LATTICE_STATE_FILE << "...";
	ofstream requestDumpF (REQUEST_LATTICE_STATE_FILE, ios::trunc);
	
	//set precision
	requestDumpF.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	requestDumpF.precision(FILE_PRECISION);

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

void usageMessage()
{
		cerr << "Usage: sim-state <filename> <mcs> [ options ]" << endl <<
		"<filename> - Binary state file to load for simulation." << endl <<
		"<mcs> - Number of monte carlo steps to run simulation for.\n\n" <<
		"[options]" << endl <<
		"--rand-seed <seed>\n" <<
		"Set the random number generator seed <seed> (where <seed> is a positive integer) to be used in simulator. This option is ignored if the simulation is being resumed.\n\n" <<
		"--anneal-step <annealstep>\n" <<
		"Lower \"Temperature\" in simulator every <annealstep> monte carlo steps where <annealstep> is an integer.\n" << endl;
		exit(1);

}

void handleArgs(int n, char* argv[])
{
	if(n<3)
	{
		usageMessage();
	}

	//read mandatory options
	stateFile = argv[1];

	if(atoi(argv[2]) < 1)
	{
		cerr << "Error: Number of monte carlo steps must be 1 or more.\n\n" << endl;
		usageMessage();
	}
	else
	{
		loopMax = atoi(argv[2]);
	}	

	//process optional arguments
	if(n>3)
	{
		//Set argument index to first optional argument
		int argIndex=3;
		while ( argIndex < n)
		{
			
			if( strcmp(argv[argIndex],"--rand-seed") ==0 )
			{
				
				//move to next argument index (even if it doesn't exist!)
				argIndex++;

				//check we have another argument to process
				if( (n -1) < argIndex)
				{
					cerr << "Error: Expected random seed value <seed> (int).\n\n" << endl;
					usageMessage();
				}

				//get seed value
				if( atoi(argv[argIndex]) < 0)
				{
					cerr << "Error: Random seed <seed> must be >= 0\n\n" << endl;
					usageMessage();
				}
				else
				{
					seedToUse = atoi(argv[argIndex]);
					cout << "#Overwriting seed with seed:" << seedToUse << endl;
				}

				argIndex++;
				continue;
			}

			if( strcmp(argv[argIndex],"--anneal-step")  ==0)
			{
				//move to next argument index (even if it doesn't exist!)
				argIndex++;

				//check we have another argument to process
				if( (n -1) < argIndex)
				{
					cerr << "Error: Expected annealing step <annealstep> (int)\n\n" << endl;
					usageMessage();
				}


				if(atoi(argv[argIndex]) <= 0)
				{
					cerr << "Error: Anealing step <annealstep> must be > 0\n\n" << endl;
					usageMessage();
				}
				else
				{
					annealStep = atoi(argv[argIndex]);
				}

				argIndex++;
				continue;

			}

			//If we get this far the argument hasn't been handled so it isn't a valid argument!
			cerr << "Argument " << argv[argIndex] << " not recongised.\n\n" << endl;
			usageMessage();

		}


	}
}
