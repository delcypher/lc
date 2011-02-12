/* This is a test for checking the Lattice::calculateEnergyOfCell() method gives
*  the same answer as a known analytical solution.
* 
*  The lattice is configured in the known minimum free energy per unit volume
*  state (which is constant).
*
*  The configuration is n = (cos(theta), sin(theta), 0)
*  theta = (PI/2)(y + 1)/(height +1)
*
* We check two things:
* - That every cell has the same calculated free energy per unit volume
* - That every cell value is near the true analytical value using <relative_error_tolerance>
*
* ./host-energy-analytical <width> <height> <cell_cell_relative_error_tolerance> <cell_analytical_relative_error_tolerance>
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "randgen.h"
#include "differentiate.h"
#include "lattice.h"
#include "exitcodes.h"

using namespace std;

//This is the analytical free energy per unit volume.
double analyticalFreeEnergy;

double calculatedFreeEnergy;

bool failed;

double relativeError(double expected, double recieved);
double cellCellTolerance;
double cellAnalyticalTolerance;

//aray of calculated cell energies
double* cellEnergy;

void cleanup();

int main(int n, char* argv[])
{
	//used to keep track of failure of all tests and so determines program's exit code
	failed =false;

	//The number of times a comparision between the firstCell and other cells in the lattice has failed.
	int failedCellCellCount=0;

	//The number of times a comparision between a cell in the lattice and the computed analytical value has failed.
	int failedCellAnalyticalCount=0;

	if(n!=5)
	{
		cerr << "Usage: " << argv[0] << 
		"<width> <height> <cell_cell_relative_error_tolerance> <cell_analytical_relative_error_tolerance>\n\n" <<
		"Relative errors should be expressed as a decimal fraction.\n";
		exit(TH_BAD_ARGUMENT);
	}


	LatticeConfig configuration;

	configuration.width = atoi(argv[1]);
	configuration.height= atoi(argv[2]);
	
	//set relative cell to cell error tolerance
	cellCellTolerance = atof(argv[3]);

	//set relative cell to analytical solution for a cell  error tolerance
	cellAnalyticalTolerance = atof(argv[4]);

	//set initial director alignment
	configuration.initialState = LatticeConfig::K1_EQUAL_K3;

	//set boundary conditions
	configuration.topBoundary = LatticeConfig::BOUNDARY_PERPENDICULAR;
	configuration.bottomBoundary = LatticeConfig::BOUNDARY_PARALLEL;
	configuration.leftBoundary = LatticeConfig::BOUNDARY_PERIODIC;
	configuration.rightBoundary = LatticeConfig::BOUNDARY_PERIODIC;

	//set lattice beta value (assume k_1 = k_3)
	configuration.beta = 1;

	//create lattice object, with (configuration, dump precision)
	Lattice nSystem = Lattice(configuration,10);

	//create array to hold calculated cell enery values
	cellEnergy = (double*) calloc((configuration.width*configuration.height),sizeof(double));

	/*loop over each cell in lattice and calculate free energy per unit volume
	*
	*/
	int index=0;
	double firstCellEnergy=0;
	for(int y=0; y < configuration.height; y++)
	{
		for(int x=0; x < configuration.width; x++)
		{
			index = x + (configuration.width)*y;
			cellEnergy[index] = nSystem.calculateEnergyOfCell(x,y);

			if(index==0)
			{
				firstCellEnergy=cellEnergy[index];
			}
			else
			{
				//compare energy of cell (x,y) against energy of cell (0,0)
				if( fabs(relativeError(firstCellEnergy,cellEnergy[index])) >= cellCellTolerance )
				{
					failed=true;
					failedCellCellCount++;
					fprintf(stdout,"firstCell: %.20f ,Cell(%d,%d): %.20f. RE:%.20f\n",
						firstCellEnergy,
						x,
						y,
						cellEnergy[index],
						relativeError(firstCellEnergy,cellEnergy[index])
						);
				}
			}
		}
	}
	


	//assume k_1 =1
	analyticalFreeEnergy=PI*PI/(8*(configuration.height +1)*(configuration.height +1));

	//loop over each calculated energy for a cell and compare to analyticalFreeEnergy
	for(int y=0; y < configuration.height; y++)
	{
		for(int x=0; x < configuration.width; x++)
		{
			index = x + (configuration.width)*y;
			if( fabs(relativeError(analyticalFreeEnergy,cellEnergy[index])) >= cellAnalyticalTolerance)
			{
				failed=true;
				fprintf(stdout,"calc: %.20f analytical: %.20f @ (%d,%d). RE:%.20f\n",
					cellEnergy[index],
					analyticalFreeEnergy,
					x,
					y,
					relativeError(analyticalFreeEnergy,cellEnergy[index])
					);
				failedCellAnalyticalCount++;	
			}

		}
	}
	
	cout << "Cell-Cell failures (RE " << cellCellTolerance 
		<< "):" << 100*( (double) failedCellCellCount)/(configuration.width*configuration.height)  << 
		"% of lattice. Cell-analytical failures (" << 
		cellAnalyticalTolerance << 
		"): " << 
		100*( (double) failedCellAnalyticalCount)/(configuration.width*configuration.height)  <<
		" % of lattice" << endl; 

	cleanup();

	if(failed)
	{
		return TH_FAIL;
	}
	else
	{
		return TH_SUCCESS;
	}
}


double relativeError(double expected, double recieved)
{
	double rError= (recieved /expected) -1;
	return rError;
}

void cleanup()
{
	free(cellEnergy);
}
