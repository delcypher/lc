/* Test that loads a small lattice and tries calculating the total
*  energy in different ways. This will only work on x86_64 architecture
*  as random.bin was made on that architecture
*/

#include <iostream>
#include <cstdlib>
#include "lattice.h"
#include "exitcodes.h"
#include <fstream>

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"

void cleanup();
double** energyrow;
double** energycol;

int main(int n, char* argv[])
{
	bool fail=false;

	if(n!=2)
	{
		cerr << "Usage: " << argv[0] << " <binary_state_file>" << endl;
		return TH_BAD_ARGUMENT;
	}
	
	char* file = argv[1];

	//set cout precision
	cout.precision(30);
	cerr.precision(30);

	//create lattice object from binary state file
	Lattice nSystem = Lattice(file);
	//create another lattice which we won't touch and will use for comparsion
	Lattice original = Lattice(file);

	//display description
	nSystem.dumpDescription(std::cout);

	energyrow = (double**) malloc(nSystem.param.width*sizeof(double*));
	if(energyrow==NULL)
	{
		cerr << "Failed to allocated memory" << endl;
		return TH_FAIL;
	}

	for(int x=0; x < nSystem.param.width; x++)
	{
		energyrow[x] = (double*) malloc(nSystem.param.height*sizeof(double));
	}

	double energyrowT=0;
	for(int y=0; y < nSystem.param.height; y++)
	{
		for(int x=0; x < nSystem.param.width; x++)
		{
			energyrow[x][y] = nSystem.calculateEnergyOfCell(x,y);
			if(original != nSystem)
				cerr << "Lattice changed at " << x << "," << y << " in row scan" << endl;
		}
	}


	//calculate total energy using column scan pattern
	energycol = (double**) malloc(nSystem.param.width*sizeof(double*));

	if(energycol==NULL)
	{
		cerr << "Failed to allocated memory" << endl;
		return TH_FAIL;
		cleanup();
	}
	for(int x=0; x < nSystem.param.width; x++)
	{
		energycol[x] = (double*) malloc(nSystem.param.height*sizeof(double));
	}

	double energycolT=0;
	for(int x=0; x < nSystem.param.width; x++)
	{
		for(int y=0; y < nSystem.param.height; y++)
		{
			energycol[x][y] = nSystem.calculateEnergyOfCell(x,y);
		}
	}
	
	//loop through calculated values and compared.
	for(int counter=0; counter < (nSystem.param.width*nSystem.param.height); counter++)
	{
		int x=counter%nSystem.param.width;
		int y=counter/nSystem.param.width;
		energyrowT +=energyrow[x][y];
		energycolT +=energycol[x][y];
		if(energyrow[x][y] != energycol[x][y])
		{
			cout << "Cell at counter: (" << x << "," << y << ") don't match! ";
			cout << "energy (row calc) from cell " << energyrow[x][y];
			cout << " energy (col calc) from cell " << energycol[x][y] << endl;
			fail =true;
		}
	}


	cout << "#Energy of Lattice (row): " << energyrowT << endl;
	cout << "#Energy of Lattice (col): " << energycolT << endl;

	if(energycolT != energyrowT)
	{
		cerr << "Error: Energies don't match!" << endl;
		ofstream file("end.dump");
		if(file.is_open())
		{
			nSystem.indexedNDump(file);
		}
		else
		{
			cerr << "failed to write end.dump" << endl;
		}
		file.close();
		fail=true;
	}
	
	cleanup();
	return fail?TH_FAIL:TH_SUCCESS;
}

void cleanup()
{
	free(energyrow);
	free(energycol);
}


