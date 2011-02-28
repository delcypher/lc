/* Test that loads a small lattice and tries calculating the total
*  energy in different ways. This will only work on x86_64 architecture
*  as random.bin was made on that architecture
*/

#include <iostream>
#include <cstdlib>
#include "lattice.h"
#include "exitcodes.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"


int main()
{
	//set cout precision
	cout.precision(30);
	cerr.precision(30);

	//create lattice object from binary state file
	Lattice nSystem = Lattice("tests/random.bin");
	Lattice original = Lattice("tests/random.bin");

	//display description
	nSystem.dumpDescription(std::cout);

	double energyrow[2][2];
	double energyrowT=0;
	for(int y=0; y < nSystem.param.height; y++)
	{
		for(int x=0; x < nSystem.param.width; x++)
		{
			energyrow[x][y] = nSystem.calculateEnergyOfCell(x,y);
			if(original != nSystem)
				cerr << "Lattice changed at " << x << "," << y << endl;
		}
	}


	//calculate total energy using column scan pattern
	double energycol[2][2];
	double energycolT=0;
	for(int x=0; x < nSystem.param.width; x++)
	{
		for(int y=0; y < nSystem.param.height; y++)
		{
			energycol[x][y] = nSystem.calculateEnergyOfCell(x,y);
		}
	}
	
	//loop through calculated values and compared.
	for(int counter=0; counter < 4; counter++)
	{
		int x=counter%nSystem.param.width;
		int y=counter/nSystem.param.width;
		energyrowT +=energyrow[x][y];
		energycolT +=energycol[x][y];
		if(energyrow[x][y] != energycol[x][y])
		{
			cerr << "Cell at counter: (" << x << "," << y << ") don't match!" << endl;
			cerr << "row energy from cell " << energyrow[x][y] << endl;
			cerr << "col energy from cell " << energycol[x][y] << endl;
		}
	}


	cout << "#Energy of Lattice (row): " << energyrowT << endl;
	cout << "#Energy of Lattice (col): " << energycolT << endl;

	if(energycolT != energyrowT)
	{
		cerr << "Error: Energies don't match!" << endl;
		return TH_FAIL;
	}

	return TH_SUCCESS;
}


