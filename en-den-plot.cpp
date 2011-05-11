/* en-den-plot : Energy density plotter
*
*  This program loads a binary state and will output data for plotting
*  the energy density as a function of x and y.

   Copyright (C) 2010 Dan Liew & Alex Allen
   
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "lattice.h"

using namespace std;


int main(int n, char* argv[])
{
	if(n <2)
	{
		cerr << "Usage: " << argv[0] << " <filename> [ -n ]" << endl <<
		"<filename> - Binary state file to read" << endl <<
		"OPTIONAL:" << endl <<
		"-n \nForce cells that are marked as being a nanoparticle to have energy density of 0. This is NOT the default behaviour!" << endl;
		exit(1);
	}
	
	//Decide if we should be outputting plot data for nanoparticle cells
	bool plotNanoparticles=true;
	if(n==3)
	{
		if( strcmp(argv[2],"-n") == 0)
		{
			cout << "#Not providing plot data for nanoparticle cells" << endl;
			plotNanoparticles=false;
		}
		else
		{
			cerr << "Error: " << argv[2] << " is not a supported argument!" << endl;
			exit(1);
		}
	}

	if(n>3)
		cerr << "Warning: Ignoring extra arguments!" << endl;

	char* loadfile = argv[1];
	
	//set cout precision
	cout.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	cout.precision(STDOE_PRECISION);
	cout << "#Displaying values to " << STDOE_PRECISION << " decimal places" << endl;

	//create lattice object from binary state file
	Lattice nSystem = Lattice(loadfile);
	
	if(nSystem.inBadState())
	{
		cerr << "Warning Lattice is in BAD state!" << endl;
	}

	//display description
	nSystem.dumpDescription(std::cout);
	
	//Write header
	cout << "#[x_prime] [y_prime] [energy density]" << endl;

	double cellEnergy=0;

	//loop over all the cells in the lattice excluding boundary cells
	for(int xPos=0; xPos < nSystem.param.width ; xPos++)
	{
		for(int yPos=0; yPos < nSystem.param.height; yPos++)
		{
			cellEnergy = nSystem.calculateEnergyOfCell(xPos,yPos);
			
			//check if cell is nanoparticle
			if(nSystem.getN(xPos,yPos)->isNanoparticle && !plotNanoparticles)
			{
				cellEnergy=0;
				cout << "#Cell (" << xPos << "," << yPos << ") is a nanoparticle, reporting 0 energy density!" << endl;
			}
			else if(nSystem.getN(xPos,yPos)->isNanoparticle && plotNanoparticles)
			{
				cout << "#Cell (" << xPos << "," << yPos << ") is a nanoparticle, but reporting energy density anyway!" << endl;
			}

			//produce output for plotting
			cout << xPos << " " << yPos << " " << cellEnergy << endl;
		}

		//add blank line to seperate scanlines for gnuplot's pm3d splot feature.
		cout << " " << endl;
	}

	cout << "#Finished" << endl;
	
	return 0;
}


