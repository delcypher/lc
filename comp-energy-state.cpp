/* This program is used to compare the lattice specified by a binary state
*  file against the minimum energy state of an analytical solution specified on
*  the command line.

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
#include "lattice.h"

using namespace std;


int main(int n, char* argv[])
{
	if(n !=4)
	{
		cerr << "Usage: " << argv[0] << " <filename> <state> <acceptible_error>" << endl <<
		"<filename> - Binary state file to read\n" <<
		"<state> - Analytical energy state (LatticeConfig::latticeState enum) to compare with\n" <<
		"<acceptible_error> - The maximum acceptible absolute error." << endl;
		exit(1);
	}

	char* loadfile = argv[1];
	LatticeConfig::latticeState state = (LatticeConfig::latticeState) atoi(argv[2]);
	double acceptibleError = atof(argv[3]);

	//set cout precision
	cout.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	cout.precision(STDOE_PRECISION);
	cout << "#Displaying values to " << STDOE_PRECISION << " decimal places" << endl;

	//create lattice object from binary state file
	Lattice nSystem = Lattice(loadfile);
	
	//Do comparison
	nSystem.energyCompareWith(state,std::cout,acceptibleError);

	return 0;
}


