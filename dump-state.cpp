/* This program is used to dump a binary state file specified
*  on the command line to standard output. This dumped state
*  is viewable using the GNUplot script ildump.gnu

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
	if(n !=2)
	{
		cerr << "Usage: " << argv[0] << " <filename>" << endl <<
		"<filename> - Binary state file to read" << endl;
		exit(1);
	}

	char* loadfile = argv[1];
	
	//set cout precision
	cout.setf(STREAM_FLOAT_FORMAT,ios::floatfield);
	cout.precision(STDOE_PRECISION);
	cout << "#Displaying values to " << STDOE_PRECISION << " decimal places" << endl;

	//create lattice object from binary state file
	Lattice nSystem = Lattice(loadfile);
	
	//display description
	nSystem.indexedNDump(std::cout);

	return 0;
}


