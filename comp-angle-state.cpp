/* This program is used to compare the angle distribution of a lattice specified by a binary state
*  file against the angular distribution of a minimum energy state of an analytical solution specified on
*  the command line.
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
	
	//Do comparision
	nSystem.angleCompareWith(state,std::cout,acceptibleError);

	return 0;
}


