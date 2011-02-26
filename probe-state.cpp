/* This program is used to view a binary state file specified
*  on the command line
*/

#include <iostream>
#include <cstdlib>
#include "lattice.h"

using namespace std;

/* Include the nanoparticle header files you wish you use here.
*  Make sure the nanoparticle is listed in the OBJECTS variable
*  in the make file too!
*/
#include "nanoparticles/circle.h"
#include "nanoparticles/ellipse.h"


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
	cout.precision(STD_PRECISION);
	cout << "#Displaying values to " << STD_PRECISION << " decimal places" << endl;

	//create lattice object from binary state file
	Lattice nSystem = Lattice(loadfile);
	
	//display description
	nSystem.dumpDescription(std::cout);

	return 0;
}


