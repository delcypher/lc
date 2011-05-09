/* dvid- Director variation in direction.
*
*  This program loads a binary state and will output data for plotting
*  of how the Director orientation changes when looked at in a particular direction specified by
*  the origin a vector r and the angle it makes with the x-axis.
*/

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "lattice.h"

using namespace std;


int main(int n, char* argv[])
{
	if(n <7)
	{
		cerr << "Usage: " << argv[0] << " <filename> <x> <y> <angle> <max_r> <angular_restriction> [ -n ]" << endl <<
		"<filename> - Binary state file to read" << endl <<
		"<x> - The x co-ordinate of the origin of r (int)" << endl <<
		"<y> - The y co-ordinate of the origin of r (int)" << endl <<
		"<angle> - The angle r makes with x-axis in radians" << endl <<
		"<max_r> - The maximum magnitude of r (float)" << endl << 
		"<angular_restriction> - The angular restriction to apply on the lattice (enum Lattice::angularRegion)\n\n" <<
		"OPTIONAL:" << endl <<
		"-n \nDo not provide plot data for cells that are marked as being a nanoparticle. This is NOT the default behaviour!" << endl;
		exit(1);
	}
	
	//Decide if we should be outputting plot data for nanoparticle cells
	bool plotNanoparticles=true;
	if(n==8)
	{
		if( strcmp(argv[7],"-n") == 0)
		{
			cout << "#Not providing plot data for nanoparticle cells" << endl;
			plotNanoparticles=false;
		}
		else
		{
			cerr << "Error: " << argv[7] << " is not a supported argument!" << endl;
			exit(1);
		}
	}

	if(n>8)
		cerr << "Warning: Ignoring extra arguments!" << endl;

	char* loadfile = argv[1];

	int xOrigin = atoi(argv[2]);
	int yOrigin = atoi(argv[3]);
	double angle = atof(argv[4]);
	double maxR = atof(argv[5]);
	Lattice::angularRegion angularRestriction = (Lattice::angularRegion) atoi(argv[6]);
	int x=0;//the cell x co-ordinate of interest.
	int y=0;//the cell y co-ordinate of interest.
	double angleInCell=0;

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

	//restrict angular range
	cout << "#Restricting angular range to " << angularRestriction << " (enum Lattice::angularRegion)" << endl;
	nSystem.restrictAngularRange(angularRestriction);
	
	//display description
	nSystem.dumpDescription(std::cout);
	
	//Write header
	cout << "#[R] [angle_(radians)] [x_cell_co-ordinate] [y_cell_co-ordinate]" << endl;

	//loop over a range of different r values
	for(double absR=0; absR<= maxR ; absR+= 1)
	{
		//calculate the x & y co-ordinate of the cell of interest
		// +0.5 is to simulate rounding casting to int.
		x= (int) (  ((double) xOrigin) + absR*cos(angle)  + 0.5 );
		y= (int) (  ((double) yOrigin) + absR*sin(angle)  + 0.5 );
		
		if(x > nSystem.param.width || y > nSystem.param.height)
		{
			cerr << "Warning: Going out of lattice @ (" << x << "," << y << ")" << endl;
		}

		//get orientation of cell
		angleInCell=nSystem.getN(x,y)->calculateAngle();

		//write file output
		if( nSystem.getN(x,y)->isNanoparticle==true)
		{
			if(plotNanoparticles)
			{
				//write plot data as normal but inform that the cell is a nanoparticle cell
				cout << "# (" << x << "," << y << ")  is a nanoparticle cell" << endl;
				cout << absR << " " << angleInCell << " " << x << " " << y << endl;
			}
			else
			{
				//don't plot cell as it's a nanoparticle cell and it's been requested that we don't plot it!
				cout << "#" << absR << " " << angleInCell << " " << x << " " << y << " NOT PLOTTING NANOPARTICLE CELL" << endl;
			}

		}
		else
		{
			//write plot data as normal
			cout << absR << " " << angleInCell << " " << x << " " << y << endl;
		}

	}

	cout << "#Finished" << endl;
	
	return 0;
}


