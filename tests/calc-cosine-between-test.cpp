/* Test harness to test the calculateCosineBetween() function.
*  This program works with two DirectorElements systematically varying 
*  the angle between them and comparing calculatCosinBetween(&a,&b) and
*  cos(bAngle - aAngle).
*  
*  The program also checks my (Dan's) assertion that if the angle between
*  a & b is > PI/2 then cos(bAngle - aAngle) is < 0.
*/
#include <iostream>
#include <cmath>
#include "directorelement.h"
#include "exitcodes.h"
#include "common.h"
#include <cstdlib>

using namespace std;

double relativeError(double received, double expected);
double toRadians(double angleInDegrees);
double toDegrees(double angleInRadians);

int main(int n, char* argv[])
{
	double calcCos;
	double realCalcCos;

	cerr.precision(20);

	if(n!=3)
	{
		cerr << "Usage: " << argv[0] << " <angular_step> <relative_error>" << endl <<
		"<angular_step> - The angle to step through in degrees." << endl <<
		"<relative_error> - The acceptable relative error (decimal fraction) between functions" << endl;
		exit(TH_BAD_ARGUMENT);
	}

	double angleStep = toRadians(atof(argv[1]));
	double acceptRE = atof(argv[2]);
	//calculated relative error
	double re=0;

	cout << "Angle step: " << atof(argv[1]) << " in degrees, " << angleStep << " in radians." << endl;

	//Start with both a & b pointing along x-axis
	DirectorElement a(1,0,0);
	DirectorElement b(1,0,0);
	
	//Move a through an angular range of [0,2*PI] is steps of angleStep
	for(double aAngle=0; aAngle <= 2*PI; aAngle += angleStep)
	{
		setDirectorAngle(&a,aAngle);

		//Move b through an angular range of [0,2*PI] is steps of angleStep
		for(double bAngle=0; bAngle <= 2*PI; bAngle += angleStep)
		{
			setDirectorAngle(&b,bAngle);

			calcCos = calculateCosineBetween(&a,&b);
			realCalcCos = cos( bAngle - aAngle);
			
			//calculate relative error
			re = relativeError(calcCos,realCalcCos);

			if(re > acceptRE)
			{
				cerr << "Error: a:" << toDegrees(aAngle) << " b:" << toDegrees(bAngle) << " RE:" << re << 
				", calculatecosineBetween(a,b)=" << calcCos << " ,cos(" << toDegrees(bAngle - aAngle)  << ")=" << realCalcCos << endl;
				exit(TH_FAIL);
			}

			//I assert that if fabs(bAngle - aAngle) > 90 degrees then calculateCosineBetween(&a,&b) < 0
			if( fabs(bAngle - aAngle) > (PI/2) && calcCos >=0 )
			{
				cerr << "Assertion failed: bAngle-aAngle:" << fabs(bAngle - aAngle) << " (radians) " << toDegrees(fabs(bAngle - aAngle)) <<
				" (degrees)" << " calculateCosineBetween(a,b)=" << calcCos << " cos(...)=" << realCalcCos << endl;
				exit(TH_FAIL);
			}


		}

	}

	return TH_SUCCESS;

}

double relativeError(double received, double expected)
{
	double re;
	re = (expected - received)/expected;
	return re;
}

double toRadians(double angleInDegrees)
{
	return (PI/180)*angleInDegrees;
}

double toDegrees(double angleInRadians)
{
	return (180/PI)*angleInRadians;
}
