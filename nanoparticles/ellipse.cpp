#include "../nanoparticle.h"
#include "ellipse.h"
#include <cmath>
#include <iostream>

using namespace std;

EllipticalNanoparticle::EllipticalNanoparticle(int xCentre, int yCentre, double aValue, double bValue, double thetaValue, enum boundary boundaryType) : 
Nanoparticle(xCentre, yCentre, Nanoparticle::ELLIPSE)
{
	mBoundary=boundaryType;
	if (aValue > 0 && bValue > 0)
	{
		a = aValue;
		b = bValue;
		theta = thetaValue;
	}
	else
	{
		cerr << "Error: a and b must be greater than 0!" << endl;
		badState=true;
	}

}

EllipticalNanoparticle::EllipticalNanoparticle(std::ifstream & stream) : Nanoparticle(stream,Nanoparticle::ELLIPSE)
{
	/* Format:
        *  <parents_data><a><b><theta><mBoundary>
        *
        */
	
	//don't set badState to true here as parent may of set it to false.

	if(!stream.good())
	{
		cerr << "Error: Couldn't construct Elliptical Nanoparticle. Stream not good" << endl;
		badState=true;		
	}

	stream.read( (char*) &a, sizeof(double));
	stream.read( (char*) &b, sizeof(double));
	stream.read( (char*) &theta, sizeof(double));
	stream.read( (char*) &mBoundary, sizeof(enum Nanoparticle::types));

	

}

bool EllipticalNanoparticle::processCell(int x, int y, enum writeMethods method, DirectorElement* element)
{
	double effectiveRadius;
	double distance;
	double pointTheta;
	double angle;
	double vectorAngle;

	// get polar coordinate for cell position
	distance= sqrt( (x - mxPos)*(x - mxPos) + (y - myPos)*(y - myPos) );
	pointTheta = atan2(y-myPos, x-mxPos);
	angle = pointTheta - theta;

	effectiveRadius = a*b / sqrt(b*b*cos(angle)*cos(angle) + a*a*sin(angle)*sin(angle) );
	
	/* We compare to (distance +0.5) not (distance) because (distance) measures from the centre of cell (mxPos,myPos)
	*  to the centre of another cell. But we care about including whole cells so we use (distance +0.5)
	*/
	if( (distance +0.5) <= effectiveRadius)
	{
		//we are in the ellipse

		switch(method)
		{
			case DRY_ADD:
				if(element->isNanoparticle == true)
				{
					//overlapping nanoparticle
					cerr << "Error: Adding nanoparticle would result in overlap with another at (" << x << "," << y << ")" << endl;
					return false;
				}

				return true;
			break;

			case ADD:
				{
					if(element->isNanoparticle == true)
					{
						//overlapping nanoparticle
						cerr << "Error: Nanoparticle would overlap itself at (" << x << "," << y << ")" << endl;
						return false;
					}	
					
					/* We're in the centre of the ellipse. The director field
					*  is not defined here so we set it to zero!
					*/
					if(distance==0)
					{
						element->isNanoparticle=true;
						element->x=0;
						element->y=0;
						return true;
					}

					double fac1 = (y- myPos)*cos(theta) - (x - mxPos)*sin(theta);
					double fac2 = (x - mxPos)*cos(theta) + (y - myPos)*sin(theta);
					double numerator=0;
					double denominator=0;

					if(mBoundary== PARALLEL)
					{
						numerator = a*a*sin(theta)*fac1 - b*b*cos(theta)*fac2;
						denominator = b*b*sin(theta)*fac2 + a*a*cos(theta)*fac1;
						vectorAngle= atan2(numerator,denominator);
					}
					else if(mBoundary == PERPENDICULAR)
					{
						numerator = b*b*sin(theta)*fac2 + a*a*cos(theta)*fac1;
						denominator = b*b*cos(theta)*fac2 - a*a*sin(theta)*fac1;
						vectorAngle= atan2(numerator,denominator);
					}
					else
					{
						cerr << "Error: boundary " << mBoundary << " (enum EllipticalNanoparticle::boundary) not supported!";
						return false;
					}

					
					//everything seems ok. We now add the particle
					element->isNanoparticle=true;
					element->x = cos(vectorAngle);
					element->y = sin(vectorAngle);
					return true;
				}
			break;

			default:
				cerr << "Error: Unknown write method!" << endl;
				return false;
		}

	}
	else
	{
		//This lattice point is outside the circle, do nothing
		return true;
	}
	
}

std::string EllipticalNanoparticle::getDescription()
{
	std::stringstream description(std::stringstream::out);
	
	description.precision(STDOE_PRECISION);

	description << "Elliptical Nanoparticle: " <<
	"a:" <<
	a <<
	", b:" <<
	b <<
	", Rotation angle w.r.t x-axis (Radians):" <<
	theta <<
	", Boundary:" << mBoundary << " (enum EllipticalNanoparticle::boundary)" <<
	", Centre @ (" <<
	mxPos <<
	"," <<
	myPos <<
	")" << ", State is " << (badState?"Bad":"Good") ;
	
	return description.str();

}

bool EllipticalNanoparticle::saveState(std::ofstream & stream)
{
	/* Format:
	*  <parents_data><a><b><theta><mBoundary>
	*
	*/

	//call parent to save its data
	if(!Nanoparticle::saveState(stream))
	{
		cerr << "Error: Failed to save EllipticalNanoparticle State. Parent call failed" << endl;
		return false;
	}
	
	stream.write( (char*) &a, sizeof(double));
	stream.write( (char*) &b, sizeof(double));
	stream.write( (char*) &theta, sizeof(double));
	stream.write( (char*) &mBoundary, sizeof(enum Nanoparticle::types));

	if(!stream.good())
	{
		cerr << "Error: Couldn't save EllipticalNanoparticle State. stream isn't good" << endl;
		return false;
	}

	return true;
}
