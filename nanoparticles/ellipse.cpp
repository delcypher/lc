#include "../nanoparticle.h"
#include "ellipse.h"
#include <cmath>
#include <iostream>

using namespace std;

EllipticalNanoparticle::EllipticalNanoparticle(int xCentre, int yCentre, double aValue, double bValue, double thetaValue, enum boundary boundaryType) : Nanoparticle(xCentre, yCentre)
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

EllipticalNanoparticle::EllipticalNanoparticle(const std::string & state) : Nanoparticle(state)
{
	//Assume parameters in std::string are in the following format
	//<xPos> <yPos> <a> <b> <theta> <boundary> <badState>
	string buffer;
	stringstream stream(state);

	//Assume mxPos and myPos handled by parent class
	stream >> buffer;
	stream >> buffer;

	//get a
	if(stream.eof())
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. a isn't specified" << endl;
		badState=true;
	}
	
	stream >> buffer;
	a = atof(buffer.c_str());
	if(a <=0)
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. a must be > 0" << endl;
		badState=true;
	}

	//get b
	if(stream.eof())
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. b isn't specified" << endl;
		badState=true;
	}
	
	stream >> buffer;
	b = atof(buffer.c_str());
	if(b <=0)
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. b must be > 0" << endl;
		badState=true;
	}

	//get theta
	if(stream.eof())
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. theta isn't specified" << endl;
		badState=true;
	}
	
	stream >> buffer;
	theta = atof(buffer.c_str());
	
	//get boundary
	if(stream.eof())
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. boundary isn't specified" << endl;
		badState=true;
	}
	
	stream >> buffer;
	mBoundary = (boundary) atoi(buffer.c_str());

	//get state
	if(stream.eof())
	{
		cerr << "Error: Can't construct EllipticalNanoparticle. state isn't specified" << endl;
		badState=true;
	}
	
	stream >> buffer;
	badState = (bool) atoi(buffer.c_str());


}


bool EllipticalNanoparticle::processCell(int x, int y, enum writeMethods method, DirectorElement* element)
{
	double r;
	double distance;
	double pointTheta;
	double angle;
	double vectorAngle;

	// get polar coordinate for cell position
	distance= sqrt( (x - mxPos)*(x - mxPos) + (y - myPos)*(y - myPos) );
	pointTheta = atan2(y-myPos, x-mxPos);
	angle = pointTheta - theta;

	r = a*b / sqrt(b*b*cos(angle)*cos(angle) + a*a*sin(angle)*sin(angle) );
	

	if(distance <= r)
	{
		//we are in the ellipse

		switch(method)
		{
			case DRY_ADD:
				if(element->isNanoparticle == 1)
				{
					//overlapping nanoparticle
					cerr << "Error: Adding nanoparticle would result in overlap with another at (" << x << "," << y << ")" << endl;
					return false;
				}

				return true;
			break;

			case ADD:
				{
					if(element->isNanoparticle == 1)
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
						element->isNanoparticle=1;
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
					else
					{
						//assume perpendicular boundary wanted
						numerator = b*b*sin(theta)*fac2 + a*a*cos(theta)*fac1;
						denominator = b*b*cos(theta)*fac2 - a*a*sin(theta)*fac1;
						vectorAngle= atan2(numerator,denominator);
					}

					
					//everything seems ok. We now add the particle
					element->isNanoparticle=1;
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
	description << "Elliptical Nanoparticle: " <<
	"a:" <<
	a <<
	", b:" <<
	b <<
	", Rotation angle w.r.t x-axis (Radians):" <<
	theta <<
	", Boundary(enum):" << mBoundary <<
	", Centre @ (" <<
	mxPos <<
	"," <<
	myPos <<
	")" << ", State is " << (badState?"Bad":"Good") ;
	
	return description.str();

}

std::string EllipticalNanoparticle::saveState()
{
	stringstream state;
	state.precision(STATE_SAVE_PRECISION);
	state << mxPos << " " << myPos << " " << a << " " << b << " " << theta << " " << mBoundary << " " << badState;
	return state.str();
}
