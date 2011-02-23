/* Implementation of Circular nanoparticle by Dan Liew & Alex Allen
*
*/
#include "../nanoparticle.h"
#include "circle.h"
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include "common.h"

using namespace std;

CircularNanoparticle::CircularNanoparticle(int xCentre, int yCentre, int radius, enum boundary boundaryType) : 
Nanoparticle(xCentre,yCentre)
{
	mBoundary=boundaryType;
	if (radius > 0)
	{
		mRadius=radius;
	}
	else
	{
		cerr << "Error: Cannot have negative radius!" << endl;
		badState=true;
	}

}

CircularNanoparticle::CircularNanoparticle(const std::string & state) : Nanoparticle(state)
{
	//Assume parameters are space seperated in the format
	// <xPos> <yPos> <radius> <boundary> <bad state>
	string buffer;
	stringstream stream(state);

	//we've already passed xPos & yPos to parent class, so skip those
	stream >> buffer;
	stream >> buffer;
	
	//set radius
	if(stream.eof())
	{
		cerr << "Error: Couldn't construct Circular nanoparticle. Can't get radius." << endl;
		badState=true;
	}

	stream >> buffer;
	mRadius = atoi(buffer.c_str());
	if(mRadius <=0)
	{
		cerr << "Error: Cannot have negative radius!" << endl;
		badState=true;
	}

	//set boundary

	if(stream.eof())
	{
		cerr << "Error: Couldn't construct Circular nanoparticle. Can't get boundary." << endl;
		badState=true;
	}
	stream >> buffer;
	mBoundary = (boundary) atoi(buffer.c_str());

	if(stream.eof())
	{
		cerr << "Error: Couldn't construct Circular nanoparticle. Can't get state... so setting bad state!" << endl;
		badState=true;
	}
	stream >> buffer;
	badState = (bool) atoi(buffer.c_str());

}

bool CircularNanoparticle::processCell(int x, int y, enum writeMethods method, DirectorElement* element)
{
	double distance;
	double vectorAngle;	
	//Calculate distance from circle centre to point of interest (x,y)
	distance= sqrt( (x - mxPos)*(x - mxPos) + (y - myPos)*(y - myPos) );
	

	if(distance <= (double) mRadius)
	{
		//we are in the circle

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
					
					/* We're in the centre of the circle. The director field
					*  is not defined here so we set it to zero!
					*/
					if(distance==0)
					{
						element->isNanoparticle=1;
						element->x=0;
						element->y=0;
						return true;
					}

					if(mBoundary== PARALLEL)
					{
						vectorAngle= atan2(-(x - mxPos),(y - myPos));
					}
					else
					{
						//assume perpendicular boundary wanted
						vectorAngle= atan2((y - myPos),(x - mxPos));
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

std::string CircularNanoparticle::getDescription()
{
	std::stringstream description (std::stringstream::out);
	description << "Circular Nanoparticle: Radius:" 
		<< mRadius << 
		", Boundary(enum):" <<
		mBoundary <<
		", Centre @ (" << 
		mxPos << 
		"," << 
		myPos << 
		")" << ", State is " << (badState?"Bad":"Good") ;
	return description.str();
}

std::string CircularNanoparticle::saveState()
{
	std::stringstream state(std::stringstream::out);
	state.precision(STATE_SAVE_PRECISION);

	state << mxPos << " " << myPos << " " << mRadius << " " << mBoundary << " " << badState; 

	return state.str();

}
