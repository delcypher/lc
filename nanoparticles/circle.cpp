/* Implementation of Circular nanoparticle class

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
#include "../nanoparticle.h"
#include "circle.h"
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

CircularNanoparticle::CircularNanoparticle(int xCentre, int yCentre, int radius, enum boundary boundaryType) : 
Nanoparticle(xCentre,yCentre,Nanoparticle::CIRCLE)
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

CircularNanoparticle::CircularNanoparticle(std::ifstream & stream) : Nanoparticle(stream, Nanoparticle::CIRCLE)
{
	/* FORMAT
	*  <parent_data><mRadius><mBoundary>
	*
	*/

	//we not set badState here as parent constructor may of set it to bad

	stream.read( (char*) &mRadius,sizeof(int));

	if(!stream.good())
	{
		cerr << "Error: Couldn't create CircularNanoparticle. Failed to read mRadius" << endl;
		badState=true;
	}

	stream.read( (char*) &mBoundary,sizeof(enum boundary));

	if(!stream.good())
	{
		cerr << "Error: Couldn't create CircularNanoparticle. Failed to read mBoundary" << endl;
		badState=true;
	}

}


bool CircularNanoparticle::processCell(int x, int y, enum writeMethods method, DirectorElement* element)
{
	double distance;
	double vectorAngle;	
	//Calculate distance from circle centre to point of interest (x,y)
	distance= sqrt( (x - mxPos)*(x - mxPos) + (y - myPos)*(y - myPos) );
	

	/* We compare to (distance +0.5) not (distance) because (distance) measures from the centre of cell (mxPos,myPos)
	*  to the centre of another cell. But we care about including whole cells so we use (distance +0.5)
	*/
	if( (distance +0.5) <= (double) mRadius)
	{
		//we are in the circle

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
					
					/* We're in the centre of the circle. The director field
					*  is not defined here so we set it to zero!
					*/
					if(distance==0)
					{
						element->isNanoparticle=true;
						element->x=0;
						element->y=0;
						return true;
					}

					if(mBoundary== PARALLEL)
					{
						vectorAngle= atan2(-(x - mxPos),(y - myPos));
					}
					else if(mBoundary == PERPENDICULAR)
					{
						vectorAngle= atan2((y - myPos),(x - mxPos));
					}
					else
					{
						cerr << "Error: boundary " << mBoundary << " (enum CircularNanoparticle::boundary) not supported!";
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
				cerr << "Error: Unknown write method " << method << " (enum Nanoparticle::writeMethods) " << endl;
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

	description.precision(STDOE_PRECISION);

	description << "Circular Nanoparticle: Radius:" 
		<< mRadius << 
		", Boundary:" <<
		mBoundary << " (enum CircularNanoparticle::boundary)" <<
		", Centre @ (" << 
		mxPos << 
		"," << 
		myPos << 
		")" << ", State is " << (badState?"Bad":"Good") ;
	return description.str();
}

bool CircularNanoparticle::saveState(std::ofstream & stream)
{
	/* Format:
	*  
	*  <parent_data><mRadius><boundary>
	*/

	//call parent so it can sort out saving its part
	if(! Nanoparticle::saveState(stream) )
	{
		return false;
	}

	
	if(!stream.good())
	{
		cerr << "Error: Cannot save CircularNanoparticle. Stream not good" << endl;
		return false;
	}

	stream.write( (char*) &mRadius, sizeof(int));

	if(!stream.good())
	{
		cerr << "Error: Cannot save CircularNanoparticle. Failed writing mRadius" << endl;
		return false;
	}

	stream.write( (char*) &mBoundary, sizeof(enum boundary));

	if(!stream.good())
	{
		cerr << "Error: Cannot save CircularNanoparticle. Failed writing mBoundary" << endl;
		return false;
	}

	return true;
}
