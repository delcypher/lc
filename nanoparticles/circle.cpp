/* Implementation of Circular nanoparticle by Dan Liew & Alex Allen
*
*/
#include "../nanoparticle.h"
#include "circle.h"
#include <math.h>
#include <stdio.h>

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
			fprintf(stderr,"Error: Cannot have negative radius!");
		}

	}

	bool CircularNanoparticle::processCell(int x, int y, enum writeMethods method, DirectorElement* element)
	{
		double distance;
		double vectorAngle;	
		//Calculate distance from circle centre to point of interest (x,y)
		distance= sqrt( pow((x - mxPos),2) + pow((y - myPos),2) );
		

		if(distance <= (double) mRadius)
		{
			//we are in the circle

			switch(method)
			{
				case DRY_ADD:
					if(element->isNanoparticle == 1)
					{
						//overlapping nanoparticle
						fprintf(stderr,"Error: Adding nanoparticle would result in overlap with another at (%d,%d)",x,y);
						return false;
					}

					return true;
				break;

				case ADD:
					{
						if(element->isNanoparticle == 1)
						{
							//overlapping nanoparticle
							fprintf(stderr,"Error: Nanoparticle would overlap itself at (%d,%d)",x,y);
							return false;
						}	
						
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
					fprintf(stderr,"Error: Unknown write method!");
					return false;
			}

		}
		else
		{
			//This lattice point is outside the circle, do nothing
			return true;
		}
		
	}


