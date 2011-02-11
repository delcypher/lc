#include "../nanoparticle.h"
#include "ellipse.h"
#include <cmath>
#include <cstdio>

	EllipticalNanoparticle::EllipticalNanoparticle(int xCentre, int yCentre, double aValue, double bValue, double thetaValue, enum boundary boundaryType) : Nanoparticle(xCentre, yCentre)
	{
		mBoundary=boundaryType;
		if (aValue > 0 || bValue > 0)
		{
			a = aValue;
			b = bValue;
			theta = thetaValue;
		}
		else
		{
			fprintf(stderr,"Error: a and b must be greater than 0!");
		}

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

		r = a*b / sqrt(a*a*cos(angle)*cos(angle) + b*b*sin(angle)*sin(angle) );
		

		if(distance <= r)
		{
			//we are in the ellipse

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


