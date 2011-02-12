#ifndef ELLIPTICAL_NANOPARTCILE

	#include "../nanoparticle.h"
	
	class EllipticalNanoparticle : public Nanoparticle
	{
		protected:
			double a ,b ,theta;
		public:
			enum boundary
			{
				PARALLEL, //to surface
				PERPENDICULAR //to surface
			} mBoundary;

			EllipticalNanoparticle(int xCentre, int yCentre, double aValue, double bValue, double thetaValue, enum boundary boundaryType);
			bool processCell(int x, int y, enum writeMethods method, DirectorElement* element);
	};

		#define ELLIPTICAL_NANOPARTICLE

#endif