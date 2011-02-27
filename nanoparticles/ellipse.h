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
			
			EllipticalNanoparticle(std::ifstream & stream);

			bool processCell(int x, int y, enum writeMethods method, DirectorElement* element);

			std::string getDescription();
			
			//override parent's method.
			virtual bool saveState(std::ofstream & stream);
			

	};

		#define ELLIPTICAL_NANOPARTICLE

#endif
