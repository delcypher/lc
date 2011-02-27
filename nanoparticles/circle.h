/* Circular Nanoparticle class by Dan Liew & Alex Allen */

#ifndef CIRCULAR_NANOPARTCILE

	#include "../nanoparticle.h"

	/* In this implementation we will assume (mxPos,myPos) refer to the centre of the 
	*  circle.
	*/
	class CircularNanoparticle : public Nanoparticle
	{
		protected:
			int mRadius;
		public:
			enum boundary
			{
				PARALLEL, //to surface
				PERPENDICULAR //to surface
			} mBoundary;

			CircularNanoparticle(int xCentre, int yCentre, int radius, enum boundary boundaryType);
				
			//Constructor for recreating the object from binary data
			CircularNanoparticle(std::ifstream & stream);

			bool processCell(int x, int y, enum writeMethods method, DirectorElement* element);

			std::string getDescription();

			//override parent class's method
			virtual bool saveState(std::ofstream & stream);

	};

		#define CIRCULAR_NANOPARTICLE

#endif
