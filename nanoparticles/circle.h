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
			CircularNanoparticle(const std::string & state);

			bool processCell(int x, int y, enum writeMethods method, DirectorElement* element);

			std::string saveState();

			std::string getDescription();
	};

		#define CIRCULAR_NANOPARTICLE

#endif
