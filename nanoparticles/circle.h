/* Circular Nanoparticle class definition
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
