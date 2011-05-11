/* EllipticalNanoparticle class definition
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
