/* Nanoparticle header file

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

#ifndef NANOPARTICLE
	
	#include "directorelement.h"
	#include <string>
	#include <cstring>
	#include <sstream>
	#include <iostream>
	#include <cstdlib>
	#include <fstream>
	#include "common.h"
	
	class Nanoparticle
	{
		protected:
			/* This is the location of the nanoparticle 
			*  in the lattice. The part of the nanoparticle (mxPos,myPos) describe
			*  is arbitary and it is up to the implementation to decide what to do.
			*/
			int mxPos,myPos;
			bool badState; //variable to indicate if Nanoparticle is in bad state (e.g. constructor doesn't work properly)

		public:
			/* writeMethods contains the possible methods of writing to 
			*  a lattice.
			*/
			enum writeMethods
			{
				/* DRY_ADD is a method where all the calculations are done to add
				*  a nanoparticle to the lattice but we don't actually do it. This allows
				*  the situation where the nanoparticle overlaps another to be caught.
				*/
				DRY_ADD,
				/* ADD is a method where all the calculations are done to add 
				*  a nanoparticle to the lattice and we ACTUALLY do it!
				*/
				ADD
			};
			
			enum types
			{
				CIRCLE,
				ELLIPSE
			};

			const enum types ID;

			//Constructor for initially setting mxPos,myPos
			Nanoparticle(int xPos, int yPos, enum Nanoparticle::types theType) : mxPos(xPos), myPos(yPos), ID(theType)
			{
				badState=false;
			}

			//Constructor for reconstructing Nanoparticle from binary data
			Nanoparticle(std::ifstream & stream, enum Nanoparticle::types theType) : ID(theType)
			{
				if(stream.good())
				{
					stream.read( (char*) &mxPos, sizeof(int));
					stream.read( (char*) &myPos, sizeof(int));
					badState = !(stream.good());
				}
				else
				{
					badState=true;
				}

			}

			/* Method designed to be used by the Lattice Class.
			*
			* This method should specify what to do with a DirectorElement at (x,y)
			* based on the given writeMethod method.
			*/
			virtual bool processCell(int x, int y, enum writeMethods method, DirectorElement* element) =0;
			
			//simple accessor methods
			int getX() { return mxPos;}
			int getY() { return myPos;}
			
			//Returns a string object giving a human readable description of the Nanoparticle 
			virtual std::string getDescription() =0;

			//Save the state of the object to a binary file. Derivitive classes should call this first in their saveState()
			virtual bool saveState(std::ofstream & stream)
			{
				if(stream.good())
				{
					stream.write( (char*) &mxPos, sizeof(int));
					stream.write( (char*) &myPos, sizeof(int));
				 	return stream.good();
				}
				else
				{
					std::cerr << "Error: Nanoparticle::saveState() failed. Stream not good!\n";
					return false;
				}	

			}

			bool inBadState() {return badState;}

	};




	#define NANOPARTICLE

#endif
