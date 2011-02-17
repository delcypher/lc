/* Nanoparticle header file by Dan Liew & Alex Allen*/

#ifndef NANOPARTICLE
	
	#include "common.h"
	#include "directorelement.h"
	#include <string>
	#include <sstream>

	class Nanoparticle
	{
		protected:
			/* This is the location of the nanoparticle 
			*  in the lattice. The part of the nanoparticle (mxPos,myPos) describe
			*  is arbitary and it is up to the implementation to decide what to do.
			*/
			int mxPos,myPos;

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

			Nanoparticle(int xPos, int yPos) : mxPos(xPos), myPos(yPos) 
			{
				//do nothing
			}
			virtual bool processCell(int x, int y, enum writeMethods method, DirectorElement* element) =0;
			
			//simple accessor methods
			int getX() { return mxPos;}
			int getY() { return myPos;}
			
			//Returns a string object describing the Nanoparticle 
			virtual std::string getDescription() =0;
	};

	#define NANOPARTICLE

#endif
