/* Nanoparticle header file by Dan Liew & Alex Allen*/

#ifndef NANOPARTICLE
	
	#include "common.h"
	#include "directorelement.h"
	#include <string>
	#include <cstring>
	#include <sstream>
	#include <iostream>
	#include <cstdlib>

	
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
				CIRCULAR,
				ELLIPTICAL
			};
			const enum types TYPE; //identifier for saving and loading state

			//Constructor for initially setting mxPos,myPos
			Nanoparticle(int xPos, int yPos, enum types theType) : mxPos(xPos), myPos(yPos), TYPE(theType)
			{
				badState=false;
			}

			/* Method designed to be used by the Lattice Class.
			*
			* This method should specify what to do with a DirectorElement at (x,y)
			* based on the given writeMethod method.
			*/
			virtual bool processCell(int x, int y, enum writeMethods method, DirectorElement* element) =0;
			
			/* Method designed to be used by Lattice class.
			*  This method should save a (non-human readable) description of the Nanoparticle to a C++ string
			*  which can be later be used by the class's constructor.
			*
			*  Derivitive classes MUST implement this method!
			*  The first part of the C++ string must be
			*  <xPos> <yPos>
			*  
			*  so it is compatible with parent class. Note saveState() should not contain any '\n'
			*/
			virtual std::string saveState() =0;

			//simple accessor methods
			int getX() { return mxPos;}
			int getY() { return myPos;}
			
			//Returns a string object giving a human readable description of the Nanoparticle 
			virtual std::string getDescription() =0;

			bool inBadState() {return badState;}
	};




	#define NANOPARTICLE

#endif
