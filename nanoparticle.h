/* Nanoparticle header file by Dan Liew & Alex Allen*/

#ifndef NANOPARTICLE
	
	#include "common.h"
	#include "directorelement.h"
	#include <string>
	#include <cstring>
	#include <sstream>
	#include <iostream>

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

			//Constructor for initially setting mxPos,myPos
			Nanoparticle(int xPos, int yPos) : mxPos(xPos), myPos(yPos) 
			{
				badState=false;
			}

			/* This constructor is designed to be used by the Lattice class
			*  so that nanoparticles can be recreated.
			* 
			*  It expects a string with space seperate values where the contents are
			*  <xPos> <yPos> <other stuff>
			*
			* <other stuff> is ignored and should be parameters needed by derivitive classes.
			*/
			Nanoparticle(const std::string & state)
			{
				/*doing it this way only works because >> operator stops at a space, any other
				* delimeter will NOT work.
				*/

				std::string buffer;
				std::stringstream stream(state);
				
				//get xPos value 
				if(stream.eof())
				{
					std::cerr << "Error: Can't construct Nanoparticle. Can't get xPos" << std::endl;
					badState=true;
				}
				stream >> buffer;
				mxPos = atoi(buffer.c_str());
				
				//get yPos
				if(stream.eof())
				{
					std::cerr << "Error: Can't construct Nanoparticle. Can't get yPos" << std::endl;
					badState=true;
				}

				stream >> buffer;
				myPos = atoi(buffer.c_str());
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

			bool isBadState() {return badState;}
	};

	#define NANOPARTICLE

#endif
