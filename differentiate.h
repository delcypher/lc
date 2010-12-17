/* Header file of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef DIFFERENTIATE
	
	#include "lattice.h"

	enum directorComponent
	{
		N_X,
		N_Y,
	};

	enum differencingSchemes
	{
		FORWARD_DIFF,
		BACKWARD_DIFF,
		CENTRAL_DIFF
	};

	/*The differencing scheme to be used by dndx() & dndy()
	* We will make this unavailable to change for now
	*
	extern enum differencingSchemes diffScheme;
	*/

	/* Calculate partial derivative of the "dirComp" component of the director of lattice "l"
	*  w.r.t to x then evaluated at point (xPos,yPos)
	* 
	*/
	float dndx(enum directorComponent dirComp, LatticeObject* l, int xPos, int yPos);


	/* Calculate partial derivative of the "dirComp" component of the director of the lattice "l" w.r.t to y then evaluated at point (xPos,yPos)
	* 
	*/
	float dndy(enum directorComponent dirComp, LatticeObject* l, int xPos, int yPos);

	#define DIFFERENTIATE 1
#endif
