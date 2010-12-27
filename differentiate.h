/* Header file of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef DIFFERENTIATE
	
	#include "lattice.h"
	
	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	float dNxdx_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	float dNxdx_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	float dNxdy_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	float dNxdy_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	float dNydx_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	float dNydx_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	float dNydy_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	float dNydy_B(const LatticeObject* l, int xPos, int yPos);
	#define DIFFERENTIATE 1
#endif
