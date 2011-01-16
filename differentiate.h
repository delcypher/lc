/* Header file of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef DIFFERENTIATE
	
	#include "lattice.h"
	
	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdx_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdx_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNxdy_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydx_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	double dNydy_B(const LatticeObject* l, int xPos, int yPos);
	#define DIFFERENTIATE 1
#endif
