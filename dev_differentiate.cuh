/* Header file of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef DIFFERENTIATE
	
	#include "lattice.h"
	
	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNxdx_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNxdx_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNxdy_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNxdy_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNydx_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNydx_B(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNydy_F(const LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ dobule dNydy_B(const LatticeObject* l, int xPos, int yPos);
	#define DIFFERENTIATE 1
#endif
