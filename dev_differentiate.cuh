/* Header file of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef DIFFERENTIATE_CUDA
	
	#include "lattice.h"
	
	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdx_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdx_B(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdy_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNxdy_B(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydx_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydx_B(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydy_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dNydy_B(LatticeObject* l, int xPos, int yPos);
	#define DIFFERENTIATE_CUDA 1
#endif
