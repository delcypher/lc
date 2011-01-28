/* Header file of differentiate functions
*  By Alex Allen & Daniel Liew (2010)
*/

#ifndef DIFFERENTIATE_CUDA
	
	#include "lattice.h"
	
	//Calculate partial derivative of Nx w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdx_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdx_B(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdy_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Nx w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNxdy_B(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydx_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to x using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydx_B(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using forward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydy_F(LatticeObject* l, int xPos, int yPos);

	//Calculate partial derivative of Ny w.r.t to y using backward differencing at point (xPos,yPos) on LatticeObject l
	__device__ double dev_dNydy_B(LatticeObject* l, int xPos, int yPos);

	#define DIFFERENTIATE_CUDA 1
#endif
