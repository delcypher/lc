/* common.h contains things needed by multiple files

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

#ifndef COMMON_2D

	//PI to 58d.p taken from Wolfram alpha ( http://www.wolframalpha.com/input/?i=pi )
	const double PI=3.1415926535897932384626433832795028841971693993751058209749;

      	/* This function returns the correct modulo for dealing with negative a. Note % does not!
         *
         * mod(a,b) = a mod b
        */
        inline int mod(int a, int b)
        {
                return (a%b + b)%b;
        }
	
	//Note the iostream dependency comes from need for the ios_base::fmtflags type
	#include <iostream>
	
	/* Set the output format:
	*  0 - significant figures
	*  std::ios::fixed - Fixed decimal places
	*  std::ios::scientific - Scientific notation
	*/
	const std::ios_base::fmtflags STREAM_FLOAT_FORMAT = std::ios::fixed;
	
	const int FILE_PRECISION=20; //Precision (d.p) used for saving state to files.
	const int STDOE_PRECISION=10; //Precision (d.p) used for outputing to std::cout and std::cerr
	#define COMMON_2D 1

#endif
