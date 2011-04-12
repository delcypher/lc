/* This is a header file for The Mersenne Twister (MT19937) that was
   Coded by Takuji Nishimura and Makoto Matsumoto.
   
   See http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html

   Header file by Dan Liew
*/

#ifndef MT_RND
	//Sets the the MT19937 random number generator seed to UNIX time (we generally use this)
	void initMTSeed();

	//You can manually set the seed using this method.
	void init_genrand(unsigned long s);

	//Another way to initialise the MT19937
	void init_by_array(unsigned long init_key[], int key_length);

	/* generates a random number on [0,0xffffffff]-interval */
	unsigned long genrand_int32(void);

	/* generates a random number on [0,0x7fffffff]-interval 
	 * NOT OPTIMISED!
	*/
	long genrand_int31(void);

	/* generates a random number on [0,1]-real-interval 
	 * NOTE THIS HAS BEEN OPTIMISED BY NOT CALLING  genrand_int32
	*/
	double genrand_real1(void);

	/* generates a random number on [0,1)-real-interval 
	 * NOT OPTIMISED!
	*/
	double genrand_real2(void);

	/* generates a random number on (0,1)-real-interval
	 * NOT OPTIMISED!
	*/
	double genrand_real3(void);

	/* generates a random number on [0,1) with 53-bit resolution
	 * NOT OPTIMISED!
	 */
	double genrand_res53(void);

	#define MT_RND 1
#endif


