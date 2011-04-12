/* ./uniform-rnd <no_to_generate> <no_of_bins>
*
*/
#include <iostream>
#include <cstdlib>
#include "exitcodes.h"
#include "mt19937ar.h"
#include <cmath>
#include <ctime>
#include <cstring>

using namespace std;

int* bin;

void cleanup();

int main(int n, char* argv[])
{
	int noToGenerate, noOfBins, binNo;
	double randomNumber, binWidth;
	unsigned long randomSeed=time(NULL); //default seed is to use UNIX time
	
	if(n<3)
	{
		cerr << "Usage: " << argv[0] << " <no_to_generate> <no_of_bins> [options]" << endl <<
		"  <no_to_generate> - How many random numbers to generate" << endl <<
		"  <no_of_bins> - How many bins to use for binning data\n\n" <<
		"Options\n\n" <<
		"--rand-seed <seed>\n" <<
		"Use <seed> as the random number generator seed where <seed> is a positive integer.\n" << endl;
		exit(TH_BAD_ARGUMENT);
	}

	noToGenerate = atoi(argv[1]);
	noOfBins = atoi(argv[2]);

	if(noToGenerate < 1)
	{
		cerr << "Error: Can only generate 1 or more numbers" << endl;
		exit(TH_BAD_ARGUMENT);
	}

	if(noOfBins < 1)
	{
		cerr << "Error: The number of bins must be 1 or more" << endl;
		exit(TH_BAD_ARGUMENT);
	}

	//handle optional arguments
	if(n>3)
	{
		int argIndex=3;
		while( argIndex < n)
		{
			if( strcmp(argv[argIndex],"--rand-seed") == 0)
			{
				argIndex++;

				if(argIndex > (n -1) )
				{
					cerr << "Error: expected random seed <seed>" << endl;
					exit(TH_BAD_ARGUMENT);
				}

				if( atoi(argv[argIndex]) < 0 )
				{
					cerr << "Error: <seed> must be >=0" << endl;
					exit(TH_BAD_ARGUMENT);
				}

				randomSeed=atoi(argv[argIndex]);

				//continue to next argument
				argIndex++;
				continue;

			}

			//bad argument
			cerr << "Error: Argument " << argv[argIndex] << " not recognised!" << endl;
			exit(TH_BAD_ARGUMENT);

		}

	}

	//allocate memory for bins (and set all values to zero)
	bin = (int*) calloc(noOfBins,sizeof(int));

	if(bin==NULL)
	{
		cerr << "Error: Couldn't allocate memory for " << noOfBins << " bins." << endl;
		exit(TH_BAD_ARGUMENT);
	}
	
	//set the random seed (use UNIX system time)
	cerr << "Using random seed: " << randomSeed << endl; 
	init_genrand(randomSeed);

	//calculate bin width
	binWidth = (double) 1/noOfBins;
	
	//calculate how many numbers to generate to do 1% of total number to generate (noToGenerate)
	int pStep = noToGenerate/100;
	if(pStep==0)
	{
		//can't have pStep ==0 as doing % 0 is not allowed.
		pStep=1;
	}

	//loop over numbers to generate
	for(int counter=1; counter <= noToGenerate; counter++)
	{
		randomNumber = genrand_real1();
		
		//loop over bins to see which one to put random number in.
		for(binNo=0; binNo < noOfBins; binNo++)
		{
			if(randomNumber >= (binNo*binWidth) && randomNumber < ((binNo +1)*binWidth) )
			{
				bin[binNo]++;
				break;
			}
		}
		
		//output % step
		if((counter % pStep) == 0)
			cerr << ( (double) counter/noToGenerate)*100 << "%\r" ;
	}
	
	//output data & calculated value.
	cout << "#[X] [COUNT]" << endl;

	double expected = (double) noToGenerate/noOfBins;
	double chiSquared=0;
	
	for(binNo=0; binNo < noOfBins; binNo++)
	{
		cout << (binNo + 0.5)*(binWidth) << " " << bin[binNo] << endl;
		chiSquared += (bin[binNo] - expected)*(bin[binNo] - expected)/(expected); 
	}

	cout.precision(6);
	cout << "#CHI^2 value:" << chiSquared << endl;

	cleanup();
	return TH_SUCCESS;
}

void cleanup()
{
	free(bin);
}
