/* ./uniform-rnd <no_to_generate> <no_of_bins>
*
*/
#include <iostream>
#include <cstdlib>
#include "exitcodes.h"
#include "randgen.h"
#include <cmath>

using namespace std;

int* bin;

void cleanup();

int main(int n, char* argv[])
{
	int noToGenerate, noOfBins, binNo;
	double randomNumber, binWidth;
	
	if(n!=3)
	{
		cerr << "Usage: ./" << argv[0] << " <no_to_generate> <no_of_bins>" << endl <<
		"  <no_to_generate> - How many random numbers to generate" << endl <<
		"  <no_of_bins> - How many bins to use for binning data" << endl;
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

	//allocate memory for bins (and set all values to zero)
	bin = (int*) calloc(noOfBins,sizeof(int));

	if(bin==NULL)
	{
		cerr << "Error: Couldn't allocate memory for " << noOfBins << " bins." << endl;
		exit(TH_BAD_ARGUMENT);
	}
	
	//set the random seed
	setSeed();

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
		randomNumber = rnd();
		//randomNumber = (double) rand() / RAND_MAX ;
		
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
