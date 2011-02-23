#include <iostream>
#include <string>
#include "nanoparticle.h"
#include "nanoparticles/circle.h"
#include "exitcodes.h"

using namespace std;

int main(int n, char* argv[])
{
	if(n!=2)
	{
		cerr << "Usage: " << argv[0] << " \"String constructor argument\" " << endl
			<< " Note the quotes are required!" << endl;
		return TH_BAD_ARGUMENT;

	}
	
	string constructorArgument(argv[1]);
	
	//constructor circular Nanoparticle
	CircularNanoparticle circle(constructorArgument);

	//print out description of Circular nanoparticle
	cout << circle.getDescription() << endl;

	cout << "saveState() = " << circle.saveState() << endl;

	return TH_SUCCESS;

}
