#include <iostream>
#include <string>
#include "nanoparticle.h"
#include "nanoparticles/ellipse.h"
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
	EllipticalNanoparticle ellipse(constructorArgument);

	//print out description of Circular nanoparticle
	cout << ellipse.getDescription() << endl;

	cout << "saveState() = " << ellipse.saveState() << endl;

	return TH_SUCCESS;

}
