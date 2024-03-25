
#include <string>

double init (int nAnts, const std::string& problem, int iterations);
	
	double optimize (int ITERATIONS);

	void route (int antk);

	void route2 (int antk);

	void updatePHEROMONES ();

	//Probability Calculation

	double PHI (int cityi, int cityj, int antk);

	//City selection

	int city ();

	//Aux functions:

	void connect_cities (int cityi, int cityj);

	void setCITYPOSITION (int city, double x, double y);
	
	double distance (int cityi, int cityj);
	
	bool exists (int cityi, int cityc);
	
	bool vizited (int antk, int c);

	double length (int antk);
	
	int valid (int antk, int iteration);
	
	//Print functions

	void printPOSITIONS();

	void printPHEROMONES ();

	void printGRAPH ();

	void printRESULTS ();
