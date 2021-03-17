#include "../include/file_helper.h"

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string>

void readFile(double* cities, int problemID){
	std::ifstream lerMapa;

//	std::string dji = "/home/b/b_mene01/tsp/djibouti.txt";
//	std::string lux = "/home/b/b_mene01/tsp/luxembourg.txt";
//	std::string cat = "/home/b/b_mene01/tsp/catar.txt";
//	std::string a280 = "/home/b/b_mene01/tsp/a280.txt";
//	std::string d198 = "/home/b/b_mene01/tsp/d198.txt";
//	std::string d1291 = "/home/b/b_mene01/tsp/d1291.txt";
//	std::string lin318 = "/home/b/b_mene01/tsp/lin318.txt";
//	std::string pcb442 = "/home/b/b_mene01/tsp/pcb442.txt";
//	std::string pcb1173 = "/home/b/b_mene01/tsp/pcb1173.txt";
//	std::string pr1002 = "/home/b/b_mene01/tsp/pr1002.txt";
//	std::string pr2392 = "/home/b/b_mene01/tsp/pr2392.txt";
//	std::string rat783 = "/home/b/b_mene01/tsp/rat783.txt";

	std::string dji = "djibouti.txt";
	std::string lux = "luxembourg.txt";
	std::string cat = "catar.txt";
	std::string a280 = "a280.txt";
	std::string d198 = "d198.txt";
	std::string d1291 = "d1291.txt";
	std::string lin318 = "lin318.txt";
	std::string pcb442 = "pcb442.txt";
	std::string pcb1173 = "pbc1173.txt";
	std::string pr1002 = "pr1002.txt";
	std::string pr2392 = "pr2392.txt";
	std::string rat783 = "rat783.txt";

	switch (problemID) {
		case 1:
			lerMapa.open(dji, std::ifstream::in);
			break;
		case 2:
			lerMapa.open(lux, std::ifstream::in);
			break;
		case 3:
			lerMapa.open(cat, std::ifstream::in);
			break;
		case 4:
			lerMapa.open(a280, std::ifstream::in);
			break;
		case 5:
			lerMapa.open(d198, std::ifstream::in);
			break;
		case 6:
			lerMapa.open(d1291, std::ifstream::in);
			break;
		case 7:
			lerMapa.open(lin318, std::ifstream::in);
			break;
		case 8:
			lerMapa.open(pcb442, std::ifstream::in);
			break;
		case 9:
			lerMapa.open(pcb1173, std::ifstream::in);
			break;
		case 10:
			lerMapa.open(pr1002, std::ifstream::in);
			break;
		case 11:
			lerMapa.open(pr2392, std::ifstream::in);
			break;
		case 12:
			lerMapa.open(rat783, std::ifstream::in);
			break;
	}

	int lines = 0;
	 if (lerMapa.is_open()) {

		while (!lerMapa.eof()) {

		int index;
		float x;
		float y;

		lerMapa >> index;
		lerMapa >> x;
		lerMapa >> y;

		cities[lines*3] = index-1;
		cities[lines*3+1] = x;
		cities[lines*3+2] = y;

		lines++;
	 }
	}
	 else{
		 printf("\n File not opened");
	 }
	lerMapa.close();
}

void write_results(int n_cities, int* best_path, double best_length, double time_elapsed){

}
