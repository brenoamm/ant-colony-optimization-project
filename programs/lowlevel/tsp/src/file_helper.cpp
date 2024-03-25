#include "../include/file_helper.h"

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string>

void readFile(double* cities, const std::string& problem){
    std::ifstream data;
    data.open(
            "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/" + problem + ".txt",
            std::ifstream::in);

	int lines = 0;
	 if (data.is_open()) {

		while (!data.eof()) {

		int index;
		float x;
		float y;

        data >> index;
        data >> x;
        data >> y;

		cities[lines*3] = index-1;
		cities[lines*3+1] = x;
		cities[lines*3+2] = y;

		lines++;
	 }
	}
	 else{
		 printf("\n File not opened");
	 }
    data.close();
}

void write_results(int n_cities, int* best_path, double best_length, double time_elapsed){

}
