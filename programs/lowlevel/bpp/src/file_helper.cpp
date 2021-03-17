#include "../include/file_helper.h"

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string>

void readBPPFile(int problemID, double n_objects, double bin_capacity, double* items){

	printf("\n\n Reading BPP File");

	std::ifstream fileReader;

	std::string file_to_read = "";

	//Problem Instances
	std::string f60 = "Falkenauer_t60_00.txt";
	std::string p201 = "201_2500_NR_0.txt";
	std::string p402 = "402_10000_NR_0.txt";
	std::string p600 = "600_20000_NR_0.txt";
	std::string p801 = "801_40000_NR_0.txt";
	std::string p1002 = "1002_80000_NR_0.txt";

	switch(problemID){
	case 0:
		file_to_read += f60;
		break;
	default:
		break;
	}

	fileReader.open(file_to_read, std::ifstream::in);

	int lines = 0;
	if (fileReader.is_open()) {

		fileReader >> n_objects;
		fileReader >> bin_capacity;

		printf("\n %f Objects, %f Bin Capacity", n_objects, bin_capacity);

		while (!fileReader.eof()) {
			float weight;
			float quantity;

			fileReader >> weight;
			fileReader >> quantity;

			items[lines*2] = weight;
			cities[lines*2+1] = quantity;
			lines++;

			printf("\n\t %f weight, %f quantity", weight, quantity);
		}
	}
	 else{
		 printf("\n File not opened");
	 }
	fileReader.close();
}
