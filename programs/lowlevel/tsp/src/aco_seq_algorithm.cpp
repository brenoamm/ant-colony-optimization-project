#include "Randoms.cpp"
#include "../include/aco_seq_algorithm.h"
#include "file_helper.cpp"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <climits>
#include <ctime>
#include <chrono>

using namespace std;

Randoms *randoms;

#define ALPHA 1
#define BETA 2
#define Q 11340
#define RO 0.5
#define TAUMAX 2
#define INITIAL_CITY 0
#define N_ROULETTE 32

int n_ants, n_cities, initial_city;

double best_length;
int *best_route;

int *connections, *routes, *roulette_cities;
double *cities, *pheromones, *delta_pheromones, *step_probabilities, *cities_distance;

double init(int nAnts, const std::string& problem, int iterations) {

	int ncities = 0;
    if (problem == "djibouti") {
        n_cities = 38;
    } else if (problem == "luxembourg") {
        n_cities = 980;
    } else if (problem == "catar") {
        n_cities = 194;
    } else if (problem == "a280") {
        n_cities = 280;
    } else if (problem == "d198") {
        n_cities = 198;
    } else if (problem == "d1291") {
        n_cities = 1291;
    } else if (problem == "lin318") {
        n_cities = 318;
    } else if (problem == "pcb442") {
        n_cities = 442;
    } else if (problem == "pcb1173") {
        n_cities = 1173;
    } else if (problem == "pr1002") {
        n_cities = 1002;
    } else if (problem == "pr2392") {
        n_cities = 2392;
    } else if (problem == "rat783") {
        n_cities = 783;
    } else {
        std::cout << "No valid import file provided. Please provide a valid import file." << std::endl;
        exit(-1);
    }

	n_ants = nAnts;

	randoms = new Randoms(15);

	//Create Structures
	roulette_cities = new int[n_cities * 32];
	connections = new int[n_cities * n_cities];
	cities = new double[n_cities * 3];
	pheromones = new double[n_cities * n_cities];
	delta_pheromones = new double[n_cities * n_cities];
	step_probabilities = new double[(n_cities - 1) * 2];
	cities_distance = new double[n_cities * n_cities];

	//Connect graphs and random pheromones
	for (int i = 0; i < n_cities; i++) {
		for (int j = 0; j < 3; j++) {
			cities[i * 3 + j] = -1.0;
		}
		for (int j = 0; j < n_cities; j++) {
			connections[i * n_cities + j] = 0;
			pheromones[i * n_cities + j] = 0.0;
			delta_pheromones[i * n_cities + j] = 0.0;
			cities_distance[i * n_cities + j] = 0;
		}
	}

	for (int i = 0; i < (n_cities - 1); i++) {
		for (int j = 0; j < 2; j++) {
			step_probabilities[i * 2 + j] = -1.0;
		}
	}

	routes = new int[n_ants * n_cities];

	for (int i = 0; i < n_ants; i++) {
		for (int j = 0; j < n_cities; j++) {
			routes[i * n_cities + j] = -1;
		}
	}

	best_length = (double) INT_MAX;
	best_route = new int[n_cities];
	for (int i = 0; i < n_cities; i++) {
		best_route[i] = -1;
	}

	readFile(cities, problem);

	for (int i = 0; i < n_cities; i++) {
		for (int f = 0; f < n_cities; f++) {
			if (i != f) {
				connect_cities(i, f);
			}
		}
	}

	for (int c_index = 0; c_index < n_cities; c_index++) {

		for (int i = 0; i < N_ROULETTE; i++) {

			double distance = 999999.9;
			double c_dist = 0.0;
			int city = -1;

			for (int j = 0; j < n_cities; j++) {

				bool check = true;

				for (int k = 0; k < i; k++) {
					if (roulette_cities[c_index * N_ROULETTE + k] == j) {
						check = false;
					}
				}

				if (c_index != j && check) {
					c_dist = cities_distance[(c_index * n_cities) + j];
					if (c_dist < distance) {
						distance = c_dist;
						city = j;
					}
				}
			}
			roulette_cities[c_index * N_ROULETTE + i] = city;
		}
	}
    printf("%s;", problem.c_str());
	double dist = optimize(iterations);
	return dist;
}

void connect_cities(int cityi, int cityj) {

	double randn = randoms->Uniforme() * TAUMAX;
	connections[cityi * n_cities + cityj] = 1;
	pheromones[cityi * n_cities + cityj] = randn;

	connections[cityj * n_cities + cityi] = 1;
	pheromones[cityj * n_cities + cityi] = randn;

	double dist = distance(cityi, cityj);
	cities_distance[cityi * n_cities + cityj] = dist;
	cities_distance[cityj * n_cities + cityi] = dist;
}

void setCITYPOSITION(int city, double x, double y) {
	cities[city * 3 + 1] = x;
	cities[city * 3 + 2] = y;
}


double distance(int cityi, int cityj) {

	double dist = sqrt(pow(cities[cityi * 3 + 1] - cities[cityj * 3 + 1], 2)+ pow(cities[cityi * 3 + 2] - cities[cityj * 3 + 2], 2));
	return dist;
}

bool exists(int cityi, int cityc) {
	return (connections[cityi * n_cities + cityc] == 1);
}

bool vizited(int antk, int c) {
	for (int l = 0; l < n_cities; l++) {
		if (routes[antk * n_cities + l] == -1) {
			break;
		}
		if (routes[antk * n_cities + l] == c) {
			return true;
		}
	}
	return false;
}


double PHI(int cityi, int cityj, int antk) {

	auto ETAij = (double) pow(1 / distance(cityi, cityj), BETA);
	auto TAUij = (double) pow(pheromones[(cityi * n_cities) + cityj],	ALPHA);

	double sum = 0.0;
	for (int c = 0; c < n_cities; c++) {
		if (exists(cityi, c) && (cityi != c)) {
			if (!vizited(antk, c)) {
				auto ETA = (double) pow(1 / distance(cityi, c), BETA);
				auto TAU = (double) pow(
						pheromones[(cityi * n_cities) + c], ALPHA);
				sum += ETA * TAU;
			}
		}
	}
	return (ETAij * TAUij) / sum;
}

double length(int antk) {
	double sum = 0.0;
	for (int j = 0; j < n_cities - 1; j++) {
		sum += distance(routes[antk * n_cities + j],
				routes[antk * n_cities + j + 1]);
	}

	sum += distance(routes[antk * n_cities], routes[antk * n_cities + n_cities -1]);
	return sum;
}

//Choose next city using the probabilistic method
int city() {
	double xi = randoms->Uniforme();

	int i = 0;
	double sum = step_probabilities[i * 2];
	while (sum < xi) {
		i++;
		sum += step_probabilities[i * 2];
	}

	return (int) step_probabilities[i * 2 + 1];
}

//Route method that considers all nodes as a possible future steps
void route(int antk) {

	routes[antk * n_cities] = initial_city;

	for (int i = 0; i < n_cities; i++) {

		int cityi = routes[antk * n_cities + i];
		int count = 0;

		for (int c = 0; c < n_cities; c++) {
			if (cityi != c) {
				if (exists(cityi, c)) {
					if (!vizited(antk, c)) {

						step_probabilities[count * 2] = PHI(cityi, c, antk);
						step_probabilities[count * 2 + 1] = (double) c;

						count++;
					}
				}
			}
		}

		// deadlock --- it reaches a place where there are no further connections
		if (0 == count) {
			return;
		}

		int nextCity = city();
		routes[(antk * n_cities) + (i + 1)] = nextCity;
	}
}

//Route method using the roulette selection
void route2(int antk) {

	double* ETA = new double[n_ants * N_ROULETTE];
	double* TAU = new double[n_ants * N_ROULETTE];
	double probability = 0.0;

	routes[antk * n_cities] = initial_city;

	//Complete tour loop
	for (int i = 0; i < n_cities; i++) {
		double sum = 0.0;

		int cityi = routes[antk * n_cities + i];
		int count = 0;

		//Iroulette tour: calculate probabilities to the 32 closest cities.
		for (int a = 0; a < N_ROULETTE; a++) {

			int c = roulette_cities[cityi * N_ROULETTE + a];

			if (cityi != c) {
				if (exists(cityi, c)) {
					if (!vizited(antk, c)) {

						ETA[antk * N_ROULETTE + a] = (double) pow(1 / distance(cityi, c), BETA);
						TAU[antk * N_ROULETTE + a] = (double) pow(pheromones[(cityi * n_cities) + c], ALPHA);

						sum += ETA[antk * N_ROULETTE + a] * TAU[antk * N_ROULETTE + a];
					}
					else{
						ETA[antk * N_ROULETTE + a] = 0;
						TAU[antk * N_ROULETTE + a] = 0;
					}
				}
				else{
					ETA[antk * N_ROULETTE + a] = 0;
					TAU[antk * N_ROULETTE + a] = 0;
				}
			}
			else{
				ETA[antk * N_ROULETTE + a] = 0;
				TAU[antk * N_ROULETTE + a] = 0;
			}
		}



		for (int a = 0; a < N_ROULETTE; a++) {

			probability = (ETA[antk * N_ROULETTE + a] * TAU[antk * N_ROULETTE + a]) / sum;

			step_probabilities[count * 2] = probability;
			step_probabilities[count * 2 + 1] = roulette_cities[cityi * N_ROULETTE + a];

			if(probability > 0.0){
				count++;
			}

			probability = 0.0;
		}

		int nextCity = 0;

		//If all cities in the roulette vector have been choosen, chose the first one available
		if (0 == count) {
			for(int nc = 0; nc < n_cities; nc++){
				if(!vizited(antk, nc)){
					nextCity = nc;
					break;
				}
			}
		}else{
			//else chose using the probabilities
			nextCity = city();
		}

		//set next city
		routes[(antk * n_cities) + (i + 1)] = nextCity;

		//reset probabilities vector
		for (int a = 0; a < N_ROULETTE; a++) {
			step_probabilities[a * 2] = 0;
			step_probabilities[a * 2 + 1] = 0;
		}
	}
}

//Method indicates if a trip is valid or not (incomplete, repeated nodes or using unexistant connections).
int valid(int antk, int iteration) {
	for (int i = 0; i < n_cities - 1; i++) {

		int cityi = routes[antk * n_cities + i];
		int cityj = routes[antk * n_cities + i + 1];

		//incomplete trip
		if (cityi < 0 || cityj < 0) {
			printf("1");
			return -1;
		}

		//impossible trip
		if (!exists(cityi, cityj)) {
			printf("\n\n 2 - Trip does not exist %i %i ", cityi, cityj);
			return -2;
		}

		//repeated city
		for (int j = 0; j < i - 1; j++) {
			if (routes[antk * n_cities + i] == routes[antk * n_cities + j]) {
				printf("\n 3 - invalid trip");
				return -3;
			}
		}
	}

	if (!exists(initial_city,routes[antk * n_cities + n_cities - 1])) {
		printf("\n 4");
		return -4;
	}

	return 0;
}

//Update pheromone method. For every ant, for every visited connection:
void updatepheromones() {

	for (int k = 0; k < n_ants; k++) {
		double rlength = length(k);

		for (int r = 0; r < n_cities - 1; r++) {

			int cityi = routes[k * n_cities + r];
			int cityj = routes[k * n_cities + r + 1];

			delta_pheromones[cityi * n_cities + cityj] += Q / rlength;
			delta_pheromones[cityj * n_cities + cityi] += Q / rlength;
		}
	}

	for (int i = 0; i < n_cities; i++) {
		for (int j = 0; j < n_cities; j++) {
			pheromones[i * n_cities + j] = (1 - RO)
					* pheromones[i * n_cities + j]
					+ delta_pheromones[i * n_cities + j];
			delta_pheromones[i * n_cities + j] = 0.0;

			pheromones[j * n_cities + i] = (1 - RO)
					* pheromones[j * n_cities + i]
					+ delta_pheromones[j * n_cities + i];
			delta_pheromones[j * n_cities + i] = 0.0;
		}
	}
}

//Runs the tour construction and update pheromene for n iterations.
double optimize(int it_n) {


	double calc_times = 0.0;
	double rlength = 0.0;

	//Iterations Loop
	for (int it = 0; it < it_n; it++) {

		auto t_start = std::chrono::high_resolution_clock::now();

		//Ant Loop
		for (int k = 0; k < n_ants; k++) {

			//Create new tour for ant k and calculate the lenght
			route2(k);
			rlength = length(k);

			//Update best solution
			if (rlength < best_length) {
				best_length = rlength;
				for (int i = 0; i < n_cities; i++) {
					best_route[i] = routes[k * n_cities + i];
				}
			}
		}

		//Time measure for tour Construction
		auto t_end = std::chrono::high_resolution_clock::now();
        calc_times +=  std::chrono::duration<double>(t_end-t_start).count();

		//Update Pheromones
		updatepheromones();

		//Reset Routes
		for (int i = 0; i < n_ants; i++) {
			for (int j = 0; j < n_cities; j++) {
				routes[i * n_cities + j] = -1;
			}
		}
	}

    printf("%d;%s;%f;", n_ants, "seq", calc_times);
	return best_length;
}