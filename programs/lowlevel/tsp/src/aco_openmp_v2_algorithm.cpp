#include "Randoms.cpp"
#include <omp.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include "file_helper.cpp"

#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2
#define Q 11340

Randoms *randoms;

using namespace std;

int n_cities = 0;
int n_ants = 0;

// ler o arquivo txt e inicia o pheromonio
void readMap(double* coord, double* phero, const std::string& problem){

	readFile(coord, problem);
	double randn = randoms -> Uniforme() * TAUMAX;
    for(int j = 0;j<n_cities;j++){
        for(int k = 0;k<n_cities;k++){
            if(j!=k){
                phero[(j*n_cities) + k] = randn;
            }
            else{
                phero[(j*n_cities) + k] = 0;
            }
        }
    }
}

void calcDist(double* dist, double* coord){
    for(int i = 0;i<n_cities;i++){
        for(int j = 0;j<n_cities;j++){
            if(i!=j){
            	double distancia = sqrt(pow(coord[j*3+1] - coord[i*3+1],2) + pow(coord[(j*3+2)] - coord[(i*3+2)],2));
                dist[(i*n_cities) + j] = distancia;
                dist[(j*n_cities) + i] = distancia;
            }else{
                dist[(i*n_cities) + j] = 0.0;
            }
        }
    }
}

bool vizited (const int* route, int cityc, int count) {

	for (int l=0; l<count; l++) {
		if (route[l] == cityc) {
			return true;
		}
	}
	return false;
}

int lock = 0;

// cria a rota
void route(double* phero, const double* dist, int sequence[]){

    int count = 0;
    double selection_prob[n_cities];
    double r = 0.0;
    double p = 0.0;
    int l = 0;

    sequence[count] = 0;
    double sumPh = 0.0;
    while(count < n_cities-1){

    	sumPh = 0.0;

    	#pragma omp parallel num_threads(4) default(none) shared(sequence, count, selection_prob, sumPh, dist, phero, n_cities, lock) private(r, p, l)
    	{

			#pragma omp parallel for reduction(+:sumPh) default(none) shared(sequence, count, selection_prob, dist, phero, n_cities, lock)
			for(int i=0;i<n_cities;i++){
				if(i != sequence[count] && !vizited(sequence, i, count)){
					double ETA = pow(1/(dist[(sequence[count]*n_cities) +i]),BETA);
					double TAU = pow(phero[(sequence[count]*n_cities)+i],ALPHA);
					sumPh += ETA * TAU;
				}
			}

			#pragma omp for
			for(int i=0;i<n_cities;i++){
//				if(i==0 && lock ==0)printf("\n level 2 - %i threads", omp_get_num_threads());
				if(i != sequence[count] && !vizited(sequence, i, count)){
					double ETA = pow(1/(dist[(sequence[count]*n_cities) +i]),BETA);
					double TAU = pow(phero[(sequence[count]*n_cities)+i],ALPHA);
					selection_prob[i] = (ETA * TAU)/sumPh;
				}
				else{
					selection_prob[i] = 0;
				}
			}
    	}

    	lock++;

        r = ((double)rand()/(RAND_MAX));
        l = 0;
        p = selection_prob[l];

        while(p<r){
        	l++;
            p+=selection_prob[l];
        }

        count++;
        sequence[count] = l;
    }
}

void pheroEvap(double* phero){
    for(int i = 0;i< n_cities;i++){
        for(int j=0;j<n_cities;j++){
            phero[(i* n_cities) + j] = (1-EVAPORATION)*phero[(i*n_cities)+j];
        }
    }
}

double pheroDeposit(const int* sequence, const double* dist, double* phero){

    double totalDist = 0;
    for(int i = 0;i<n_cities-1;i++){
        totalDist+= dist[(sequence[i]*n_cities) + sequence[i+1]];
    }
    totalDist+=dist[(sequence[n_cities-1]*n_cities) + sequence[0]];

    for(int i = 0;i<n_cities-1;i++){

        phero[(sequence[i]*n_cities) + sequence[i+1]] += double(Q/totalDist);
        phero[(sequence[i+1]*n_cities) + sequence[i]] = phero[(sequence[i]*n_cities) + sequence[i+1]];
    }
    phero[(sequence[n_cities-1]*n_cities)+sequence[0]] += double(Q/totalDist);
    phero[(sequence[0]*n_cities)+ sequence[n_cities-1]] = phero[(sequence[n_cities-1]*n_cities) + sequence[0]];

    return totalDist;
}

double run(const std::string&  problem, int nants, int iterations)
{
	randoms = new Randoms(15);


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

	n_ants =  nants;

	omp_set_nested(1);
    omp_set_num_threads(4);
	omp_set_dynamic(0);

    srand(time(NULL));

	//int bestSequence[NCITY];
	double bestRoute = 9999999999999999.9;
	int inter = iterations;
	double aux;

	double diste[n_ants];
	double* coord;
	double* phero;
	double* dist;

	int sequence[n_ants][n_cities];
	int bestSequence[n_cities];

	coord = (double *)malloc((n_cities*3) * sizeof(double));
	phero = (double *)malloc((n_cities*n_cities) * sizeof(double));
	dist = (double *)malloc((n_cities*n_cities) * sizeof(double));


	readMap(coord, phero, problem); // read txt with the citys position

	calcDist(dist, coord); // calculates the distances of each city

	while(inter>0){

//		#pragma omp parallel num_threads(2)
//		{
//			#pragma omp for
			for(int i = 0;i<n_ants;i++){
				route(phero, dist, sequence[i]);
			}
//		}

		pheroEvap(phero);

		for(int i = 0;i<n_ants;i++){
			diste[i] = pheroDeposit(sequence[i],dist,phero);
		}
		for (int i = 0;i<n_ants;i++) {
			if(bestRoute>diste[i]){
				bestRoute = diste[i];
				for(int j = 0;j<n_cities;j++){
					bestSequence[j] = sequence[i][j];
				}
			}
		}
		inter -=1;
	}

	return bestRoute;
}
