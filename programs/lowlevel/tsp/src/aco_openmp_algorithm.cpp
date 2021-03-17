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
void readMap(double* coord, double* phero, int problem){

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

// calcula as distancias
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

bool vizited (int* route, int cityc, int count) {

	for (int l=0; l<count; l++) {
		if (route[l] == cityc) {
//			printf("\n Visited city %i result true",  cityc);
			return true;
		}
	}
//	printf("\n Visited city %i result false",  cityc);
	return false;
}

// cria a rota
void route(double* phero, double* dist, int sequence[]){

    int count = 0;
    bool aux = true;
    double selection_prob[n_cities];
    double sum_prob = 0.0;
    double r = 0.0;
    double p = 0.0;
    int l = 0;

    //sequence[k] = (double)(rand()%n_cities);

    //Initial City
    sequence[count] = 0;

    double sumPh = 0.0;

    while(count < n_cities-1){

    	sumPh = 0.0;
    	sum_prob = 0.0;

    	for(int i=0;i<n_cities;i++){
			if(i != sequence[count] && !vizited(sequence, i, count)){
				sumPh += pow(phero[(sequence[count]*n_cities)+i],ALPHA) * pow(1/(dist[(sequence[count]*n_cities) +i]),BETA);
			}
    	}

    	for(int i=0;i<n_cities;i++){
        	if(i != sequence[count] && !vizited(sequence, i, count)){
            		selection_prob[i] = pow(phero[(sequence[count]*n_cities) + i],ALPHA) * pow(1/(dist[(sequence[count]*n_cities) + i]),BETA);
//            		printf("\n seqk %i", sequence[count]);
//            		printf("\n sel1 %f", selection_prob[i]);

            		selection_prob[i] = selection_prob[i]/sumPh;
//            		printf("\n sel2 %f", selection_prob[i]);
            }
        	else{
        		selection_prob[i] = 0;
        	}

            sum_prob += selection_prob[i];
        }



        r = ((double)rand() / (RAND_MAX));
//        printf("\n Random %f", r);

        r = r*sum_prob;
        l = 0;
        p = selection_prob[l];

//        printf("\n Probability %f", p);
//        printf("\n Random %f", r);

        while(p<r){

        	l++;

            p+=selection_prob[l];

//            printf("\n New Probability %f", p);
        }

        count++;

//        printf("\n Choice %i", l);
//        printf("\n COunt %i", count);
        sequence[count] = l;

//        printf("\n\n Sequence %i city %i", count, l);
    }
}

// evapora o pheromonio
void pheroEvap(double* phero){

    for(int i = 0;i< n_cities;i++){
        for(int j=0;j<n_cities;j++){
            phero[(i* n_cities) + j] = (1-EVAPORATION)*phero[(i*n_cities)+j];
        }
    }

}

// deposita o pheromonio
double pheroDeposit(int* sequence, double* dist, double* phero){

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

double run(int problem, int nants, int iterations, int runs, int n_threads)
{
	randoms = new Randoms(15);

//	printf("\n Problem : %i", problem);
	//pre-processing

	switch (problem) {
		case 1:
			n_cities = 38; //Djbouti
			break;
		case 2:
			n_cities = 980; //Luxemburg
			break;
		default:
			n_cities = 194; //Catar
		}

//	printf("\n N cities : %i", n_cities);

	n_ants =  nants;

    omp_set_num_threads(n_threads);

    double bestAll = 9999999999999999.9;
    double a = omp_get_wtime();
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

		#pragma omp parallel for
		for(int i = 0;i<n_ants;i++){
			route(phero, dist, sequence[i]);
		}

		pheroEvap(phero);

		for(int i = 0;i<n_ants;i++){
			diste[i] = pheroDeposit(sequence[i],dist,phero);
		}

		for(int i = 0;i<n_ants;i++){
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
