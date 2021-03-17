#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <cstdio>
#include <ctime>

#include <malloc.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../include/acoCuda.cuh"

#include "Randoms.cpp"

#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2

#define NBLOCKS 16
#define BLOCK_SIZE 32
#define GRAPH_SIZE 38

__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_BLOCK_SIZE;
__device__ int d_GRAPH_SIZE;


std::string::size_type sz;

Randoms *randoms;

// nvcc -o a.out acoCuda.cu
// ./a.out 38 50 50 3 0

using namespace std;

void readMap(double* coord, double* phero, double* dist, int NCITY){ // small

    string posicao;
    ifstream lerMapa;
    lerMapa.open("mapa.txt");
    double randn = 0.0;

    for(int j = 0;j<NCITY;j++){
        for(int k = 0;k<NCITY;k++){
            if(j!=k){
            	randn = randoms -> Uniforme() * TAUMAX;
            	phero[(j*NCITY) + k] = randn;
            	phero[(k*NCITY) + j] = randn;
            }
            else{
                phero[(j*NCITY) + k] = 0;
                phero[(k*NCITY) + j] = 0;
            }
        }
    }

    int i = 0;
    while(!lerMapa.eof()){
        getline(lerMapa,posicao);

        string auxiliar1;
        string auxiliar2;

        for(int j = 0;j<12;j++){
            auxiliar1 += posicao[3 + j];
            auxiliar2 += posicao[16 + j];
//            printf("\n Aux1: %s", auxiliar1.c_str());
//            printf("\n Aux2: %c", auxiliar2);
        }
        coord[(i*2)] = std::stod (auxiliar1.c_str(),&sz);
        coord[(i*2) + 1] = atof(auxiliar2.c_str());
//        printf("\n Coordenadas : x %f y %f", coord[(i*2)], coord[((i*2)+1)]);
        i+=1;

    }
    lerMapa.close();
}

void readMap2(double* coord, double* phero, double* dist, int NCITY){ // large

    string posicao;
    ifstream lerMapa;
    lerMapa.open("mapa2.txt");

    for(int j = 0;j<NCITY;j++){
        for(int k = 0;k<NCITY;k++){
            if(j!=k){
                phero[(j*NCITY) + k] = PHERINIT;
            }
            else{
                phero[(j*NCITY) + k] = 0;
            }
        }
    }

    int i = 0;
    while(!lerMapa.eof()){
        getline(lerMapa,posicao);
        string auxiliar1;
        string auxiliar2;
        for(int j = 0;j<10;j++){
            auxiliar1 += posicao[4 + j];
        }
        for(int j = 0;j<9;j++){
            auxiliar2 += posicao[15 + j];
        }
        coord[(i*2)] = atof(auxiliar1.c_str());
        coord[(i*2) + 1] = atof(auxiliar2.c_str());
	//printf("%f \n",coord[(i*2)]);
	//printf("%f \n",coord[(i*2) + 1]);
	i+=1;

    }
    lerMapa.close();
}

void readMap3(double* coord, double* phero, double* dist, int NCITY){ // medium

    string posicao;
    ifstream lerMapa;
    lerMapa.open("mapa3.txt");

    for(int j = 0;j<NCITY;j++){
        for(int k = 0;k<NCITY;k++){
            if(j!=k){
                phero[(j*NCITY) + k] = PHERINIT;
            }
            else{
                phero[(j*NCITY) + k] = 0;
            }
        }
    }

    int i = 0;
    while(!lerMapa.eof()){
        getline(lerMapa,posicao);
        string auxiliar1;
        string auxiliar2;
        for(int j = 0;j<10;j++){
            auxiliar1 += posicao[4 + j];
            auxiliar2 += posicao[15 + j];
        }
        coord[(i*2)] = atof(auxiliar1.c_str());
        coord[(i*2) + 1] = atof(auxiliar2.c_str());
        i+=1;

    }
    lerMapa.close();
}

__global__ void setup_rand_kernel(curandState * state, unsigned long seed) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, id, 0, &state[id]);

//	curand_init(1234, id, 0, &state[id]);

  if(id == 0){
	  d_PHERINIT = 0.005;
	  d_EVAPORATION = 0.5;
	  d_ALPHA = 1;
	  d_BETA = 2;
	  d_TAUMAX = 2;
  }
}

__global__ void calcDist(double* dist, double* coord, int NCITY){

    int c_index = threadIdx.x;

    for(int j = 0 ;j<NCITY;j++){
         if(c_index!=j){
            dist[(c_index*NCITY) + j] = sqrt(pow(coord[j*2] - coord[c_index*2],2) + pow(coord[(j*2) + 1] - coord[(c_index*2) + 1],2));
            dist[(j*NCITY) + c_index] = dist[(c_index*NCITY) + j];
         }else{
            dist[(c_index*NCITY) + j] = 0.0;
         }
    }
}


__global__ void route(double* phero, double* dist, int* sequence, int NCITY){

    int antN = threadIdx.x;

    // initial position
    int stepIndex = 0;

    //aux variables
    bool aux = true;
    double selection_prob[38];
    double sum_prob = 0.0;
    double random = 0.0;
    double prob = 0.0;

    //iterator
    int l = 0;

    curandState_t state;
    curand_init(1234, antN,0, &state);

    //set initial city 0
    sequence[(antN*NCITY) + stepIndex] = 0; //(double)(curand(&state)%NCITY);
//  sequence[(antN*NCITY) + NCITY-1]=-1;

    double sumPh = 0.0;

    // while the path is not complete
    while(stepIndex < NCITY-1){

    	sumPh = 0;

    	for(int i=0;i<NCITY;i++){
        	aux = true;
        	for(int j = 0;j<=stepIndex;j++){
            		if(sequence[(antN*NCITY) + j]==i){
                		aux=false;
                		break;
            		}
        	}
        	if(aux==true){
//        				printf("\n\n Pheromonio %f", phero[(sequence[(antN*NCITY) + stepIndex]*NCITY)+i]);
//        				printf("\n Dist %f", dist[(sequence[(antN*NCITY) + stepIndex]*NCITY)+i]);
            		sumPh += pow(phero[(sequence[(antN*NCITY) + stepIndex]*NCITY)+i],ALPHA) * pow(1/(dist[(sequence[(antN*NCITY) + stepIndex]*NCITY) +i]),BETA);
        	}
    	}

        sum_prob = 0;

        for(int i = 0;i<NCITY;i++){
            aux = true;
            for(int j = 0;j<=stepIndex;j++){
                if(i==sequence[(antN*NCITY) + j]){
                    selection_prob[i]=0;
                    aux = false;
                    break;
                }
            }
            if(aux==true){
                selection_prob[i] = pow(phero[(sequence[(antN*NCITY) + stepIndex]*NCITY) + i],ALPHA) * pow(1/(dist[(sequence[(antN*NCITY) + stepIndex]*NCITY) + i]),BETA);
                selection_prob[i] = selection_prob[i]/sumPh;

//                if(antN == 0 && stepIndex == 0){
//                	printf("\n\n Prob %f" , selection_prob[i]);
//                }
            }
            sum_prob += selection_prob[i];
        }
        random = (double)(curand(&state)%100001)/100000;
        random = random*sum_prob;
        l = 0;
        prob = selection_prob[l];
        while(prob<random){
            l+=1;
            prob+=selection_prob[l];

//            if(antN == 0 && stepIndex == 1){
//				printf("\n\n R %f p %f" , r, p);
//			}
        }
//        if(antN == 0 && stepIndex == 0){
//        	printf("\n\n Proxima cidade %i" , l);
//        }
        stepIndex+=1;
        sequence[(antN*NCITY) + stepIndex] = l;
    }
}

__global__ void route2(double* PHERO, double* DIST, int* ROUTES, double* PROBS, int NCITY, curandState* rand_states){

//	int local_ant_index = threadIdx.x;
//	int antN = blockIdx.x * blockDim.x + threadIdx.x;

	int antN = blockIdx.x * blockDim.x + threadIdx.x;
	int initialCity = 0;

//	__shared__ int local_routes[BLOCK_SIZE * NCITY];

	ROUTES[antN * NCITY] = initialCity;

	for (int i=0; i < NCITY-1; i++) {

			int cityi = ROUTES[antN*NCITY+i];
			int count = 0;

			double testvar = 0;

//			printf("\n\n Step %i Current CIty %i", i, cityi);
//			printf("\n Probabilities: \n");

			for (int c = 0; c < NCITY; c++) {

				if (cityi == c || vizited(antN, c, ROUTES, NCITY, i)) {
					PROBS[antN*NCITY+c] = 0;
				}else{
					PROBS[antN*NCITY+c] = PHI (cityi, c, antN, NCITY, DIST, PHERO, ROUTES, i);
					count++;
				}
				testvar += PROBS[antN*NCITY+c];
//				printf("%f ", PROBS[antN*NCITY+c]);
			}

//			printf("\n\n Teste VAR : %f" , testvar);

			// deadlock --- it reaches a place where there are no further connections
			if (0 == count) {
				return;
			}

			int nextCity = city(antN, NCITY, PROBS, rand_states);

			ROUTES[(antN * NCITY) + (i + 1)] = nextCity;
		}

	__syncthreads();
}

__global__ void route3(double* PHERO, double* DIST, int* ROUTES, int NCITYs, curandState* rand_states){

	int local_ant_index = threadIdx.x;
	int antN = blockIdx.x * blockDim.x + threadIdx.x;

	int initialCity = 0;
	int NCITY = 38;

	__shared__ int local_routes[BLOCK_SIZE * GRAPH_SIZE];
	__shared__ double local_probs[BLOCK_SIZE * GRAPH_SIZE];


	local_routes[local_ant_index*NCITY] = initialCity;

	for (int i=0; i < NCITY-1; i++) {

		int cityi = local_routes[local_ant_index*NCITY+i];
		int count = 0;

		for (int c = 0; c < NCITY; c++) {

			if (cityi == c || vizited(local_ant_index, c, local_routes, NCITY, i)) {
				local_probs[local_ant_index*NCITY+c] = 0;
			}else{
				local_probs[local_ant_index*NCITY+c] = PHI (cityi, c, local_ant_index, NCITY, DIST, PHERO, local_routes, i);
				count++;
			}
		}


		// deadlock --- it reaches a place where there are no further connections
		if (0 == count) {
			return;
		}

		int nextCity = city(local_ant_index, NCITY, local_probs, rand_states);

		local_routes[(local_ant_index * NCITY) + (i + 1)] = nextCity;
	}

	for(int j = 0 ; j < NCITY ; j++){
		ROUTES[(antN * NCITY) + j] = local_routes[(local_ant_index * NCITY)+j];
	}
}

__global__ void d_updatePHEROMONES (int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES, double* PHEROMONES, double* DELTAPHEROMONES, double* DIST, double* routes_distance, double* bestRoute, int* d_best_sequence) {

//	printf("\n\n\n updatePHEROMONES: ");

	int Q = 11340;
	double RO = 0.5;

	for (int k=0; k<NUMBEROFANTS[0]; k++) {

		double rlength = d_length(k, NUMBEROFCITIES[0], ROUTES, DIST);
		routes_distance[k] = rlength;

//		printf("\n Distances : %f", rlength);

		for (int r=0; r < NUMBEROFCITIES[0]-1; r++) {

			int cityi = ROUTES[k * NUMBEROFCITIES[0] + r];
			int cityj = ROUTES[k * NUMBEROFCITIES[0] + r + 1];

			DELTAPHEROMONES[cityi* NUMBEROFCITIES[0] + cityj] += Q / rlength;
			DELTAPHEROMONES[cityj* NUMBEROFCITIES[0] + cityi] += Q / rlength;
		}

		if(routes_distance[k] < bestRoute[0]){
			bestRoute[0] = routes_distance[k];
			for (int count=0; count < NUMBEROFCITIES[0]; count++) {
				d_best_sequence[count] = ROUTES[k * NUMBEROFCITIES[0]+count];
			}
		}
	}

	for (int i=0; i<NUMBEROFCITIES[0]; i++) {
		for (int j=0; j<NUMBEROFCITIES[0]; j++) {
			PHEROMONES[i * NUMBEROFCITIES[0] + j] = (1 - RO) * PHEROMONES[i * NUMBEROFCITIES[0] +j] + DELTAPHEROMONES[i * NUMBEROFCITIES[0] +j];
			DELTAPHEROMONES[i * NUMBEROFCITIES[0] +j] = 0.0;

			PHEROMONES[j * NUMBEROFCITIES[0] + i] = (1 - RO) * PHEROMONES[j * NUMBEROFCITIES[0] +i] + DELTAPHEROMONES[j * NUMBEROFCITIES[0] +i];
			DELTAPHEROMONES[j * NUMBEROFCITIES[0] +i] = 0.0;
		}
	}

	__syncthreads();
}

__device__ bool vizited(int antk, int c, int* ROUTES, int NUMBEROFCITIES, int step) {

	for (int l=0; l < step; l++) {
		if (ROUTES[antk*NUMBEROFCITIES+l] == c) {
			return true;
		}
	}
	return false;
}

__device__ double PHI (int cityi, int cityj, int antk, int NUMBEROFCITIES ,double* dist, double* PHEROMONES, int* ROUTES, int step) {

//	printf("\n\n Probabilidades ant: %i" , antk);

	double dista = dist[cityi*NUMBEROFCITIES+cityj];

	double ETAij = (double) pow (1 / dista , d_BETA);
	double TAUij = (double) pow (PHEROMONES[(cityi * NUMBEROFCITIES) + cityj],   d_ALPHA);

	double sum = 0.0;
	for (int c=0; c < NUMBEROFCITIES; c++) {
		if (cityi != c && !vizited(antk, c, ROUTES, NUMBEROFCITIES, step)){
			double ETA = (double) pow (1 / dist[cityi*NUMBEROFCITIES+c], d_BETA);
			double TAU = (double) pow (PHEROMONES[(cityi*NUMBEROFCITIES)+c],   d_ALPHA);
			sum += ETA * TAU;
		}
	}
	return (ETAij * TAUij) / sum;
}

__device__ int city (int antK, int NCITIES, double* PROBS, curandState* rand_states) {

    double random = curand_uniform(&rand_states[antK]);

	int i = 0;

	double sum = PROBS[antK*NCITIES];
//	printf("\n SUM %f", sum);
	while (sum < random){ // && i < NCITIES-1) {
		i++;
		sum += PROBS[antK*NCITIES+i];
//		printf("\n SUM2 %f", sum);
	}

	//printf("\n\n Next CIty %i", i);
	return (int) i;
}

__device__ double d_length (int antk, int NUMBEROFCITIES, int* ROUTES, double* DIST) {
	double sum = 0.0;
	for (int j=0; j<NUMBEROFCITIES-1; j++) {

		int cityi = ROUTES[antk*NUMBEROFCITIES+j];
		int cityj = ROUTES[antk*NUMBEROFCITIES+j+1];

		sum += DIST[cityi*NUMBEROFCITIES + cityj];
	}
	return sum;
}

void pheroEvap(double* phero, int NCITY){

    for(int i = 0;i< NCITY;i++){
        for(int j=0;j<NCITY;j++){
            phero[(i* NCITY) + j] = (1-EVAPORATION)*phero[(i*NCITY)+j];
        }
    }

}

double pheroDeposit(int* sequence, int formiga, double* dist, double* phero, int NCITY){


//	printf("\n\n Phero Deposit : %i \n" , formiga );

    double totalDist = 0;
    for(int i = 0;i<NCITY;i++){

	//cout << ((sequence[formiga*NCITY + i])*NCITY) << " " << sequence[formiga*NCITY + i+1] << " " << formiga<< " "<<NCITY<< " "<<i << endl;
        totalDist+= dist[((sequence[formiga*NCITY + i])*NCITY) + sequence[formiga*NCITY + i+1]];
//        printf(" %i " , sequence[formiga*NCITY + i]);
    }
    totalDist+=dist[(sequence[formiga*NCITY + NCITY-1]*NCITY) + sequence[formiga*NCITY]];
//    printf("\n\n Total DIst : %f" , totalDist );

    for(int i = 0;i<NCITY-1;i++){

        phero[(sequence[formiga*NCITY + i]*NCITY) + sequence[formiga*NCITY + i+1]] += double(1/totalDist);
        phero[(sequence[formiga*NCITY + i+1]*NCITY) + sequence[formiga*NCITY + i]] = phero[(sequence[formiga*NCITY + i]*NCITY) + sequence[formiga*NCITY + i+1]];
    }
    phero[(sequence[formiga*NCITY + NCITY-1]*NCITY)+sequence[formiga*NCITY]] += double(1/totalDist);
    phero[(sequence[formiga*NCITY]*NCITY)+ sequence[formiga*NCITY + NCITY-1]] = phero[(sequence[formiga*NCITY + NCITY-1]*NCITY) + sequence[formiga*NCITY]];

    return totalDist;
}

void updatePHEROMONES (int NUMBEROFANTS, int NUMBEROFCITIES, int* ROUTES, double* PHEROMONES, double* DIST, double* routes_distance, double bestRoute) {

//	printf("\n\n\n updatePHEROMONES: ");

	int Q = 11340;
	double RO = 0.5;

	double DELTAPHEROMONES[NUMBEROFCITIES*NUMBEROFCITIES];

	for (int k=0; k<NUMBEROFANTS; k++) {

		double rlength = length(k, NUMBEROFCITIES, ROUTES, DIST);
		routes_distance[k] = rlength;

//		printf("\n Distances : %f", rlength);

		for (int r=0; r < NUMBEROFCITIES-1; r++) {

			int cityi = ROUTES[k * NUMBEROFCITIES + r];
			int cityj = ROUTES[k * NUMBEROFCITIES + r + 1];

			DELTAPHEROMONES[cityi* NUMBEROFCITIES + cityj] += Q / rlength;
			DELTAPHEROMONES[cityj* NUMBEROFCITIES + cityi] += Q / rlength;
		}
	}

	for (int i=0; i<NUMBEROFCITIES; i++) {
		for (int j=0; j<NUMBEROFCITIES; j++) {
			PHEROMONES[i * NUMBEROFCITIES + j] = (1 - RO) * PHEROMONES[i * NUMBEROFCITIES +j] + DELTAPHEROMONES[i * NUMBEROFCITIES +j];
			DELTAPHEROMONES[i * NUMBEROFCITIES +j] = 0.0;

			PHEROMONES[j * NUMBEROFCITIES + i] = (1 - RO) * PHEROMONES[j * NUMBEROFCITIES +i] + DELTAPHEROMONES[j * NUMBEROFCITIES +i];
			DELTAPHEROMONES[j * NUMBEROFCITIES +i] = 0.0;
		}
	}
}

double length (int antk, int NUMBEROFCITIES, int* ROUTES, double* DIST) {
	double sum = 0.0;
	for (int j=0; j<NUMBEROFCITIES-1; j++) {

		int cityi = ROUTES[antk*NUMBEROFCITIES+j];
		int cityj = ROUTES[antk*NUMBEROFCITIES+j+1];

		sum += DIST[cityi*NUMBEROFCITIES + cityj];
	}
	return sum;
}

int main(int argc, char** argv)
{

	printf("\n  ------------ Starting Execution ------------");

	randoms = new Randoms(15);

	int GPU_N;
	const int MAX_GPU_COUNT = 1;

	cudaGetDeviceCount(&GPU_N);

	if (GPU_N > MAX_GPU_COUNT) {
		GPU_N = MAX_GPU_COUNT;
	}

	printf("\n CUDA-capable device count: %i", GPU_N);

	// create stream array - create one stream per GPU
	cudaStream_t stream[GPU_N];

	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(i);
		cudaDeviceReset();
		cudaStreamCreate(&stream[i]);
	}

	printf("\n Streams Created \n");

    int NCITY;
    stringstream ncity(argv[1]);
    ncity >> NCITY;
    int NANT;
    stringstream ant(argv[2]);
    ant >> NANT;
    int INTER;
    stringstream inter(argv[3]);
    inter >> INTER;
    int EXEC;
    stringstream exec(argv[4]);
    exec >> EXEC;
    int problem;
    stringstream prob(argv[5]);
    prob >> problem;

//    srand(time(NULL));

    //host variables
    double bestAll = 999999.9;
    double* coord;
    double* phero;
    double* dist;
	int* sequence;
//	double* routes_distance;
	int* best_sequence;

	//device variables
    double* d_coord;
    double* d_phero;
    double* d_delta_phero;
    double* d_dist;
    double* d_routes_distance;
    int* d_seq;
    double* d_bestRoute;
    int* d_nants;
    int* d_ncities;
    int* d_best_sequence;

    //Init Random Generators
    curandState* d_rand_states_ind;
	cudaMalloc(&d_rand_states_ind, NANT * NCITY * sizeof(curandState));

    //alloc host variables
	sequence = (int *)malloc(NCITY*NANT*sizeof(int));
    coord = (double *)malloc((NCITY*2) * sizeof(double));
    phero = (double *)malloc((NCITY*NCITY) * sizeof(double));
    dist = (double *)malloc((NCITY*NCITY) * sizeof(double));
//    routes_distance = (double *)malloc((NCITY*NANT) * sizeof(double));
    best_sequence = (int *)malloc((NCITY) * sizeof(int));

    //alloc device variables
    cudaMalloc((void**) &d_coord, NCITY*2*sizeof(double));
    cudaMalloc((void**) &d_phero, NCITY*NCITY*sizeof(double));
    cudaMalloc((void**) &d_delta_phero, NCITY*NCITY*sizeof(double));
    cudaMalloc((void**) &d_dist, NCITY*NCITY*sizeof(double));
    cudaMalloc((void**) &d_seq, NANT*NCITY*sizeof(int));
    cudaMalloc((void**) &d_routes_distance, NANT*NCITY*sizeof(double));
    cudaMalloc((void**) &d_bestRoute, sizeof(double));
    cudaMalloc((void**) &d_nants, sizeof(int));
    cudaMalloc((void**) &d_ncities, sizeof(int));
    cudaMalloc((void**) &d_best_sequence, NCITY*sizeof(int));

    //cudaMemcpy(coord, d_coord, NCITY*2*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(phero, d_phero, NCITY*NCITY*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(dist, d_dist, NCITY*NCITY*sizeof(double), cudaMemcpyHostToDevice);

    setup_rand_kernel<<<NANT, NCITY, 0, stream[0]>>>(d_rand_states_ind, time(NULL));

    double total_execution_time = 0.0;
    double actual_execution_time = 0.0;

    std::clock_t start;

    //Init executions
    for(int j = 0;j<EXEC;j++){

		double bestRoute = 999999.9;
		int inter = INTER;


		for(int p=0;p<NCITY*NANT;p++){
			sequence[p]=0;
		}
		// ----------------------------------------------------------------------APAGAR DPS
			//cout << "teste2" << endl;

		if(problem==0){
			readMap(coord,phero,dist, NCITY); // read txt with the citys position
		}else if(problem==1){
			readMap2(coord,phero,dist, NCITY); // read txt with the citys position
		}else if(problem==2){
			readMap3(coord,phero,dist, NCITY); // read txt with the citys position
		}
		else{
			cout << "ERRO NO PROBLEMA, DIGITAR 0,1,OU 2 " << endl;
		}
		//cout << "teste3" << endl;

		cudaMemcpy(d_coord, coord, NCITY*2*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_phero, phero, NCITY*NCITY*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bestRoute, &bestRoute, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_nants, &NANT, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ncities, &NCITY, sizeof(int), cudaMemcpyHostToDevice);

		calcDist<<<1, NCITY>>>(d_dist, d_coord, NCITY); // calculates the distances of each city

	//	cudaMemcpy(d_coord, coord, NCITY*2*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(dist, d_dist, NCITY*NCITY*sizeof(double), cudaMemcpyDeviceToHost);

		//cout << "teste4" << endl;
		//cudaMemcpy(&d_phero, &phero, NCITY*NCITY*sizeof(double), cudaMemcpyDeviceToHost);
		//cout << "teste5" << endl;

		start = std::clock();

		while(inter>0){

			route3<<<NBLOCKS, BLOCK_SIZE>>>(d_phero, d_dist, d_seq, NCITY, d_rand_states_ind);
//			cudaMemcpy(sequence, d_seq, NCITY*NANT*sizeof(int), cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			//cout << "teste6" << endl;
			//pheroEvap(phero, NCITY);

//			updatePHEROMONES(NANT, NCITY, sequence, phero, dist, routes_distance, bestRoute);

			d_updatePHEROMONES<<<1,1>>>(d_nants, d_ncities, d_seq, d_phero, d_delta_phero, d_dist, d_routes_distance, d_bestRoute, d_best_sequence);

			cudaDeviceSynchronize();

//			for(int i = 0;i<NANT;i++){
//				if(routes_distance[i] < bestRoute){
//					bestRoute = routes_distance[i];
//					printf("\n New Best Route : %f", bestRoute);
//
//					for(int j = 0;j<NCITY;j++){
//						best_sequence[j] = sequence[i*NCITY+j];
//					}
//				}
//			}

//			for(int count = 0 ; count < NCITY ; count++){
//				printf(" %i " , sequence[count]);
//			}

			inter -=1;


		}

		actual_execution_time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		total_execution_time += actual_execution_time;
		std::cout<<"\n Execution: " << j <<" Elapsed Time: "<< actual_execution_time <<"";

		cudaMemcpy(best_sequence, d_best_sequence, NCITY*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&bestRoute, d_bestRoute, sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		cout << "\n The best execution of execution  " << j << " was:  " << bestRoute <<endl;
		cout << "\n THe best path was:" <<endl;

		for(int i = 0;i<NCITY;i++){
			cout <<  best_sequence[i] << " ";
		}

		cout << "\n END OF EXECUTION : \n\n" <<endl;

		//cout << endl;
		if(bestAll>bestRoute){
			bestAll = bestRoute;
		}


    }




    cout << "\n\n\n\n The best result from all " << EXEC << " exetucions was :  " << bestAll << " Total time elapsed was: " << total_execution_time <<endl;

    cudaFree(d_coord);
    cudaFree(d_phero);
    cudaFree(d_delta_phero);
    cudaFree(d_dist);
    cudaFree(d_routes_distance);
    cudaFree(d_seq);
    cudaFree(d_bestRoute);
    cudaFree(d_nants);
    cudaFree(d_ncities);
    cudaFree(d_best_sequence);


    return 0;
}
