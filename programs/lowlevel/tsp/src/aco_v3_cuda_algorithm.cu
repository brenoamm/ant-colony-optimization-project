#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <malloc.h>
#include <curand_kernel.h>
#include <ctime>
#include <chrono>
#include <timer.h>

#define CUDA_ERROR_CHECK

// #include "../include/aco_v3_cuda_algorithm.cuh"
#include "../include/aco_v3_cuda_algorithm.cuh"
#include "Randoms.cpp"

#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2
#define IROULETE 32

#define BLOCK_SIZE 32

__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA;
__device__ double d_TAUMAX;
__device__ int d_BLOCK_SIZE;
__device__ int d_GRAPH_SIZE;

int n_blocks = 0;

std::string::size_type sz;
Randoms *randoms;
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void readMap(double *coord, double *phero, int n_cities, int problem) { // small

    std::ifstream citiesMap;

    std::string dji = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/djibouti.txt";
    std::string lux = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/luxembourg.txt";
    std::string cat = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/catar.txt";
    std::string a280 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/a280.txt";
    std::string d198 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/d198.txt";
    std::string d1291 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/d1291.txt";
    std::string lin318 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/lin318.txt";
    std::string pcb442 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/pcb442.txt";
    std::string pcb1173 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/pbc1173.txt";
    std::string pr1002 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/pr1002.txt";
    std::string pr2392 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/pr2392.txt";
    std::string rat783 = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/rat783.txt";


    switch (problem) {
        case 1:
            citiesMap.open(dji, std::ifstream::in);
            break;
        case 2:
            citiesMap.open(lux, std::ifstream::in);
            break;
        case 3:
            citiesMap.open(cat, std::ifstream::in);
            break;
        case 4:
            citiesMap.open(a280, std::ifstream::in);
            break;
        case 5:
            citiesMap.open(d198, std::ifstream::in);
            break;
        case 6:
            citiesMap.open(d1291, std::ifstream::in);
            break;
        case 7:
            citiesMap.open(lin318, std::ifstream::in);
            break;
        case 8:
            citiesMap.open(pcb442, std::ifstream::in);
            break;
        case 9:
            citiesMap.open(pcb1173, std::ifstream::in);
            break;
        case 10:
            citiesMap.open(pr1002, std::ifstream::in);
            break;
        case 11:
            citiesMap.open(pr2392, std::ifstream::in);
            break;
        case 12:
            citiesMap.open(rat783, std::ifstream::in);
            break;
        default:
            printf("Invalid problem number - must be between 1-12.");
            exit(1);
    }

    if (citiesMap.is_open()) {

        double randn;

        for (int j = 0; j < n_cities; j++) {
            for (int k = 0; k < n_cities; k++) {
                if (j != k) {
                    randn = randoms->Uniforme() * TAUMAX;
                    phero[(j * n_cities) + k] = randn;
                    phero[(k * n_cities) + j] = randn;
                } else {
                    phero[(j * n_cities) + k] = 0;
                    phero[(k * n_cities) + j] = 0;
                }
            }
        }

        int i = 0;

        double index, x, y;

        index = 0.0;
        x = 0.0;
        y = 0.0;

        while (!citiesMap.eof()) {

            citiesMap >> index;
            citiesMap >> x;
            citiesMap >> y;

            coord[(i * 2)] = (double) x;
            coord[(i * 2) + 1] = (double) y;
            i += 1;
        }

    } else {
        printf(" File not opened\n");
    }
    citiesMap.close();
}


__global__ void setup_rand_kernel(curandState *state, unsigned long seed) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, id, 0, &state[id]);
//	curand_init(1234, id, 0, &state[id]);

    if (id == 0) {
        d_PHERINIT = 0.005;
        d_EVAPORATION = 0.5;
        d_ALPHA = 1;
        d_BETA = 2;
        d_TAUMAX = 2;
    }

    __syncthreads();
}

//Calculate distance between cities and chose the closest 32 cities for the I-Roulette.
__global__ void calculate_distance_kernel(double *dist, double *coord, int n_cities) {

    int c_index = threadIdx.x;

    for (int j = 0; j < n_cities; j++) {
        if (c_index != j) {
            dist[(c_index * n_cities) + j] = sqrt(
                    pow(coord[j * 2] - coord[c_index * 2], 2) + pow(coord[(j * 2) + 1] - coord[(c_index * 2) + 1], 2));
            dist[(j * n_cities) + c_index] = dist[(c_index * n_cities) + j];
        } else {
            dist[(c_index * n_cities) + j] = 0.0;
        }
    }
}

__global__ void calculate_iroulette_kernel(const double *dist, int *iroulette, int n_cities) {

    int c_index = threadIdx.x;

    //Get the 32 closest nodes for each node.
    for (int i = 0; i < IROULETE; i++) {

        double distance = 999999.9;
        double c_dist;
        int city = -1;
        for (int j = 0; j < n_cities; j++) {
            bool check = true;
            for (int k = 0; k < i; k++) {
                if (iroulette[c_index * IROULETE + k] == j) {
                    check = false;
                }
            }

            if (c_index != j && check) {
                c_dist = dist[(c_index * n_cities) + j];
                if (c_dist < distance) {
                    distance = c_dist;
                    city = j;
                }
            }
        }
        iroulette[c_index * IROULETE + i] = city;
    }
}

__global__ void
route_kernel(int n_cities, int *routes, double *c_phero, const double *c_dist, double *d_probs, const int *iroulette,
              curandState *rand_states, double *d_sum) { // double *d_eta, double *d_tau, double *d_sum) {

    size_t ant_index = blockIdx.x;
    size_t dim_index = threadIdx.x;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    int next_city = -1;

    routes[ant_index * n_cities] = 0;
    __shared__ double ds_eta[32];
    __shared__ double ds_tau[32];
    // __shared__ double ds_sum[1];
    d_sum[ant_index] = 0.0;
    //d_eta[index] = 0.0;
    //d_tau[index] = 0.0;
    d_probs[index] = 0.0;

    // Loop to build complete route
    for (int i = 0; i < n_cities - 1; i++) {
        int cityi = routes[ant_index * n_cities + i];
        next_city = iroulette[(cityi * IROULETE) + dim_index];
        if (cityi != next_city && !visited(ant_index, next_city, routes, n_cities, i)) {
            ds_eta[dim_index] = (double) pow(1 / c_dist[cityi * n_cities + next_city], d_BETA);
            ds_tau[dim_index] = (double) pow(c_phero[(cityi * n_cities) + next_city], d_ALPHA);
        }

        //synchronize
        __syncthreads();

        if (dim_index == 0) {
            for (int j = 0; j < IROULETE; j++) {
                d_sum[ant_index] += ds_eta[j] * ds_tau[j];
            }
        }
        __syncthreads();
        if (index == 0 ) {
            for (int j = 0; j < IROULETE; j++) {
                d_sum[ant_index] += ds_eta[j] * ds_tau[j];
            }
        }
        //synchronize
        __syncthreads();

        if (cityi == next_city || visited(ant_index, next_city, routes, n_cities, i)) {
            d_probs[index] = 0;
        } else {
            d_probs[index] = ds_eta[dim_index] * ds_tau[dim_index] / d_sum[ant_index];
        }
        __syncthreads();

        // Choose next city. - For some reason with shared memory if produces memory error. Route is 9999??
        if (dim_index == 0) {
           /* if (d_sum[ant_index] > 0.0) {
                int nextCity = city(ant_index, d_probs, rand_states);
                routes[(ant_index * n_cities) + (i + 1)] = iroulette[cityi * IROULETE + nextCity];
            } else {*/
                int nc;
                for (nc = 0; nc < n_cities; nc++) {
                    if (!visited(ant_index, nc, routes, n_cities, i)) {
                        break;
                    }
                }
                routes[(ant_index * n_cities) + (i + 1)] = nc;
            //}

            // Clean for next iteration.
            d_sum[ant_index] = 0.0;
        }
        //d_eta[index] = 0.0;
        //d_tau[index] = 0.0;

        //synchronize
        __syncthreads();
    }
}

__device__ bool visited(int antk, int c, const int *ROUTES, int NUMBEROFCITIES, int step) {
    for (int l = 0; l <= step; l++) {
        if (ROUTES[antk * NUMBEROFCITIES + l] == c) {
            return true;
        }
    }
    return false;
}

__global__ void evaporation_kernel(double *c_phero) {

    int x_index = blockIdx.x * blockDim.x + threadIdx.x;

    double RO = 0.5;

    if (blockIdx.x != threadIdx.x) {
        c_phero[x_index] = (1 - RO) * c_phero[x_index];
    }

}

__global__ void
pheromone_deposit_kernel(int NUMBEROFCITIES, int *ROUTES, double *c_phero, double *DIST) {

    int Q = 11340;

    int ant_k = blockIdx.x;
    int city_id = threadIdx.x;

    double rlength = d_length(ant_k, NUMBEROFCITIES, ROUTES, DIST);

    __syncthreads();

    int cityi = ROUTES[ant_k * NUMBEROFCITIES + city_id];
    int cityj = ROUTES[ant_k * NUMBEROFCITIES + city_id + 1];

    double delta_pheromone = Q / rlength;

    atomicAdd(&c_phero[cityi * NUMBEROFCITIES + cityj], delta_pheromone);
    atomicAdd(&c_phero[cityj * NUMBEROFCITIES + cityi], delta_pheromone);

    __syncthreads();
}

__global__ void
calc_route_distance(const int * d_seq, const double * d_dist, double * d_routes_distance, int n_cities) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    for (int i = 0; i < n_cities; i++) {
        sum += d_dist[d_seq[index * n_cities + i] * n_cities + d_seq[index * n_cities + (i + 1)]];
    }
    d_routes_distance[index] = sum;
}

__global__ void
best_ant_kernel(const int n_ants, const int n_cities, const int *ROUTES, const double *routes_distance,
                double * bestRoute, int *d_best_sequence) {
    for (int k = 0; k < n_ants; k++) {
        if (routes_distance[k] < bestRoute[0] && routes_distance[k] != 0.00) {
            bestRoute[0] = routes_distance[k];
            for (int count = 0; count < n_cities; count++) {
                d_best_sequence[count] = ROUTES[k * n_cities + count];
            }
        }
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ double PHI(int cityi, int cityj, int n_cities, const double *c_dist, double *c_phero, double sum) {
    double dista = c_dist[cityi * n_cities + cityj];

    auto ETAij = (double) pow(1 / dista, d_BETA);
    auto TAUij = (double) pow(c_phero[(cityi * n_cities) + cityj], d_ALPHA);

    return (ETAij * TAUij) / sum;
}

__device__ int city(int antK, double *PROBS, curandState *rand_states) {

    double random = curand_uniform(&rand_states[antK]);

    int i = 0;

    double sum = PROBS[antK * IROULETE];

    while (sum < random) {
        i++;
        sum += PROBS[antK * IROULETE + i];
    }

    return (int) i;
}

__device__ double d_length(int antk, int n_cities, int *ROUTES, double *DIST) {

    double sum = 0.0;

    for (int j = 0; j < n_cities - 1; j++) {

        int cityi = ROUTES[antk * n_cities + j];
        int cityj = ROUTES[antk * n_cities + j + 1];

        sum += DIST[cityi * n_cities + cityj];
    }

    int cityi = ROUTES[antk * n_cities + n_cities - 1];
    int cityj = ROUTES[antk * n_cities];

    sum += DIST[cityi * n_cities + cityj];

    return sum;
}
msl::Timer* startTiming() {
#ifdef MPI_VERSION
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    return new msl::Timer();
}
double stopTiming(msl::Timer* timer) {
    double total_time = timer->stop();
    return total_time;
}
double run_aco(int n_ants, int n_iterations, int problem) {

    int n_cities = 0;
    n_blocks = n_ants / BLOCK_SIZE;

    switch (problem) {
        case 1:
            n_cities = 38; //Djbouti
            break;
        case 2:
            n_cities = 980; //Luxemburg
            break;
        case 3:
            n_cities = 194; //Catar
            break;
        case 4:
            n_cities = 280;
            break;
        case 5:
            n_cities = 198;
            break;
        case 6:
            n_cities = 1291;
            break;
        case 7:
            n_cities = 318;
            break;
        case 8:
            n_cities = 442;
            break;
        case 9:
            n_cities = 1173;
            break;
        case 10:
            n_cities = 1002;
            break;
        case 11:
            n_cities = 2392;
            break;
        case 12:
            n_cities = 783;
            break;
    }

    randoms = new Randoms(15);

    int GPU_N;
    const int MAX_GPU_COUNT = 1;

    cudaGetDeviceCount(&GPU_N);

    if (GPU_N > MAX_GPU_COUNT) {
        GPU_N = MAX_GPU_COUNT;
    }

    cudaStream_t stream[GPU_N];

    for (int i = 0; i < GPU_N; ++i) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, i);
        // In case you need GPU Information.
        // printf("\n");
        // std::cout << "Device " << i << ":\t " << deviceProp.name << std::endl;
        // std::cout << "Compute Capability:\t " << deviceProp.major << "." << deviceProp.minor << std::endl;
        // std::cout << "Total Global Memory:\t " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        // std::cout << "Number of Multiprocessors:\t " << deviceProp.multiProcessorCount << std::endl;
        // Add more properties as needed

        // Check for additional device capabilities and properties
        if (deviceProp.major != 8 || deviceProp.minor != 6) {
            std::cout << "Project is build with CUDA 8.6 - if you have another GPU you need to adjust the CMake File" << std::endl;
        }
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }

    //device variables
    double *d_coord;
    double *d_phero;
    double *d_dist;
    double *d_routes_distance;
    double *d_probs;

/*    double *d_eta;
    double *d_tau;*/
    double *d_sum;

    int *d_iroulette;
    int *d_seq;
    int *d_best_sequence;

    // Init Random Generators
    curandState *d_rand_states_ind;
    cudaMalloc(&d_rand_states_ind, n_ants * n_cities * sizeof(curandState));

    // Allocate host variables.
    auto *coord = new double[n_cities * 2];
    auto *phero = new double[n_cities * n_cities];
    auto *dist = new double[n_cities * n_cities];
    int *best_sequence = new int[n_cities];

    //alloc device variables
    cudaMalloc((void **) &d_coord, n_cities * 2 * sizeof(double));
    cudaMalloc((void **) &d_phero, n_cities * n_cities * sizeof(double));
    cudaMalloc((void **) &d_dist, n_cities * n_cities * sizeof(double));
    cudaMalloc((void **) &d_iroulette, n_cities * IROULETE * sizeof(int));
    cudaMalloc((void **) &d_seq, n_ants * n_cities * sizeof(int));
    cudaMalloc((void **) &d_routes_distance, n_ants * n_cities * sizeof(double));
    cudaMalloc((void **) &d_best_sequence, n_cities * sizeof(int));
    cudaMalloc((void **) &d_probs, n_ants * IROULETE * sizeof(double));

    cudaMalloc((void **) &d_sum, n_ants * sizeof(double));
/*    cudaMalloc((void **) &d_eta, n_ants * IROULETE * sizeof(double));
    cudaMalloc((void **) &d_tau, n_ants * IROULETE * sizeof(double));*/
    double host_bestRoute = 9999.9999;
    double* d_bestRoute;
    cudaMalloc((void**) &d_bestRoute, sizeof(double));
    cudaMemcpy(d_bestRoute, &host_bestRoute, sizeof(double), cudaMemcpyHostToDevice);

    setup_rand_kernel<<<n_ants, n_cities, 0, stream[0]>>>(d_rand_states_ind, time(nullptr));

    readMap(coord, phero, n_cities, problem);

    double bestRoute = 999999.9;
    cudaMemcpy(d_phero, phero, n_cities * n_cities * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord, coord, n_cities * 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // Calculates the distances between each city.
    calculate_distance_kernel<<<1, n_cities>>>(d_dist, d_coord, n_cities); // calculates the distances of each city
    // Save the distance of the 32 closest cities.
    calculate_iroulette_kernel<<<1, n_cities>>>(d_dist, d_iroulette, n_cities);

    cudaDeviceSynchronize();

    //Execution Time measure
    double mean_times = 0.0;
    int iteration = 0;
    double calctime = 0.0;
    msl::Timer* timer = startTiming();
    while (iteration < n_iterations) {
        // Measuring the time for route calculation.
        auto t_start = std::chrono::high_resolution_clock::now();
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        route_kernel<<<n_ants, IROULETE>>>(n_cities, d_seq, d_phero, d_dist, d_probs, d_iroulette, d_rand_states_ind, d_sum); // d_eta, d_tau, d_sum);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        mean_times += std::chrono::duration<double>(t_end - t_start).count();

        evaporation_kernel<<<n_cities, n_cities>>>(d_phero);
        pheromone_deposit_kernel<<<n_ants, (n_cities - 1)>>>(n_cities, d_seq, d_phero, d_dist);
        cudaDeviceSynchronize();
        iteration++;
    }
    calctime += stopTiming(timer);

    dim3 dimBlock(1024);
    dim3 dimGrid((n_ants + dimBlock.x - 1) / n_ants);
    calc_route_distance<<<dimGrid, dimBlock>>>(d_seq, d_dist, d_routes_distance, n_cities);

    best_ant_kernel<<<1, 1>>>(n_ants, n_cities, d_seq, d_routes_distance, d_bestRoute, d_best_sequence);
    cudaMemcpy(&host_bestRoute, d_bestRoute, sizeof(double), cudaMemcpyDeviceToHost);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(best_sequence, d_best_sequence, n_cities * sizeof(int), cudaMemcpyDeviceToHost);

    // Only for checking path.
    printf("%d;%s;%f;", n_ants, "v3", calctime);
    /*printf("\n Best PATH: \t %f \n", host_bestRoute);
    for (int var = 0; var < n_cities; ++var) {
        printf(" %i ", best_sequence[var]);
    }*/
    return host_bestRoute;
}
