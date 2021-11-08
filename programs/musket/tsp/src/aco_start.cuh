#include <cuda.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <array>
#include <vector>
#include <sstream>
#include <chrono>
#include <curand_kernel.h>
#include <curand.h>
#include <malloc.h>
#include <cuda_runtime.h>
//#include <limits>
#include <memory>
#include <cstddef>
#include <type_traits>
#include <ctime>
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>

#include "../include/musket.cuh"
#include "../include/aco_iroulette_0.cuh"

#include "Randoms.cpp"

int ants = 256;
int ncities = 256;
auto bestroute = 9.99999999E7;
auto try_bestroute = 9.99999999E7;
const int IROULETE = 32;
const double PHERINIT = 0.005;
const double EVAPORATION = 0.5;
const int ALPHA = 2;
const int BETA = 2;
const int TAUMAX = 2;
const int block_size = 64;

//Device Variables
__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_block_size;
__device__ int d_GRAPH_SIZE;

int n_blocks =0;

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
//Read Map and start graphs
void readMap(mkt::DArray<double> cities, mkt::DArray<double> phero, mkt::DArray<double> dist, int n_cities, int problem){ // small

}

struct Calculate_distance_map_index_in_place_array_functor{

    Calculate_distance_map_index_in_place_array_functor(const mkt::DArray<double>& _cities) : cities(_cities){}

    ~Calculate_distance_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int i, double y){
        double returner = 0.0;
        int j = ((i) / (ncities));
        int currentcity = (static_cast<int>((i)) % (ncities));

        if(((j) != (currentcity))){
            double difference = (cities.get_global(((j) * 2)) - cities.get_global(((currentcity) * 2)));

            if(((difference) < 0)){
                difference = ((difference) * -(1));
            }
            double nextdifference = (cities.get_global((((j) * 2) + 1)) - cities.get_global((((currentcity) * 2) + 1)));

            if(((nextdifference) < 0)){
                nextdifference = ((nextdifference) * -(1));
            }
            double pow2 = 2.0;
            double first = __powf((difference), (pow2));
            double second = __powf((nextdifference), (pow2));
            returner = sqrtf(((first) + (second)));
        }
        return (returner);
    }

    void init(int device){
        cities.init(device);
    }

    size_t get_smem_bytes(){
        size_t result = 0;
        return result;
    }

    int ncities;
    mkt::DeviceArray<double> cities;
};
struct Calculate_iroulette_map_index_in_place_array_functor{

    Calculate_iroulette_map_index_in_place_array_functor(const mkt::DArray<int>& _d_iroulette, const mkt::DArray<double>& _distance) : d_iroulette(_d_iroulette), distance(_distance){}

    ~Calculate_iroulette_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int cityindex, int value){
        int c_index = (cityindex);
        for(int i = 0; ((i) < (IROULETE)); i++){
            double citydistance = 999999.9;
            double c_dist = 0.0;
            int cityy = -(1);
            for(int j = 0; ((j) < (ncities)); j++){
                bool check = true;
                for(int k = 0; ((k) < (i)); k++){

                    if((d_iroulette.get_global((((c_index) * (IROULETE)) + (k))) == (j))){
                        check = false;
                    }
                }

                if(((c_index) != (j))){

                    if(((check) == true)){
                        c_dist = distance.get_global((((c_index) * (ncities)) + (j)));

                        if(((c_dist) < (citydistance))){
                            citydistance = (c_dist);
                            cityy = (j);
                        }
                    }
                }
            }
            d_iroulette.set_global((((c_index) * (IROULETE)) + (i)), (cityy));
        }
        return (value);
    }

    void init(int device){
        d_iroulette.init(device);
        distance.init(device);
    }

    size_t get_smem_bytes(){
        size_t result = 0;
        return result;
    }
    int ncities;

    mkt::DeviceArray<int> d_iroulette;
    mkt::DeviceArray<double> distance;
};
struct Route_kernel2_map_index_in_place_array_functor{

    Route_kernel2_map_index_in_place_array_functor(const mkt::DArray<int>& _antss, const mkt::DArray<int>& _d_routes, const mkt::DArray<int>& _d_iroulette, const mkt::DArray<double>& _distance, const mkt::DArray<double>& _phero, const mkt::DArray<double>& _d_probabilities) : antss(_antss), d_routes(_d_routes), d_iroulette(_d_iroulette), distance(_distance), phero(_phero), d_probabilities(_d_probabilities){}

    ~Route_kernel2_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int Index, int value){
        curandState_t curand_state;
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(clock64(), id, 0, &curand_state);

        int newroute = 0;
        int ant_index = Index;
        double sum = 0.0;
        int next_city = -(1);
        double ETA = 0.0;
        double TAU = 0.0;

        //Ants start randomly
		int initial_city = static_cast<int>(curand_uniform(&curand_state) * (static_cast<double>((ncities)) - 0.0) + 0.0);
        if (ant_index < ants) {

            for (int i = 0; ((i) < ((ncities-1))); i++) {
                int cityi = d_routes.get_global((((ant_index) * (ncities)) + (i)));
                int count = 0;
                for (int c = 0; ((c) < (IROULETE)); c++) {
                    next_city = d_iroulette.get_global((((cityi) * (IROULETE)) + (c)));
                    int visited = 0;
                    for (int l = 0; ((l) <= (i)); l++) {
                        if ((d_routes.get_global((((ant_index) * (ncities)) + (l))) == (next_city))) {
                            visited = 1;
                        }
                    }

                    if (!(visited)) {
                        int indexpath = (((cityi) * (ncities)) + (next_city));
                        double firstnumber = (1 / distance.get_global((indexpath)));
                        ETA = static_cast<double>(__powf((firstnumber), (BETA)));
                        TAU = static_cast<double>(__powf(phero.get_global((indexpath)),(ALPHA)));
                        sum += ((ETA) * (TAU));
                    }
                }

                for (int c = 0; ((c) < (IROULETE)); c++) {
                    next_city = d_iroulette.get_global((((cityi) * (IROULETE)) + (c)));
                    int visited = 0;
                    for (int l = 0; ((l) <= (i)); l++) {
                        if ((d_routes.get_global((((ant_index) * (ncities)) + (l))) == (next_city))) {
                            visited = 1;
                        }
                    }

                    if((visited)){
                        d_probabilities.set_global((((ant_index) * (ncities)) + (c)), 0.0);
                    } else {
                        double dista = static_cast<double>(distance.get_global((((cityi) * (ncities)) + (next_city))));
                        double ETAij = static_cast<double>(__powf((1 / (dista)), (BETA)));
                        double TAUij = static_cast<double>(__powf(phero.get_global((((cityi) * (ncities)) + (next_city))),(ALPHA)));
                        d_probabilities.set_global((((ant_index) * (ncities)) + (c)), ((ETAij * TAUij) / sum));
                        count++;
                    }
                }


                if ((0 == (count))) {
                    int breaknumber = 0;
                    for (int nc = 0; ((nc) < (ncities)); nc++) {
                        int visited = 0;
                        for (int l = 0; ((l) <= (i)); l++) {
                            if ((d_routes.get_global((((ant_index) * (ncities)) + (l))) ==
                                 (nc))) {
                                visited = 1;
                            }
                        }
                        if (!(visited)) {
                            breaknumber = (nc);
                            nc = ncities;
                        }
                    }
                    newroute = (breaknumber);
                } else {
                    double random = curand_uniform(&curand_state);
                    int ii = -1;
                    double summ = 0.0;;

                    for (int check = 1; ((check) > 0); check++) {

                    	if (((summ) > (random))) {
							check = -(2);
						} else{
							ii = ii +1;
							summ += d_probabilities.get_global((((ant_index) * (ncities)) + (ii)));
						}
                    }
                    int chosen_city = (ii);
                    newroute = d_iroulette.get_global((((cityi) * (IROULETE)) + (chosen_city)));
                }

                d_routes.set_global((((ant_index) * (ncities)) + ((i) + 1)), (newroute));
                sum = 0.0;
            }
        }
        return value;
    }

    void init(int device){
        antss.init(device);
        d_routes.init(device);
        d_iroulette.init(device);
        distance.init(device);
        phero.init(device);
        d_probabilities.init(device);
    }

    size_t get_smem_bytes(){
        size_t result = 0;
        return result;
    }
    int ants;
    int ncities;

    mkt::DeviceArray<int> antss;
    mkt::DeviceArray<int> d_routes;
    mkt::DeviceArray<int> d_iroulette;
    mkt::DeviceArray<double> distance;
    mkt::DeviceArray<double> phero;
    mkt::DeviceArray<double> d_probabilities;
};

struct Update_pheromones0_map_index_in_place_array_functor{

    Update_pheromones0_map_index_in_place_array_functor(const mkt::DArray<int>& _d_routes, const mkt::DArray<double>& _distance, const mkt::DArray<double>& _d_routes_distance, const mkt::DArray<double>& _d_delta_phero) : d_routes(_d_routes), distance(_distance), d_routes_distance(_d_routes_distance), d_delta_phero(_d_delta_phero){}

    ~Update_pheromones0_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int Index, int value){
        int Q = 11340;
        double RO = 0.05;
        int k = (Index);
        double sum = 0.0;
        if (Index <= ants) {
            for (int j = 0; ((j) < ((ncities) - 1)); j++) {
                int cityii = d_routes.get_global((((k) * (ncities)) + (j)));
                int cityjj = d_routes.get_global(((((k) * (ncities)) + (j)) + 1));
                sum += distance.get_global((((cityii) * (ncities)) + (cityjj)));
            }
            int cityiii = d_routes.get_global(((((k) * (ncities)) + (ncities)) - 1));
            int cityjjj = d_routes.get_global(((k) * (ncities)));
            sum += distance.get_global((((cityiii) * (ncities)) + (cityjjj)));
            double rlength = (sum);
            d_routes_distance.set_global((k), (rlength));
            for (int r = 0; ((r) < ((ncities) - 1)); r++) {
                int cityi = d_routes.get_global((((k) * (ncities)) + (r)));
                int cityj = d_routes.get_global(((((k) * (ncities)) + (r)) + 1));
                double delta = d_delta_phero.get_global(((cityi) * (ncities)) + (cityj));
                d_delta_phero.set_global((((cityi) * (ncities)) + (cityj)), delta + ((Q) / (rlength)));
                d_delta_phero.set_global((((cityj) * (ncities)) + (cityi)), delta + ((Q) / (rlength)));
            }
        }
        return (value);
    }

    void init(int device){
        d_routes.init(device);
        distance.init(device);
        d_routes_distance.init(device);
        d_delta_phero.init(device);
    }

    size_t get_smem_bytes(){
        size_t result = 0;
        return result;
    }

    int ncities;
    int ants;

    mkt::DeviceArray<int> d_routes;
    mkt::DeviceArray<double> distance;
    mkt::DeviceArray<double> d_routes_distance;
    mkt::DeviceArray<double> d_delta_phero;
};
struct Update_pheromones1_map_index_in_place_array_functor{

    Update_pheromones1_map_index_in_place_array_functor(const mkt::DArray<double>& _d_routes_distance, const mkt::DArray<int>& _best_sequence, const mkt::DArray<int>& _d_routes) : d_routes_distance(_d_routes_distance), best_sequence(_best_sequence), d_routes(_d_routes){}

    ~Update_pheromones1_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int Index, int value){
        int k = (Index);

        if((d_routes_distance.get_global((k)) == (bestRoute))){
		bestRoute = d_routes_distance.get_global((k));
            for(int count = 0; ((count) < (ncities)); count++){
                best_sequence.set_global((count), d_routes.get_global((((k) * (ncities)) + (count))));
            }
        }
        return (value);
    }

    void init(int device){
        d_routes_distance.init(device);
        best_sequence.init(device);
        d_routes.init(device);
    }

    size_t get_smem_bytes(){
        size_t result = 0;
        return result;
    }

    double bestRoute;
    int ncities;
    mkt::DeviceArray<double> d_routes_distance;
    mkt::DeviceArray<int> best_sequence;
    mkt::DeviceArray<int> d_routes;
};
struct Update_pheromones2_map_index_in_place_array_functor{

    Update_pheromones2_map_index_in_place_array_functor(const mkt::DArray<double>& _phero, const mkt::DArray<double>& _d_delta_phero) : phero(_phero), d_delta_phero(_d_delta_phero){}

    ~Update_pheromones2_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int Index, int value){
    	int Q = 11340;
		double RO = 0.5;
        int i = (Index);
        for(int j = 0; ((j) < (ncities)); j++){
        	double new_phero =  (((1 - (RO)) * phero.get_global((((i) * (ncities)) + (j)))) + d_delta_phero.get_global((((i) * (ncities)) + (j))));

        	if(new_phero > 2.0){
        		new_phero = 2.0;
        	}
        	if(new_phero < 0.1){
				new_phero = 0.1;
			}

            phero.set_global((((i) * (ncities)) + (j)), new_phero);
            phero.set_global((((j) * (ncities)) + (i)), new_phero);
            d_delta_phero.set_global((((i) * (ncities)) + (j)), 0.0);
            d_delta_phero.set_global((((j) * (ncities)) + (i)), 0.0);
        }
        return (value);
    }

    void init(int device){
        phero.init(device);
        d_delta_phero.init(device);
    }

    size_t get_smem_bytes(){
        size_t result = 0;
        return result;
    }
    int ncities;

    mkt::DeviceArray<double> phero;
    mkt::DeviceArray<double> d_delta_phero;
};




template<>
double mkt::reduce_min(mkt::DArray<double>& a){
    double local_result = 2147483647;

    const int gpu_elements = a.get_size_gpu();
    int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
    int blocks = (gpu_elements + threads - 1) / threads;
    cudaSetDevice(0);
    double* d_odata;
    cudaMalloc((void**) &d_odata, blocks * sizeof(double));
    double* devptr = a.get_device_pointer(0);

    mkt::kernel::reduce_min_call(gpu_elements, devptr, d_odata, threads, blocks, mkt::cuda_streams[0], 0);

    // fold on gpus: step 2
    while(blocks > 1){
        int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
        int blocks_2 = (blocks + threads_2 - 1) / threads_2;
        mkt::kernel::reduce_min_call(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
        blocks = blocks_2;
    }

    // copy final sum from device to host
    cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
    mkt::sync_streams();
    cudaFree(d_odata);

    return local_result;
}

void printfDarrayDouble(mkt::DArray<double> dArray, char* name, int length) {
    dArray.update_self();
    printf("\n%c: \n", name);
    for (int z = 0; z < length; z++) {
        printf("%0.5f,", dArray[z]);
    }
}
void printfDarrayInt(mkt::DArray<int> dArray, char* name, int length) {
    dArray.update_self();
    printf("\n%c: \n", name);
    for (int z = 0; z < length; z++) {
        printf("%d,", dArray[z]);
    }
}

int run_aco(int ant, int iterations, int problem) {
    mkt::init();

    int n_cities = 0;
    int n_ants = ant;
    int n_blocks = n_ants/block_size;

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
            n_cities =  1291;
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
            n_cities =  783;
            break;
    }
    randoms = new Randoms(15);
    ncities = n_cities;
    ants = n_ants;
    int GPU_N;
    const int MAX_GPU_COUNT = 1;

    cudaGetDeviceCount(&GPU_N);

    if (GPU_N > MAX_GPU_COUNT) {
        GPU_N = MAX_GPU_COUNT;
    }

    // create stream array - create one stream per GPU
    cudaStream_t stream[GPU_N];

    for (int i = 0; i < GPU_N; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    curandState* d_rand_states_ind;
    cudaMalloc(&d_rand_states_ind, n_ants * n_cities * sizeof(curandState));


    std::chrono::high_resolution_clock::time_point init_start = std::chrono::high_resolution_clock::now();

    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
    int citysquared = n_cities * n_cities;

    std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();

    // Initializes Musket data structs.
    mkt::DArray<double> cities(0, n_cities * 2, n_cities * 2, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<int> city(0, n_cities, n_cities, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<int> antss(0, ant, ant, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<double> phero(0, citysquared, citysquared, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<double> phero_new(0, citysquared, citysquared, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<double> distance(0, citysquared, citysquared, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<int> best_sequence(0, 512, 512, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<double> d_delta_phero(0, citysquared, citysquared, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<double> d_routes_distance(0, ant, ant, 9999.999, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<double> d_probabilities(0, n_cities * ant, n_cities * ant, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<int> d_iroulette(0, n_cities * IROULETE, n_cities * IROULETE, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
    mkt::DArray<int> d_routes(0, n_cities * ants, n_cities * ants, 0, 1, 0, 0, mkt::DIST, mkt::COPY);

    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(timer_end - timer_start).count();

    // Initializes Skeletons.
    Calculate_distance_map_index_in_place_array_functor calculate_distance_map_index_in_place_array_functor{cities};
    Calculate_iroulette_map_index_in_place_array_functor calculate_iroulette_map_index_in_place_array_functor{d_iroulette, distance};
    Route_kernel2_map_index_in_place_array_functor route_kernel2_map_index_in_place_array_functor{antss, d_routes, d_iroulette, distance, phero, d_probabilities};
    Update_pheromones0_map_index_in_place_array_functor update_pheromones0_map_index_in_place_array_functor{d_routes, distance, d_routes_distance, d_delta_phero};
    Update_pheromones1_map_index_in_place_array_functor update_pheromones1_map_index_in_place_array_functor{d_routes_distance, best_sequence, d_routes};
    Update_pheromones2_map_index_in_place_array_functor update_pheromones2_map_index_in_place_array_functor{phero, d_delta_phero};
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point init_end = std::chrono::high_resolution_clock::now();
    double init = std::chrono::duration<double>(init_end - init_start).count();
    printf("%0.5f;", init);
    std::chrono::high_resolution_clock::time_point readandfill_start = std::chrono::high_resolution_clock::now();

    std::ifstream lerMapa;


    std::string dji = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/djibouti.txt";
    std::string lux = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/luxembourg.txt";
    std::string cat = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/catar.txt";
    std::string a280 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/a280.txt";
    std::string d198 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/d198.txt";
    std::string d1291 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/d1291.txt";
    std::string lin318 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/lin318.txt";
    std::string pcb442 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/pcb442.txt";
    std::string pcb1173 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/pcb1173.txt";
    std::string pr1002 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/pr1002.txt";
    std::string pr2392 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/pr2392.txt";
    std::string rat783 = "/home/ninaherrmann/Research/HLPP2020/public_programs/Musket_Program/tsplib/rat783.txt";

    switch (problem) {
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

    if (lerMapa.is_open()) {

        double randn = 0.0;
        for(int j = 0;j<n_cities;j++){
            for(int k = 0;k<n_cities;k++){
                if(j!=k){
                    randn = randoms -> Uniforme() * TAUMAX;
                    phero[(j*n_cities) + k] = randn;
                    phero[(k*n_cities) + j] = randn;
                }
                else{
                    phero[(j*n_cities) + k] = 0.0;
                    phero[(k*n_cities) + j] = 0.0;
                }
            }
        }

        int i = 0;

        double index, x, y;

        index = 0.0;
        x = 0.0;
        y = 0.0;

        while(!lerMapa.eof() && index <= n_cities){

            lerMapa >> index;
            lerMapa >> x;
            lerMapa >> y;
            if (x != x) {printf("x:%0.5f,", x);}
            if (y != y) {printf("x:%0.5f,", y);}

            cities[(i*2)] = (double)x;
            cities[(i*2) + 1] = (double)y;

            i+=1;
        }

    }    else{
        printf(" File not opened\n");
    }
    lerMapa.close();
    cities.update_devices();
    phero.update_devices();
    distance.update_devices();
    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point readandfill_end = std::chrono::high_resolution_clock::now();
    double readandfill = std::chrono::duration<double>(readandfill_end - readandfill_start).count();
    printf("%0.5f;", readandfill);
    double start = 0.0;
    start = std::clock();
    std::chrono::high_resolution_clock::time_point kernel1_start = std::chrono::high_resolution_clock::now();

    calculate_distance_map_index_in_place_array_functor.ncities = ncities;

    mkt::map_index_in_place<double, Calculate_distance_map_index_in_place_array_functor>(distance, calculate_distance_map_index_in_place_array_functor);

    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point kernel1_end = std::chrono::high_resolution_clock::now();
    double seconds_kernel1 = std::chrono::duration<double>(kernel1_end - kernel1_start).count();
    printf("%0.5f;", seconds_kernel1);
    std::chrono::high_resolution_clock::time_point kernel2_start = std::chrono::high_resolution_clock::now();

    calculate_iroulette_map_index_in_place_array_functor.ncities = ncities;
    mkt::map_index_in_place<int, Calculate_iroulette_map_index_in_place_array_functor>(city, calculate_iroulette_map_index_in_place_array_functor);
    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point kernel2_end = std::chrono::high_resolution_clock::now();
    double seconds_kernel2 = std::chrono::duration<double>(kernel2_end - kernel2_start).count();
    printf("%0.5f;", seconds_kernel2);

    double seconds_minkernel = 0.0;
    double seconds_updatephero0 = 0.0;
    double seconds_kernel3 = 0.0;
    double seconds_updatephero12 = 0.0;
    for(int i = 0; ((i) < (iterations)); i++){

        std::chrono::high_resolution_clock::time_point kernel3_start = std::chrono::high_resolution_clock::now();

        route_kernel2_map_index_in_place_array_functor.ants = (ants);
        route_kernel2_map_index_in_place_array_functor.ncities = (ncities);
        mkt::map_index_in_place64<int, Route_kernel2_map_index_in_place_array_functor>(antss, route_kernel2_map_index_in_place_array_functor);
        mkt::sync_streams();
        std::chrono::high_resolution_clock::time_point kernel3_end = std::chrono::high_resolution_clock::now();
        seconds_kernel3 += std::chrono::duration<double>(kernel3_end - kernel3_start).count();
        std::chrono::high_resolution_clock::time_point updatephero0_start = std::chrono::high_resolution_clock::now();
        update_pheromones0_map_index_in_place_array_functor.ncities = ncities;
        update_pheromones0_map_index_in_place_array_functor.ants = ants;
        mkt::map_index_in_place<int, Update_pheromones0_map_index_in_place_array_functor>(antss, update_pheromones0_map_index_in_place_array_functor);
        mkt::sync_streams();
        std::chrono::high_resolution_clock::time_point updatephero0_end = std::chrono::high_resolution_clock::now();
        seconds_updatephero0 += std::chrono::duration<double>(updatephero0_end - updatephero0_start).count();

        std::chrono::high_resolution_clock::time_point minkernel_start = std::chrono::high_resolution_clock::now();
        try_bestroute = mkt::reduce_min<double>(d_routes_distance);
        mkt::sync_streams();
        std::chrono::high_resolution_clock::time_point minkernel_end = std::chrono::high_resolution_clock::now();
        seconds_minkernel += std::chrono::duration<double>(minkernel_end - minkernel_start).count();

        std::chrono::high_resolution_clock::time_point updatephero12_start = std::chrono::high_resolution_clock::now();

        update_pheromones1_map_index_in_place_array_functor.bestRoute = (try_bestroute);
        update_pheromones1_map_index_in_place_array_functor.ncities = (ncities);

        mkt::map_index_in_place<int, Update_pheromones1_map_index_in_place_array_functor>(antss, update_pheromones1_map_index_in_place_array_functor);

        update_pheromones2_map_index_in_place_array_functor.ncities = (ncities);

        mkt::map_index_in_place<int, Update_pheromones2_map_index_in_place_array_functor>(city, update_pheromones2_map_index_in_place_array_functor);
        mkt::sync_streams();
        std::chrono::high_resolution_clock::time_point updatephero12_end = std::chrono::high_resolution_clock::now();
        seconds_updatephero12 += std::chrono::duration<double>(updatephero12_end - updatephero12_start).count();

    }
	mkt::sync_streams();
    double d_seconds_minkernel = seconds_minkernel/iterations;
    double d_seconds_updatephero0 = seconds_updatephero0/iterations;
    double d_seconds_kernel3 = seconds_kernel3/iterations;
    double d_seconds_updatephero12 = seconds_updatephero12/iterations;
    printf("%0.5f;%0.5f;%0.5f;%0.5f;", d_seconds_kernel3, d_seconds_updatephero0, d_seconds_minkernel, d_seconds_updatephero12);
    printf("%0.5f;", try_bestroute);
    mkt::sync_streams();

    return EXIT_SUCCESS;
}
