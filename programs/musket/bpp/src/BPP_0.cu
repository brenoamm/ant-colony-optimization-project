#include <cuda.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <array>
#include <vector>
#include <sstream>
#include <fstream>

#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <memory>
#include <cstddef>
#include <type_traits>

#include "../include/musket.cuh"
#include "../include/BPP_0.cuh"
#include "Randoms.cpp"

Randoms *randoms;

const int BETA = 1;
const double EVAPORATION = 0.5;
const int TAUMAX = 2;
const int Q = 32;
int itemtypes = 50;
int itemcount = 59;
auto bin_capacity = 1000;
bool PRINT = true;
bool PALMA = true;

struct Copybppitemsquantity_map_index_in_place_array_functor{

    Copybppitemsquantity_map_index_in_place_array_functor(const mkt::DArray<int>& _bpp_items_quantity) : bpp_items_quantity(_bpp_items_quantity){}

    ~Copybppitemsquantity_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int indexx, int valuee){
      int new_index = ((indexx) % (itemtypess));
      // printf("%d;%d;%d;%d;%d\n", indexx, itemtypess, valuee, new_index, bpp_items_quantity.get_global((new_index)));
      return bpp_items_quantity.get_global((new_index));
    }

    void init(int device){
      bpp_items_quantity.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }

    int itemtypess;

    mkt::DeviceArray<int> bpp_items_quantity;
};
struct Copybppitemsweight_map_index_in_place_array_functor{

    Copybppitemsweight_map_index_in_place_array_functor(const mkt::DArray<int>& _bpp_items_weight) : bpp_items_weight(_bpp_items_weight){}

    ~Copybppitemsweight_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int indexx, int valuee){
      int new_index = ((indexx) % (itemtypess));

      return bpp_items_weight.get_global((new_index))/* TODO: For multiple GPUs*/;
    }

    void init(int device){
      bpp_items_weight.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }

    int itemtypess;

    mkt::DeviceArray<int> bpp_items_weight;
};
struct Packing_kernel_map_index_in_place_array_functor{

    Packing_kernel_map_index_in_place_array_functor(const mkt::DArray<int>& _d_bins, const mkt::DArray<int>& _copy_bpp_items_quantity, const mkt::DArray<int>& _bpp_items_quantity, const mkt::DArray<double>& _d_eta, const mkt::DArray<double>& _d_tau, const mkt::DArray<double>& _d_probabilities, const mkt::DArray<int>& _bpp_items_weight, const mkt::DArray<double>& _d_phero, curandState* _d_rand_states_ind) : d_bins(_d_bins), copy_bpp_items_quantity(_copy_bpp_items_quantity), bpp_items_quantity(_bpp_items_quantity), d_eta(_d_eta), d_tau(_d_tau), d_probabilities(_d_probabilities), bpp_items_weight(_bpp_items_weight), d_phero(_d_phero),  d_rand_states_ind(_d_rand_states_ind){}

    ~Packing_kernel_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int iindex, int y){

      int ant_index = (iindex);
      int object_bin_index = ((ant_index) * (itemcountt));
      int bins_used = 0;

      int bpp_items_prefix = (ant_index) * (itemtypess);
      int object_weightmax = 0;

      int actual_bin_weight = 0;
      int n_items_in_actual_bin = 0;
      int possible_items_to_this_bin = 0;

      int object_index = 0;
      int object_quantity = 0;
      int new_object_weight = 0;

      //ADD heaviest Object - ok
      for (int i = 0; ((i) < (itemtypess)); i++) {
        copy_bpp_items_quantity.set_global(bpp_items_prefix + i, bpp_items_quantity.get_global(i));

        new_object_weight = bpp_items_weight.get_global(i);
        object_quantity = copy_bpp_items_quantity.get_global(i);

        if((object_quantity > 0) && (new_object_weight > object_weightmax)){
          object_index = i;
          object_weightmax = new_object_weight;
        }
      }

      d_bins.set_global(((ant_index) * (itemcountt)), (object_index));

      copy_bpp_items_quantity.set_global(((bpp_items_prefix) + (object_index)), (copy_bpp_items_quantity.get_global(((bpp_items_prefix) + (object_index))) - 1));

      n_items_in_actual_bin++;
      actual_bin_weight += (object_weightmax);
      bins_used++;

      int weight_object_j;
      int object_i;
      int quantity_object_j;

      for (int i = 0; ((i) < ((itemcountt) - 1)); i++) {
        
        double eta_tau_sum = 0.0;
        possible_items_to_this_bin = 0;

        //Search POssible Items
        for (int j = 0; ((j) < (itemtypess)); j++) {

          d_eta.set_global(((bpp_items_prefix) + (j)), 0.0);
          d_tau.set_global(((bpp_items_prefix) + (j)), 0.0);
          d_probabilities.set_global(((bpp_items_prefix) + (j)), 0.0);

          weight_object_j = bpp_items_weight.get_global((j));
          quantity_object_j = copy_bpp_items_quantity.get_global(bpp_items_prefix + j);

          if (((quantity_object_j) > 0) && ((weight_object_j) <= ((bin_capacity2) - (actual_bin_weight)))) {

              for (int k = 0; ((k) < (n_items_in_actual_bin)); k++) {
                object_i = d_bins.get_global((((object_bin_index) + (i)) - (k)));
                d_eta.set_global(((bpp_items_prefix) + (j)), d_phero.get_global(object_i * (int) itemtypess + j));
              }

              d_eta.set_global(((bpp_items_prefix) + (j)), (d_eta.get_global(((bpp_items_prefix) + (j))) / (n_items_in_actual_bin)));
              d_tau.set_global(((bpp_items_prefix) + (j)), (double) pow(weight_object_j, BETA));
              eta_tau_sum = eta_tau_sum + (d_eta.get_global(((bpp_items_prefix) + (j))) * d_tau.get_global(((bpp_items_prefix) + (j))));
              possible_items_to_this_bin++;
          }
        }

        if (((possible_items_to_this_bin) > 0)) {

          //Calculate Probabilities
          for (int j = 0; ((j) < (itemtypess)); j++) {
            double tmp = d_eta.get_global(bpp_items_prefix + j);
            double tmp2 = d_tau.get_global(bpp_items_prefix + j);
            double thisthat = ((tmp * tmp2) / (eta_tau_sum));

            d_probabilities.set_global((bpp_items_prefix + j), thisthat);
          }
          eta_tau_sum = 0.0;

          //Perform probabilistic selection
          double random = curand_uniform(&d_rand_states_ind[ant_index]);
          int select_index = 0;
          int object_j = 0;
          double sum = 0.0;
          double prob = 0.0;

          while ((sum <= random) && (select_index < itemtypess)){

            prob = d_probabilities.get_global(bpp_items_prefix+select_index);
            if(prob > 0.0){
              sum += prob;
              object_j = select_index;
            }

            select_index++;
          }

          d_bins.set_global(ant_index * (int) itemcountt + i + 1, (object_j));

          weight_object_j = bpp_items_weight.get_global(object_j);
          actual_bin_weight += (weight_object_j);

          copy_bpp_items_quantity.set_global((bpp_items_prefix + object_j),(copy_bpp_items_quantity.get_global(bpp_items_prefix + object_j) - 1));

          n_items_in_actual_bin++;

        } else {

          bins_used++;

          object_index = 0;
          object_weightmax = 0;

          for (int k = 0; ((k) < (itemtypess)); k++) {
            object_quantity = copy_bpp_items_quantity.get_global((bpp_items_prefix + k));
            new_object_weight = bpp_items_weight.get_global((k));

            if (((object_quantity) > 0) && (((new_object_weight) > (object_weightmax)))) {              
                object_index = (k);
                object_weightmax = (new_object_weight);
              }
          }

          copy_bpp_items_quantity.set_global((bpp_items_prefix + object_index), (copy_bpp_items_quantity.get_global(bpp_items_prefix + object_index) - 1));
          d_bins.set_global(((((ant_index) * ((itemcountt))) + (i)) + 1), (object_index));

          n_items_in_actual_bin = 1;
          actual_bin_weight = (object_weightmax);

//          if(ant_index == 0){
//            printf("\n New Bin %i: \n\t Add %i - Weight %i",bins_used, object_index, object_weightmax);
//          }
        }
      }

      return (bins_used);
    }

    void init(int device){
      d_bins.init(device);
      copy_bpp_items_quantity.init(device);
      bpp_items_quantity.init(device);
      d_eta.init(device);
      d_tau.init(device);
      d_probabilities.init(device);
      bpp_items_weight.init(device);
      d_phero.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }

    int object_weight;
    int itemtypess;
    int itemcountt;
    int BETA2;
    int bin_capacity2;

    curandState* d_rand_states_ind;
    mkt::DeviceArray<int> d_bins;
    mkt::DeviceArray<int> copy_bpp_items_quantity;
    mkt::DeviceArray<int> bpp_items_quantity;
    mkt::DeviceArray<double> d_eta;
    mkt::DeviceArray<double> d_tau;
    mkt::DeviceArray<double> d_probabilities;
    mkt::DeviceArray<int> bpp_items_weight;
    mkt::DeviceArray<double> d_phero;
};
struct Evaporation_kernel_map_index_in_place_array_functor{

    Evaporation_kernel_map_index_in_place_array_functor(const mkt::DArray<double>& _d_phero) : d_phero(_d_phero){}

    ~Evaporation_kernel_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int iindex, double y){
      double result = 0.0;
      double RO = (EVAPORATION2);

      if((((iindex) % (itemtypess)) != 0)){
        result = ((1 - (RO)) * d_phero.get_global((iindex))/* TODO: For multiple GPUs*/);
      }
      return (result);
    }

    void init(int device){
      d_phero.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }

    int itemtypess;
    double EVAPORATION2;

    mkt::DeviceArray<double> d_phero;
};

struct Update_pheromones_kernel_map_index_in_place_array_functor{

    Update_pheromones_kernel_map_index_in_place_array_functor(const mkt::DArray<int>& _d_fitness, const mkt::DArray<int>& _d_bins, const mkt::DArray<double>& _d_phero, const mkt::DArray<int>& _bpp_items_weight) : d_fitness(_d_fitness), d_bins(_d_bins), d_phero(_d_phero), bpp_items_weight(_bpp_items_weight){}

    ~Update_pheromones_kernel_map_index_in_place_array_functor() {}
    
    __device__
    auto operator()(int iindex, int value){
      int ant_index = (iindex);

      double ant_fitness = (d_fitness.get_global((ant_index)));
      double actual_bin_weight = 0.0;
      int actual_bin_object_index = 0;
      int actual_bin_n_objects = 0;

      for (int i = 0; ((i) < (itemcountt)); i++) {

        int object_i = d_bins.get_global((((ant_index) * (itemcountt)) + (i)));
        double object_weight = bpp_items_weight.get_global(object_i);

        if ((((actual_bin_weight) + (object_weight)) <= (bin_capacity2))) {
          actual_bin_n_objects = ((actual_bin_n_objects) + 1);
          actual_bin_weight = ((actual_bin_weight) + (object_weight));
        } else {
          for (int j = 0; ((j) < (actual_bin_n_objects)); j++) {
            for (int k = ((j) + 1); ((k) < (actual_bin_n_objects)); k++) {

              int object_i = d_bins.get_global(((((ant_index) * (itemcountt)) + (actual_bin_object_index)) + (j)));
              int object_j = d_bins.get_global(((((ant_index) * (itemcountt)) + (actual_bin_object_index)) + (k)));

              double delta_pheromone = ((Q) / ant_fitness);

              d_phero.set_global((((object_i) * (itemtypee)) + (object_j)), ((delta_pheromone) + d_phero.get_global((((object_i) * (itemtypee)) + (object_j)))));
              d_phero.set_global((((object_j) * (itemtypee)) + (object_i)), ((delta_pheromone) + d_phero.get_global((((object_j) * (itemtypee)) + (object_i)))));
            }
          }
          actual_bin_n_objects = 1;
          actual_bin_weight = (object_weight);
          actual_bin_object_index = (i);
        }
      }

      //printf("%d;", value);
      return (value);
    }

    void init(int device){
      d_fitness.init(device);
      d_bins.init(device);
      d_phero.init(device);
      bpp_items_weight.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }
    int itemcountt;
    int itemtypee;
    int bin_capacity2;

    mkt::DeviceArray<int> d_fitness;
    mkt::DeviceArray<int> d_bins;
    mkt::DeviceArray<double> d_phero;
    mkt::DeviceArray<int> bpp_items_weight;
};

template<>
int mkt::reduce_min<int>(mkt::DArray<int>& a){
  int local_result = std::numeric_limits<int>::max();

  const int gpu_elements = a.get_size_gpu();
  int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
  int blocks = (gpu_elements + threads - 1) / threads;

  //cudaSetDevice(0);
  int* d_odata;
  cudaMalloc((void**) &d_odata, blocks * sizeof(int));
  int* devptr = a.get_device_pointer(0);

  mkt::kernel::reduce_min_call(gpu_elements, devptr, d_odata, threads, blocks, mkt::cuda_streams[0], 0);

  // fold on gpus: step 2
  while(blocks > 1){
    int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
    int blocks_2 = (blocks + threads_2 - 1) / threads_2;
    mkt::kernel::reduce_min_call(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
    blocks = blocks_2;
  }

  // copy final sum from device to host
  cudaMemcpyAsync(&local_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
  mkt::sync_streams();
  cudaFree(d_odata);

  return local_result;
}

__global__ void setup_rand_kernel(curandState * state, unsigned long seed) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  curand_init(seed, id, 0, &state[id]);
//	curand_init(1234, id, 0, &state[id]);
}


int main(int argc, char** argv) {
  mkt::init();

  char *n_iterationschar = argv[1];
  int n_iterations = atoi(n_iterationschar);
  char *problemchar = argv[2];
  int problem = atoi(problemchar);
  char *antschar = argv[3];
  int ants = atoi(antschar);

  randoms = new Randoms(15);
  std::ifstream fileReader;

  //Problem Instances
  std::string file_to_read = "";

  //Problem Instances
  //std::string f60 = "/home/n/n_herr03/BPP/BPP/source/bpp/Falkenauer_t60_00.txt";
  //std::string p201 = "/home/n/n_herr03/BPP/BPP/source/bpp/201_2500_NR_0.txt";
  //std::string p402 = "/home/n/n_herr03/BPP/BPP/source/bpp/402_10000_NR_0.txt";
  //std::string p600 = "/home/n/n_herr03/BPP/BPP/source/bpp/600_20000_NR_0.txt";
  //std::string p801 = "/home/n/n_herr03/BPP/BPP/source/bpp/801_40000_NR_0.txt";
  //std::string p1002 = "/home/n/n_herr03/BPP/BPP/source/bpp/1002_80000_NR_0.txt";


//if(PALMA){-
    std::string f60 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/MusketProgram/Falkenauer_t60_00.txt";
    std::string p201 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/MusketProgram/201_2500_NR_0.txt";
    std::string p402 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/MusketProgram/402_10000_NR_0.txt";
    std::string p600 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/MusketProgram/600_20000_NR_0.txt";
    std::string p801 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/MusketProgram/801_40000_NR_0.txt";
    std::string p1002 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/MusketProgram/1002_80000_NR_0.txt";
//+
  //}

  switch(problem){
    case 0:
      fileReader.open(f60, std::ifstream::in);
      break;
    case 1:
      fileReader.open(p201, std::ifstream::in);
      break;
    case 2:
      fileReader.open(p402, std::ifstream::in);
      break;
    case 3:
      fileReader.open(p600, std::ifstream::in);
      break;
    case 4:
      fileReader.open(p801, std::ifstream::in);
      break;
    case 5:
      fileReader.open(p1002, std::ifstream::in);
      break;
    default:
      break;
  }

  if (fileReader.is_open()) {
    fileReader >> itemtypes;
    fileReader >> bin_capacity;
  }
  fileReader.close();
  int pheromone_matrix_size = itemtypes * itemtypes;

  mkt::DArray<double> d_phero(0, pheromone_matrix_size, pheromone_matrix_size, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> d_fitness(0, ants, ants, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_probabilities(0, ants*itemtypes, ants*itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_eta(0, ants*itemtypes, ants*itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_tau(0, ants*itemtypes, ants*itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> bpp_items_weight(0, itemtypes, itemtypes, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> bpp_items_quantity(0, itemtypes, itemtypes, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> copy_bpp_items_quantity(0, itemtypes*ants, itemtypes*ants, 0, 1, 0, 0, mkt::DIST, mkt::COPY);

  curandState* d_rand_states_ind;
  cudaMalloc(&d_rand_states_ind, ants * sizeof(curandState));
  setup_rand_kernel<<<ants, 1, 0>>>(d_rand_states_ind, time(NULL));

  d_fitness.update_devices();
  d_probabilities.update_devices();
  d_eta.update_devices();
  d_tau.update_devices();
  
  double randn;

  for(int j = 0;j<itemtypes;j++){
    for(int k = 0;k<itemtypes;k++){
      randn = randoms -> Uniforme() * TAUMAX;
      d_phero[(j*itemtypes) + k] = randn;
      d_phero[(k*itemtypes) + j] = randn;
    }
  }
  d_phero.update_devices();

  int lines = 0;
  double total = 0.0;
  switch(problem){
    case 0:
      fileReader.open(f60, std::ifstream::in);
      break;
    case 1:
      fileReader.open(p201, std::ifstream::in);
      break;
    case 2:
      fileReader.open(p402, std::ifstream::in);
      break;
    case 3:
      fileReader.open(p600, std::ifstream::in);
      break;
    case 4:
      fileReader.open(p801, std::ifstream::in);
      break;
    case 5:
      fileReader.open(p1002, std::ifstream::in);
      break;
    default:
      break;
  }
  if (fileReader.is_open()) {

    fileReader >> itemtypes;
    fileReader >> bin_capacity;

    while (lines < itemtypes && !fileReader.eof()) {
      double weight;
      double quantity;

      fileReader >> weight;
      fileReader >> quantity;

      bpp_items_weight[lines] = weight;
      bpp_items_quantity[lines] = quantity;
      total+=quantity;

      lines++;
    }
  }
  else{
    printf("\nFile not opened");
  }

  bpp_items_weight.update_devices();
  bpp_items_quantity.update_devices();

  itemcount = total;

  (PRINT && !PALMA)?printf("\nSetup Description"):printf("");
  (PRINT && !PALMA)?printf("\n\tObject Types: %d" , itemtypes):printf("");
  (PRINT && !PALMA)?printf("\n\tObject Total: %d" , itemcount):printf("");
  (PRINT && !PALMA)?printf("\n\tAnts: %d \n\tProblem %d:\n", ants, problem):printf("");

  fileReader.close();
  (PRINT && !PALMA)?printf("\t\t%d itemstypes \n\t\t%d items \n\t\t%d capacity\n\n", itemtypes, itemcount, (bin_capacity)):printf("");

  mkt::sync_streams();

  std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
  int best_fitness = 999999;

  printf("\n%d; %d; %d; %d;", ants, problem,itemtypes,itemcount);

  mkt::DArray<int> d_bins(0, ants*itemcount, ants*itemcount, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
  d_bins.update_devices();

  Copybppitemsquantity_map_index_in_place_array_functor copybppitemsquantity_map_index_in_place_array_functor{bpp_items_quantity};
  Copybppitemsweight_map_index_in_place_array_functor copybppitemsweight_map_index_in_place_array_functor{bpp_items_weight};
  Packing_kernel_map_index_in_place_array_functor packing_kernel_map_index_in_place_array_functor{d_bins, copy_bpp_items_quantity, bpp_items_quantity, d_eta, d_tau, d_probabilities, bpp_items_weight, d_phero, d_rand_states_ind};
  Evaporation_kernel_map_index_in_place_array_functor evaporation_kernel_map_index_in_place_array_functor{d_phero};
  Update_pheromones_kernel_map_index_in_place_array_functor update_pheromones_kernel_map_index_in_place_array_functor{d_fitness, d_bins, d_phero, bpp_items_weight};

  int BLOCK_SIZE = 256;

  int n_blocks = ants / BLOCK_SIZE;
  int n_threads = ants / n_blocks;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  double packt = 0.0;

  mkt::sync_streams();
  std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();

  for(int iterate = 0; ((iterate) < (n_iterations)); iterate++){

    int maxobject = 0;

    packing_kernel_map_index_in_place_array_functor.object_weight = (maxobject);
    packing_kernel_map_index_in_place_array_functor.itemtypess = (itemtypes);
    packing_kernel_map_index_in_place_array_functor.itemcountt = (itemcount);
    packing_kernel_map_index_in_place_array_functor.BETA2 = (BETA);
    packing_kernel_map_index_in_place_array_functor.bin_capacity2 = (bin_capacity);

    mkt::map_index_in_place<int, Packing_kernel_map_index_in_place_array_functor>(d_fitness, packing_kernel_map_index_in_place_array_functor, n_threads, n_blocks);

    evaporation_kernel_map_index_in_place_array_functor.itemtypess = (itemtypes);
    evaporation_kernel_map_index_in_place_array_functor.EVAPORATION2 = (EVAPORATION);
    mkt::map_index_in_place<double, Evaporation_kernel_map_index_in_place_array_functor>(d_phero, evaporation_kernel_map_index_in_place_array_functor, itemtypes, itemtypes);

    int new_best_fitness = mkt::reduce_min<int>(d_fitness);
    if (best_fitness > new_best_fitness) best_fitness = new_best_fitness;
    update_pheromones_kernel_map_index_in_place_array_functor.itemtypee = (itemtypes);
    update_pheromones_kernel_map_index_in_place_array_functor.itemcountt = (itemcount);
    update_pheromones_kernel_map_index_in_place_array_functor.bin_capacity2 = (bin_capacity);

    mkt::map_index_in_place<int, Update_pheromones_kernel_map_index_in_place_array_functor>(d_fitness, update_pheromones_kernel_map_index_in_place_array_functor, n_threads, n_blocks);
    //(PRINT && !PALMA)?printf("\nBest Fitness (Number of bins used) Iteration %d: %d", iterate, best_fitness):printf("");
  }

  mkt::sync_streams();
  std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
  double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();

  if (PRINT & !PALMA) {
    printf("\nResults:");
    printf("\n\tSeconds: %.5f;", complete_seconds);
    printf("\n\tFitness: %d;\n", best_fitness);
  }
  if (PALMA) {
    printf("%.5f; ", complete_seconds);
    printf(" %d;", best_fitness);
  }
  return EXIT_SUCCESS;
}
