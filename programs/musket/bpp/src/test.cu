#include <cuda.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <array>
#include <vector>
#include <sstream>
#include <fstream>
#include <boost/program_options.hpp>

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
//namespace bpo = boost::program_options;
//bpo::variables_map g_prog_options;

const int ants = 256;
const int BETA = 1;
const double EVAPORATION = 0.5;
const double PHERINIT = 0.005;
const int ALPHA = 1;
const int TAUMAX = 2;
const int BLOCK_SIZE = 32;
const int Q = 32;
int itemtypes = 50;
int itemcount = 59;
const int pheromone_matrix_size = 2500;
const int antssquaredtimes = 12800;
const int antssquaredcount = 15104;
auto bin_capacity = 1000;
const int itemtypesdoubled = 100;

struct Packing_kernel_map_index_in_place_array_functor{

    Packing_kernel_map_index_in_place_array_functor(const mkt::DArray<double>& _d_bpp_items_copy, const mkt::DArray<double>& _bpp_items, const mkt::DArray<int>& _d_bins, const mkt::DArray<double>& _d_eta, const mkt::DArray<double>& _d_tau, const mkt::DArray<double>& _d_probabilities, const mkt::DArray<double>& _d_phero) : d_bpp_items_copy(_d_bpp_items_copy), bpp_items(_bpp_items), d_bins(_d_bins), d_eta(_d_eta), d_tau(_d_tau), d_probabilities(_d_probabilities), d_phero(_d_phero){}

    ~Packing_kernel_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int iindex, int y){
      int ant_index = (iindex);
      int object_bin_index = ((ant_index) * (itemtypes));
      int bins_used = 0;
      for(int i = 0; ((i) < (static_cast<int>(2) * (itemtypess))); i++){
        int indexx = (((static_cast<int>((ant_index)) * (itemtypess)) * 2) + (i));
        d_bpp_items_copy.set_global((indexx), bpp_items.get_global((i))/* TODO: For multiple GPUs*/);
      }
      double actual_bin_weight = 0.0;
      double n_items_in_actual_bin = 0.0;
      int possible_items_to_this_bin = 0;
      int bpp_items_prefix = ((static_cast<int>((ant_index)) * (itemtypess)) * 2);
      int object_index = 0;
      double object_weight = 0.0;
      double object_quantity = 0.0;
      double new_object_weight = 0.0;
      for(int i = 0; ((i) < (itemtypes)); i++){
        new_object_weight = d_bpp_items_copy.get_global(((bpp_items_prefix) + (2 * (i))))/* TODO: For multiple GPUs*/;
        object_quantity = d_bpp_items_copy.get_global((((bpp_items_prefix) + (2 * (i))) + 1))/* TODO: For multiple GPUs*/;

        if(((object_quantity) > 0)){

          if(((new_object_weight) > (object_weight))){
            object_index = (i);
            object_weight = (new_object_weight);
          }
        }
      }
      d_bins.set_global(((ant_index) * static_cast<int>((itemtypes))), (object_index));
      d_bpp_items_copy.set_global((((bpp_items_prefix) + ((object_index) * 2)) + 1), (d_bpp_items_copy.get_global((((bpp_items_prefix) + ((object_index) * 2)) + 1))/* TODO: For multiple GPUs*/ - 1));
      n_items_in_actual_bin = ((n_items_in_actual_bin) + 1);
      actual_bin_weight += (object_weight);
      bins_used = ((bins_used) + 1);
      double weight_object_j = 0.0;
      for(int i = 0; ((i) < ((itemcountt) - 1)); i++){
        double eta_tau_sum = 0.0;
        for(int j = 0; ((j) < (itemtypess)); j++){
          d_eta.set_global(((object_bin_index) + (j)), 0.0);
          d_tau.set_global(((object_bin_index) + (j)), 0.0);
          d_probabilities.set_global(((object_bin_index) + (j)), 0.0);
          weight_object_j = d_bpp_items_copy.get_global(((bpp_items_prefix) + (2 * (j))))/* TODO: For multiple GPUs*/;
          double quantity_object_j = d_bpp_items_copy.get_global((((bpp_items_prefix) + (2 * (j))) + 1))/* TODO: For multiple GPUs*/;

          if(((quantity_object_j) > 0)){

            if(((weight_object_j) < ((bin_capacity) - (actual_bin_weight)))){

              if(((actual_bin_weight) == 0)){
                d_eta.set_global(((object_bin_index) + (j)), 1.0);
              }
              else {
                for(int k = 0; ((k) < (n_items_in_actual_bin)); k++){
                  int object_i = d_bins.get_global((((object_bin_index) + (i)) - (k)))/* TODO: For multiple GPUs*/;
                  d_eta.set_global(((object_bin_index) + (j)), d_phero.get_global((((object_i) * static_cast<int>((itemtypes))) + (j)))/* TODO: For multiple GPUs*/);
                }
                d_eta.set_global(((object_bin_index) + (j)), (d_eta.get_global(((object_bin_index) + (j)))/* TODO: For multiple GPUs*/ / (n_items_in_actual_bin)));
              }
              d_tau.set_global(((object_bin_index) + (j)), static_cast<double>(__powf((weight_object_j), (BETA))));
              eta_tau_sum += (d_eta.get_global(((object_bin_index) + (j)))/* TODO: For multiple GPUs*/ * d_tau.get_global(((object_bin_index) + (j)))/* TODO: For multiple GPUs*/);
              possible_items_to_this_bin = ((possible_items_to_this_bin) + 1);
            }
          }
        }

        if(((possible_items_to_this_bin) > 0)){
          for(int j = 0; ((j) < (itemtypes)); j++){
            d_probabilities.set_global(((object_bin_index) + (j)), ((d_eta.get_global(((object_bin_index) + (j)))/* TODO: For multiple GPUs*/ * d_tau.get_global(((object_bin_index) + (j)))/* TODO: For multiple GPUs*/) / (eta_tau_sum)));
            d_eta.set_global(((object_bin_index) + (j)), 0.0);
            d_tau.set_global(((object_bin_index) + (j)), 0.0);
          }
          eta_tau_sum = 0.0;
          double random = 0.0;
          int object_j = 0;
          double sum = d_probabilities.get_global((object_bin_index))/* TODO: For multiple GPUs*/;
          for(int s = 0; ((s) > -(1)); s++){
            object_j = ((object_j) + 1);
            sum = ((sum) + d_probabilities.get_global(((object_bin_index) + (object_j)))/* TODO: For multiple GPUs*/);

            if(((sum) < (random))){
              s = -(2);
            }
          }
          d_bins.set_global(((((ant_index) * static_cast<int>((itemtypes))) + (i)) + 1), (object_j));
          weight_object_j = d_bpp_items_copy.get_global(((bpp_items_prefix) + (2 * (object_j))))/* TODO: For multiple GPUs*/;
          actual_bin_weight += (weight_object_j);
          d_bpp_items_copy.set_global((((bpp_items_prefix) + (2 * (object_j))) + 1), (d_bpp_items_copy.get_global((((bpp_items_prefix) + (2 * (object_j))) + 1))/* TODO: For multiple GPUs*/ - 1));
          n_items_in_actual_bin = ((n_items_in_actual_bin) + 1);
          possible_items_to_this_bin = 0;
        }
        else {
          possible_items_to_this_bin = 0;
          actual_bin_weight = 0.0;
          actual_bin_weight = 0.0;
          object_index = 0;
          object_weight = 0.0;
          object_quantity = 0.0;
          new_object_weight = 0.0;
          for(int k = 0; ((k) < (itemtypes)); k++){
            object_quantity = d_bpp_items_copy.get_global((((bpp_items_prefix) + (2 * (k))) + 1))/* TODO: For multiple GPUs*/;
            new_object_weight = d_bpp_items_copy.get_global(((bpp_items_prefix) + (2 * (k))))/* TODO: For multiple GPUs*/;

            if(((object_quantity) > 0)){

              if(((new_object_weight) > (object_weight))){
                object_index = (k);
                object_weight = (new_object_weight);
              }
            }
          }
          d_bpp_items_copy.set_global((((bpp_items_prefix) + ((object_index) * 2)) + 1), (d_bpp_items_copy.get_global((((bpp_items_prefix) + ((object_index) * 2)) + 1))/* TODO: For multiple GPUs*/ - 1));
          d_bins.set_global(((((ant_index) * static_cast<int>((itemtypes))) + (i)) + 1), (object_index));
          n_items_in_actual_bin = (n_items_in_actual_bin);
          actual_bin_weight += (object_weight);
          bins_used = ((bins_used) + 1);
        }
      }
      return (bins_used);
    }

    void init(int device){
      d_bpp_items_copy.init(device);
      bpp_items.init(device);
      d_bins.init(device);
      d_eta.init(device);
      d_tau.init(device);
      d_probabilities.init(device);
      d_phero.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }

    int itemtypess;
    int itemcountt;
    int BETA2;
    int bin_capacity2;

    mkt::DeviceArray<double> d_bpp_items_copy;
    mkt::DeviceArray<double> bpp_items;
    mkt::DeviceArray<int> d_bins;
    mkt::DeviceArray<double> d_eta;
    mkt::DeviceArray<double> d_tau;
    mkt::DeviceArray<double> d_probabilities;
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

    Update_pheromones_kernel_map_index_in_place_array_functor(const mkt::DArray<int>& _d_fitness, const mkt::DArray<int>& _d_bins, const mkt::DArray<double>& _d_phero) : d_fitness(_d_fitness), d_bins(_d_bins), d_phero(_d_phero){}

    ~Update_pheromones_kernel_map_index_in_place_array_functor() {}

    __device__
    auto operator()(int iindex, int value){
      int ant_index = (iindex);
      double ant_fitness = (d_fitness.get_global((ant_index))/* TODO: For multiple GPUs*/ * 1.0);
      double actual_bin_weight = 0.0;
      int actual_bin_object_index = 0;
      int actual_bin_n_objects = 0;
      for(int i = 0; ((i) < (itemcountt)); i++){
        double object_weight = static_cast<double>(d_bins.get_global((((ant_index) * (itemcountt)) + (i)))/* TODO: For multiple GPUs*/);

        if((((actual_bin_weight) + (object_weight)) < (bin_capacity2))){
          actual_bin_n_objects = ((actual_bin_n_objects) + 1);
          actual_bin_weight = ((actual_bin_weight) + (object_weight));
        }
        else {
          for(int j = 0; ((j) < (actual_bin_n_objects)); j++){
            for(int k = ((j) + 1); ((k) < (actual_bin_n_objects)); k++){
              int object_i = d_bins.get_global(((((ant_index) * (itemcountt)) + (actual_bin_object_index)) + (j)))/* TODO: For multiple GPUs*/;
              int object_j = d_bins.get_global(((((ant_index) * (itemcountt)) + (actual_bin_object_index)) + (k)))/* TODO: For multiple GPUs*/;
              double delta_pheromone = ((Q) / (d_fitness.get_global((ant_index))/* TODO: For multiple GPUs*/ * 1.0));
              d_phero.set_global((((object_i) * (itemcountt)) + (object_j)), ((delta_pheromone) + d_phero.get_global((((object_i) * (itemcountt)) + (object_j)))/* TODO: For multiple GPUs*/));
              d_phero.set_global((((object_j) * (itemcountt)) + (object_i)), ((delta_pheromone) + d_phero.get_global((((object_j) * (itemcountt)) + (object_i)))/* TODO: For multiple GPUs*/));
            }
          }
          actual_bin_n_objects = 1;
          actual_bin_weight = (object_weight);
          actual_bin_object_index = (i);
        }
      }
      return (value);
    }

    void init(int device){
      d_fitness.init(device);
      d_bins.init(device);
      d_phero.init(device);
    }

    size_t get_smem_bytes(){
      size_t result = 0;
      return result;
    }

    int itemcountt;
    int bin_capacity2;

    mkt::DeviceArray<int> d_fitness;
    mkt::DeviceArray<int> d_bins;
    mkt::DeviceArray<double> d_phero;
};

template<>
int mkt::reduce_min<int>(mkt::DArray<int>& a){
  double local_result = std::numeric_limits<double>::max();

  const int gpu_elements = a.get_size_gpu();
  int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
  int blocks = (gpu_elements + threads - 1) / threads;
  cudaSetDevice(0);
  double* d_odata;
  cudaMalloc((void**) &d_odata, blocks * sizeof(double));
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
  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
  mkt::sync_streams();
  cudaFree(d_odata);

  return local_result;
}

int main(int argc, char** argv) {
  mkt::init();


  mkt::sync_streams();
  std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
  //int runs = g_prog_options["runs"].as<int>();
  //int n_iterations = g_prog_options["iterations"].as<int>();
  //int problem = g_prog_options["problem"].as<int>();
  int problem = 0;

  randoms = new Randoms(15);

  printf("\n\nReading BPP File");

  std::ifstream fileReader;

  //Problem Instances
  std::string file_to_read = "";

  //Problem Instances
  std::string f60 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/source/bpp/Falkenauer_t60_00.txt";
  std::string p201 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/source/bpp/201_2500_NR_0.txt";
  std::string p402 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/source/bpp/402_10000_NR_0.txt";
  std::string p600 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/source/bpp/600_20000_NR_0.txt";
  std::string p801 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/source/bpp/801_40000_NR_0.txt";
  std::string p1002 = "/home/schredder/Research/HLPP/2020/ACO_Breno/BPP/source/bpp/1002_80000_NR_0.txt";


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
  int pheromone_matrix_size = itemtypes * itemtypes;

  mkt::DArray<double> d_phero(0, pheromone_matrix_size, pheromone_matrix_size, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_delta_phero(0, pheromone_matrix_size, pheromone_matrix_size, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> d_fitness(0, ants, ants, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_probabilities(0, ants*itemtypes, ants*itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_eta(0, ants*itemtypes, ants*itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_tau(0, ants*itemtypes, ants*itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> bpp_items(0, itemtypes, itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_bpp_items_copy(0, itemtypes, itemtypes, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> d_bins(0, ants*itemtypes, ants*itemtypes, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<int> d_best_solution(0, itemtypes, itemtypes, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
  mkt::DArray<double> d_rand_states_ind(0, ants, ants, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);

  double randn = 0.0;

  for(int j = 0;j<itemtypes;j++){
    for(int k = 0;k<itemtypes;k++){
      if(j!= k){
        randn = randoms -> Uniforme() * TAUMAX;
        d_phero[(j*itemtypes) + k] = randn;
        d_phero[(k*itemtypes) + j] = randn;
      }
      else{
        d_phero[(j*itemtypes) + k] = 0.0;
        d_phero[(k*itemtypes) + j] = 0.0;
      }
    }
  }
  d_phero.update_devices();

  Packing_kernel_map_index_in_place_array_functor packing_kernel_map_index_in_place_array_functor{d_bpp_items_copy, bpp_items, d_bins, d_eta, d_tau, d_probabilities, d_phero};
  Evaporation_kernel_map_index_in_place_array_functor evaporation_kernel_map_index_in_place_array_functor{d_phero};
  Update_pheromones_kernel_map_index_in_place_array_functor update_pheromones_kernel_map_index_in_place_array_functor{d_fitness, d_bins, d_phero};

  int lines = 0;
  double total = 0.0;

  if (fileReader.is_open()) {

    fileReader >> itemtypes;
    fileReader >> bin_capacity;
    while (lines < itemtypes && !fileReader.eof()) {
      double weight;
      double quantity;

      fileReader >> weight;
      fileReader >> quantity;

      bpp_items[lines*2] = weight;
      bpp_items[lines*2+1] = quantity;
      (lines < 2) ? printf("%d weight %d quantity\n", weight, quantity): printf("");

      total+=quantity;

      lines++;
    }
  }
  else{
    printf("\nFile not opened");
  }
  /*for (int o = 0; o < 100; o++) {
    printf("%.2f; ", bpp_items[o]);
    if (o % 32 == 0) {printf("\n");}
  }*/
  itemcount = total;

  printf("\nObject Types: %d" , itemtypes);
  printf("\nObject Total: %d" , itemcount);

  fileReader.close();
  printf("\n\nProcessing %d itemstypes %d items %d capacity \n\n", itemtypes, itemcount, (bin_capacity));

  mkt::sync_streams();
  std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
  double best_fitness = 999999.9;
  double mean_times = 0.0;
  int n_iterations = 2;
  for(int iterate = 0; ((iterate) < (n_iterations)); iterate++){
    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
    packing_kernel_map_index_in_place_array_functor.itemtypess = (itemtypes);
    packing_kernel_map_index_in_place_array_functor.itemcountt = (itemcount);
    packing_kernel_map_index_in_place_array_functor.BETA2 = (BETA);
    packing_kernel_map_index_in_place_array_functor.bin_capacity2 = (bin_capacity);
    mkt::map_index_in_place<int, Packing_kernel_map_index_in_place_array_functor>(d_fitness, packing_kernel_map_index_in_place_array_functor);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    /*d_fitness.update_self();
    printf("d_fitness");
		for (int o = 0; o < 100; o++) {
      if (o % 32 == 0) {printf("\n");}
      printf("%.2f; ", d_fitness[o]);
    }*/

    evaporation_kernel_map_index_in_place_array_functor.itemtypess = (itemtypes);
    evaporation_kernel_map_index_in_place_array_functor.EVAPORATION2 = (EVAPORATION);
    mkt::map_index_in_place<double, Evaporation_kernel_map_index_in_place_array_functor>(d_phero, evaporation_kernel_map_index_in_place_array_functor);
    d_fitness.update_self();
    printf("d_fitness");
    for (int o = 0; o < 100; o++) {
      if (o % 32 == 0) {printf("\n");}
      printf("%.2f; ", d_fitness[o]);
    }
    update_pheromones_kernel_map_index_in_place_array_functor.itemcountt = (itemcount);
    update_pheromones_kernel_map_index_in_place_array_functor.bin_capacity2 = (bin_capacity);
    mkt::map_index_in_place<int, Update_pheromones_kernel_map_index_in_place_array_functor>(d_fitness, update_pheromones_kernel_map_index_in_place_array_functor);

    best_fitness = mkt::reduce_min<int>(d_fitness);
    /*d_fitness.update_self();
    for (int o = 0; o < 256; o++) {
      printf("%.2f; ", d_fitness[o]);
      if (o % 32 == 0) {printf("\n");}
    }*/
    mkt::sync_streams();
    std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
  }
  mkt::sync_streams();
  std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration<double>(timer_end - timer_start).count();

  mkt::sync_streams();
  std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
  double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();
  printf("Complete execution time: %.5fs\n", complete_seconds);
  printf("Best Fitness: %.5f\n\n", best_fitness);

  printf("Execution time: %.5fs\n", seconds);
  printf("Threads: %i\n", 1);
  printf("Processes: %i\n", 1);

  return EXIT_SUCCESS;
}
