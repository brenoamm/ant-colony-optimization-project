cmake_minimum_required(VERSION 3.10)
project(aco_iroulette VERSION 1.0.0 LANGUAGES CXX CUDA)

SET(CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -Wall -Wextra -DMPICH_IGNORE_CXX_SEEK -std=c++14")
SET(CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -fopt-info-vec-optimized -Wall -Wextra -DMPICH_IGNORE_CXX_SEEK -std=c++14 " )
SET(CMAKE_CXX_FLAGS_NVPROF "-O3 -g -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14")
SET(CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14")
SET(CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14")
SET(CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-O3 -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK -std=c++14")

set(CMAKE_CUDA_HOST_FLAGS " -Xcompiler ")
set(CMAKE_CUDA_HOST_LINKER_FLAGS " -Xlinker ")

# packages
find_package(OpenMP REQUIRED)
foreach (flag ${OpenMP_CXX_FLAGS})
	string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
	string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
endforeach (flag ${OpenMP_CXX_FLAGS})

# append host flags to "normal" flags
string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_FLAGS})
string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_LINKER_FLAGS})

SET( CMAKE_CUDA_FLAGS_DEV "-g -G -O0 -arch=compute_61 -code=sm_61 -use_fast_math -restrict -Xptxas -O0 -Xcompiler -O0,-g,-march=native,-Wall,-Wextra,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
SET( CMAKE_CUDA_FLAGS_TEST "-g -G -O0 -arch=compute_61 -code=sm_61 -use_fast_math -restrict -Xptxas -O0 -Xcompiler -O3,-g,-march=native,-fopt-info-vec-optimized,-Wall,-Wextra,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
SET( CMAKE_CUDA_FLAGS_NVPROF "-g -G -pg -O3 -arch=compute_50 -code=sm_50 -use_fast_math -w -restrict -Xcompiler -O3,-g,-DNDEBUG,-march=native,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
SET( CMAKE_CUDA_FLAGS_BENCHMARK "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3" )
SET( CMAKE_CUDA_FLAGS_BENCHMARKPALMA "-O3 -arch=compute_35 -code=sm_35" )
SET( CMAKE_CUDA_FLAGS_BENCHMARKTAURUS "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=haswell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")	

# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

add_executable(aco_iroulette_0 ${PROJECT_SOURCE_DIR}/src/aco_iroulette_0.cu)
target_compile_features(aco_iroulette_0 PRIVATE cxx_std_14)
target_include_directories(aco_iroulette_0 PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(aco_iroulette_0 PRIVATE)
target_compile_options(aco_iroulette_0 PRIVATE )
target_link_libraries(aco_iroulette_0 PRIVATE ${OpenMP_CXX_LIBRARIES})
