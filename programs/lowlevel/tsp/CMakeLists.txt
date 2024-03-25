# Top CMakeLists.txt     
cmake_minimum_required(VERSION 3.5)
project(aco_low_level)

# required macros
include(CheckCXXCompilerFlag)
enable_testing()
find_package(MPI REQUIRED)

# config status messages
message(STATUS "Host System: " ${CMAKE_HOST_SYSTEM})

message(STATUS "compiler id: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "compiler flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "compiler flags debug: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "compiler flags release: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "compiler flags minsizerel: " ${CMAKE_CXX_FLAGS_MINSIZEREL})
message(STATUS "compiler flags relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})


# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

find_package(MPI REQUIRED)
### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)

message(STATUS "mpi compiler: " ${MPI_CXX_COMPILER})
message(STATUS "mpi compile flags: " ${MPI_CXX_COMPILE_FLAGS})
message(STATUS "mpi link flags: " ${MPI_CXX_LINK_FLAGS})
message(STATUS "mpi libraries: " ${MPI_CXX_LIBRARIES})
message(STATUS "mpi include paths: " ${MPI_CXX_INCLUDE_PATH})

find_package(OpenMP)

if(OPENMP_FOUND)
    ### output found OpenMP CXX compiler flags
    message(STATUS "OpenMP CXX compiler flags: " ${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

find_package(CUDA)

## set compiler for CUDA Host Code
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

## set compiler flags for SEQ, CPU, PHI, and GPU

# check supported flags
# common
CHECK_CXX_COMPILER_FLAG(-Wall compiler_flag_wall)
CHECK_CXX_COMPILER_FLAG(-march=native compiler_flag_marchnative)
CHECK_CXX_COMPILER_FLAG(-fno-strict-aliasing compiler_flag_fno_strict_aliasing)
CHECK_CXX_COMPILER_FLAG(-DMPICH_IGNORE_CXX_SEEK compiler_flag_dmpich_ignore_cxx_seek)

# seq
CHECK_CXX_COMPILER_FLAG(-Wno-unknown-pragmas compiler_flag_wno_unknown_pragmas)

# phi
CHECK_CXX_COMPILER_FLAG(-Wno-reorder compiler_flag_wno_reorder)
CHECK_CXX_COMPILER_FLAG(-mmic compiler_flag_mmic)
CHECK_CXX_COMPILER_FLAG(-vec-report1 compiler_flag_vec_report1)
CHECK_CXX_COMPILER_FLAG(-DMPICH_SKIP_MPICXX compiler_flag_dmpich_skip_mpicxx)

## determine which variants are built
set(BUILD_SEQ true)

if(OPENMP_FOUND)
    set(BUILD_CPU true)
endif(OPENMP_FOUND)

if(OPENMP_FOUND AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND compiler_flag_mmic)
    set(BUILD_PHI true)
endif(OPENMP_FOUND AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND compiler_flag_mmic)

if(CUDA_FOUND)
    set(BUILD_GPU true)
endif(CUDA_FOUND)


# set the supported flags
if(compiler_flag_wall)
    set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -Wall)
    set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-Wall\"")
endif(compiler_flag_wall)

if(compiler_flag_marchnative)
    set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -march=native)
    set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-march=native\"")
endif(compiler_flag_marchnative)

if(compiler_flag_fno_strict_aliasing)
    set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -fno-strict-aliasing)
    set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-fno-strict-aliasing\"")
endif(compiler_flag_fno_strict_aliasing)

if(compiler_flag_dmpich_ignore_cxx_seek)
    set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -DMPICH_IGNORE_CXX_SEEK)
    set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-DMPICH_IGNORE_CXX_SEEK\"")
endif(compiler_flag_dmpich_ignore_cxx_seek)

# seq
if(compiler_flag_wno_unknown_pragmas)
    set(SEQ_COMPILER_OPTIONS ${SEQ_COMPILER_OPTIONS} -Wno-unknown-pragmas)
endif(compiler_flag_wno_unknown_pragmas)

# Phi
if(compiler_flag_wno_reorder)
    set(PHI_COMPILER_OPTIONS ${PHI_COMPILER_OPTIONS} -Wno-reorder)
endif(compiler_flag_wno_reorder)

if(compiler_flag_mmic)
    set(PHI_COMPILER_OPTIONS ${PHI_COMPILER_OPTIONS} -mmic)
endif(compiler_flag_mmic)

if(compiler_flag_vec_report1)
    set(PHI_COMPILER_OPTIONS ${PHI_COMPILER_OPTIONS} -vec-report1)
endif(compiler_flag_vec_report1)

if(compiler_flag_dmpich_skip_mpicxx)
    set(PHI_COMPILER_OPTIONS ${PHI_COMPILER_OPTIONS} -DMPICH_SKIP_MPICXX)
endif(compiler_flag_dmpich_skip_mpicxx)


## special treatment for CUDA
if(BUILD_GPU)
    #include directories and 
    CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/detail ${MPI_CXX_INCLUDE_PATH})

    ## NVCC flags
    SET(CUDA_PROPAGATE_HOST_FLAGS "OFF") # these are manually set via -Xcompiler flag ( -Xcompiler "-flag -flag" )

    set(GPU_COMPILER_OPTIONS "-arch=compute_61 -code=sm_61 --use_fast_math") # C++11 lambdas are still experimental in CUDA 7.5. This flag may be removed with CUDA 8.0 
    set(GPU_COMPILER_OPTIONS_DEBUG "-g;-G;-O0")
    set(GPU_COMPILER_OPTIONS_RELEASE "-O3")
    set(GPU_COMPILER_OPTIONS_MINSIZEREL "-O2")
    set(GPU_COMPILER_OPTIONS_RELWITHDEBINFO "-g;-G;-O2")

    if (OPENMP_FOUND)
        foreach(flag ${OpenMP_CXX_FLAGS})
            string(CONCAT ESCAPED_OPENMP_FLAGS ${ESCAPED_OPENMP_FLAGS} ",\"${flag}\"")
        endforeach(flag ${OpenMP_CXX_FLAGS})
        string(CONCAT GPU_COMPILER_OPTIONS_HOST ${GPU_COMPILER_OPTIONS_HOST} "${ESCAPED_OPENMP_FLAGS}") 
    endif (OPENMP_FOUND)

    string(CONCAT GPU_HOST_COMPILER_OPTIONS_DEBUG "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_DEBUG})
    string(CONCAT GPU_HOST_COMPILER_OPTIONS_RELEASE "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_RELEASE})
    string(CONCAT GPU_HOST_COMPILER_OPTIONS_MINSIZEREL "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_MINSIZEREL})
    string(CONCAT GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_RELWITHDEBINFO})

endif(BUILD_GPU)



##################################################################################################################################
###############   build PSO 
##################################################################################################################################

find_package(Boost COMPONENTS program_options timer)

if(Boost_FOUND)

	message(STATUS "Boost include path: " ${Boost_INCLUDE_DIR})

  if(BUILD_SEQ)
    add_executable(aco_seq_ref ${PROJECT_SOURCE_DIR}/src/aco_seq.cpp ${PROJECT_SOURCE_DIR}/src/aco_seq_algorithm.cpp ${PROJECT_SOURCE_DIR}/src/Randoms.cpp)
    target_compile_options(aco_seq_ref PUBLIC $<$<CONFIG:Debug>:${COMMON_DEBUG_COMPILER_OPTIONS}> $<$<CONFIG:Release>:${COMMON_RELEASE_COMPILER_OPTIONS}> $<$<CONFIG:RelWithDebInfo>:${COMMON_RELWITHDEBINFO_COMPILER_OPTIONS}> $<$<CONFIG:MinSizeRel>:${COMMON_MINSIZEREL_COMPILER_OPTIONS}>)
    target_compile_features(aco_seq_ref PUBLIC cxx_auto_type cxx_lambdas cxx_nullptr cxx_uniform_initialization)
    target_include_directories(aco_seq_ref PRIVATE ${Boost_INCLUDE_DIRS} )
	target_link_libraries(aco_seq_ref ${Boost_LIBRARIES} rt)
  
  endif(BUILD_SEQ)
  
  if(BUILD_CPU)
	add_executable(aco_openmp_ref ${PROJECT_SOURCE_DIR}/src/aco_openmp.cpp ${PROJECT_SOURCE_DIR}/src/aco_openmp_algorithm.cpp)
    	target_compile_options(aco_openmp_ref PUBLIC ${OpenMP_CXX_FLAGS} $<$<CONFIG:Debug>:${COMMON_DEBUG_COMPILER_OPTIONS}> $<$<CONFIG:Release>:${COMMON_RELEASE_COMPILER_OPTIONS}> $<$<CONFIG:RelWithDebInfo>:		${COMMON_RELWITHDEBINFO_COMPILER_OPTIONS}> $<$<CONFIG:MinSizeRel>:${COMMON_MINSIZEREL_COMPILER_OPTIONS}>)
    	target_compile_features(aco_openmp_ref PUBLIC cxx_auto_type cxx_lambdas cxx_nullptr cxx_uniform_initialization)
    	target_include_directories(aco_openmp_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
    	target_link_libraries(aco_openmp_ref ${Boost_LIBRARIES} -fopenmp gomp rt)

add_executable(aco_openmp_v2_ref ${PROJECT_SOURCE_DIR}/src/aco_openmp_v2.cpp ${PROJECT_SOURCE_DIR}/src/aco_openmp_v2_algorithm.cpp)
    	target_compile_options(aco_openmp_v2_ref PUBLIC ${OpenMP_CXX_FLAGS} $<$<CONFIG:Debug>:${COMMON_DEBUG_COMPILER_OPTIONS}> $<$<CONFIG:Release>:${COMMON_RELEASE_COMPILER_OPTIONS}> $<$<CONFIG:RelWithDebInfo>:		${COMMON_RELWITHDEBINFO_COMPILER_OPTIONS}> $<$<CONFIG:MinSizeRel>:${COMMON_MINSIZEREL_COMPILER_OPTIONS}>)
    	target_compile_features(aco_openmp_v2_ref PUBLIC cxx_auto_type cxx_lambdas cxx_nullptr cxx_uniform_initialization)
    	target_include_directories(aco_openmp_v2_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_PATH}>)
    	target_link_libraries(aco_openmp_v2_ref ${Boost_LIBRARIES} -fopenmp gomp rt)
  endif(BUILD_CPU)

if(BUILD_GPU)
#CUDA_ADD_EXECUTABLE(aco_cuda_ref ${PROJECT_SOURCE_DIR}/src/aco_cuda.cu ${PROJECT_SOURCE_DIR}/src/aco_cuda_algorithm.cu OPTIONS ${GPU_COMPILER_OPTIONS} DEBUG ${GPU_COMPILER_OPTIONS_DEBUG} ${GPU_HOST_COMPILER_OPTIONS_DEBUG} RELEASE ${GPU_COMPILER_OPTIONS_RELEASE} ${GPU_HOST_COMPILER_OPTIONS_RELEASE} RELWITHDEBINFO ${GPU_COMPILER_OPTIONS_RELWITHDEBINFO} ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} MINSIZEREL ${GPU_COMPILER_OPTIONS_MINSIZEREL} ${GPU_HOST_COMPILER_OPTIONS_MINSIZEREL})
#target_include_directories(aco_cuda_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
#target_link_libraries(aco_cuda_ref ${Boost_LIBRARIES} rt)

#CUDA_ADD_EXECUTABLE(aco_cuda_v1_ref ${PROJECT_SOURCE_DIR}/src/aco_v1_cuda.cu ${PROJECT_SOURCE_DIR}/src/aco_v1_cuda_algorithm.cu OPTIONS ${GPU_COMPILER_OPTIONS} DEBUG ${GPU_COMPILER_OPTIONS_DEBUG} ${GPU_HOST_COMPILER_OPTIONS_DEBUG} RELEASE ${GPU_COMPILER_OPTIONS_RELEASE} ${GPU_HOST_COMPILER_OPTIONS_RELEASE} RELWITHDEBINFO ${GPU_COMPILER_OPTIONS_RELWITHDEBINFO} ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} MINSIZEREL ${GPU_COMPILER_OPTIONS_MINSIZEREL} ${GPU_HOST_COMPILER_OPTIONS_MINSIZEREL})
#target_include_directories(aco_cuda_v1_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
#target_link_libraries(aco_cuda_v1_ref ${Boost_LIBRARIES} rt)

CUDA_ADD_EXECUTABLE(aco_cuda_v2_ref ${PROJECT_SOURCE_DIR}/src/aco_v2_cuda.cu ${PROJECT_SOURCE_DIR}/src/aco_v2_cuda_algorithm.cu OPTIONS ${GPU_COMPILER_OPTIONS} DEBUG ${GPU_COMPILER_OPTIONS_DEBUG} ${GPU_HOST_COMPILER_OPTIONS_DEBUG} RELEASE ${GPU_COMPILER_OPTIONS_RELEASE} ${GPU_HOST_COMPILER_OPTIONS_RELEASE} RELWITHDEBINFO ${GPU_COMPILER_OPTIONS_RELWITHDEBINFO} ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} MINSIZEREL ${GPU_COMPILER_OPTIONS_MINSIZEREL} ${GPU_HOST_COMPILER_OPTIONS_MINSIZEREL}) 
target_include_directories(aco_cuda_v2_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(aco_cuda_v2_ref ${Boost_LIBRARIES} rt ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
# target_compile_options(aco_cuda_v2_ref ${MPI_CXX_COMPILE_FLAGS} ${COMMON_COMPILER_OPTIONS})

#CUDA_ADD_EXECUTABLE(aco_cuda_v3_ref ${PROJECT_SOURCE_DIR}/src/aco_v3_cuda.cu ${PROJECT_SOURCE_DIR}/src/aco_v3_cuda_algorithm.cu OPTIONS ${GPU_COMPILER_OPTIONS} DEBUG ${GPU_COMPILER_OPTIONS_DEBUG} ${GPU_HOST_COMPILER_OPTIONS_DEBUG} RELEASE ${GPU_COMPILER_OPTIONS_RELEASE} ${GPU_HOST_COMPILER_OPTIONS_RELEASE} RELWITHDEBINFO ${GPU_COMPILER_OPTIONS_RELWITHDEBINFO} ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} MINSIZEREL ${GPU_COMPILER_OPTIONS_MINSIZEREL} ${GPU_HOST_COMPILER_OPTIONS_MINSIZEREL})
#target_include_directories(aco_cuda_v3_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_PATH})
#target_link_libraries(aco_cuda_v3_ref ${Boost_LIBRARIES} rt ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
# target_compile_options(aco_cuda_v3_ref ${MPI_CXX_COMPILE_FLAGS} ${COMMON_COMPILER_OPTIONS})

#CUDA_ADD_EXECUTABLE(mknap_aco_cuda_ref ${PROJECT_SOURCE_DIR}/src/mknap_aco_cuda.cu ${PROJECT_SOURCE_DIR}/src/mknap_aco_cuda_algorithm.cu OPTIONS ${GPU_COMPILER_OPTIONS} DEBUG ${GPU_COMPILER_OPTIONS_DEBUG} ${GPU_HOST_COMPILER_OPTIONS_DEBUG} RELEASE ${GPU_COMPILER_OPTIONS_RELEASE} ${GPU_HOST_COMPILER_OPTIONS_RELEASE} RELWITHDEBINFO ${GPU_COMPILER_OPTIONS_RELWITHDEBINFO} ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} MINSIZEREL ${GPU_COMPILER_OPTIONS_MINSIZEREL} ${GPU_HOST_COMPILER_OPTIONS_MINSIZEREL})
#target_include_directories(mknap_aco_cuda_ref PUBLIC ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
#target_link_libraries(mknap_aco_cuda_ref ${Boost_LIBRARIES} rt)

endif(BUILD_GPU)
endif(Boost_FOUND)

file(COPY ${PROJECT_SOURCE_DIR}/tsplib/djibouti.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/catar.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/luxembourg.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/a280.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/d198.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/d1291.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/lin318.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/pcb442.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/pcb1173.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/pr1002.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/pr2392.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tsplib/rat783.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

