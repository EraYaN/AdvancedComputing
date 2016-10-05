### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
    )

# Specify include directories
include_directories(
    ../../tclap-1.2.1/include
    ../../date-master
	../acs-shared
    )

# Specify library paths
link_directories(
    ../out
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    benchmark
    main.cpp
    cuda/cuda-constrast.cu
    cuda/cuda-histogram.cu
	cuda/cuda-grayscale.cu
	cuda/cuda-smooth.cu
	cpu/cpu-constrast.cu
    cpu/cpu-histogram.cu
	cpu/cpu-grayscale.cu
	cpu/cpu-smooth.cu
    checkCudaCall.h
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    benchmark
    -lacs-shared
)