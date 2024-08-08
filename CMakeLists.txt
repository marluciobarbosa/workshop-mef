cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(FEMSolver LANGUAGES CXX)
cmake_policy(SET CMP0146 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define o diretório de saída para os binários
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Diretórios de origem e libs externas
set(EXTERNAL_LIBS_LOCATION ${CMAKE_SOURCE_DIR}/libs)
file(MAKE_DIRECTORY "${EXTERNAL_LIBS_LOCATION}")

# Opções de Build
option(USE_KOKKOS "Usar Kokkos" ON)
option(USE_CUDA "Usar CUDA" ON)
option(USE_OPENMP "Usar OpenMP" ON)
option(USE_OPENACC "Usar OpenACC" ON)
option(USE_OPENACC_GPU "Usar OpenACC" OFF)
option(USE_PTHREAD "Usar Pthread" OFF)
option(USE_NVHPC "Usar NVIDIA HPC SDK Compiler" OFF)

# Configuração do compilador para NVIDIA HPC SDK, se necessário
if(USE_NVHPC)
    set(CMAKE_CXX_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvc++)
    set(CMAKE_C_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvc)
endif()

# Incluir GitUtils
include(CMake/GitUtils.cmake)

# Incluir e configurar Eigen
find_package(Eigen3 3.4 QUIET)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    set(EIGEN_TARGET Eigen3::Eigen)
else()
    set(EIGEN_TARGET "")
    git_clone(
        QUIET
        PROJECT_NAME    eigen
        GIT_URL         https://gitlab.com/libeigen/eigen.git
        GIT_BRANCH      3.4
        DIRECTORY	    ${EXTERNAL_LIBS_LOCATION}
    )
    include_directories("${eigen_SOURCE_DIR}")
endif()

# Detectar e configurar BLAS e LAPACK
find_package(BLAS QUIET)
find_package(LAPACK QUIET)
if(LAPACK_FOUND AND BLAS_FOUND)
  set(LAPACK_LIBRARIES lapack)
  set(BLAS_LIBRARIES blas)
  include(CheckFunctionExists)
  check_function_exists(LAPACKE_dgeqrf LAPACKE_WORKS)
endif()

# Inclusão de Threads
if(USE_PTHREAD)
    find_package(Threads QUIET)
endif()

# Inclusão de OpenMP
if(USE_OPENMP)
    find_package(OpenMP QUIET)
endif()

# Inclusão de OpenACC
if(USE_OPENACC)
    find_package(OpenACC REQUIRED)
endif()

# Adicionar subdiretórios para cada componente
add_subdirectory(src/basic_fem)
add_subdirectory(src/openmp_fem)
add_subdirectory(src/openacc_fem)
# add_subdirectory(src/cuda_fem)
add_subdirectory(src/kokkos_fem)
add_subdirectory(src/stdpar)

# Inclui sub-diretórios condicionalmente
# add_subdirectory(src/basic_fem)
# add_subdirectory(src/stdpar)
# if(USE_OPENMP)
#   add_subdirectory(src/openmp_fem)
# endif()
# if(USE_OPENACC)
#   add_subdirectory(src/openacc_fem)
# endif()
# if(USE_CUDA)
#   add_subdirectory(src/cuda_fem)
# endif()
# if(USE_KOKKOS)
#   add_subdirectory(src/kokkos_fem)
# endif()