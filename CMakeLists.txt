cmake_minimum_required(VERSION 3.10)
project(FEMSolver LANGUAGES CXX C)
cmake_policy(SET CMP0146 NEW)

# Defina o padrão C++ e habilite C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Diretório onde os arquivos de código-fonte estão localizados
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

# Bibliotecas externas
include(CMake/GitUtils.cmake)
set(EXTERNAL_LIBS_LOCATION ${CMAKE_SOURCE_DIR}/libs)
# Criar o diretório libs se ele não existir
file(MAKE_DIRECTORY "${EXTERNAL_LIBS_LOCATION}")

# Opções para o Build
set(USE_CUDA OFF CACHE BOOL "Usar CUDA")
set(USE_OPENMP ON CACHE BOOL "Usar OpenMP")
set(USE_OPENACC ON CACHE BOOL "Usar OpenACC")
set(USE_PTHREAD OFF CACHE BOOL "Usar Pthread")
# Opção para selecionar o compilador NVIDIA HPC SDK
set(USE_NVHPC OFF CACHE BOOL "Usar NVIDIA HPC SDK Compiler")

if(USE_NVHPC)
    set(CMAKE_CXX_COMPILER nvc++)
    set(CMAKE_C_COMPILER nvc)
endif()

# Configurações de Build
if(USE_OPENMP)
  set(Kokkos_ENABLE_OPENMP ON CACHE INTERNAL "Kokkos_ENABLE_OPENMP" FORCE)
  set(Kokkos_ENABLE_PTHREAD OFF CACHE INTERNAL "Kokkos_ENABLE_PTHREAD" FORCE)
  set(Kokkos_ENABLE_SERIAL OFF CACHE INTERNAL "Kokkos_ENABLE_SERIAL" FORCE)
elseif(USE_PTHREAD)
  set(Kokkos_ENABLE_OPENMP OFF CACHE INTERNAL "Kokkos_ENABLE_OPENMP" FORCE)
  set(Kokkos_ENABLE_PTHREAD ON CACHE INTERNAL "Kokkos_ENABLE_PTHREAD" FORCE)
  set(Kokkos_ENABLE_SERIAL OFF CACHE INTERNAL "Kokkos_ENABLE_SERIAL" FORCE)
else()
  set(Kokkos_ENABLE_OPENMP OFF CACHE INTERNAL "Kokkos_ENABLE_OPENMP" FORCE)
  set(Kokkos_ENABLE_PTHREAD OFF CACHE INTERNAL "Kokkos_ENABLE_PTHREAD" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE INTERNAL "Kokkos_ENABLE_SERIAL" FORCE)
endif()

if(USE_CUDA)
  set(Kokkos_ENABLE_CUDA ON CACHE INTERNAL "Kokkos_ENABLE_CUDA" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE INTERNAL "Kokkos_ENABLE_CUDA_LAMBDA" FORCE)
  set(Kokkos_ENABLE_CUDA_LDG_INTRINSIC ON CACHE INTERNAL "Kokkos_ENABLE_CUDA_LDG_INTRINSIC" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE INTERNAL "Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE" FORCE)

  find_package(CUDA QUIET REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -G ") 
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -use_fast_math")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}  -use_fast_math")
endif()

# Inclui Blas como depedencia do projeto
find_package(BLAS QUIET)
find_package(LAPACK QUIET)
set(LAPACK_LIBRARIES lapack)
set(BLAS_LIBRARIES blas)

if(LAPACK_FOUND AND BLAS_FOUND)
  set(lapack_libraries "${LAPACK_LIBRARIES}")
  set(blas_libraries "${BLAS_LIBRARIES}")
  include(CheckFunctionExists)
  check_function_exists(LAPACKE_dgeqrf LAPACKE_WORKS)
endif()

# Inclui OpenMP como depedencia do projeto
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OpenMP_FOUND)

# Incluir CUDA como depedencia do projeto, se possível
find_package(CUDA QUIET)
if(CUDA_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CUDA_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CUDA_EXE_LINKER_FLAGS}")
endif(CUDA_FOUND)

# Inclui Kokkos Projeto como depedencia do projeto
git_clone(
        QUIET  
        PROJECT_NAME    kokkos
        GIT_URL         https://github.com/kokkos/kokkos.git
        GIT_BRANCH      master
        DIRECTORY	    ${EXTERNAL_LIBS_LOCATION}
)
add_subdirectory(${kokkos_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kokkos)
include_directories("${kokkos_SOURCE_DIR}/core/src")
include_directories("${kokkos_SOURCE_DIR}")

# Inclui Eigen como depedencia do projeto
git_clone(
        QUIET
        PROJECT_NAME    eigen
        GIT_URL         https://gitlab.com/libeigen/eigen.git
        GIT_BRANCH      master
        DIRECTORY	    ${EXTERNAL_LIBS_LOCATION}
)
include_directories("${eigen_SOURCE_DIR}")

# Adicione uma função para definir o executável e os arquivos de origem
function(add_fem_executable name src_file)
    add_executable(${name} ${src_file})
    target_include_directories(${name} PRIVATE ${SOURCE_DIR}
        ${lapack_libraries}
        ${blas_libraries})
endfunction()

# Compilar a versão básica 1D
add_fem_executable(basic_fem_1D ${SOURCE_DIR}/1D/basic_fem.cpp)

# Compilar a versão com OpenMP 1D
if(OpenMP_CXX_FOUND)
    add_fem_executable(openmp_fem_1D ${SOURCE_DIR}/1D/openmp_fem.cpp)
    target_link_libraries(openmp_fem_1D PRIVATE OpenMP::OpenMP_CXX
        ${lapack_libraries}
        ${blas_libraries})
endif()

# Compilar a versão com CUDA 1D
if(CUDA_FOUND)
    add_executable(cuda_fem_1D ${SOURCE_DIR}/1D/cuda_fem.cpp)
    set_target_properties(cuda_fem_1D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Compilar a versão com Kokkos 1D
add_fem_executable(kokkos_fem_1D ${SOURCE_DIR}/1D/kokkos_fem.cpp)
add_dependencies(kokkos_fem_1D kokkos)
target_link_libraries(kokkos_fem_1D PRIVATE Kokkos::kokkos
    ${lapack_libraries}
    ${blas_libraries})
set_target_properties(kokkos_fem_1D PROPERTIES
    COMPILE_OPTIONS "${KOKKOS_COMPILE_OPTIONS}"
    LINK_OPTIONS "${KOKKOS_LINK_OPTIONS}"
)

# Compilar a versão básica 2D
add_fem_executable(basic_fem_2D ${SOURCE_DIR}/2D/basic_fem.cpp)

# Compilar a versão com OpenMP 2D
add_fem_executable(openmp_fem_2D ${SOURCE_DIR}/2D/openmp_fem.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(openmp_fem_2D PRIVATE OpenMP::OpenMP_CXX
        ${lapack_libraries}
        ${blas_libraries})
endif()

# Compilar a versão com OpenACC 2D
if(USE_OPENACC)
    add_executable(openacc_fem_2D ${SOURCE_DIR}/2D/openacc_fem.cpp)
    if(USE_NVHPC)
        if(CUDA_FOUND)
            set_target_properties(openacc_fem_2D PROPERTIES COMPILE_FLAGS "-acc -ta=tesla:cuda" LINK_FLAGS "-acc -ta=tesla:cuda")
        else()
            set_target_properties(openacc_fem_2D PROPERTIES COMPILE_FLAGS "-acc" LINK_FLAGS "-acc")
        endif()
    else()
        set_target_properties(openacc_fem_2D PROPERTIES COMPILE_FLAGS "-fopenacc" LINK_FLAGS "-fopenacc")
    endif()
endif()

# Compilar a versão com CUDA 2D
if(CUDA_FOUND)
    add_executable(cuda_fem_2D ${SOURCE_DIR}/2D/cuda_fem.cpp)
    set_target_properties(cuda_fem_2D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Compilar a versão com Kokkos 2D
add_fem_executable(kokkos_fem_2D ${SOURCE_DIR}/2D/kokkos_fem.cpp)
add_dependencies(kokkos_fem_2D kokkos)
target_link_libraries(kokkos_fem_2D PRIVATE Kokkos::kokkos
    ${lapack_libraries}
    ${blas_libraries})
set_target_properties(kokkos_fem_2D PROPERTIES
    COMPILE_OPTIONS "${KOKKOS_COMPILE_OPTIONS}"
    LINK_OPTIONS "${KOKKOS_LINK_OPTIONS}"
)