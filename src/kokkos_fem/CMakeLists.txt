# Configurações para o Kokkos
set(Kokkos_ENABLE_CUDA OFF CACHE INTERNAL "Kokkos_ENABLE_CUDA" FORCE)
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

set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION OFF)

if(USE_CUDA)
  set(Kokkos_ENABLE_CUDA ON CACHE INTERNAL "Kokkos_ENABLE_CUDA" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE INTERNAL "Kokkos_ENABLE_CUDA_LAMBDA" FORCE)
  #set(Kokkos_ENABLE_CUDA_LDG_INTRINSIC ON CACHE INTERNAL "Kokkos_ENABLE_CUDA_LDG_INTRINSIC" FORCE) # depreciado
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OFF CACHE INTERNAL "Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE" FORCE)

  # find_package(CUDA QUIET REQUIRED)
  #message(STATUS ">>>>>>>>>>>>>>>>> ********************** >>>>>>>>>>>>>>>>>>> ${CUDA_INCLUDE_DIRS}")
  #include_directories(${CUDA_INCLUDE_DIRS})
#   include_directories("/usr/local/cuda/include")
#   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -G ") 
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -use_fast_math")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}  -use_fast_math")
endif()

# Incluir e configurar Kokkos
git_clone(
    QUIET  
    PROJECT_NAME kokkos
    GIT_URL https://github.com/kokkos/kokkos.git
    GIT_BRANCH master
    DIRECTORY ${EXTERNAL_LIBS_LOCATION}
)
add_subdirectory(${EXTERNAL_LIBS_LOCATION}/kokkos ${CMAKE_BINARY_DIR}/kokkos)
include_directories(${EXTERNAL_LIBS_LOCATION}/kokkos/core/src)


add_executable(kokkos_fem kokkos_fem.cpp)
add_dependencies(kokkos_fem kokkos)
target_link_libraries(kokkos_fem PRIVATE 
                        Kokkos::kokkos
                        ${EIGEN_TARGET}
                        ${LAPACK_LIBRARIES}
                        ${BLAS_LIBRARIES}
                    )
set_target_properties(kokkos_fem PROPERTIES
                    COMPILE_OPTIONS "${KOKKOS_COMPILE_OPTIONS}"
                    LINK_OPTIONS "${KOKKOS_LINK_OPTIONS}"
                )                    