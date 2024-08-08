add_executable(cuda_fem cuda_fem.cpp)

enable_language(CUDA)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cuda_fem PRIVATE 
        OpenMP::OpenMP_CXX
        ${EIGEN_TARGET}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES})
else()
        target_link_libraries(cuda_fem PRIVATE 
        ${EIGEN_TARGET}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES})
endif()

set_target_properties(cuda_fem PROPERTIES COMPILE_FLAGS "-cuda" LINK_FLAGS "-cuda")
