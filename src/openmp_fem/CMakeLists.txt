add_executable(openmp_fem openmp_fem.cpp)

target_link_libraries(openmp_fem PRIVATE 
                        OpenMP::OpenMP_CXX
                        ${EIGEN_TARGET}
                        ${LAPACK_LIBRARIES}
                        ${BLAS_LIBRARIES}
                    )

set_target_properties(openmp_fem PROPERTIES
        COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
        LINK_OPTIONS "${OpenMP_EXE_LINKER_FLAGS}"
    )                    