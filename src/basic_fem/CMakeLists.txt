add_executable(basic_fem basic_fem.cpp)
target_link_libraries(basic_fem PRIVATE 
                        ${EIGEN_TARGET}
                        ${LAPACK_LIBRARIES}
                        ${BLAS_LIBRARIES}
                    )