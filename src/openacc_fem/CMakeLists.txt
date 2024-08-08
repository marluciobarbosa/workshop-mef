add_executable(openacc_fem openacc_fem.cpp)

if(USE_NVHPC OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    if(USE_OPENACC_GPU)
        # set_target_properties(openacc_fem_2D PROPERTIES COMPILE_FLAGS "-acc -gpu=ccall  -Minfo=accel" LINK_FLAGS "-acc  -gpu=ccall")
        set_target_properties(openacc_fem PROPERTIES COMPILE_FLAGS "-acc=gpu -gpu=ccall"  LINK_FLAGS "-acc=gpu  -gpu=ccall")
    else()
        set_target_properties(openacc_fem PROPERTIES COMPILE_FLAGS "-acc=multicore" LINK_FLAGS "-acc=multicore")
    endif()
    target_link_libraries(openacc_fem PRIVATE 
        stdc++
        ${EIGEN_TARGET}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
    )
else()
    set_target_properties(openacc_fem PROPERTIES COMPILE_FLAGS "-fopenacc" LINK_FLAGS "-fopenacc")
    target_link_libraries(openacc_fem PRIVATE 
        stdc++
        ${EIGEN_TARGET}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
    )
endif()