add_executable(stdpar_fem stdpar_fem.cpp)
if(USE_NVHPC OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # if(USE_NVHPC)
        if(USE_CUDA)
            set_target_properties(stdpar_fem PROPERTIES COMPILE_FLAGS "-stdpar=gpu"  LINK_FLAGS "-stdpar=gpu")
        else()
            set_target_properties(stdpar_fem PROPERTIES COMPILE_FLAGS "-stdpar=multicore"  LINK_FLAGS "-stdpar=multicore")
        endif()
        target_link_libraries(stdpar_fem PRIVATE stdc++)
else()
    find_package(TBB REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(stdpar_fem PRIVATE 
            TBB::tbb    
            OpenMP::OpenMP_CXX
            ${EIGEN_TARGET}
            ${lapack_libraries}
            ${blas_libraries})

        set_target_properties(stdpar_fem PROPERTIES
            COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
            LINK_OPTIONS "${OpenMP_EXE_LINKER_FLAGS}"
        ) 
    endif()
endif()