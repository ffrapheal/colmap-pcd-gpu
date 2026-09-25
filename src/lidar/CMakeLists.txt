set(FOLDER_NAME "lidar")

set(LIDAR_SOURCES
    lidar_point.h lidar_point.cc
    pt_type.h
    pcd_projection.h pcd_projection.cc
    kdtree.h kdtree.cc
    ply.h ply.cc
)

if(GPU_BA_CUDA_ENABLED)
    list(APPEND LIDAR_SOURCES
        incremental_causal_lidar_map.h
        incremental_causal_lidar_map.cc)
endif()

COLMAP_ADD_SOURCES(${LIDAR_SOURCES})

COLMAP_ADD_TEST(pcd_projection_test pcd_projection_test.cc)

if(GPU_BA_CUDA_ENABLED)
    add_library(colmap_cuda_lidar_normals_lib STATIC
                cuda_normal_estimation.h
                cuda_normal_estimation.cu)
    set_target_properties(colmap_cuda_lidar_normals_lib PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    target_link_libraries(colmap_cuda_lidar_normals_lib PUBLIC CUDA::cudart)

    add_executable(colmap_cuda_lidar_normals
                   cuda_normal_estimation_main.cc)
    set_target_properties(colmap_cuda_lidar_normals PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(colmap_cuda_lidar_normals
                          PRIVATE colmap colmap_cuda_lidar_normals_lib)

    if(TESTS_ENABLED)
        add_executable(incremental_causal_lidar_map_test
                       incremental_causal_lidar_map_test.cc)
        set_target_properties(incremental_causal_lidar_map_test PROPERTIES
            FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(incremental_causal_lidar_map_test
                              PRIVATE colmap colmap_cuda_lidar_normals_lib
                                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("${FOLDER_NAME}/incremental_causal_lidar_map_test"
                 incremental_causal_lidar_map_test)

        add_executable(cuda_normal_estimation_test
                       cuda_normal_estimation_test.cc)
        set_target_properties(cuda_normal_estimation_test PROPERTIES
            FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(cuda_normal_estimation_test
                              PRIVATE colmap colmap_cuda_lidar_normals_lib
                                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("${FOLDER_NAME}/cuda_normal_estimation_test"
                 cuda_normal_estimation_test)
    endif()
endif()
