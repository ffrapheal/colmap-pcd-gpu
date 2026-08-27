set(FOLDER_NAME "lidar")

COLMAP_ADD_SOURCES(
    lidar_point.h lidar_point.cc
    pt_type.h
    pcd_projection.h pcd_projection.cc
    kdtree.h kdtree.cc
    ply.h ply.cc
)

if(GPU_BA_CUDA_ENABLED)
    add_executable(colmap_cuda_lidar_normals
                   cuda_normal_estimation.h
                   cuda_normal_estimation.cu
                   cuda_normal_estimation_main.cc)
    set_target_properties(colmap_cuda_lidar_normals PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    target_link_libraries(colmap_cuda_lidar_normals PRIVATE colmap CUDA::cudart)
endif()
