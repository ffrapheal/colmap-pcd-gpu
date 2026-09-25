# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "sfm")

set(SFM_SOURCES
    incremental_mapper.h incremental_mapper.cc
    incremental_triangulator.h incremental_triangulator.cc
    online_local_ba_postprocess.h online_local_ba_postprocess.cc
    active_covisibility_graph.h active_covisibility_graph.cc
    online_dual_selection.h online_dual_selection.cc
)

if(GPU_BA_CUDA_ENABLED)
    list(APPEND SFM_SOURCES
         online_lidar_association.h online_lidar_association.cc
         online_lidar_ba_intent.h online_lidar_ba_intent.cc
         online_local_ba_executor.h online_local_ba_executor.cc)
endif()

COLMAP_ADD_SOURCES(${SFM_SOURCES})

COLMAP_ADD_TEST(incremental_mapper_test incremental_mapper_test.cc)

COLMAP_ADD_TEST(active_covisibility_graph_test
                active_covisibility_graph_test.cc)

COLMAP_ADD_TEST(online_dual_selection_test
                online_dual_selection_test.cc)

COLMAP_ADD_TEST(online_local_ba_postprocess_test
                online_local_ba_postprocess_test.cc)

if(GPU_BA_CUDA_ENABLED AND TESTS_ENABLED)
    add_executable(online_lidar_association_test
                   online_lidar_association_test.cc)
    set_target_properties(online_lidar_association_test PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(online_lidar_association_test
                          PRIVATE colmap colmap_cuda_lidar_normals_lib
                                  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("${FOLDER_NAME}/online_lidar_association_test"
             online_lidar_association_test)

    add_executable(online_lidar_ba_intent_test
                   online_lidar_ba_intent_test.cc)
    set_target_properties(online_lidar_ba_intent_test PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(online_lidar_ba_intent_test
                          PRIVATE colmap gpu_ba_custom_cuda
                                  colmap_cuda_lidar_normals_lib
                                  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("${FOLDER_NAME}/online_lidar_ba_intent_test"
             online_lidar_ba_intent_test)

    add_executable(online_local_ba_executor_test
                   online_local_ba_executor_test.cc)
    set_target_properties(online_local_ba_executor_test PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(online_local_ba_executor_test
                          PRIVATE colmap gpu_ba_custom_cuda
                                  colmap_cuda_lidar_normals_lib
                                  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("${FOLDER_NAME}/online_local_ba_executor_test"
             online_local_ba_executor_test)
endif()
