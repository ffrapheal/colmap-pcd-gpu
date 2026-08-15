set(FOLDER_NAME "gpu_ba")

COLMAP_ADD_SOURCES(
    snapshot.h snapshot.cc
    snapshot_recorder.h snapshot_recorder.cc
)

if(GPU_BA_TESTS_ENABLED)
    add_executable(gpu_ba_snapshot_test snapshot_test.cc)
    set_target_properties(gpu_ba_snapshot_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_snapshot_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_snapshot_test" gpu_ba_snapshot_test)
endif()
