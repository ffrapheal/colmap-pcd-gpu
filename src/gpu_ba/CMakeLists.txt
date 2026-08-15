set(FOLDER_NAME "gpu_ba")

COLMAP_ADD_SOURCES(
    fixed_linearization.h fixed_linearization.cc
    linearization.h linearization.cc
    snapshot.h snapshot.cc
    snapshot_recorder.h snapshot_recorder.cc
    validation.h validation.cc
)

if(GPU_BA_TESTS_ENABLED)
    add_executable(gpu_ba_fixed_linearization_test
                   fixed_linearization_test.cc)
    set_target_properties(gpu_ba_fixed_linearization_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_fixed_linearization_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_fixed_linearization_test"
             gpu_ba_fixed_linearization_test)

    add_executable(gpu_ba_snapshot_test snapshot_test.cc)
    set_target_properties(gpu_ba_snapshot_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_snapshot_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_snapshot_test" gpu_ba_snapshot_test)

    add_executable(gpu_ba_linearization_test linearization_test.cc)
    set_target_properties(gpu_ba_linearization_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_linearization_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_linearization_test" gpu_ba_linearization_test)

    add_executable(gpu_ba_validation_test validation_test.cc)
    set_target_properties(gpu_ba_validation_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_validation_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_validation_test" gpu_ba_validation_test)
endif()
