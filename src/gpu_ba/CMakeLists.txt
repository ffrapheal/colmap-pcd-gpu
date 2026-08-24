set(FOLDER_NAME "gpu_ba")

COLMAP_ADD_SOURCES(
    ba_types.h
    active_solve_view.h active_solve_view.cc
    host_ba_graph.h host_ba_graph.cc
    active_ba_solve_spec.h active_ba_solve_spec.cc
    ceres_envelope.h ceres_envelope.cc
    ceres_fidelity.h ceres_fidelity.cc
    fixed_linearization.h fixed_linearization.cc
    linearization.h linearization.cc
    snapshot.h snapshot.cc
    snapshot_recorder.h snapshot_recorder.cc
    validation.h validation.cc
)

if(GPU_BA_CUDA_ENABLED)
    add_library(gpu_ba_custom_cuda STATIC
                custom_cuda.h custom_cuda.cu custom_cuda_fp32.cu
                custom_cuda_audit.cc host_problem_store.cc
                host_problem_store.h host_problem_store_internal.h)
    set_target_properties(gpu_ba_custom_cuda PROPERTIES
        FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    if(GPU_BA_CUDA_PERFORMANCE)
        target_compile_definitions(gpu_ba_custom_cuda
                                   PRIVATE GPU_BA_CUDA_PERFORMANCE)
    else()
        target_compile_options(gpu_ba_custom_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
    target_compile_definitions(gpu_ba_custom_cuda
                               PRIVATE GPU_BA_CUDA_ENABLED)
    target_link_libraries(gpu_ba_custom_cuda
                          PUBLIC colmap CUDA::cudart CUDA::cublas
                                 CUDA::cusolver CUDA::cuda_driver)
endif()

if(GPU_BA_TESTS_ENABLED)
    add_executable(gpu_ba_active_solve_view_test active_solve_view_test.cc)
    set_target_properties(gpu_ba_active_solve_view_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_active_solve_view_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_active_solve_view_test"
             gpu_ba_active_solve_view_test)

    add_executable(gpu_ba_host_ba_graph_test host_ba_graph_test.cc)
    set_target_properties(gpu_ba_host_ba_graph_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_host_ba_graph_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_host_ba_graph_test" gpu_ba_host_ba_graph_test)

    add_executable(gpu_ba_ceres_envelope_test ceres_envelope_test.cc)
    set_target_properties(gpu_ba_ceres_envelope_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_ceres_envelope_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_ceres_envelope_test"
             gpu_ba_ceres_envelope_test)

    add_executable(gpu_ba_ceres_fidelity_test ceres_fidelity_test.cc)
    set_target_properties(gpu_ba_ceres_fidelity_test PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(gpu_ba_ceres_fidelity_test colmap
                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    add_test("gpu_ba/gpu_ba_ceres_fidelity_test"
             gpu_ba_ceres_fidelity_test)

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

    if(GPU_BA_CUDA_ENABLED)
        add_executable(gpu_ba_custom_cuda_test custom_cuda_test.cc)
        set_target_properties(gpu_ba_custom_cuda_test PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_test
                              gpu_ba_custom_cuda colmap
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("gpu_ba/gpu_ba_custom_cuda_test"
                 gpu_ba_custom_cuda_test)

        add_executable(gpu_ba_native_cuda_bridge_test
                       native_cuda_bridge_test.cc)
        set_target_properties(gpu_ba_native_cuda_bridge_test PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_native_cuda_bridge_test
                              gpu_ba_custom_cuda colmap
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("gpu_ba/gpu_ba_native_cuda_bridge_test"
                 gpu_ba_native_cuda_bridge_test)

        add_executable(gpu_ba_custom_cuda_cleanup_failure_helper
                       custom_cuda_cleanup_failure_helper.cc)
        set_target_properties(gpu_ba_custom_cuda_cleanup_failure_helper
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_cleanup_failure_helper
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_v2_golden
                       custom_cuda_v2_golden.cc)
        set_target_properties(gpu_ba_custom_cuda_v2_golden PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_v2_golden
                              gpu_ba_custom_cuda colmap)
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_v2_golden"
            COMMAND gpu_ba_custom_cuda_v2_golden
                    /tmp/colmap-pcd-gpu-v2-golden-ctest)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_v2_golden"
            PROPERTIES LABELS "cuda_correctness;canonical_v2" TIMEOUT 60)

        add_executable(gpu_ba_custom_cuda_v1_compatibility
                       custom_cuda_v1_compatibility.cc)
        set_target_properties(gpu_ba_custom_cuda_v1_compatibility PROPERTIES
            FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_v1_compatibility
                              gpu_ba_custom_cuda colmap)
        set(v1_compat_snapshot
            /home/nvidia/colmap-PCD-gpu-artifacts/oracles/phase5p5-ceres-replay-fidelity-v1/reg2-envelope8/snapshots/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json)
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_v1_normal_compatibility"
            COMMAND gpu_ba_custom_cuda_v1_compatibility
                    ${v1_compat_snapshot} normal
                    19e41076d424dbbf90afbfd63bc8e90eb606372933a0cb040214e7e359b22e79
                    9c7e2ce8c209f77ee20b13fba79df7b8d9c1bdc96b033d839a0b51d5eeea644c
                    /tmp/colmap-pcd-gpu-v1-normal-compatibility.json)
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_v1_pending_compatibility"
            COMMAND gpu_ba_custom_cuda_v1_compatibility
                    ${v1_compat_snapshot} pending_linearization
                    1204dcd0a5cf03c4fb2b14d526f3e0d95ce1a3bd6118c48dddccc1f4ae3acab1
                    92bebdc19e3f495a7c411d95cea421b51c4fc7a5cd8910c0065fa870e49c80b5
                    /tmp/colmap-pcd-gpu-v1-pending-compatibility.json)
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_v1_preparation_compatibility"
            COMMAND gpu_ba_custom_cuda_v1_compatibility
                    ${v1_compat_snapshot} synthetic_preparation
                    ec3d8438d64c323762d8d9c0203774aa173186dbb6551944be6deeb91d6c38d5
                    33c81e64e6f924bd4f34d4a51c565b5b40e1adb82a70049d7822f91a23b50dfd
                    /tmp/colmap-pcd-gpu-v1-preparation-compatibility.json)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_v1_normal_compatibility"
            "gpu_ba/gpu_ba_custom_cuda_v1_pending_compatibility"
            "gpu_ba/gpu_ba_custom_cuda_v1_preparation_compatibility"
            PROPERTIES LABELS "cuda_correctness;v1_compatibility" TIMEOUT 120)
        foreach(teardown_type IN ITEMS get_device set_device solver blas event
                                       stream allocation)
            add_test(
                NAME "gpu_ba/gpu_ba_custom_cuda_cleanup_${teardown_type}"
                COMMAND gpu_ba_custom_cuda_cleanup_failure_helper
                        ${teardown_type})
            set_tests_properties(
                "gpu_ba/gpu_ba_custom_cuda_cleanup_${teardown_type}"
                PROPERTIES LABELS "cuda_correctness;resource_cleanup;fresh_exec"
                           TIMEOUT 120)
        endforeach()

        add_executable(gpu_ba_custom_cuda_replay custom_cuda_replay.cc)
        set_target_properties(gpu_ba_custom_cuda_replay PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_replay
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_layer_b_replay
                       custom_cuda_layer_b_replay.cc)
        set_target_properties(gpu_ba_custom_cuda_layer_b_replay
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_layer_b_replay
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_layer_c_replay
                       custom_cuda_layer_c_replay.cc)
        set_target_properties(gpu_ba_custom_cuda_layer_c_replay
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_layer_c_replay
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_full_lm_replay
                       custom_cuda_full_lm_replay.cc)
        set_target_properties(gpu_ba_custom_cuda_full_lm_replay
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_full_lm_replay
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_precision_replay
                       custom_cuda_precision_replay.cc)
        set_target_properties(gpu_ba_custom_cuda_precision_replay
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_precision_replay
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_precision_state_compare
                       custom_cuda_precision_state_compare.cc)
        set_target_properties(gpu_ba_custom_cuda_precision_state_compare
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_precision_state_compare
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_precision_sequence
                       custom_cuda_precision_sequence.cc)
        set_target_properties(gpu_ba_custom_cuda_precision_sequence
                              PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_precision_sequence
                              gpu_ba_custom_cuda colmap)

        add_executable(gpu_ba_custom_cuda_global120_diagnostic
                       custom_cuda_global120_diagnostic.cc)
        set_target_properties(gpu_ba_custom_cuda_global120_diagnostic PROPERTIES
            FOLDER ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(gpu_ba_custom_cuda_global120_diagnostic
                              gpu_ba_custom_cuda colmap)

        # Production snapshot gate.  The driver asserts the documented
        # global120 strict-failure status instead of converting it to a pass.
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_production_fidelity"
            COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_custom_cuda_production_fidelity.sh
                    ${CMAKE_BINARY_DIR}
                    /home/nvidia/colmap-PCD-gpu-artifacts
                    /home/nvidia/colmap-PCD-gpu-artifacts/oracles/phase7p1a-current-linearization-cache-r1/ctest-production-fidelity)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_production_fidelity"
            PROPERTIES LABELS "production_fidelity;cuda_correctness"
                       TIMEOUT 1800)
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_stale_report_test"
            COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_custom_cuda_production_fidelity.sh
                    ${CMAKE_BINARY_DIR}
                    /home/nvidia/colmap-PCD-gpu-artifacts
                    /home/nvidia/colmap-PCD-gpu-artifacts/oracles/phase7p1a-current-linearization-cache-r1/ctest-production-fidelity
                    --stale-report-test)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_stale_report_test"
            PROPERTIES LABELS "production_fidelity;stale_report"
                       TIMEOUT 60)

        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_state_comparator_test"
            COMMAND gpu_ba_custom_cuda_full_lm_replay --state-comparator-self-test)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_state_comparator_test"
            PROPERTIES LABELS "cuda_correctness;state_comparator"
                       TIMEOUT 60)

        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_cache_safety_test"
            COMMAND gpu_ba_custom_cuda_test --run_test=CacheSafetySelfTest)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_cache_safety_test"
            PROPERTIES LABELS "cuda_correctness;cache_safety"
                       TIMEOUT 120)

        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_memory_accounting_test"
            COMMAND gpu_ba_custom_cuda_test --run_test=MemoryAccountingSelfTest)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_memory_accounting_test"
            PROPERTIES LABELS "cuda_correctness;memory_accounting"
                       TIMEOUT 60)

        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_instrumentation_env_invalid"
            COMMAND ${CMAKE_COMMAND} -E env
                    COLMAP_PCD_GPU_BA_INSTRUMENTATION=invalid
                    $<TARGET_FILE:gpu_ba_custom_cuda_full_lm_replay>
                    /home/nvidia/colmap-PCD-gpu-artifacts/snapshots/phase2-canonical/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json
                    /tmp/gpu_ba_invalid_instrumentation_should_not_exist.json
                    8 instrumentation_ab)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_instrumentation_env_invalid"
            PROPERTIES LABELS "cuda_correctness;instrumentation"
                       WILL_FAIL TRUE TIMEOUT 30)
        add_test(
            NAME "gpu_ba/gpu_ba_custom_cuda_linearization_cache_env_invalid"
            COMMAND ${CMAKE_COMMAND} -E env
                    COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE=invalid
                    $<TARGET_FILE:gpu_ba_custom_cuda_full_lm_replay>
                    /home/nvidia/colmap-PCD-gpu-artifacts/snapshots/phase2-canonical/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json
                    /tmp/gpu_ba_invalid_linearization_cache_should_not_exist.json
                    8 instrumentation_ab)
        set_tests_properties(
            "gpu_ba/gpu_ba_custom_cuda_linearization_cache_env_invalid"
            PROPERTIES LABELS "cuda_correctness;current_linearization_cache"
                       WILL_FAIL TRUE TIMEOUT 30)
    endif()
endif()
