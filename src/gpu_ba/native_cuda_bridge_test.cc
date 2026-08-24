#define TEST_NAME "gpu_ba/native_cuda_bridge"
#include "util/testing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "gpu_ba/custom_cuda.h"

namespace colmap {
namespace gpu_ba {
namespace {

struct ShadowFixture {
  HostBaGraphStore store{44};
  CatalogReadLease lease;
  DenseActiveState dense_state;
  BaSolveIntent intent;
  Snapshot reference;
};

CudaFullLmOptions NativeFullLmOptions(
    const NativeCudaResolvedConfig& config) {
  CudaFullLmOptions options;
  options.arithmetic_precision = config.arithmetic_precision;
  options.device_context_mode = config.device_context;
  options.hot_kernel_mode = config.hot_kernel;
  options.execution_profile = config.execution_profile;
  options.audit_profile = config.audit_profile;
  options.performance_mode = config.performance_mode;
  options.current_linearization_cache_mode = config.linearization_cache;
  options.pair_chunk_limit_bytes_for_testing =
      config.pair_chunk_limit_bytes;
  options.max_num_iterations = -1;
  options.max_num_consecutive_invalid_steps = -1;
  options.function_tolerance = -1.0;
  options.gradient_tolerance = -1.0;
  options.parameter_tolerance = -1.0;
  options.max_solver_time_in_seconds = config.max_solver_time_in_seconds;
  options.initial_trust_region_radius = config.initial_trust_region_radius;
  options.min_trust_region_radius = config.min_trust_region_radius;
  options.max_trust_region_radius = config.max_trust_region_radius;
  options.min_relative_decrease = config.min_relative_decrease;
  options.layer_c.schur_contribution_backend = config.schur_backend;
  options.layer_c.schur_segment_size_for_testing = config.schur_segment_size;
  options.layer_c.layer_b.layer_a.device = config.device;
  options.layer_c.layer_b.layer_a.block_size = config.block_size;
  options.layer_c.layer_b.layer_a.memory_mode = config.memory_mode;
  options.layer_c.layer_b.layer_a.residual_order = config.residual_order;
  options.layer_c.layer_b.loss_mode = config.loss_mode;
  options.layer_c.layer_b.loss_scale = config.loss_scale;
  options.layer_c.layer_b.cost_reduction_threads =
      config.cost_reduction_threads;
  options.layer_c.layer_b.reduction_mode = config.reduction_mode;
  options.layer_c.layer_b.min_lm_diagonal = config.min_lm_diagonal;
  options.layer_c.layer_b.max_lm_diagonal = config.max_lm_diagonal;
  options.layer_c.layer_b.hessian_assembly_backend = config.hessian_backend;
  options.layer_c.layer_b.hessian_segment_size_for_testing =
      config.hessian_segment_size;
  return options;
}

bool BuildShadowFixture(ShadowFixture* fixture, std::string* error) {
  HostBaGraphColdInput graph;
  graph.owner_epoch = 44;
  graph.topology_revision = 3;
  graph.cameras.push_back({7, 4, 640, 480, 8});
  graph.images.push_back({20, 7, true});
  graph.images.push_back({21, 7, true});
  graph.points.push_back({30});
  graph.observations.push_back({20, 0, 30, {{10.0, 11.0}}});
  graph.observations.push_back({21, 1, 30, {{12.0, 13.0}}});
  HostBaGraphUpdateResult update;
  if (!fixture->store.ColdBuild(graph, &update, error)) return false;
  fixture->lease = fixture->store.AcquireReadLease();

  fixture->dense_state.owner_epoch = 44;
  fixture->dense_state.state_generation = 5;
  DenseCameraState camera;
  camera.camera_slot = 0;
  camera.state_generation = 5;
  camera.parameters = {500.0, 500.0, 320.0, 240.0,
                       0.01,  -0.01, 0.001, -0.001};
  fixture->dense_state.cameras.push_back(camera);
  DenseImageState image;
  image.image_slot = 0;
  image.state_generation = 5;
  image.quaternion = {{2.0, 0.0, 0.0, 0.0}};
  image.translation = {{0.1, 0.2, 0.3}};
  fixture->dense_state.images.push_back(image);
  image.image_slot = 1;
  image.quaternion = {{0.0, 2.0, 0.0, 0.0}};
  image.translation = {{0.4, 0.5, 0.6}};
  fixture->dense_state.images.push_back(image);
  DensePointState point;
  point.point_slot = 0;
  point.state_generation = 5;
  point.xyz = {{1.0, 2.0, 3.0}};
  fixture->dense_state.points.push_back(point);

  fixture->intent.owner_epoch = 44;
  fixture->intent.catalog_revision = fixture->lease.topology_revision();
  fixture->intent.catalog_generation = fixture->lease.generation();
  fixture->intent.selection_revision = 2;
  fixture->intent.kind = BaKind::kLocal;
  fixture->intent.config.resolved = true;
  fixture->intent.config.config_generation = 6;
  fixture->intent.config.arithmetic_precision =
      CudaArithmeticPrecision::kFp32MixedStable;
  fixture->intent.config.hessian_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  fixture->intent.config.schur_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  fixture->intent.config.hot_kernel = CudaHotKernelMode::kTransformed;
  fixture->intent.config.execution_profile =
      CudaExecutionProfile::kCompactControl;
  fixture->intent.config.device_context = CudaDeviceContextMode::kDeviceControl;
  fixture->intent.config.memory_mode = CudaMemoryMode::kExplicitDeviceCopy;
  fixture->intent.config.reduction_mode =
      CudaReductionMode::kParallelDeterministic;
  fixture->intent.config.audit_profile = CudaAuditProfile::kProduction;
  fixture->intent.config.linearization_cache =
      CudaCurrentLinearizationCacheMode::kEnabled;
  fixture->intent.config.residual_order =
      CudaResidualOrder::kSourceInsertion;
  fixture->intent.config.loss_mode = CudaLossMode::kSoftL1;
  fixture->intent.config.loss_scale = 1.0;
  fixture->intent.config.max_num_iterations = 2;
  fixture->intent.config.initial_trust_region_radius = 1.0;
  fixture->intent.active_image_slots.push_back(0);
  fixture->intent.explicit_variable_point_slots.push_back(0);
  fixture->intent.translation_subsets.push_back({0, 2, {0, 0, 0}});
  CameraParameterPolicy camera_policy;
  camera_policy.camera_slot = 0;
  camera_policy.constant = true;
  fixture->intent.camera_policies.push_back(camera_policy);
  LidarConstraintRecord lidar;
  lidar.point_slot = 0;
  lidar.constraint_slot = 2;
  lidar.physical_identity = 0xabc00002ull;
  lidar.lidar_type = 1;
  lidar.plane = {{0.0, 0.0, 1.0, -3.0}};
  lidar.lidar_xyz = {{1.0, 2.0, 3.0}};
  lidar.weight = 2.0;
  lidar.search_range = 0.1;
  lidar.point_state_generation = 5;
  fixture->intent.lidar.lidar_map_generation = 7;
  fixture->intent.lidar.match_config_generation = 8;
  fixture->intent.lidar.constraints.push_back(lidar);
  fixture->intent.source_insertion_order = {
      {ResidualKind::kVisual, 0},
      {ResidualKind::kLidar, 2},
      {ResidualKind::kVisual, 1}};

  fixture->reference.metadata.ba_kind = BaKind::kLocal;
  fixture->reference.metadata.loss_function = "SOFT_L1";
  fixture->reference.metadata.lidar_residual_mode = "legacy_exact";
  fixture->reference.metadata.max_num_iterations = 2;
  fixture->reference.metadata.max_consecutive_invalid_steps = 10;
  fixture->reference.metadata.function_tolerance = 0.0;
  fixture->reference.metadata.gradient_tolerance = 0.0;
  fixture->reference.metadata.parameter_tolerance = 0.0;
  CameraSnapshot reference_camera;
  reference_camera.camera_id = 7;
  reference_camera.model_id = 4;
  reference_camera.width = 640;
  reference_camera.height = 480;
  reference_camera.constant = true;
  reference_camera.params = camera.parameters;
  fixture->reference.cameras.push_back(reference_camera);
  for (const DenseImageState& source : fixture->dense_state.images) {
    ImageSnapshot reference_image;
    reference_image.image_id = source.image_slot == 0 ? 20 : 21;
    reference_image.camera_id = 7;
    reference_image.selected = source.image_slot == 0;
    reference_image.pose_constant = source.image_slot != 0;
    reference_image.has_pose_parameter_blocks = source.image_slot == 0;
    reference_image.constant_tvec_mask = source.image_slot == 0 ? 2 : 0;
    reference_image.qvec = source.image_slot == 0
        ? std::array<double, 4>{{1.0, 0.0, 0.0, 0.0}}
        : std::array<double, 4>{{0.0, 1.0, 0.0, 0.0}};
    reference_image.tvec = source.translation;
    fixture->reference.images.push_back(reference_image);
  }
  PointSnapshot reference_point;
  reference_point.point3D_id = 30;
  reference_point.xyz = point.xyz;
  fixture->reference.points.push_back(reference_point);
  ObservationSnapshot observation;
  observation.source_index = 0;
  observation.image_id = 20;
  observation.point2D_idx = 0;
  observation.point3D_id = 30;
  observation.xy = {{10.0, 11.0}};
  fixture->reference.observations.push_back(observation);
  observation.source_index = 2;
  observation.image_id = 21;
  observation.point2D_idx = 1;
  observation.pose_constant = true;
  observation.xy = {{12.0, 13.0}};
  fixture->reference.observations.push_back(observation);
  LidarSnapshot reference_lidar;
  reference_lidar.source_index = 1;
  reference_lidar.point3D_id = 30;
  reference_lidar.lidar_type = 1;
  reference_lidar.search_range = 0.1;
  reference_lidar.has_search_range = true;
  reference_lidar.weight = 2.0;
  reference_lidar.lidar_xyz = lidar.lidar_xyz;
  reference_lidar.plane = lidar.plane;
  fixture->reference.lidar.push_back(reference_lidar);
  fixture->reference.source_insertion_order = {
      {0, ResidualKind::kVisual, 20, 0, 30},
      {1, ResidualKind::kLidar, 0, 0, 30},
      {2, ResidualKind::kVisual, 21, 1, 30}};
  fixture->reference.parameter_blocks_source_order = {
      {0, ParameterKind::kQuaternion, 20, 4, 3, false},
      {1, ParameterKind::kTranslation, 20, 3, 2, false},
      {2, ParameterKind::kPoint3D, 30, 3, 3, false},
      {3, ParameterKind::kCamera, 7, 8, 8, true}};
  return true;
}

bool BuildAcceptedCommitFixture(ShadowFixture* fixture, std::string* error) {
  HostBaGraphColdInput graph;
  graph.owner_epoch = 44;
  graph.topology_revision = 4;
  graph.cameras.push_back({7, 4, 0, 0, 8});
  graph.images.push_back({11, 7, true});
  graph.images.push_back({12, 7, true});
  graph.points.push_back({101});
  graph.points.push_back({102});
  graph.observations.push_back({11, 1, 101, {{360.0, 190.0}}});
  graph.observations.push_back({12, 1, 101, {{350.0, 200.0}}});
  graph.observations.push_back({11, 2, 102, {{300.0, 260.0}}});
  HostBaGraphUpdateResult update;
  if (!fixture->store.ColdBuild(graph, &update, error)) return false;
  fixture->lease = fixture->store.AcquireReadLease();

  fixture->dense_state.owner_epoch = graph.owner_epoch;
  fixture->dense_state.state_generation = 9;
  DenseCameraState camera;
  camera.camera_slot = 0;
  camera.state_generation = fixture->dense_state.state_generation;
  camera.parameters = {610.0, 605.0, 320.0, 240.0,
                       -0.03, 0.004, 0.001, -0.0007};
  fixture->dense_state.cameras.push_back(camera);
  DenseImageState image;
  image.image_slot = 0;
  image.state_generation = fixture->dense_state.state_generation;
  image.quaternion = {{1.0, 0.0, 0.0, 0.0}};
  image.translation = {{0.1, -0.2, 0.3}};
  fixture->dense_state.images.push_back(image);
  image.image_slot = 1;
  image.translation = {{-0.2, 0.1, 0.4}};
  fixture->dense_state.images.push_back(image);
  DensePointState point;
  point.point_slot = 0;
  point.state_generation = fixture->dense_state.state_generation;
  point.xyz = {{0.2, -0.1, 3.0}};
  fixture->dense_state.points.push_back(point);
  point.point_slot = 1;
  point.xyz = {{-0.4, 0.3, 4.0}};
  fixture->dense_state.points.push_back(point);

  fixture->intent.owner_epoch = graph.owner_epoch;
  fixture->intent.catalog_revision = fixture->lease.topology_revision();
  fixture->intent.catalog_generation = fixture->lease.generation();
  fixture->intent.selection_revision = 3;
  fixture->intent.kind = BaKind::kLocal;
  fixture->intent.config.resolved = true;
  fixture->intent.config.config_generation = 7;
  fixture->intent.config.arithmetic_precision = CudaArithmeticPrecision::kFp64;
  fixture->intent.config.device_context = CudaDeviceContextMode::kDeviceControl;
  fixture->intent.config.memory_mode = CudaMemoryMode::kExplicitDeviceCopy;
  fixture->intent.config.reduction_mode =
      CudaReductionMode::kParallelDeterministic;
  fixture->intent.config.audit_profile = CudaAuditProfile::kProduction;
  fixture->intent.config.linearization_cache =
      CudaCurrentLinearizationCacheMode::kEnabled;
  fixture->intent.config.hessian_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  fixture->intent.config.schur_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  fixture->intent.config.hot_kernel = CudaHotKernelMode::kTransformed;
  fixture->intent.config.execution_profile =
      CudaExecutionProfile::kCompactControl;
  fixture->intent.config.residual_order = CudaResidualOrder::kSourceInsertion;
  fixture->intent.config.loss_mode = CudaLossMode::kTrivial;
  fixture->intent.config.loss_scale = 1.0;
  fixture->intent.config.max_num_iterations = 2;
  fixture->intent.config.max_consecutive_invalid_steps = 10;
  fixture->intent.config.function_tolerance = 0.0;
  fixture->intent.config.gradient_tolerance = 0.0;
  fixture->intent.config.parameter_tolerance = 0.0;
  fixture->intent.active_image_slots.push_back(0);
  fixture->intent.explicit_variable_point_slots.push_back(0);
  fixture->intent.translation_subsets.push_back({0, 1u << 1, {0, 0, 0}});
  CameraParameterPolicy camera_policy;
  camera_policy.camera_slot = 0;
  camera_policy.constant = true;
  fixture->intent.camera_policies.push_back(camera_policy);
  fixture->intent.point_policies.push_back({0, false, 0, false, 0.0});
  fixture->intent.point_policies.push_back({1, true, 0, false, 0.0});
  LidarConstraintRecord lidar;
  lidar.point_slot = 0;
  lidar.constraint_slot = 0;
  lidar.physical_identity = 3;
  lidar.plane = {{0.2, -0.3, 0.7, -1.8}};
  lidar.weight = 4.0;
  lidar.point_state_generation = fixture->dense_state.state_generation;
  fixture->intent.lidar.lidar_map_generation = 1;
  fixture->intent.lidar.match_config_generation = 1;
  fixture->intent.lidar.constraints.push_back(lidar);
  fixture->intent.source_insertion_order = {
      {ResidualKind::kVisual, 0},
      {ResidualKind::kVisual, 1},
      {ResidualKind::kVisual, 2},
      {ResidualKind::kLidar, 0}};

  fixture->reference.metadata.snapshot_id = "cuda-layer-b-native-accepted";
  fixture->reference.metadata.loss_function = "TRIVIAL";
  fixture->reference.metadata.lidar_residual_mode = "legacy_exact";
  fixture->reference.metadata.max_num_iterations = 2;
  fixture->reference.metadata.max_consecutive_invalid_steps = 10;
  fixture->reference.metadata.function_tolerance = 0.0;
  fixture->reference.metadata.gradient_tolerance = 0.0;
  fixture->reference.metadata.parameter_tolerance = 0.0;
  CameraSnapshot reference_camera;
  reference_camera.camera_id = 7;
  reference_camera.model_id = 4;
  reference_camera.constant = true;
  reference_camera.params = camera.parameters;
  fixture->reference.cameras.push_back(reference_camera);
  ImageSnapshot reference_image;
  reference_image.image_id = 11;
  reference_image.camera_id = 7;
  reference_image.selected = true;
  reference_image.pose_constant = false;
  reference_image.has_pose_parameter_blocks = true;
  reference_image.constant_tvec_mask = 1u << 1;
  reference_image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  reference_image.tvec = {{0.1, -0.2, 0.3}};
  fixture->reference.images.push_back(reference_image);
  reference_image.image_id = 12;
  reference_image.selected = false;
  reference_image.pose_constant = true;
  reference_image.has_pose_parameter_blocks = false;
  reference_image.constant_tvec_mask = 0;
  reference_image.tvec = {{-0.2, 0.1, 0.4}};
  fixture->reference.images.push_back(reference_image);
  PointSnapshot reference_point;
  reference_point.point3D_id = 101;
  reference_point.xyz = {{0.2, -0.1, 3.0}};
  fixture->reference.points.push_back(reference_point);
  reference_point.point3D_id = 102;
  reference_point.constant = true;
  reference_point.xyz = {{-0.4, 0.3, 4.0}};
  fixture->reference.points.push_back(reference_point);
  const auto add_observation = [&](const uint64_t source,
                                   const uint32_t image_id,
                                   const uint32_t point2D_idx,
                                   const uint64_t point3D_id,
                                   const std::array<double, 2>& xy) {
    ObservationSnapshot observation;
    observation.source_index = source;
    observation.image_id = image_id;
    observation.point2D_idx = point2D_idx;
    observation.point3D_id = point3D_id;
    observation.pose_constant = image_id == 12;
    observation.xy = xy;
    fixture->reference.observations.push_back(observation);
    fixture->reference.source_insertion_order.push_back(
        {source, ResidualKind::kVisual, image_id, point2D_idx, point3D_id});
  };
  add_observation(0, 11, 1, 101, {{360.0, 190.0}});
  add_observation(1, 12, 1, 101, {{350.0, 200.0}});
  add_observation(2, 11, 2, 102, {{300.0, 260.0}});
  LidarSnapshot reference_lidar;
  reference_lidar.source_index = 3;
  reference_lidar.point3D_id = 101;
  reference_lidar.weight = 4.0;
  reference_lidar.plane = lidar.plane;
  fixture->reference.lidar.push_back(reference_lidar);
  fixture->reference.source_insertion_order.push_back(
      {3, ResidualKind::kLidar, 0, 0, 101});
  fixture->reference.canonical_order =
      fixture->reference.source_insertion_order;
  fixture->reference.parameter_blocks_source_order = {
      {0, ParameterKind::kQuaternion, 11, 4, 3, false},
      {1, ParameterKind::kTranslation, 11, 3, 2, false},
      {2, ParameterKind::kPoint3D, 101, 3, 3, false}};
  return true;
}

BOOST_AUTO_TEST_CASE(SyntheticLegacyAndNativeKernelBoundaryPreparationParity) {
  ShadowFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildShadowFixture(&fixture, &error), error);
  const DenseActiveState entry_state = fixture.dense_state;
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer state;
  NativeHostSolvePreparationRuntime preparation;
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                            fixture.lease, fixture.intent, fixture.dense_state,
                            &view, &state, &preparation, &error),
                        error);
  BOOST_REQUIRE_EQUAL(view.active_camera_slots.size(), 1);
  BOOST_REQUIRE_EQUAL(view.active_image_slots.size(), 1);
  BOOST_REQUIRE_EQUAL(view.boundary_image_slots.size(), 1);
  BOOST_REQUIRE_EQUAL(view.active_point_slots.size(), 1);
  BOOST_CHECK_EQUAL(view.catalog.cameras()[view.active_camera_slots[0]].camera_id,
                    7);
  BOOST_CHECK_EQUAL(view.catalog.images()[view.active_image_slots[0]].image_id,
                    20);
  BOOST_CHECK_EQUAL(
      view.catalog.images()[view.boundary_image_slots[0]].image_id, 21);
  BOOST_CHECK_EQUAL(view.catalog.points()[view.active_point_slots[0]].point3D_id,
                    30);
  BOOST_REQUIRE_EQUAL(view.fixed.images.size(), 2);
  BOOST_CHECK_EQUAL(view.fixed.images[0].translation_subset_mask, 2);
  BOOST_CHECK_EQUAL(view.fixed.images[0].pose_constant, 0);
  BOOST_CHECK_EQUAL(view.fixed.images[1].boundary_pose, 1);
  BOOST_REQUIRE_EQUAL(view.lidar.constraints.size(), 1);
  BOOST_CHECK_EQUAL(view.lidar.constraints[0].point_slot, 0);
  BOOST_CHECK_EQUAL(view.lidar.constraints[0].lidar_type,
                    fixture.reference.lidar[0].lidar_type);
  BOOST_CHECK_EQUAL(view.lidar.constraints[0].weight,
                    fixture.reference.lidar[0].weight);
  BOOST_CHECK_EQUAL_COLLECTIONS(view.lidar.constraints[0].plane.begin(),
                                view.lidar.constraints[0].plane.end(),
                                fixture.reference.lidar[0].plane.begin(),
                                fixture.reference.lidar[0].plane.end());
  CudaLayerBOptions options;
  options.layer_a.residual_order = CudaResidualOrder::kSourceInsertion;
  options.loss_mode = CudaLossMode::kSoftL1;
  options.loss_scale = 1.0;
  options.hessian_assembly_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  options.hessian_segment_size_for_testing = 64;
  NativePreparationComparisonResult comparison;
  BOOST_REQUIRE_MESSAGE(CompareNativePreparationToLegacyForTesting(
                            fixture.reference, view, state, options, 0,
                            CudaHotKernelMode::kTransformed,
                            CudaSchurContributionBackend::kSegmentedTransformed,
                            64, &comparison, &error),
                        error);
  BOOST_CHECK(comparison.source_indices_dense);
  BOOST_CHECK(comparison.cost_entries_exact);
  BOOST_CHECK(comparison.layer_b_exact);
  BOOST_CHECK(comparison.layer_c_exact);
  LegacyKernelInputBundle native;
  BOOST_REQUIRE_MESSAGE(BuildLegacyKernelInputBundle(view, state, &native,
                                                     &error), error);
  BOOST_REQUIRE_EQUAL(native.visual().size(), 2);
  BOOST_REQUIRE_EQUAL(native.lidar().size(), 1);
  BOOST_CHECK_EQUAL(native.visual()[0].source_index, 0);
  BOOST_CHECK_EQUAL(native.lidar()[0].source_index, 1);
  BOOST_CHECK_EQUAL(native.visual()[1].source_index, 2);
  BOOST_CHECK_EQUAL(native.runtime().legacy_kernel_input_bundle_calls, 1);
  BOOST_CHECK_EQUAL(native.runtime().build_cuda_layer_a_inputs_calls, 1);
  BOOST_CHECK_EQUAL(native.runtime().build_static_layout_calls, 1);
  BOOST_CHECK_EQUAL(native.runtime().build_cost_layout_calls, 1);
  BOOST_CHECK_EQUAL(native.runtime().build_layer_b_topology_calls, 1);
  BOOST_CHECK_EQUAL(native.runtime().build_layer_c_topology_calls, 1);
  BOOST_REQUIRE_EQUAL(view.parameter_ordinals.size(),
                      fixture.reference.parameter_blocks_source_order.size());
  for (size_t i = 0; i < view.parameter_ordinals.size(); ++i) {
    const ParameterOrdinal& native_parameter = view.parameter_ordinals[i];
    const ParameterBlockSnapshot& reference_parameter =
        fixture.reference.parameter_blocks_source_order[i];
    BOOST_CHECK(native_parameter.kind == reference_parameter.kind);
    uint64_t native_entity_id = 0;
    if (native_parameter.kind == ParameterKind::kQuaternion ||
        native_parameter.kind == ParameterKind::kTranslation) {
      native_entity_id = view.catalog.images()[native_parameter.entity_slot]
                             .image_id;
    } else if (native_parameter.kind == ParameterKind::kPoint3D) {
      native_entity_id = view.catalog.points()[native_parameter.entity_slot]
                             .point3D_id;
    } else {
      native_entity_id = view.catalog.cameras()[native_parameter.entity_slot]
                             .camera_id;
    }
    BOOST_CHECK_EQUAL(native_entity_id, reference_parameter.entity_id);
    BOOST_CHECK_EQUAL(native_parameter.ambient_size,
                      reference_parameter.ambient_size);
    BOOST_CHECK_EQUAL(native_parameter.tangent_size,
                      reference_parameter.tangent_size);
    BOOST_CHECK_EQUAL(native_parameter.constant,
                      reference_parameter.constant);
  }
  BOOST_CHECK_EQUAL(fixture.dense_state.images[0].quaternion[0],
                    entry_state.images[0].quaternion[0]);
  BOOST_CHECK_EQUAL(fixture.dense_state.images[1].quaternion[1],
                    entry_state.images[1].quaternion[1]);
}

BOOST_AUTO_TEST_CASE(TestOnlyActiveStateCopyIsNotReportedAsDeviceDownload) {
  ShadowFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildShadowFixture(&fixture, &error), error);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer state;
  NativeHostSolvePreparationRuntime preparation;
  BOOST_REQUIRE(materializer.Materialize(fixture.lease, fixture.intent,
                                         fixture.dense_state, &view, &state,
                                         &preparation, &error));
  DenseActiveState downloaded;
  BOOST_REQUIRE_MESSAGE(CopyActiveStateForTesting(view, state, &downloaded,
                                                  &error),
                        error);
  BOOST_CHECK_EQUAL(downloaded.owner_epoch, view.identity.owner_epoch);
  BOOST_CHECK_EQUAL(downloaded.cameras.size(), state.cameras.size());
  BOOST_CHECK_EQUAL(downloaded.images.size(), state.images.size());
  BOOST_CHECK_EQUAL(downloaded.points.size(), state.points.size());
  BOOST_CHECK_EQUAL(downloaded.images[0].quaternion[0], 1.0);
}

BOOST_AUTO_TEST_CASE(NativeRequestRunsSharedLmControllerAndDownloadsDeviceState) {
  ShadowFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildShadowFixture(&fixture, &error), error);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer state;
  NativeHostSolvePreparationRuntime preparation;
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                            fixture.lease, fixture.intent,
                            fixture.dense_state, &view, &state, &preparation,
                            &error),
                        error);
  CudaFullLmOptions options = NativeFullLmOptions(view.config);
  NativeCudaSolveRequest request;
  request.view = &view;
  request.initial_state = &state;
  request.options = &options;
  CudaFullLmResult legacy_result;
  std::string legacy_error;
  const bool legacy_success = RunCustomCudaSolve(
      fixture.reference, options, &legacy_result, &legacy_error);
  BOOST_REQUIRE_MESSAGE(legacy_success, legacy_error);
  BaSolveResult result;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(request, &result, &error), error);
  BOOST_CHECK(result.success);
  BOOST_CHECK_EQUAL(result.success, legacy_result.success);
  BOOST_CHECK_EQUAL(result.termination_reason,
                    legacy_result.termination_reason);
  BOOST_CHECK_EQUAL(result.trial_iterations, legacy_result.trial_iterations);
  BOOST_CHECK_EQUAL(result.accepted_steps, legacy_result.accepted_steps);
  BOOST_CHECK_EQUAL(result.accepted_commits, legacy_result.accepted_commits);
  BOOST_CHECK_EQUAL(result.rejected_steps, legacy_result.rejected_steps);
  BOOST_CHECK_EQUAL(result.invalid_steps, legacy_result.invalid_steps);
  BOOST_CHECK_EQUAL(result.final_internal_state_epoch,
                    legacy_result.runtime.final_internal_state_epoch);
  int32_t trace_decisions = 0;
  int32_t trace_commits = 0;
  for (const CudaLmIteration& iteration : legacy_result.trace) {
    if (iteration.iteration == 0) continue;
    trace_decisions += iteration.accepted_decision ? 1 : 0;
    trace_commits += iteration.accepted_commit_success ? 1 : 0;
  }
  BOOST_CHECK_EQUAL(trace_decisions, legacy_result.accepted_decisions);
  BOOST_CHECK_EQUAL(trace_commits, legacy_result.accepted_commits);
  BOOST_CHECK_EQUAL(result.initial_cost, legacy_result.initial_cost);
  BOOST_CHECK_EQUAL(result.final_cost, legacy_result.final_cost);
  BOOST_CHECK_GT(result.trial_iterations, 0);
  BOOST_CHECK(result.runtime.native_lm_controller_handoff_complete);
  BOOST_CHECK_EQUAL(result.runtime.dense_active_state_device_download_calls, 1);
  BOOST_CHECK_EQUAL(result.final_state.cameras.size(), state.cameras.size());
  BOOST_CHECK_EQUAL(result.final_state.images.size(), state.images.size());
  BOOST_CHECK_EQUAL(result.final_state.points.size(), state.points.size());
  BOOST_CHECK_EQUAL(result.final_state.state_generation,
                    state.state_generation +
                        legacy_result.runtime.final_internal_state_epoch);
  BOOST_TEST_MESSAGE("native full-LM trace: trials="
                     << result.trial_iterations
                     << " decisions=" << legacy_result.accepted_decisions
                     << " commits=" << result.accepted_commits
                     << " rejects=" << result.rejected_steps
                     << " invalid=" << result.invalid_steps
                     << " final_epoch=" << result.final_internal_state_epoch);
  BOOST_CHECK(std::isfinite(result.initial_cost));
  BOOST_CHECK(std::isfinite(result.final_cost));
  BOOST_CHECK_LE(result.final_cost, result.initial_cost);
  BOOST_CHECK_EQUAL(result.runtime.legacy_kernel_input_bundle_calls, 1);
  BOOST_CHECK_EQUAL(result.runtime.build_cuda_layer_a_inputs_calls, 1);
  BOOST_CHECK_EQUAL(result.runtime.build_static_layout_calls, 1);
  BOOST_CHECK_EQUAL(result.runtime.build_cost_layout_calls, 1);
  BOOST_CHECK_EQUAL(result.runtime.build_layer_b_topology_calls, 1);
  BOOST_CHECK_EQUAL(result.runtime.build_layer_c_topology_calls, 1);
  BOOST_REQUIRE_EQUAL(result.final_state.cameras.size(),
                      legacy_result.final_state.cameras.size());
  BOOST_REQUIRE_EQUAL(result.final_state.images.size(),
                      legacy_result.final_state.images.size());
  BOOST_REQUIRE_EQUAL(result.final_state.points.size(),
                      legacy_result.final_state.points.size());
  for (size_t i = 0; i < result.final_state.cameras.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        result.final_state.cameras[i].parameters.begin(),
        result.final_state.cameras[i].parameters.end(),
        legacy_result.final_state.cameras[i].params.begin(),
        legacy_result.final_state.cameras[i].params.end());
  }
  for (size_t i = 0; i < result.final_state.images.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        result.final_state.images[i].quaternion.begin(),
        result.final_state.images[i].quaternion.end(),
        legacy_result.final_state.images[i].qvec.begin(),
        legacy_result.final_state.images[i].qvec.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        result.final_state.images[i].translation.begin(),
        result.final_state.images[i].translation.end(),
        legacy_result.final_state.images[i].tvec.begin(),
        legacy_result.final_state.images[i].tvec.end());
  }
  for (size_t i = 0; i < result.final_state.points.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        result.final_state.points[i].xyz.begin(),
        result.final_state.points[i].xyz.end(),
        legacy_result.final_state.points[i].xyz.begin(),
        legacy_result.final_state.points[i].xyz.end());
  }
}

BOOST_AUTO_TEST_CASE(
    NativeAcceptedCommitSwapsCurrentSlotAndDownloadsCommittedState) {
  ShadowFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildAcceptedCommitFixture(&fixture, &error), error);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer state;
  NativeHostSolvePreparationRuntime preparation;
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                            fixture.lease, fixture.intent,
                            fixture.dense_state, &view, &state, &preparation,
                            &error),
                        error);
  CudaFullLmOptions options = NativeFullLmOptions(view.config);
  CudaFullLmResult legacy;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(fixture.reference, options, &legacy, &error), error);
  NativeCudaSolveRequest request{&view, &state, &options};
  BaSolveResult native;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(request, &native, &error), error);
  BOOST_REQUIRE_GE(legacy.accepted_commits, 1);
  BOOST_REQUIRE_GE(legacy.runtime.final_internal_state_epoch, 1);
  BOOST_CHECK_EQUAL(native.success, legacy.success);
  BOOST_CHECK_EQUAL(native.termination_reason, legacy.termination_reason);
  BOOST_CHECK_EQUAL(native.trial_iterations, legacy.trial_iterations);
  BOOST_CHECK_EQUAL(native.accepted_commits, legacy.accepted_commits);
  BOOST_CHECK_EQUAL(native.rejected_steps, legacy.rejected_steps);
  BOOST_CHECK_EQUAL(native.invalid_steps, legacy.invalid_steps);
  BOOST_CHECK_EQUAL(native.final_internal_state_epoch,
                    legacy.runtime.final_internal_state_epoch);
  BOOST_CHECK_EQUAL(native.initial_cost, legacy.initial_cost);
  BOOST_CHECK_EQUAL(native.final_cost, legacy.final_cost);
  BOOST_CHECK_EQUAL(native.final_state.state_generation,
                    state.state_generation + native.final_internal_state_epoch);
  BOOST_CHECK_EQUAL(native.runtime.dense_active_state_device_download_calls, 1);
  BOOST_REQUIRE_EQUAL(native.final_state.cameras.size(),
                      legacy.final_state.cameras.size());
  BOOST_REQUIRE_EQUAL(native.final_state.images.size(),
                      legacy.final_state.images.size());
  BOOST_REQUIRE_EQUAL(native.final_state.points.size(),
                      legacy.final_state.points.size());
  for (size_t i = 0; i < native.final_state.cameras.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.final_state.cameras[i].parameters.begin(),
        native.final_state.cameras[i].parameters.end(),
        legacy.final_state.cameras[i].params.begin(),
        legacy.final_state.cameras[i].params.end());
  }
  for (size_t i = 0; i < native.final_state.images.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.final_state.images[i].quaternion.begin(),
        native.final_state.images[i].quaternion.end(),
        legacy.final_state.images[i].qvec.begin(),
        legacy.final_state.images[i].qvec.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.final_state.images[i].translation.begin(),
        native.final_state.images[i].translation.end(),
        legacy.final_state.images[i].tvec.begin(),
        legacy.final_state.images[i].tvec.end());
  }
  for (size_t i = 0; i < native.final_state.points.size(); ++i) {
    const uint64_t point3D_id =
        fixture.lease.points()[native.final_state.points[i].point_slot]
            .point3D_id;
    const auto legacy_point = std::find_if(
        legacy.final_state.points.begin(), legacy.final_state.points.end(),
        [point3D_id](const PointSnapshot& point) {
          return point.point3D_id == point3D_id;
        });
    BOOST_REQUIRE(legacy_point != legacy.final_state.points.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.final_state.points[i].xyz.begin(),
        native.final_state.points[i].xyz.end(),
        legacy_point->xyz.begin(), legacy_point->xyz.end());
  }
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap
