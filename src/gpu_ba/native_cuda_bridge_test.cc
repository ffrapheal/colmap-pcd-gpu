#define TEST_NAME "gpu_ba/native_cuda_bridge"
#include "util/testing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "base/reconstruction.h"
#include "gpu_ba/custom_cuda.h"
#include "gpu_ba/native_graph_problem_store.h"

namespace colmap {
namespace gpu_ba {
namespace {

struct ShadowFixture {
  HostBaGraphStore store{44};
  std::shared_ptr<DeviceBaProblemStoreHandle> device_store =
      CreateDeviceBaProblemStore(44);
  CatalogReadLease lease;
  DenseActiveState dense_state;
  BaSolveIntent intent;
  Snapshot reference;
};

struct DirectNativeFixture {
  Reconstruction reconstruction;
  CorrespondenceGraph graph;
  point3D_t first_point_id = 0;
  point3D_t second_point_id = 0;
  uint64_t owner_epoch = 4044;
  bool journal_started = false;
  std::unique_ptr<GpuBaHostProblemStore> store;
  CudaHostStoreBinding binding;
  CudaFullLmOptions options;
  NativeBaSolveIntent intent;

  ~DirectNativeFixture() {
    if (store != nullptr) {
      std::string ignored;
      store->Shutdown(&ignored);
      store.reset();
    }
    if (journal_started) reconstruction.EndStructureJournal();
  }
};

CudaFullLmOptions NativeFullLmOptions(
    const NativeCudaResolvedConfig& config) {
  CudaFullLmOptions options;
  options.arithmetic_precision = config.arithmetic_precision;
  options.device_context_mode = config.device_context;
  options.hot_kernel_mode = config.hot_kernel;
  options.execution_profile = config.execution_profile;
  options.audit_profile = config.audit_profile;
  options.prepared_selection_cache_mode = config.prepared_selection_cache;
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

bool BuildDirectNativeFixture(DirectNativeFixture* fixture,
                              std::string* error) {
  fixture->first_point_id = fixture->reconstruction.AddPoint3D(
      Eigen::Vector3d(1.0, 2.0, 3.0), Track());
  fixture->second_point_id = fixture->reconstruction.AddPoint3D(
      Eigen::Vector3d(-1.0, 0.5, 4.0), Track());

  Camera camera;
  camera.InitializeWithId(OpenCVCameraModel::model_id, 500.0, 640, 480);
  camera.SetCameraId(7);
  fixture->reconstruction.AddCamera(camera);

  const auto add_image = [&](const image_t image_id,
                             const double xy_offset) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(7);
    image.SetName(std::to_string(image_id));
    image.SetQvec(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    image.SetTvec(Eigen::Vector3d(0.01 * image_id, 0.0, 8.0));
    image.SetRegistered(true);
    std::vector<Eigen::Vector2d> points2D;
    for (point2D_t index = 0; index < 4; ++index) {
      points2D.emplace_back(xy_offset + index,
                            xy_offset + 10.0 + index);
    }
    fixture->graph.AddImage(image_id, points2D.size());
    image.SetPoints2D(points2D);
    fixture->reconstruction.AddImage(image);
  };
  add_image(20, 200.0);
  add_image(21, 210.0);
  add_image(22, 220.0);
  fixture->reconstruction.SetUp(&fixture->graph);
  fixture->reconstruction.AddObservation(
      fixture->first_point_id, TrackElement{20, 1});
  fixture->reconstruction.AddObservation(
      fixture->first_point_id, TrackElement{21, 2});
  fixture->reconstruction.AddObservation(
      fixture->first_point_id, TrackElement{22, 0});
  fixture->reconstruction.AddObservation(
      fixture->second_point_id, TrackElement{20, 3});
  fixture->reconstruction.AddObservation(
      fixture->second_point_id, TrackElement{21, 0});

  fixture->reconstruction.BeginStructureJournal(fixture->owner_epoch, 64);
  fixture->journal_started = true;
  fixture->store.reset(
      new GpuBaHostProblemStore(&fixture->reconstruction,
                                fixture->owner_epoch));
  fixture->binding.store = fixture->store.get();
  fixture->binding.owner_epoch = fixture->owner_epoch;
  fixture->binding.mode = CudaHostProblemStoreMode::kHostPreparedStore;

  NativeBaSolveIntent& intent = fixture->intent;
  intent.owner_epoch = fixture->owner_epoch;
  intent.reconstruction_identity =
      reinterpret_cast<uintptr_t>(&fixture->reconstruction);
  intent.expected_topology_revision =
      fixture->reconstruction.StructureRevision();
  intent.selection_revision = 12;
  intent.kind = BaKind::kLocal;
  intent.visual_observation_scope =
      NativeBaVisualObservationScope::kActiveImagesOnly;
  intent.config.resolved = true;
  intent.config.config_generation = 13;
  intent.config.arithmetic_precision = CudaArithmeticPrecision::kFp64;
  intent.config.hessian_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  intent.config.schur_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  intent.config.hot_kernel = CudaHotKernelMode::kTransformed;
  intent.config.execution_profile =
      CudaExecutionProfile::kCompactControl;
  intent.config.residual_order = CudaResidualOrder::kSourceInsertion;
  intent.config.prepared_selection_cache =
      CudaPreparedSelectionCacheMode::kEnabled;
  intent.config.loss_mode = CudaLossMode::kTrivial;
  intent.config.loss_scale = 1.0;
  intent.active_image_ids = {21, 20};
  intent.explicit_variable_point_ids = {fixture->first_point_id};
  intent.lidar_map_generation = 17;
  intent.lidar_match_config_generation = 13;
  intent.online_lidar_identity.valid = true;
  intent.online_lidar_identity.trigger_image_id = 21;
  intent.online_lidar_identity.map_version = 17;
  intent.online_lidar_identity.max_scan_index = 17;
  intent.online_lidar_identity.snapshot_sha256.fill(0x41);
  intent.online_lidar_identity.geometry_sha256.fill(0x52);
  intent.online_lidar_identity.association_sha256.fill(0x63);

  NativeBaLidarConstraint first;
  first.point3D_id = fixture->first_point_id;
  first.constraint_slot = 1;
  first.physical_identity = 1;
  first.association_id = 0;
  first.owner_image_id = 21;
  first.owner_point2D_idx = 2;
  first.lidar_type = 1;
  first.frozen_point3D_xyz = {{1.0, 2.0, 3.0}};
  first.plane = {{0.0, 0.0, 1.0, -3.0}};
  first.lidar_xyz = {{1.1, 2.1, 3.1}};
  first.weight = 2.0;
  first.search_range = 0.25;
  intent.lidar_constraints.push_back(first);

  NativeBaLidarConstraint second;
  second.point3D_id = fixture->second_point_id;
  second.constraint_slot = 0;
  second.physical_identity = 2;
  second.association_id = 1;
  second.owner_image_id = 20;
  second.owner_point2D_idx = 3;
  second.lidar_type = 2;
  second.frozen_point3D_xyz = {{-1.0, 0.5, 4.0}};
  second.plane = {{1.0, 0.0, 0.0, 1.0}};
  second.lidar_xyz = {{-1.1, 0.6, 4.1}};
  second.weight = 3.0;
  second.search_range = 0.5;
  intent.lidar_constraints.push_back(second);

  fixture->options = NativeFullLmOptions(intent.config);
  error->clear();
  return true;
}

bool PrepareDirectNative(DirectNativeFixture* fixture,
                         const NativeBaSolveIntent& intent,
                         PreparedNativeActiveSolve* prepared,
                         std::string* error) {
  return PrepareCudaNativeBaSolve(intent, &fixture->reconstruction,
                                  fixture->options, fixture->binding,
                                  prepared, error);
}

void UseLegacyLidarProvenance(NativeBaSolveIntent* intent) {
  intent->visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  intent->online_lidar_identity = NativeBaOnlineLidarIdentity();
  for (NativeBaLidarConstraint& constraint : intent->lidar_constraints) {
    constraint.association_id = kBaGraphInvalidAssociationId;
    constraint.owner_image_id = kBaGraphInvalidSlot;
    constraint.owner_point2D_idx = kBaGraphInvalidSlot;
    constraint.frozen_point3D_xyz = {{0.0, 0.0, 0.0}};
  }
}

const LidarConstraintRecord* FindLidarRecord(
    const NativeHostSolveView& view, const uint64_t physical_identity) {
  const std::vector<LidarConstraintRecord>& constraints =
      view.LidarConstraints();
  const auto found = std::find_if(
      constraints.begin(), constraints.end(),
      [physical_identity](const LidarConstraintRecord& constraint) {
        return constraint.physical_identity == physical_identity;
      });
  return found == constraints.end() ? nullptr : &*found;
}

void CheckPackedInputsEqual(const LegacyKernelInputBundle& lhs,
                            const LegacyKernelInputBundle& rhs) {
  BOOST_REQUIRE_EQUAL(lhs.visual().size(), rhs.visual().size());
  BOOST_REQUIRE_EQUAL(lhs.lidar().size(), rhs.lidar().size());
  for (size_t index = 0; index < lhs.visual().size(); ++index) {
    const CudaVisualInput& a = lhs.visual()[index];
    const CudaVisualInput& b = rhs.visual()[index];
    BOOST_CHECK_EQUAL(a.source_index, b.source_index);
    BOOST_CHECK_EQUAL(a.image_id, b.image_id);
    BOOST_CHECK_EQUAL(a.point3D_id, b.point3D_id);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.quaternion),
                                  std::end(a.quaternion),
                                  std::begin(b.quaternion),
                                  std::end(b.quaternion));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.translation),
                                  std::end(a.translation),
                                  std::begin(b.translation),
                                  std::end(b.translation));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.point), std::end(a.point),
                                  std::begin(b.point), std::end(b.point));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.camera), std::end(a.camera),
                                  std::begin(b.camera), std::end(b.camera));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.observation),
                                  std::end(a.observation),
                                  std::begin(b.observation),
                                  std::end(b.observation));
  }
  for (size_t index = 0; index < lhs.lidar().size(); ++index) {
    const CudaLidarInput& a = lhs.lidar()[index];
    const CudaLidarInput& b = rhs.lidar()[index];
    BOOST_CHECK_EQUAL(a.source_index, b.source_index);
    BOOST_CHECK_EQUAL(a.point3D_id, b.point3D_id);
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.point), std::end(a.point),
                                  std::begin(b.point), std::end(b.point));
    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a.plane), std::end(a.plane),
                                  std::begin(b.plane), std::end(b.plane));
    BOOST_CHECK_EQUAL(a.weight, b.weight);
    BOOST_CHECK_EQUAL(a.mode, b.mode);
    BOOST_CHECK_EQUAL(a.near_zero_threshold, b.near_zero_threshold);
  }
}

void CheckOnlineRecordMatches(const NativeHostSolveView& view,
                              const NativeBaLidarConstraint& expected) {
  const LidarConstraintRecord* record =
      FindLidarRecord(view, expected.physical_identity);
  BOOST_REQUIRE(record != nullptr);
  BOOST_CHECK_EQUAL(record->constraint_slot, expected.constraint_slot);
  BOOST_CHECK_EQUAL(record->association_id, expected.association_id);
  BOOST_CHECK_EQUAL(record->owner_image_id, expected.owner_image_id);
  BOOST_CHECK_EQUAL(record->owner_point2D_idx, expected.owner_point2D_idx);
  BOOST_CHECK_EQUAL(record->lidar_type, expected.lidar_type);
  BOOST_CHECK_EQUAL_COLLECTIONS(record->frozen_point3D_xyz.begin(),
                                record->frozen_point3D_xyz.end(),
                                expected.frozen_point3D_xyz.begin(),
                                expected.frozen_point3D_xyz.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(record->plane.begin(), record->plane.end(),
                                expected.plane.begin(), expected.plane.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(record->lidar_xyz.begin(),
                                record->lidar_xyz.end(),
                                expected.lidar_xyz.begin(),
                                expected.lidar_xyz.end());
  BOOST_CHECK_EQUAL(record->weight, expected.weight);
  BOOST_CHECK_EQUAL(record->search_range, expected.search_range);

  const std::vector<ResidualOrdinal>& ordinals = view.ResidualOrdinals();
  const auto ordinal = std::find_if(
      ordinals.begin(), ordinals.end(),
      [&](const ResidualOrdinal& value) {
        return value.kind == ResidualKind::kLidar &&
               value.source_slot == expected.constraint_slot;
      });
  BOOST_REQUIRE(ordinal != ordinals.end());
  BOOST_CHECK_EQUAL(ordinal->physical_identity, expected.physical_identity);
  BOOST_CHECK_EQUAL(ordinal->association_id, expected.association_id);
  BOOST_CHECK_EQUAL(ordinal->owner_image_id, expected.owner_image_id);
  BOOST_CHECK_EQUAL(ordinal->owner_point2D_idx,
                    expected.owner_point2D_idx);
}

bool BuildShadowFixture(ShadowFixture* fixture, std::string* error) {
  HostBaGraphColdInput graph;
  graph.owner_epoch = 44;
  graph.topology_revision = 3;
  graph.cameras.push_back({7, 4, 640, 480, 8});
  graph.images.push_back({20, 7, true});
  graph.images.push_back({21, 7, true});
  graph.points.push_back({30});
  graph.points.push_back({31});
  graph.observations.push_back({20, 0, 30, {{10.0, 11.0}}});
  graph.observations.push_back({21, 1, 30, {{12.0, 13.0}}});
  graph.observations.push_back({20, 2, 31, {{99.0, 101.0}}});
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
  image.quaternion = {{0.0, 1.0, 0.0, 0.0}};
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
  fixture->intent.active_visual_observation_slots_explicit = true;
  fixture->intent.active_visual_observation_slots.push_back(0);
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

BOOST_AUTO_TEST_CASE(
    DirectNativeActiveOnlyOnlineProvenanceAndPackingParity) {
  DirectNativeFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildDirectNativeFixture(&fixture, &error), error);
  NativeBaSolveIntent active_only_intent = fixture.intent;
  active_only_intent.config.prepared_selection_cache =
      CudaPreparedSelectionCacheMode::kDisabled;
  PreparedNativeActiveSolve prepared;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, active_only_intent, &prepared, &error),
      error);
  const NativeHostSolveView* view = prepared.view();
  const ActiveStateBuffer* state = prepared.initial_state();
  BOOST_REQUIRE(view != nullptr);
  BOOST_REQUIRE(state != nullptr);
  BOOST_CHECK(view->identity.visual_observation_scope ==
              NativeBaVisualObservationScope::kActiveImagesOnly);
  BOOST_CHECK(SameNativeBaOnlineLidarIdentity(
      view->identity.online_lidar_identity,
      active_only_intent.online_lidar_identity));
  BOOST_CHECK_EQUAL(view->identity.online_lidar_identity.map_version,
                    view->identity.online_lidar_identity.max_scan_index);
  BOOST_CHECK_EQUAL(view->identity.online_lidar_identity.trigger_image_id, 21);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      view->identity.online_lidar_identity.snapshot_sha256.begin(),
      view->identity.online_lidar_identity.snapshot_sha256.end(),
      active_only_intent.online_lidar_identity.snapshot_sha256.begin(),
      active_only_intent.online_lidar_identity.snapshot_sha256.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      view->identity.online_lidar_identity.geometry_sha256.begin(),
      view->identity.online_lidar_identity.geometry_sha256.end(),
      active_only_intent.online_lidar_identity.geometry_sha256.begin(),
      active_only_intent.online_lidar_identity.geometry_sha256.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      view->identity.online_lidar_identity.association_sha256.begin(),
      view->identity.online_lidar_identity.association_sha256.end(),
      active_only_intent.online_lidar_identity.association_sha256.begin(),
      active_only_intent.online_lidar_identity.association_sha256.end());

  BOOST_REQUIRE_EQUAL(view->ActiveImageSlots().size(), 2);
  BOOST_CHECK_EQUAL(
      view->catalog.images()[view->ActiveImageSlots()[0]].image_id, 21);
  BOOST_CHECK_EQUAL(
      view->catalog.images()[view->ActiveImageSlots()[1]].image_id, 20);
  BOOST_CHECK(view->BoundaryImageSlots().empty());
  BOOST_REQUIRE_EQUAL(view->VisualObservationSlots().size(), 4);
  const std::array<uint32_t, 4> expected_images{{21, 21, 20, 20}};
  const std::array<uint32_t, 4> expected_point2D{{0, 2, 1, 3}};
  for (size_t index = 0; index < expected_images.size(); ++index) {
    const HostBaObservationSlot& observation =
        view->catalog.observations()[view->VisualObservationSlots()[index]];
    BOOST_CHECK_EQUAL(
        view->catalog.images()[observation.image_slot].image_id,
        expected_images[index]);
    BOOST_CHECK_EQUAL(observation.point2D_idx, expected_point2D[index]);
  }
  const HostBaPointSlot* first_point =
      view->catalog.FindPointById(fixture.first_point_id);
  BOOST_REQUIRE(first_point != nullptr);
  const auto point_policy = std::find_if(
      view->Fixed().points.begin(), view->Fixed().points.end(),
      [&](const PointFixedPolicyResult& value) {
        return value.point_slot == first_point->header.slot;
      });
  BOOST_REQUIRE(point_policy != view->Fixed().points.end());
  BOOST_CHECK_EQUAL(point_policy->constant, 0);

  BOOST_REQUIRE_EQUAL(view->LidarConstraints().size(), 2);
  CheckOnlineRecordMatches(*view, active_only_intent.lidar_constraints[0]);
  CheckOnlineRecordMatches(*view, active_only_intent.lidar_constraints[1]);
  for (const ResidualOrdinal& ordinal : view->ResidualOrdinals()) {
    if (ordinal.kind != ResidualKind::kVisual) continue;
    BOOST_CHECK_EQUAL(ordinal.association_id,
                      kBaGraphInvalidAssociationId);
    BOOST_CHECK_EQUAL(ordinal.owner_image_id, kBaGraphInvalidSlot);
    BOOST_CHECK_EQUAL(ordinal.owner_point2D_idx, kBaGraphInvalidSlot);
  }

  LegacyKernelInputBundle layer_a_only;
  BOOST_REQUIRE_MESSAGE(
      BuildCudaLayerAInputs(*view, *state, &layer_a_only, &error), error);
  LegacyKernelInputBundle full_online;
  BOOST_REQUIRE_MESSAGE(
      BuildLegacyKernelInputBundle(*view, *state, &full_online, &error),
      error);
  CheckPackedInputsEqual(layer_a_only, full_online);
  BOOST_REQUIRE_EQUAL(full_online.visual().size(), 4);
  BOOST_REQUIRE_EQUAL(full_online.lidar().size(), 2);
  for (size_t index = 0; index < full_online.visual().size(); ++index) {
    BOOST_CHECK_EQUAL(full_online.visual()[index].source_index, index);
    BOOST_CHECK_EQUAL(full_online.visual()[index].image_id,
                      expected_images[index]);
  }
  BOOST_CHECK_EQUAL(full_online.lidar()[0].source_index, 4);
  BOOST_CHECK_EQUAL(full_online.lidar()[1].source_index, 5);
  BOOST_CHECK_EQUAL(full_online.lidar()[0].point3D_id,
                    fixture.first_point_id);
  BOOST_CHECK_EQUAL(full_online.lidar()[1].point3D_id,
                    fixture.second_point_id);

  NativeHostSolveView legacy_provenance_view = *view;
  legacy_provenance_view.identity.visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  legacy_provenance_view.identity.online_lidar_identity =
      NativeBaOnlineLidarIdentity();
  legacy_provenance_view.lidar.online_lidar_identity =
      NativeBaOnlineLidarIdentity();
  for (LidarConstraintRecord& constraint :
       legacy_provenance_view.lidar.constraints) {
    constraint.association_id = kBaGraphInvalidAssociationId;
    constraint.owner_image_id = kBaGraphInvalidSlot;
    constraint.owner_point2D_idx = kBaGraphInvalidSlot;
    constraint.frozen_point3D_xyz = {{0.0, 0.0, 0.0}};
  }
  for (ResidualOrdinal& ordinal : legacy_provenance_view.residual_ordinals) {
    if (ordinal.kind != ResidualKind::kLidar) continue;
    ordinal.association_id = kBaGraphInvalidAssociationId;
    ordinal.owner_image_id = kBaGraphInvalidSlot;
    ordinal.owner_point2D_idx = kBaGraphInvalidSlot;
  }
  BOOST_REQUIRE_MESSAGE(ValidateNativeHostSolveView(
                            legacy_provenance_view, *state, &error),
                        error);
  LegacyKernelInputBundle legacy_provenance;
  BOOST_REQUIRE_MESSAGE(BuildLegacyKernelInputBundle(
                            legacy_provenance_view, *state,
                            &legacy_provenance, &error),
                        error);
  CheckPackedInputsEqual(full_online, legacy_provenance);
  prepared.Release();

  NativeBaSolveIntent legacy_scope_intent = active_only_intent;
  UseLegacyLidarProvenance(&legacy_scope_intent);
  PreparedNativeActiveSolve legacy_scope;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, legacy_scope_intent, &legacy_scope,
                          &error),
      error);
  BOOST_REQUIRE(legacy_scope.view() != nullptr);
  BOOST_REQUIRE_EQUAL(legacy_scope.view()->BoundaryImageSlots().size(), 1);
  BOOST_CHECK_EQUAL(
      legacy_scope.view()
          ->catalog.images()[legacy_scope.view()->BoundaryImageSlots()[0]]
          .image_id,
      22);
  BOOST_REQUIRE_EQUAL(legacy_scope.view()->VisualObservationSlots().size(), 5);
  const HostBaObservationSlot& expanded =
      legacy_scope.view()->catalog.observations()[
          legacy_scope.view()->VisualObservationSlots().back()];
  BOOST_CHECK_EQUAL(
      legacy_scope.view()->catalog.images()[expanded.image_slot].image_id, 22);
  BOOST_CHECK_EQUAL(expanded.point2D_idx, 0);
  const auto legacy_point_policy = std::find_if(
      legacy_scope.view()->Fixed().points.begin(),
      legacy_scope.view()->Fixed().points.end(),
      [&](const PointFixedPolicyResult& value) {
        return value.point_slot == first_point->header.slot;
      });
  BOOST_REQUIRE(legacy_point_policy !=
                legacy_scope.view()->Fixed().points.end());
  BOOST_CHECK_EQUAL(legacy_point_policy->constant, 0);
  legacy_scope.Release();
}

BOOST_AUTO_TEST_CASE(DirectNativePreparedCacheKeysOnlineProvenance) {
  DirectNativeFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildDirectNativeFixture(&fixture, &error), error);

  PreparedNativeActiveSolve cold;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, fixture.intent, &cold, &error), error);
  BOOST_CHECK_EQUAL(cold.runtime().host_plan_hits, 0);
  BOOST_CHECK_EQUAL(cold.runtime().host_plan_misses, 1);
  BOOST_REQUIRE(cold.view() != nullptr);
  BOOST_REQUIRE(cold.view()->prepared_plan != nullptr);
  const std::shared_ptr<const PreparedSelectionPlan> original_plan =
      cold.view()->prepared_plan;
  const NativeHostSolveViewIdentity original_identity = cold.view()->identity;
  const std::vector<LidarConstraintRecord> original_records =
      cold.view()->LidarConstraints();
  cold.Release();

  PreparedNativeActiveSolve hot;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, fixture.intent, &hot, &error), error);
  BOOST_CHECK_EQUAL(hot.runtime().host_plan_hits, 1);
  BOOST_CHECK_EQUAL(hot.runtime().host_plan_misses, 0);
  BOOST_REQUIRE(hot.view() != nullptr);
  BOOST_CHECK(hot.view()->prepared_plan.get() == original_plan.get());
  BOOST_CHECK(SameNativeBaOnlineLidarIdentity(
      hot.view()->identity.online_lidar_identity,
      original_identity.online_lidar_identity));
  BOOST_REQUIRE_EQUAL(hot.view()->LidarConstraints().size(),
                      original_records.size());
  for (size_t index = 0; index < original_records.size(); ++index) {
    BOOST_CHECK_EQUAL(hot.view()->LidarConstraints()[index].association_id,
                      original_records[index].association_id);
    BOOST_CHECK_EQUAL(hot.view()->LidarConstraints()[index].owner_image_id,
                      original_records[index].owner_image_id);
    BOOST_CHECK_EQUAL(hot.view()->LidarConstraints()[index].owner_point2D_idx,
                      original_records[index].owner_point2D_idx);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        hot.view()->LidarConstraints()[index].frozen_point3D_xyz.begin(),
        hot.view()->LidarConstraints()[index].frozen_point3D_xyz.end(),
        original_records[index].frozen_point3D_xyz.begin(),
        original_records[index].frozen_point3D_xyz.end());
  }
  hot.Release();

  const auto expect_miss = [&](const NativeBaSolveIntent& candidate) {
    PreparedNativeActiveSolve miss;
    error.clear();
    BOOST_REQUIRE_MESSAGE(
        PrepareDirectNative(&fixture, candidate, &miss, &error), error);
    BOOST_CHECK_EQUAL(miss.runtime().host_plan_hits, 0);
    BOOST_CHECK_EQUAL(miss.runtime().host_plan_misses, 1);
    BOOST_REQUIRE(miss.view() != nullptr);
    BOOST_REQUIRE(miss.view()->prepared_plan != nullptr);
    BOOST_CHECK(miss.view()->prepared_plan.get() != original_plan.get());
    BOOST_CHECK(SameNativeBaOnlineLidarIdentity(
        miss.view()->identity.online_lidar_identity,
        candidate.online_lidar_identity));
    for (const NativeBaLidarConstraint& expected :
         candidate.lidar_constraints) {
      CheckOnlineRecordMatches(*miss.view(), expected);
    }
    miss.Release();
  };

  NativeBaSolveIntent candidate = fixture.intent;
  candidate.online_lidar_identity.snapshot_sha256[0] ^= 0x1;
  expect_miss(candidate);
  candidate = fixture.intent;
  candidate.online_lidar_identity.geometry_sha256[0] ^= 0x1;
  expect_miss(candidate);
  candidate = fixture.intent;
  candidate.online_lidar_identity.association_sha256[0] ^= 0x1;
  expect_miss(candidate);
  candidate = fixture.intent;
  candidate.lidar_constraints[0].owner_image_id = 20;
  candidate.lidar_constraints[0].owner_point2D_idx = 1;
  expect_miss(candidate);
  candidate = fixture.intent;
  std::swap(candidate.lidar_constraints[0].association_id,
            candidate.lidar_constraints[1].association_id);
  std::swap(candidate.lidar_constraints[0].physical_identity,
            candidate.lidar_constraints[1].physical_identity);
  expect_miss(candidate);

  PreparedNativeActiveSolve original_again;
  BOOST_REQUIRE_MESSAGE(PrepareDirectNative(
                            &fixture, fixture.intent, &original_again, &error),
                        error);
  BOOST_CHECK_EQUAL(original_again.runtime().host_plan_hits, 1);
  BOOST_REQUIRE(original_again.view() != nullptr);
  BOOST_CHECK(original_again.view()->prepared_plan.get() ==
              original_plan.get());
  BOOST_CHECK(SameNativeBaOnlineLidarIdentity(
      original_again.view()->identity.online_lidar_identity,
      fixture.intent.online_lidar_identity));
  CheckOnlineRecordMatches(*original_again.view(),
                           fixture.intent.lidar_constraints[0]);
  CheckOnlineRecordMatches(*original_again.view(),
                           fixture.intent.lidar_constraints[1]);
  original_again.Release();
}

BOOST_AUTO_TEST_CASE(DirectNativeOnlineIdentitySurvivesEmptyConstraintCache) {
  DirectNativeFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildDirectNativeFixture(&fixture, &error), error);
  NativeBaSolveIntent intent = fixture.intent;
  intent.lidar_constraints.clear();

  PreparedNativeActiveSolve cold;
  BOOST_REQUIRE_MESSAGE(PrepareDirectNative(&fixture, intent, &cold, &error),
                        error);
  BOOST_REQUIRE(cold.view() != nullptr);
  BOOST_CHECK(cold.view()->LidarConstraints().empty());
  BOOST_CHECK(cold.view()->identity.online_lidar_identity.valid);
  BOOST_CHECK(SameNativeBaOnlineLidarIdentity(
      cold.view()->identity.online_lidar_identity,
      intent.online_lidar_identity));
  BOOST_REQUIRE(cold.view()->prepared_plan != nullptr);
  BOOST_CHECK(cold.view()->prepared_plan->online_lidar_identity.valid);
  cold.Release();

  PreparedNativeActiveSolve hot;
  BOOST_REQUIRE_MESSAGE(PrepareDirectNative(&fixture, intent, &hot, &error),
                        error);
  BOOST_CHECK_EQUAL(hot.runtime().host_plan_hits, 1);
  BOOST_REQUIRE(hot.view() != nullptr);
  BOOST_CHECK(hot.view()->LidarConstraints().empty());
  BOOST_CHECK(SameNativeBaOnlineLidarIdentity(
      hot.view()->identity.online_lidar_identity,
      intent.online_lidar_identity));
  hot.Release();
}

BOOST_AUTO_TEST_CASE(DirectNativeOnlineProvenanceFailsClosedBeforeCacheReuse) {
  DirectNativeFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildDirectNativeFixture(&fixture, &error), error);
  PreparedNativeActiveSolve primed;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, fixture.intent, &primed, &error), error);
  primed.Release();

  const auto reject = [&](const NativeBaSolveIntent& candidate,
                          const char* context) {
    BOOST_TEST_CONTEXT(context) {
      PreparedNativeActiveSolve rejected;
      BOOST_REQUIRE_MESSAGE(
          PrepareDirectNative(&fixture, fixture.intent, &rejected, &error),
          error);
      BOOST_REQUIRE(rejected.valid());
      BOOST_CHECK_EQUAL(rejected.runtime().host_plan_hits, 1);
      error.clear();
      BOOST_CHECK(
          !PrepareDirectNative(&fixture, candidate, &rejected, &error));
      BOOST_CHECK(!rejected.valid());
      BOOST_CHECK(rejected.view() == nullptr);
      BOOST_CHECK(rejected.initial_state() == nullptr);
      BOOST_CHECK(!error.empty());
    }
  };

  NativeBaSolveIntent candidate = fixture.intent;
  candidate.visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  reject(candidate, "online identity with legacy scope");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.visual_observation_scope =
      NativeBaVisualObservationScope::kActiveImagesOnly;
  reject(candidate, "active-only scope without online identity");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.online_lidar_identity.trigger_image_id = 21;
  reject(candidate, "invalid identity retains trigger");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.online_lidar_identity.map_version = 17;
  reject(candidate, "invalid identity retains map version");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.online_lidar_identity.snapshot_sha256[0] = 1;
  reject(candidate, "invalid identity retains digest");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.lidar_constraints[0].association_id = 0;
  reject(candidate, "legacy constraint retains association");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.lidar_constraints[0].owner_image_id = 21;
  candidate.lidar_constraints[0].owner_point2D_idx = 2;
  reject(candidate, "legacy constraint retains owner");
  candidate = fixture.intent;
  UseLegacyLidarProvenance(&candidate);
  candidate.lidar_constraints[0].frozen_point3D_xyz[0] = 1.0;
  reject(candidate, "legacy constraint retains frozen point");
  candidate = fixture.intent;
  candidate.online_lidar_identity.snapshot_sha256.fill(0);
  reject(candidate, "zero snapshot digest");
  candidate = fixture.intent;
  candidate.online_lidar_identity.geometry_sha256.fill(0);
  reject(candidate, "zero geometry digest");
  candidate = fixture.intent;
  candidate.online_lidar_identity.association_sha256.fill(0);
  reject(candidate, "zero association digest");
  candidate = fixture.intent;
  candidate.online_lidar_identity.snapshot_sha256.fill(0);
  candidate.online_lidar_identity.geometry_sha256.fill(0);
  candidate.online_lidar_identity.association_sha256.fill(0);
  reject(candidate, "all zero digests");
  candidate = fixture.intent;
  candidate.online_lidar_identity.trigger_image_id = kBaGraphInvalidSlot;
  reject(candidate, "invalid trigger sentinel");
  candidate = fixture.intent;
  candidate.online_lidar_identity.trigger_image_id = 22;
  reject(candidate, "trigger outside active images");
  candidate = fixture.intent;
  candidate.online_lidar_identity.map_version += 1;
  reject(candidate, "map and max scan mismatch");
  candidate = fixture.intent;
  ++candidate.lidar_map_generation;
  reject(candidate, "LiDAR map generation differs from online identity");
  candidate = fixture.intent;
  ++candidate.lidar_match_config_generation;
  reject(candidate, "LiDAR match generation differs from resolved config");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].owner_image_id = 22;
  candidate.lidar_constraints[0].owner_point2D_idx = 0;
  reject(candidate, "owner outside active images");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].owner_image_id = 999;
  reject(candidate, "owner does not exist");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].owner_point2D_idx = 99;
  reject(candidate, "owner point2D missing");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].owner_point2D_idx = 0;
  reject(candidate, "owner point2D does not reference point");
  candidate = fixture.intent;
  candidate.lidar_constraints[1].point3D_id = fixture.first_point_id;
  candidate.lidar_constraints[1].owner_image_id = 20;
  candidate.lidar_constraints[1].owner_point2D_idx = 1;
  candidate.lidar_constraints[1].frozen_point3D_xyz = {{1.0, 2.0, 3.0}};
  reject(candidate, "duplicate point association");
  candidate = fixture.intent;
  candidate.lidar_constraints[1].constraint_slot = 1;
  reject(candidate, "duplicate constraint slot");
  candidate = fixture.intent;
  candidate.lidar_constraints[1].constraint_slot = 2;
  reject(candidate, "non-dense constraint slot");
  candidate = fixture.intent;
  candidate.lidar_constraints[1].association_id = 0;
  reject(candidate, "duplicate association id");
  candidate = fixture.intent;
  candidate.lidar_constraints[1].association_id = 2;
  reject(candidate, "non-dense association id");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].physical_identity = 0;
  reject(candidate, "zero physical identity");
  candidate = fixture.intent;
  candidate.lidar_constraints[1].physical_identity =
      candidate.lidar_constraints[0].physical_identity;
  reject(candidate, "duplicate physical identity");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].physical_identity = 3;
  reject(candidate, "physical identity differs from association plus one");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].plane[0] =
      std::numeric_limits<double>::quiet_NaN();
  reject(candidate, "non-finite constraint");
  candidate = fixture.intent;
  candidate.lidar_constraints[0].frozen_point3D_xyz[0] += 0.25;
  reject(candidate, "frozen point mismatch");

  PreparedNativeActiveSolve recovered;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, fixture.intent, &recovered, &error),
      error);
  BOOST_CHECK_EQUAL(recovered.runtime().host_plan_hits, 1);
  BOOST_REQUIRE(recovered.view() != nullptr);
  CheckOnlineRecordMatches(*recovered.view(),
                           fixture.intent.lidar_constraints[0]);
  recovered.Release();
}

BOOST_AUTO_TEST_CASE(DirectNativeFailedReuseReleasesPreparedState) {
  DirectNativeFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildDirectNativeFixture(&fixture, &error), error);
  PreparedNativeActiveSolve prepared;
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, fixture.intent, &prepared, &error), error);
  BOOST_REQUIRE(prepared.valid());

  NativeBaSolveIntent invalid = fixture.intent;
  invalid.visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  error.clear();
  BOOST_CHECK(!PrepareDirectNative(&fixture, invalid, &prepared, &error));
  BOOST_CHECK(!error.empty());
  BOOST_CHECK(!prepared.valid());
  BOOST_CHECK(prepared.view() == nullptr);
  BOOST_CHECK(prepared.initial_state() == nullptr);

  Camera extra_camera;
  extra_camera.InitializeWithId(OpenCVCameraModel::model_id, 500.0, 640, 480);
  extra_camera.SetCameraId(8);
  fixture.reconstruction.AddCamera(extra_camera);
  NativeBaSolveIntent recovered = fixture.intent;
  recovered.expected_topology_revision =
      fixture.reconstruction.StructureRevision();
  BOOST_REQUIRE_MESSAGE(
      PrepareDirectNative(&fixture, recovered, &prepared, &error), error);
  BOOST_CHECK(prepared.valid());
  BOOST_CHECK_EQUAL(prepared.runtime().reader_busy, 0);
  BOOST_CHECK_EQUAL(prepared.runtime().catalog_delta_updates, 1);
  prepared.Release();
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
  BOOST_REQUIRE_EQUAL(view.visual_observation_slots.size(), 2);
  BOOST_CHECK(view.identity.visual_observation_scope ==
              NativeBaVisualObservationScope::
                  kLegacyExplicitPointTrackExpansion);
  BOOST_CHECK(!view.identity.online_lidar_identity.valid);
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
  for (const ResidualOrdinal& ordinal : view.residual_ordinals) {
    if (ordinal.kind != ResidualKind::kVisual) continue;
    BOOST_CHECK_EQUAL(ordinal.association_id,
                      kBaGraphInvalidAssociationId);
    BOOST_CHECK_EQUAL(ordinal.owner_image_id, kBaGraphInvalidSlot);
    BOOST_CHECK_EQUAL(ordinal.owner_point2D_idx, kBaGraphInvalidSlot);
  }
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
  request.device_store = fixture.device_store;
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
  BOOST_CHECK_EQUAL(result.runtime.dense_active_state_device_download_calls, 0);
  BOOST_CHECK_GT(result.runtime.variable_state_delta_device_download_calls, 0);
  BOOST_CHECK(result.variable_delta.cameras.empty());
  BOOST_CHECK_EQUAL(result.variable_delta.images.size(), 1);
  BOOST_CHECK_EQUAL(result.variable_delta.points.size(), 1);
  BOOST_CHECK_EQUAL(result.variable_delta.state_generation,
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
  BOOST_CHECK_EQUAL(result.runtime.legacy_kernel_input_bundle_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.build_cuda_layer_a_inputs_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.build_static_layout_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.build_cost_layout_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.build_layer_b_topology_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.build_layer_c_topology_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.repeated_residual_state_packing_bytes, 0);
  BOOST_CHECK_GT(result.runtime.device_store_full_upload_calls, 0);
  const VariableImageStateDelta& image = result.variable_delta.images.front();
  const VariablePointStateDelta& point = result.variable_delta.points.front();
  BOOST_CHECK_EQUAL(fixture.lease.images()[image.image_slot].image_id, 20);
  BOOST_CHECK_EQUAL(fixture.lease.points()[point.point_slot].point3D_id, 30);
  const auto legacy_image = std::find_if(
      legacy_result.final_state.images.begin(),
      legacy_result.final_state.images.end(),
      [](const ImageSnapshot& value) { return value.image_id == 20; });
  const auto legacy_point = std::find_if(
      legacy_result.final_state.points.begin(),
      legacy_result.final_state.points.end(),
      [](const PointSnapshot& value) { return value.point3D_id == 30; });
  BOOST_REQUIRE(legacy_image != legacy_result.final_state.images.end());
  BOOST_REQUIRE(legacy_point != legacy_result.final_state.points.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(image.quaternion.begin(), image.quaternion.end(),
                                legacy_image->qvec.begin(),
                                legacy_image->qvec.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(image.translation.begin(),
                                image.translation.end(),
                                legacy_image->tvec.begin(),
                                legacy_image->tvec.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(point.xyz.begin(), point.xyz.end(),
                                legacy_point->xyz.begin(),
                                legacy_point->xyz.end());
  BaSolveResult repeated;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(request, &repeated, &error), error);
  BOOST_CHECK_EQUAL(repeated.runtime.device_store_full_upload_calls, 0);
  BOOST_CHECK_EQUAL(repeated.runtime.device_store_reuse_calls, 1);
  BOOST_CHECK_EQUAL(repeated.runtime.legacy_kernel_input_bundle_calls, 0);
  BOOST_CHECK_EQUAL(repeated.final_cost, result.final_cost);
}

BOOST_AUTO_TEST_CASE(
    NativeDeviceStorePatchGrowthContextAndFailedPublishRecover) {
  ShadowFixture fixture;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildShadowFixture(&fixture, &error), error);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer state;
  NativeHostSolvePreparationRuntime preparation;
  const auto materialize = [&]() {
    view = NativeHostSolveView();
    state = ActiveStateBuffer();
    preparation = NativeHostSolvePreparationRuntime();
    fixture.intent.catalog_revision = fixture.lease.topology_revision();
    fixture.intent.catalog_generation = fixture.lease.generation();
    return materializer.Materialize(fixture.lease, fixture.intent,
                                    fixture.dense_state, &view, &state,
                                    &preparation, &error);
  };
  BOOST_REQUIRE_MESSAGE(materialize(), error);
  CudaFullLmOptions options = NativeFullLmOptions(view.config);
  const auto run_native = [&](const CudaFullLmOptions& run_options,
                              BaSolveResult* result) {
    NativeCudaSolveRequest request{&view, &state, &run_options,
                                   fixture.device_store};
    return RunCustomCudaSolve(request, result, &error);
  };
  BaSolveResult cold;
  BOOST_REQUIRE_MESSAGE(run_native(options, &cold), error);
  BOOST_CHECK_EQUAL(cold.runtime.device_store_full_upload_calls, 1);
  const uint64_t cold_generation = cold.runtime.device_store_generation;
  uint64_t cold_context_queries = 0;
  BOOST_REQUIRE_MESSAGE(GetDeviceBaProblemStoreContextQueryCountForTesting(
                            fixture.device_store, &cold_context_queries, &error),
                        error);
  BOOST_CHECK_EQUAL(cold_context_queries, 6);
  BaSolveResult unchanged_hot;
  BOOST_REQUIRE_MESSAGE(run_native(options, &unchanged_hot), error);
  BOOST_CHECK_EQUAL(unchanged_hot.runtime.device_store_reuse_calls, 1);
  BOOST_CHECK_EQUAL(unchanged_hot.runtime.device_store_full_upload_calls, 0);
  uint64_t unchanged_hot_context_queries = 0;
  BOOST_REQUIRE_MESSAGE(GetDeviceBaProblemStoreContextQueryCountForTesting(
                            fixture.device_store,
                            &unchanged_hot_context_queries, &error),
                        error);
  BOOST_CHECK_EQUAL(unchanged_hot_context_queries, cold_context_queries);

  view = NativeHostSolveView();
  fixture.lease = CatalogReadLease();
  CoalescedBaGraphMutation xy_patch;
  xy_patch.owner_epoch = 44;
  xy_patch.revision_before = 3;
  xy_patch.revision_after = 4;
  xy_patch.observation_upserts.push_back(
      {20, 0, 30, {{18.0, 19.0}}});
  HostBaGraphUpdateResult update;
  BOOST_REQUIRE_MESSAGE(
      fixture.store.ApplyCoalescedMutation(xy_patch, &update, &error), error);
  fixture.lease = fixture.store.AcquireReadLease();
  ++fixture.intent.selection_revision;
  BOOST_REQUIRE_MESSAGE(materialize(), error);
  options = NativeFullLmOptions(view.config);
  fixture.reference.observations[0].xy = {{18.0, 19.0}};
  CudaFullLmResult patch_reference;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(fixture.reference, options, &patch_reference, &error),
      error);
  BaSolveResult patched;
  BOOST_REQUIRE_MESSAGE(run_native(options, &patched), error);
  BOOST_CHECK_EQUAL(patched.runtime.device_store_patch_upload_calls, 1);
  BOOST_CHECK_EQUAL(patched.runtime.device_store_full_upload_calls, 0);
  BOOST_CHECK_EQUAL(patched.runtime.device_store_growth_d2d_calls, 0);
  BOOST_CHECK_GT(patched.runtime.device_store_patch_upload_bytes, 0);
  BOOST_CHECK_EQUAL(patched.initial_cost, patch_reference.initial_cost);
  BOOST_CHECK_EQUAL(patched.runtime.device_store_generation,
                    cold_generation + 1);

  view = NativeHostSolveView();
  fixture.lease = CatalogReadLease();
  CoalescedBaGraphMutation growth;
  growth.owner_epoch = 44;
  growth.revision_before = 4;
  growth.revision_after = 5;
  for (uint32_t index = 0; index < 62; ++index) {
    const uint32_t image_id = 1000 + index;
    const uint64_t point_id = 2000 + index;
    growth.image_upserts.push_back({image_id, 7, true});
    growth.point_upserts.push_back({point_id});
    growth.observation_upserts.push_back(
        {image_id, 0, point_id,
         {{static_cast<double>(index), static_cast<double>(index + 1)}}});
  }
  BOOST_REQUIRE_MESSAGE(
      fixture.store.ApplyCoalescedMutation(growth, &update, &error), error);
  fixture.lease = fixture.store.AcquireReadLease();
  ++fixture.intent.selection_revision;
  BOOST_REQUIRE_MESSAGE(materialize(), error);
  options = NativeFullLmOptions(view.config);
  BaSolveResult grown;
  BOOST_REQUIRE_MESSAGE(run_native(options, &grown), error);
  BOOST_CHECK_EQUAL(grown.runtime.device_store_patch_upload_calls, 1);
  BOOST_CHECK_EQUAL(grown.runtime.device_store_full_upload_calls, 0);
  BOOST_CHECK_EQUAL(grown.runtime.device_store_growth_d2d_calls, 1);
  BOOST_CHECK_GT(grown.runtime.device_store_growth_d2d_bytes, 0);
  BOOST_CHECK_EQUAL(grown.initial_cost, patched.initial_cost);
  uint64_t growth_context_queries = 0;
  BOOST_REQUIRE_MESSAGE(GetDeviceBaProblemStoreContextQueryCountForTesting(
                            fixture.device_store, &growth_context_queries,
                            &error),
                        error);

  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
  BaSolveResult pool_reset_reuse;
  BOOST_REQUIRE_MESSAGE(run_native(options, &pool_reset_reuse), error);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_reuse_calls, 1);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_full_upload_calls, 0);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_full_upload_bytes, 0);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_patch_upload_calls, 0);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_patch_upload_bytes, 0);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_growth_d2d_calls, 0);
  BOOST_CHECK_EQUAL(pool_reset_reuse.runtime.device_store_generation,
                    grown.runtime.device_store_generation);
  uint64_t reset_context_queries = 0;
  BOOST_REQUIRE_MESSAGE(GetDeviceBaProblemStoreContextQueryCountForTesting(
                            fixture.device_store, &reset_context_queries,
                            &error),
                        error);
  BOOST_CHECK_EQUAL(reset_context_queries, growth_context_queries + 6);

  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
  BOOST_REQUIRE_MESSAGE(FailNextDeviceBaProblemStoreContextQueryForTesting(
                            fixture.device_store, &error),
                        error);
  BaSolveResult context_query_failure;
  BOOST_CHECK(!run_native(options, &context_query_failure));
  BOOST_CHECK(error.find("pointer context query failure injected") !=
              std::string::npos);
  BaSolveResult context_recovery;
  BOOST_REQUIRE_MESSAGE(run_native(options, &context_recovery), error);
  BOOST_CHECK_EQUAL(context_recovery.runtime.device_store_reuse_calls, 0);
  BOOST_CHECK_EQUAL(context_recovery.runtime.device_store_full_upload_calls, 1);
  BOOST_CHECK_GT(context_recovery.runtime.device_store_generation,
                 pool_reset_reuse.runtime.device_store_generation);
  uint64_t recovery_context_queries = 0;
  BOOST_REQUIRE_MESSAGE(GetDeviceBaProblemStoreContextQueryCountForTesting(
                            fixture.device_store, &recovery_context_queries,
                            &error),
                        error);
  BOOST_CHECK_EQUAL(recovery_context_queries, reset_context_queries + 6);

  CudaFullLmOptions tiny_budget = options;
  tiny_budget.layer_c.layer_b.memory_budget_override_bytes = 1;
  BaSolveResult budget_failure;
  BOOST_CHECK(!run_native(tiny_budget, &budget_failure));
  BOOST_CHECK(error.find("INSUFFICIENT_GPU_MEMORY") != std::string::npos);

  view = NativeHostSolveView();
  fixture.lease = CatalogReadLease();
  CoalescedBaGraphMutation second_patch;
  second_patch.owner_epoch = 44;
  second_patch.revision_before = 5;
  second_patch.revision_after = 6;
  second_patch.observation_upserts.push_back(
      {20, 0, 30, {{20.0, 21.0}}});
  BOOST_REQUIRE_MESSAGE(
      fixture.store.ApplyCoalescedMutation(second_patch, &update, &error),
      error);
  fixture.lease = fixture.store.AcquireReadLease();
  ++fixture.intent.selection_revision;
  BOOST_REQUIRE_MESSAGE(materialize(), error);
  options = NativeFullLmOptions(view.config);
  BOOST_REQUIRE_MESSAGE(FailNextDeviceBaProblemStorePublishForTesting(
                            fixture.device_store, &error),
                        error);
  BaSolveResult injected;
  BOOST_CHECK(!run_native(options, &injected));
  BOOST_CHECK(error.find("publish failure injected") != std::string::npos);
  BaSolveResult recovered;
  BOOST_REQUIRE_MESSAGE(run_native(options, &recovered), error);
  BOOST_CHECK_EQUAL(recovered.runtime.device_store_reuse_calls, 0);
  BOOST_CHECK_EQUAL(recovered.runtime.device_store_full_upload_calls, 1);
  BOOST_CHECK_EQUAL(recovered.runtime.device_store_generation,
                    context_recovery.runtime.device_store_generation + 1);
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
  NativeCudaSolveRequest request{&view, &state, &options,
                                 fixture.device_store};
  BaSolveResult native;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(request, &native, &error), error);
  BOOST_REQUIRE_GE(legacy.accepted_commits, 1);
  BOOST_REQUIRE_GE(legacy.runtime.final_internal_state_epoch, 1);
  BOOST_CHECK_EQUAL(native.success, legacy.success);
  BOOST_CHECK_EQUAL(native.termination_reason, legacy.termination_reason);
  BOOST_CHECK_EQUAL(native.trial_iterations, legacy.trial_iterations);
  BOOST_CHECK_EQUAL(native.accepted_commits, legacy.accepted_commits);
  BOOST_CHECK_EQUAL(native.accepted_decisions, legacy.accepted_decisions);
  BOOST_CHECK_EQUAL(native.rejected_steps, legacy.rejected_steps);
  BOOST_CHECK_EQUAL(native.invalid_steps, legacy.invalid_steps);
  BOOST_CHECK_EQUAL(native.final_internal_state_epoch,
                    legacy.runtime.final_internal_state_epoch);
  double legacy_max_backward_error = 0.0;
  for (const CudaLmIteration& iteration : legacy.trace) {
    legacy_max_backward_error =
        std::max(legacy_max_backward_error, iteration.backward_error);
  }
  BOOST_CHECK_EQUAL(native.backward_error_samples, legacy.trace.size());
  BOOST_CHECK_EQUAL(native.max_backward_error, legacy_max_backward_error);
  BOOST_CHECK(std::isfinite(native.max_backward_error));
  BOOST_CHECK_EQUAL(native.initial_cost, legacy.initial_cost);
  BOOST_CHECK_EQUAL(native.final_cost, legacy.final_cost);
  BOOST_CHECK_EQUAL(native.variable_delta.state_generation,
                    state.state_generation + native.final_internal_state_epoch);
  BOOST_CHECK_EQUAL(native.runtime.dense_active_state_device_download_calls, 0);
  BOOST_CHECK_GT(native.runtime.variable_state_delta_device_download_calls, 0);
  BOOST_CHECK(native.variable_delta.cameras.empty());
  for (size_t i = 0; i < native.variable_delta.images.size(); ++i) {
    const uint32_t image_id =
        fixture.lease.images()[native.variable_delta.images[i].image_slot]
            .image_id;
    const auto legacy_image = std::find_if(
        legacy.final_state.images.begin(), legacy.final_state.images.end(),
        [image_id](const ImageSnapshot& value) {
          return value.image_id == image_id;
        });
    BOOST_REQUIRE(legacy_image != legacy.final_state.images.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.variable_delta.images[i].quaternion.begin(),
        native.variable_delta.images[i].quaternion.end(),
        legacy_image->qvec.begin(), legacy_image->qvec.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.variable_delta.images[i].translation.begin(),
        native.variable_delta.images[i].translation.end(),
        legacy_image->tvec.begin(), legacy_image->tvec.end());
  }
  for (size_t i = 0; i < native.variable_delta.points.size(); ++i) {
    const uint64_t point3D_id =
        fixture.lease.points()[native.variable_delta.points[i].point_slot]
            .point3D_id;
    const auto legacy_point = std::find_if(
        legacy.final_state.points.begin(), legacy.final_state.points.end(),
        [point3D_id](const PointSnapshot& point) {
          return point.point3D_id == point3D_id;
        });
    BOOST_REQUIRE(legacy_point != legacy.final_state.points.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        native.variable_delta.points[i].xyz.begin(),
        native.variable_delta.points[i].xyz.end(),
        legacy_point->xyz.begin(), legacy_point->xyz.end());
  }
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap
