#define TEST_NAME "gpu_ba/active_solve_view"
#include "util/testing.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gpu_ba/active_solve_view.h"
#include "gpu_ba/host_problem_store_internal.h"

namespace colmap {
namespace gpu_ba {
namespace {

CameraSnapshot Camera() {
  CameraSnapshot value;
  value.camera_id = 7;
  value.model_id = 4;
  value.width = 1920;
  value.height = 1080;
  value.constant = true;
  value.params = {1000.0, 1000.0, 960.0, 540.0,
                  0.01, -0.01, 0.001, -0.001};
  return value;
}

ImageSnapshot Image(const uint32_t id, const uint8_t mask = 0) {
  ImageSnapshot value;
  value.image_id = id;
  value.camera_id = 7;
  value.selected = true;
  value.pose_constant = false;
  value.has_pose_parameter_blocks = true;
  value.constant_tvec_mask = mask;
  value.tvec = {{0.01 * id, 0.0, 1.0}};
  return value;
}

PointSnapshot Point(const uint64_t id, const bool constant = false) {
  PointSnapshot value;
  value.point3D_id = id;
  value.constant = constant;
  value.xyz = {{0.001 * id, 0.2, 4.0}};
  return value;
}

ObservationSnapshot Observation(const uint64_t source,
                                const uint32_t image,
                                const uint32_t point2D,
                                const uint64_t point) {
  ObservationSnapshot value;
  value.source_index = source;
  value.image_id = image;
  value.point2D_idx = point2D;
  value.point3D_id = point;
  value.xy = {{100.0 + source, 200.0 - source}};
  return value;
}

void AppendPoseParameters(const ImageSnapshot& image,
                          CudaSolveProblem* problem) {
  uint64_t source = problem->parameter_blocks_source_order.size();
  problem->parameter_blocks_source_order.push_back(
      {source++, ParameterKind::kQuaternion, image.image_id, 4, 3,
       image.pose_constant});
  const uint32_t free_translation =
      3u - static_cast<uint32_t>((image.constant_tvec_mask & 1u) != 0) -
      static_cast<uint32_t>((image.constant_tvec_mask & 2u) != 0) -
      static_cast<uint32_t>((image.constant_tvec_mask & 4u) != 0);
  problem->parameter_blocks_source_order.push_back(
      {source, ParameterKind::kTranslation, image.image_id, 3,
       free_translation, image.pose_constant || free_translation == 0});
}

CudaSolveProblem GlobalProblem() {
  CudaSolveProblem problem;
  problem.metadata.ba_kind = BaKind::kGlobal;
  problem.metadata.loss_function = "SOFT_L1";
  problem.cameras = {Camera()};
  problem.images = {Image(10), Image(11, 1), Image(12)};
  problem.points = {Point(100), Point(101)};
  problem.observations = {
      Observation(0, 10, 0, 100), Observation(1, 11, 0, 100),
      Observation(2, 11, 1, 101), Observation(3, 12, 0, 101)};
  for (const ImageSnapshot& image : problem.images) {
    AppendPoseParameters(image, &problem);
  }
  for (const PointSnapshot& point : problem.points) {
    const uint64_t source = problem.parameter_blocks_source_order.size();
    problem.parameter_blocks_source_order.push_back(
        {source, ParameterKind::kPoint3D, point.point3D_id, 3, 3,
         point.constant});
  }
  problem.parameter_blocks_source_order.push_back(
      {problem.parameter_blocks_source_order.size(), ParameterKind::kCamera,
       7, 8, 8, true});
  for (const ObservationSnapshot& value : problem.observations) {
    problem.source_insertion_order.push_back(
        {value.source_index, ResidualKind::kVisual, value.image_id,
         value.point2D_idx, value.point3D_id});
  }
  return problem;
}

CudaSolveProblem LocalProblem(const CudaSolveProblem& global,
                              const size_t begin) {
  CudaSolveProblem local;
  local.metadata = global.metadata;
  local.metadata.ba_kind = BaKind::kLocal;
  local.cameras = global.cameras;
  local.images = {global.images[begin], global.images[begin + 1]};
  local.points = {global.points[begin]};
  local.observations = {global.observations[2 * begin],
                        global.observations[2 * begin + 1]};
  for (const ImageSnapshot& image : local.images) {
    AppendPoseParameters(image, &local);
  }
  local.parameter_blocks_source_order.push_back(
      {local.parameter_blocks_source_order.size(), ParameterKind::kPoint3D,
       local.points[0].point3D_id, 3, 3, false});
  local.parameter_blocks_source_order.push_back(
      {local.parameter_blocks_source_order.size(), ParameterKind::kCamera,
       7, 8, 8, true});
  for (const ObservationSnapshot& value : local.observations) {
    local.source_insertion_order.push_back(
        {value.source_index, ResidualKind::kVisual, value.image_id,
         value.point2D_idx, value.point3D_id});
  }
  return local;
}

IndexedActiveSolveConfig Config(const CudaArithmeticPrecision precision) {
  IndexedActiveSolveConfig value;
  value.effective_config_resolved = true;
  value.config_generation =
      precision == CudaArithmeticPrecision::kFp64 ? 1 : 2;
  value.arithmetic_precision = precision;
  value.hessian_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  value.schur_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  value.hot_kernel = CudaHotKernelMode::kTransformed;
  value.execution_profile = CudaExecutionProfile::kCompactControl;
  value.loss_mode = CudaLossMode::kSoftL1;
  value.loss_scale = 1.0;
  return value;
}

HostIndexedCatalogData HostData(const CudaSolveProblem& problem,
                                const uint64_t owner,
                                const uint64_t revision) {
  HostIndexedCatalogData data;
  data.owner_epoch = owner;
  data.revision = revision;
  data.generation = revision;
  for (const CameraSnapshot& camera : problem.cameras) {
    data.cameras.push_back({camera.camera_id, camera.model_id, camera.width,
                            camera.height, camera.params.size()});
  }
  for (const ImageSnapshot& image : problem.images) {
    data.images.push_back({image.image_id, image.camera_id, true});
  }
  for (const PointSnapshot& point : problem.points) {
    HostCatalogPoint value;
    value.point3D_id = point.point3D_id;
    for (const ObservationSnapshot& observation : problem.observations) {
      if (observation.point3D_id == point.point3D_id) {
        value.track.push_back(
            {observation.image_id, observation.point2D_idx});
      }
    }
    data.points.push_back(std::move(value));
  }
  for (const ObservationSnapshot& observation : problem.observations) {
    data.observations.push_back({observation.point3D_id, observation.image_id,
                                 observation.point2D_idx, observation.xy});
  }
  return data;
}

BOOST_AUTO_TEST_CASE(StableCatalogSupportsDifferentActiveSets) {
  constexpr uint64_t kOwner = 101;
  const CudaSolveProblem global = GlobalProblem();
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      catalog.EnsureProblemStaticData(kOwner, 1, global, &update, &error),
      error);
  const uint32_t shared_slot = catalog.FindImage(11)->header.slot;
  MapperStaticCatalogStableTables tables;
  BOOST_REQUIRE_MESSAGE(catalog.ExportStableTables(&tables, &error), error);
  BOOST_CHECK_EQUAL(tables.observations.size(), 4);
  BOOST_CHECK_EQUAL(tables.image_incidences.size(), 4);
  BOOST_CHECK_EQUAL(tables.point_incidences.size(), 4);

  IndexedActiveSolveMaterializer materializer;
  IndexedActiveSolveDescriptor first;
  IndexedActiveSolveDescriptor second;
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                            kOwner, catalog, LocalProblem(global, 0),
                            Config(CudaArithmeticPrecision::kFp64), &first,
                            &error),
                        error);
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                            kOwner, catalog, LocalProblem(global, 1),
                            Config(CudaArithmeticPrecision::kFp32MixedStable),
                            &second, &error),
                        error);
  BOOST_CHECK_EQUAL(catalog.FindImage(11)->header.slot, shared_slot);
  BOOST_CHECK_EQUAL(first.identity.catalog_generation,
                    second.identity.catalog_generation);
  BOOST_CHECK_NE(first.identity.solve_view_generation,
                 second.identity.solve_view_generation);
  BOOST_CHECK(first.config.arithmetic_precision ==
              CudaArithmeticPrecision::kFp64);
  BOOST_CHECK(second.config.arithmetic_precision ==
              CudaArithmeticPrecision::kFp32MixedStable);
}

BOOST_AUTO_TEST_CASE(DescriptorPreservesFixedMasksAndSourceIdentity) {
  constexpr uint64_t kOwner = 102;
  CudaSolveProblem problem = GlobalProblem();
  problem.points[1].constant = true;
  for (ParameterBlockSnapshot& parameter :
       problem.parameter_blocks_source_order) {
    if (parameter.kind == ParameterKind::kPoint3D &&
        parameter.entity_id == 101) {
      parameter.constant = true;
      parameter.tangent_size = 0;
    }
  }
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.EnsureProblemStaticData(
      kOwner, 1, problem, &update, &error));
  IndexedActiveSolveMaterializer materializer;
  IndexedActiveSolveDescriptor descriptor;
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                            kOwner, catalog, problem,
                            Config(CudaArithmeticPrecision::kFp64),
                            &descriptor, &error),
                        error);
  BOOST_REQUIRE_EQUAL(descriptor.images.size(), 3);
  BOOST_CHECK_EQUAL(descriptor.images[1].constant_tvec_mask, 1);
  BOOST_CHECK_EQUAL(descriptor.images[1].pose_dimension, 5);
  BOOST_REQUIRE_EQUAL(descriptor.points.size(), 2);
  BOOST_CHECK_EQUAL(descriptor.points[1].point_index, -1);
  BOOST_REQUIRE_EQUAL(descriptor.visual.size(), 4);
  for (size_t i = 0; i < descriptor.visual.size(); ++i) {
    BOOST_CHECK_EQUAL(descriptor.visual[i].source_index, i);
  }
}

BOOST_AUTO_TEST_CASE(AppendAndTombstoneDoNotReuseStableSlots) {
  constexpr uint64_t kOwner = 103;
  CudaSolveProblem problem = GlobalProblem();
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.EnsureProblemStaticData(
      kOwner, 1, problem, &update, &error));
  const uint32_t old_point = catalog.FindPoint(100)->header.slot;
  const uint32_t old_observation =
      catalog.FindObservation(10, 0)->header.slot;

  CudaSolveProblem appended;
  appended.points = {Point(102)};
  appended.observations = {Observation(10, 10, 0, 102)};
  MapperStaticCatalogTombstones tombstones;
  tombstones.points = {100};
  tombstones.observations = {{{10, 0}}, {{11, 0}}};
  BOOST_REQUIRE_MESSAGE(catalog.ApplyDelta(
                            kOwner, 2, appended, tombstones, &update, &error),
                        error);
  BOOST_CHECK(catalog.FindPoint(100) == nullptr);
  BOOST_CHECK_GT(catalog.FindPoint(102)->header.slot, old_point);
  BOOST_CHECK_GT(catalog.FindObservation(10, 0)->header.slot,
                 old_observation);
  BOOST_CHECK_EQUAL(update.tombstoned_points, 1);
  BOOST_CHECK_EQUAL(update.tombstoned_observations, 2);
}

BOOST_AUTO_TEST_CASE(DeviceCatalogLayoutAndIdentityBindArenaGeneration) {
  constexpr uint64_t kOwner = 104;
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.EnsureProblemStaticData(
      kOwner, 1, GlobalProblem(), &update, &error));
  MapperStaticCatalogStableTables tables;
  BOOST_REQUIRE(catalog.ExportStableTables(&tables, &error));
  DeviceCatalogLayoutSpec spec;
  spec.base_offset = 17;
  spec.arena_capacity = 1ull << 20;
  DeviceCatalogLayout layout;
  BOOST_REQUIRE_MESSAGE(
      PlanPersistentDeviceCatalogLayout(tables, spec, &layout, &error), error);
  BOOST_CHECK_EQUAL(layout.begin % 256, 0);
  BOOST_CHECK_EQUAL(layout.image_incidences.elements, 4);
  BOOST_CHECK_EQUAL(layout.point_incidences.elements, 4);

  DeviceCatalogPhysicalIdentity identity;
  identity.device_ordinal = 0;
  identity.context_incarnation = 9;
  identity.owner_epoch = kOwner;
  identity.catalog_revision = catalog.revision();
  identity.catalog_generation = catalog.generation();
  identity.arena_generation = 3;
  PersistentDeviceCatalogState state;
  BOOST_REQUIRE_MESSAGE(state.Publish(identity, layout, &error), error);
  BOOST_CHECK(state.CanReuse(identity));
  DeviceCatalogPhysicalIdentity stale = identity;
  ++stale.arena_generation;
  BOOST_CHECK(!state.CanReuse(stale));
  stale = identity;
  ++stale.catalog_revision;
  BOOST_CHECK(!state.CanReuse(stale));
  state.Invalidate();
  BOOST_CHECK(!state.valid());
}

BOOST_AUTO_TEST_CASE(ReassignmentPreservesIncidenceCountsAndStableIdentity) {
  constexpr uint64_t kOwner = 105;
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.ReconcileFullStaticData(
      HostData(GlobalProblem(), kOwner, 1), &update, &error));
  MapperStaticCatalogStableTables before;
  BOOST_REQUIRE(catalog.ExportStableTables(&before, &error));
  const uint32_t old_observation = catalog.FindObservation(10, 0)->header.slot;
  const uint32_t old_point = catalog.FindPoint(100)->header.slot;
  BOOST_REQUIRE_EQUAL(before.points[old_point].track_length, 2);
  BOOST_REQUIRE_EQUAL(before.points[old_point].observation_count, 2);

  CudaSolveProblem appended;
  appended.points = {Point(102)};
  appended.observations = {Observation(8, 10, 0, 102)};
  MapperStaticCatalogTombstones tombstones;
  tombstones.observations = {{{10, 0}}};
  BOOST_REQUIRE_MESSAGE(catalog.ApplyDelta(
                            kOwner, 2, appended, tombstones, &update, &error),
                        error);
  MapperStaticCatalogStableTables after;
  BOOST_REQUIRE(catalog.ExportStableTables(&after, &error));
  const auto* reassigned = catalog.FindObservation(10, 0);
  BOOST_REQUIRE(reassigned != nullptr);
  BOOST_CHECK_GT(reassigned->header.slot, old_observation);
  BOOST_CHECK_EQUAL(reassigned->source_identity,
                    (uint64_t{10} << 32) | uint64_t{0});
  BOOST_CHECK_EQUAL(after.observations[old_observation].header.alive, 0);
  BOOST_CHECK_EQUAL(after.points[old_point].track_length, 1);
  BOOST_CHECK_EQUAL(after.points[old_point].observation_count, 2);
  const MapperCatalogPointSlot* new_point = catalog.FindPoint(102);
  BOOST_REQUIRE(new_point != nullptr);
  BOOST_CHECK_EQUAL(new_point->track_length, 1);
  BOOST_CHECK_EQUAL(new_point->observation_count, 1);
}

BOOST_AUTO_TEST_CASE(FailedDeltaAndDanglingFullDataDoNotPublish) {
  constexpr uint64_t kOwner = 106;
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.ReconcileFullStaticData(
      HostData(GlobalProblem(), kOwner, 1), &update, &error));
  const uint64_t generation = catalog.generation();
  const uint64_t revision = catalog.revision();
  const uint64_t observation_slots = catalog.observation_slot_count();

  CudaSolveProblem invalid_append;
  invalid_append.observations = {Observation(9, 10, 7, 9999)};
  MapperStaticCatalogTombstones no_tombstones;
  BOOST_CHECK(!catalog.ApplyDelta(kOwner, 2, invalid_append, no_tombstones,
                                  &update, &error));
  BOOST_CHECK_EQUAL(catalog.generation(), generation);
  BOOST_CHECK_EQUAL(catalog.revision(), revision);
  BOOST_CHECK_EQUAL(catalog.observation_slot_count(), observation_slots);

  HostIndexedCatalogData dangling = HostData(GlobalProblem(), kOwner, 2);
  dangling.cameras.clear();
  BOOST_CHECK(!catalog.ReconcileFullStaticData(dangling, &update, &error));
  BOOST_CHECK_EQUAL(catalog.generation(), generation);
  BOOST_CHECK_EQUAL(catalog.revision(), revision);
  BOOST_CHECK_EQUAL(catalog.observation_slot_count(), observation_slots);
}

BOOST_AUTO_TEST_CASE(PointMergeTombstonesOldIncidenceAndAppendsNewSlots) {
  constexpr uint64_t kOwner = 107;
  const CudaSolveProblem original = GlobalProblem();
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.ReconcileFullStaticData(
      HostData(original, kOwner, 1), &update, &error));
  const uint32_t old_point_max = std::max(
      catalog.FindPoint(100)->header.slot,
      catalog.FindPoint(101)->header.slot);

  CudaSolveProblem merged;
  merged.points = {Point(200)};
  for (size_t index = 0; index < original.observations.size(); ++index) {
    const ObservationSnapshot& old = original.observations[index];
    merged.observations.push_back(
        Observation(index + 20, old.image_id, old.point2D_idx, 200));
  }
  MapperStaticCatalogTombstones tombstones;
  tombstones.points = {100, 101};
  for (const ObservationSnapshot& observation : original.observations) {
    tombstones.observations.push_back(
        {{observation.image_id, observation.point2D_idx}});
  }
  BOOST_REQUIRE_MESSAGE(catalog.ApplyDelta(
                            kOwner, 2, merged, tombstones, &update, &error),
                        error);
  BOOST_CHECK(catalog.FindPoint(100) == nullptr);
  BOOST_CHECK(catalog.FindPoint(101) == nullptr);
  BOOST_REQUIRE(catalog.FindPoint(200) != nullptr);
  BOOST_CHECK_GT(catalog.FindPoint(200)->header.slot, old_point_max);
  BOOST_CHECK_EQUAL(catalog.FindPoint(200)->track_length, 4);
  BOOST_CHECK_EQUAL(update.tombstoned_observations, 4);
}

BOOST_AUTO_TEST_CASE(EntityDeleteCascadesAliveObservations) {
  constexpr uint64_t kOwner = 109;
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.ReconcileFullStaticData(
      HostData(GlobalProblem(), kOwner, 1), &update, &error));
  MapperStaticCatalogTombstones tombstones;
  tombstones.cameras = {7};
  CudaSolveProblem no_appends;
  BOOST_REQUIRE_MESSAGE(catalog.ApplyDelta(
                            kOwner, 2, no_appends, tombstones, &update, &error),
                        error);
  BOOST_CHECK(catalog.FindCamera(7) == nullptr);
  BOOST_CHECK(catalog.FindImage(10) == nullptr);
  BOOST_CHECK(catalog.FindImage(11) == nullptr);
  BOOST_CHECK(catalog.FindImage(12) == nullptr);
  BOOST_CHECK(catalog.FindObservation(10, 0) == nullptr);
  BOOST_REQUIRE(catalog.FindPoint(100) != nullptr);
  BOOST_REQUIRE(catalog.FindPoint(101) != nullptr);
  BOOST_CHECK_EQUAL(catalog.FindPoint(100)->track_length, 0);
  BOOST_CHECK_EQUAL(catalog.FindPoint(101)->track_length, 0);
  BOOST_CHECK_EQUAL(update.tombstoned_observations, 4);
}

BOOST_AUTO_TEST_CASE(ResolvedConfigurationMismatchIsRejected) {
  constexpr uint64_t kOwner = 108;
  const CudaSolveProblem problem = GlobalProblem();
  MapperStaticProblemDataCatalog catalog(kOwner);
  MapperStaticCatalogUpdateResult update;
  std::string error;
  BOOST_REQUIRE(catalog.ReconcileFullStaticData(
      HostData(problem, kOwner, 1), &update, &error));
  MapperStaticCatalogStableTables tables;
  BOOST_REQUIRE(catalog.ExportStableTables(&tables, &error));
  IndexedActiveSolveMaterializer materializer;
  IndexedActiveSolveDescriptor descriptor;
  const IndexedActiveSolveConfig config =
      Config(CudaArithmeticPrecision::kFp32MixedStable);
  BOOST_REQUIRE(materializer.Materialize(
      kOwner, catalog, problem, config, &descriptor, &error));
  BOOST_REQUIRE(ValidateIndexedCatalogSolveBinding(
      tables, descriptor, config, &error));
  IndexedActiveSolveConfig mismatch = config;
  mismatch.loss_scale = 2.0;
  BOOST_CHECK(!ValidateIndexedCatalogSolveBinding(
      tables, descriptor, mismatch, &error));
  mismatch = config;
  mismatch.execution_profile = CudaExecutionProfile::kBaseline;
  BOOST_CHECK(!ValidateIndexedCatalogSolveBinding(
      tables, descriptor, mismatch, &error));
  MapperStaticCatalogStableTables stale_tables = tables;
  ++stale_tables.catalog_revision;
  BOOST_CHECK(!ValidateIndexedCatalogSolveBinding(
      stale_tables, descriptor, config, &error));
  stale_tables = tables;
  ++stale_tables.abi_version;
  BOOST_CHECK(!ValidateIndexedCatalogSolveBinding(
      stale_tables, descriptor, config, &error));
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap
