#define TEST_NAME "gpu_ba/host_ba_graph"
#include "util/testing.h"

#include <cmath>
#include <string>

#include "gpu_ba/custom_cuda.h"
#include "gpu_ba/host_ba_graph.h"

namespace colmap {
namespace gpu_ba {
namespace {

HostBaGraphColdInput MakeGraph() {
  HostBaGraphColdInput input;
  input.owner_epoch = 101;
  input.topology_revision = 1;
  input.cameras.push_back({7, 4, 1920, 1080, 8});
  input.images.push_back({20, 7, true});
  input.images.push_back({21, 7, true});
  input.points.push_back({30});
  input.observations.push_back({20, 0, 30, {{100.0, 200.0}}});
  input.observations.push_back({21, 1, 30, {{110.0, 210.0}}});
  return input;
}

DenseActiveState MakeState() {
  DenseActiveState state;
  state.owner_epoch = 101;
  state.state_generation = 9;
  DenseCameraState camera;
  camera.camera_slot = 0;
  camera.state_generation = 9;
  camera.parameters = {1000.0, 1000.0, 960.0, 540.0,
                       0.01,   -0.01,  0.001, -0.001};
  state.cameras.push_back(camera);
  DenseImageState image;
  image.image_slot = 0;
  image.state_generation = 9;
  image.quaternion = {{2.0, 0.0, 0.0, 0.0}};
  image.translation = {{0.1, 0.2, 0.3}};
  state.images.push_back(image);
  image.image_slot = 1;
  image.quaternion = {{0.0, 1.0, 0.0, 0.0}};
  image.translation = {{0.4, 0.5, 0.6}};
  state.images.push_back(image);
  DensePointState point;
  point.point_slot = 0;
  point.state_generation = 9;
  point.xyz = {{1.0, 2.0, 3.0}};
  state.points.push_back(point);
  return state;
}

HostBaGraphColdInput MakeLargeGraph(const size_t observation_count) {
  HostBaGraphColdInput input;
  input.owner_epoch = 303;
  input.topology_revision = 1;
  input.cameras.push_back({7, 4, 1920, 1080, 8});
  input.images.reserve(observation_count);
  input.points.reserve(observation_count);
  input.observations.reserve(observation_count);
  for (size_t i = 0; i < observation_count; ++i) {
    const uint32_t image_id = static_cast<uint32_t>(10000 + i);
    const uint64_t point_id = static_cast<uint64_t>(20000 + i);
    input.images.push_back({image_id, 7, true});
    input.points.push_back({point_id});
    input.observations.push_back(
        {image_id,
         0,
         point_id,
         {{static_cast<double>(i), static_cast<double>(i + 1)}}});
  }
  return input;
}

BaSolveIntent MakeIntent(const CatalogReadLease& lease) {
  BaSolveIntent intent;
  intent.owner_epoch = lease.owner_epoch();
  intent.catalog_revision = lease.topology_revision();
  intent.catalog_generation = lease.generation();
  intent.selection_revision = 4;
  intent.kind = BaKind::kLocal;
  intent.config.resolved = true;
  intent.config.config_generation = 8;
  intent.config.arithmetic_precision =
      CudaArithmeticPrecision::kFp32MixedStable;
  intent.config.hessian_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  intent.config.schur_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  intent.config.hot_kernel = CudaHotKernelMode::kTransformed;
  intent.config.execution_profile = CudaExecutionProfile::kCompactControl;
  intent.config.residual_order = CudaResidualOrder::kCanonical;
  intent.config.loss_mode = CudaLossMode::kSoftL1;
  intent.config.loss_scale = 1.0;
  intent.active_image_slots.push_back(0);
  intent.explicit_variable_point_slots.push_back(0);
  intent.translation_subsets.push_back({0, 2, {0, 0, 0}});
  CameraParameterPolicy camera;
  camera.camera_slot = 0;
  camera.constant = true;
  intent.camera_policies.push_back(camera);
  LidarConstraintRecord lidar;
  lidar.point_slot = 0;
  lidar.constraint_slot = 3;
  lidar.physical_identity = 900;
  lidar.lidar_type = 1;
  lidar.plane = {{0.0, 0.0, 1.0, -3.0}};
  lidar.lidar_xyz = {{1.0, 2.0, 3.0}};
  lidar.weight = 2.0;
  lidar.search_range = 0.2;
  lidar.point_state_generation = 9;
  intent.lidar.lidar_map_generation = 11;
  intent.lidar.match_config_generation = 12;
  intent.lidar.constraints.push_back(lidar);
  intent.source_insertion_order = {
      {ResidualKind::kVisual, 0},
      {ResidualKind::kVisual, 1},
      {ResidualKind::kLidar, 3}};
  return intent;
}

BOOST_AUTO_TEST_CASE(ColdBuildLeaseAndNativeView) {
  HostBaGraphStore store(101);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE_MESSAGE(store.ColdBuild(MakeGraph(), &update, &error), error);
  BOOST_CHECK(update.published);
  BOOST_CHECK_EQUAL(update.capacity_growth_events, 6);
  BOOST_CHECK_EQUAL(update.capacity_growth_copy_bytes, 0);
  BOOST_CHECK_EQUAL(update.hash_rehash_events, 4);
  BOOST_CHECK_EQUAL(update.hash_rehash_entries, 0);
  BOOST_CHECK_EQUAL(store.generation(), 1);
  const CatalogReadLease lease = store.AcquireReadLease();
  BOOST_REQUIRE(lease.valid());
  BOOST_CHECK(store.IsCurrent(lease));
  BOOST_CHECK_EQUAL(lease.images().size, 2);
  BOOST_CHECK_EQUAL(lease.points()[0].track_length, 2);

  DenseActiveState state = MakeState();
  const DenseActiveState entry_state = state;
  BaSolveIntent intent = MakeIntent(lease);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer active_state;
  NativeHostSolvePreparationRuntime runtime;
  BOOST_REQUIRE_MESSAGE(materializer.Materialize(lease, intent, state, &view,
                                                 &active_state, &runtime,
                                                 &error),
                        error);
  BOOST_CHECK_EQUAL(view.active_image_slots.size(), 1);
  BOOST_CHECK_EQUAL(view.boundary_image_slots.size(), 1);
  BOOST_CHECK_EQUAL(view.visual_observation_slots.size(), 2);
  BOOST_CHECK_EQUAL(view.active_point_slots.size(), 1);
  BOOST_CHECK_EQUAL(view.lidar.constraints.size(), 1);
  BOOST_CHECK_EQUAL(view.residual_ordinals.size(), 3);
  BOOST_CHECK(view.residual_ordinals[0].kind == ResidualKind::kVisual);
  BOOST_CHECK(view.residual_ordinals[1].kind == ResidualKind::kVisual);
  BOOST_CHECK(view.residual_ordinals[2].kind == ResidualKind::kLidar);
  for (size_t i = 0; i < view.residual_ordinals.size(); ++i) {
    BOOST_CHECK_EQUAL(view.residual_ordinals[i].source_insertion_index, i);
    BOOST_CHECK_EQUAL(view.residual_ordinals[i].execution_ordinal, i);
  }
  BOOST_CHECK_NE(view.residual_ordinals[0].physical_identity,
                 view.residual_ordinals[0].source_insertion_index);
  BOOST_REQUIRE_EQUAL(view.parameter_ordinals.size(), 4);
  BOOST_CHECK(view.parameter_ordinals[0].kind == ParameterKind::kQuaternion);
  BOOST_CHECK(view.parameter_ordinals[1].kind == ParameterKind::kTranslation);
  BOOST_CHECK_EQUAL(view.parameter_ordinals[1].translation_subset_mask, 2);
  BOOST_CHECK_EQUAL(view.parameter_ordinals[1].tangent_size, 2);
  BOOST_CHECK(view.parameter_ordinals[2].kind == ParameterKind::kPoint3D);
  BOOST_CHECK(view.parameter_ordinals[3].kind == ParameterKind::kCamera);
  BOOST_REQUIRE_EQUAL(active_state.images.size(), 2);
  BOOST_CHECK_EQUAL(active_state.images[0].quaternion[0], 1.0);
  BOOST_CHECK_EQUAL(active_state.images[1].quaternion[1], 1.0);
  BOOST_CHECK_EQUAL(state.images[0].quaternion[0],
                    entry_state.images[0].quaternion[0]);
  BOOST_CHECK_EQUAL(state.images[1].quaternion[1],
                    entry_state.images[1].quaternion[1]);
  BOOST_CHECK_EQUAL(runtime.quaternion_normalizations, 2);
}

BOOST_AUTO_TEST_CASE(TransactionNoopReassignAndFailureDoNotPartiallyPublish) {
  HostBaGraphStore store(101);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE(store.ColdBuild(MakeGraph(), &update, &error));
  CatalogReadLease generation1 = store.AcquireReadLease();
  CatalogReadLease generation1_copy = generation1;

  CoalescedBaGraphMutation noop;
  noop.owner_epoch = 101;
  noop.revision_before = 1;
  noop.revision_after = 2;
  noop.observation_upserts.push_back({20, 99, 30, {{1.0, 2.0}}});
  noop.tombstone_observations.push_back({20, 99});
  HostBaGraphColdInput busy_rebuild = MakeGraph();
  busy_rebuild.topology_revision = 2;
  BOOST_CHECK(!store.ColdBuild(busy_rebuild, &update, &error));
  BOOST_CHECK(update.reader_busy);
  BOOST_CHECK(!store.ApplyCoalescedMutation(noop, &update, &error));
  BOOST_CHECK(update.reader_busy);
  BOOST_CHECK_EQUAL(generation1.topology_revision(), 1);
  generation1 = CatalogReadLease();
  BOOST_CHECK(!store.ApplyCoalescedMutation(noop, &update, &error));
  BOOST_CHECK(update.reader_busy);
  generation1_copy = CatalogReadLease();
  BOOST_REQUIRE_MESSAGE(store.ApplyCoalescedMutation(noop, &update, &error),
                        error);
  BOOST_CHECK(update.semantic_noop);
  BOOST_CHECK_EQUAL(store.generation(), 1);
  BOOST_CHECK_EQUAL(store.topology_revision(), 2);
  BOOST_CHECK(!store.IsCurrent(generation1));

  CoalescedBaGraphMutation wrong_owner;
  wrong_owner.owner_epoch = 202;
  wrong_owner.revision_before = 2;
  wrong_owner.revision_after = 3;
  BOOST_CHECK(!store.ApplyCoalescedMutation(wrong_owner, &update, &error));
  BOOST_CHECK(store.AcquireReadLease().valid());
  CoalescedBaGraphMutation stale_duplicate;
  stale_duplicate.owner_epoch = 101;
  stale_duplicate.revision_before = 1;
  stale_duplicate.revision_after = 2;
  BOOST_CHECK(
      !store.ApplyCoalescedMutation(stale_duplicate, &update, &error));
  BOOST_CHECK(store.AcquireReadLease().valid());
  CoalescedBaGraphMutation malformed;
  malformed.owner_epoch = 101;
  malformed.revision_before = 2;
  malformed.revision_after = 2;
  BOOST_CHECK(!store.ApplyCoalescedMutation(malformed, &update, &error));
  BOOST_CHECK(store.AcquireReadLease().valid());

  CoalescedBaGraphMutation accepted_invalid;
  accepted_invalid.owner_epoch = 101;
  accepted_invalid.revision_before = 2;
  accepted_invalid.revision_after = 3;
  accepted_invalid.observation_upserts.push_back(
      {20, 2, 999, {{3.0, 4.0}}});
  BOOST_CHECK(
      !store.ApplyCoalescedMutation(accepted_invalid, &update, &error));
  BOOST_CHECK(!update.published);
  BOOST_CHECK(update.full_rebuild_required);
  BOOST_CHECK_EQUAL(store.topology_revision(), 2);
  BOOST_CHECK_EQUAL(store.generation(), 1);
  BOOST_CHECK(!store.AcquireReadLease().valid());
  HostBaGraphColdInput accepted_failure_rebuild = MakeGraph();
  accepted_failure_rebuild.topology_revision = 3;
  BOOST_REQUIRE_MESSAGE(
      store.ColdBuild(accepted_failure_rebuild, &update, &error), error);
  {
    const CatalogReadLease lease = store.AcquireReadLease();
    BOOST_REQUIRE(lease.valid());
    BOOST_CHECK_EQUAL(lease.topology_revision(), 3);
    BOOST_CHECK(lease.FindObservation(20, 2) == nullptr);
  }

  CoalescedBaGraphMutation reassign;
  reassign.owner_epoch = 101;
  reassign.revision_before = 3;
  reassign.revision_after = 4;
  reassign.camera_upserts.push_back({8, 4, 1920, 1080, 8});
  reassign.image_upserts.push_back({21, 8, true});
  reassign.point_upserts.push_back({31});
  reassign.observation_upserts.push_back(
      {21, 1, 31, {{110.0, 210.0}}});
  BOOST_REQUIRE_MESSAGE(
      store.ApplyCoalescedMutation(reassign, &update, &error), error);
  CatalogReadLease generation2 = store.AcquireReadLease();
  BOOST_REQUIRE_EQUAL(generation2.points().size, 2);
  BOOST_CHECK_EQUAL(generation2.points()[0].track_length, 1);
  BOOST_CHECK_EQUAL(generation2.points()[1].track_length, 1);
  BOOST_CHECK_EQUAL(generation2.FindObservation(21, 1)->point_slot, 1);
  BOOST_CHECK_EQUAL(
      generation2.images()[generation2.FindObservation(21, 1)->image_slot]
          .camera_slot,
      1);

  BaSolveIntent stale = MakeIntent(generation2);
  stale.catalog_generation = 1;
  DenseActiveState state = MakeState();
  DensePointState point31;
  point31.point_slot = 1;
  point31.state_generation = 9;
  point31.xyz = {{2.0, 3.0, 4.0}};
  state.points.push_back(point31);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer active;
  NativeHostSolvePreparationRuntime runtime;
  BOOST_CHECK(!materializer.Materialize(generation2, stale, state, &view,
                                        &active, &runtime, &error));
  const uint64_t generation2_value = generation2.generation();
  generation2 = CatalogReadLease();

  CoalescedBaGraphMutation erase;
  erase.owner_epoch = 101;
  erase.revision_before = 4;
  erase.revision_after = 5;
  erase.tombstone_point_ids.push_back(31);
  BOOST_REQUIRE(store.ApplyCoalescedMutation(erase, &update, &error));
  BOOST_CHECK(!store.AcquireReadLease().points()[1].header.alive);
  CoalescedBaGraphMutation revive;
  revive.owner_epoch = 101;
  revive.revision_before = 5;
  revive.revision_after = 6;
  revive.point_upserts.push_back({31});
  revive.observation_upserts.push_back({21, 1, 31, {{110.0, 210.0}}});
  BOOST_REQUIRE(store.ApplyCoalescedMutation(revive, &update, &error));
  BOOST_CHECK(store.AcquireReadLease().points()[1].header.alive);
  BOOST_CHECK_EQUAL(store.AcquireReadLease().FindObservation(21, 1)->point_slot,
                    1);

  CoalescedBaGraphMutation full_rebuild;
  full_rebuild.owner_epoch = 101;
  full_rebuild.revision_before = 6;
  full_rebuild.revision_after = 7;
  full_rebuild.force_full_rebuild = true;
  BOOST_CHECK(!store.ApplyCoalescedMutation(full_rebuild, &update, &error));
  BOOST_CHECK(update.full_rebuild_required);
  BOOST_CHECK_EQUAL(store.topology_revision(), 6);
  BOOST_CHECK(!store.AcquireReadLease().valid());

  HostBaGraphColdInput invalid_rebuild = MakeGraph();
  invalid_rebuild.topology_revision = 7;
  invalid_rebuild.cameras.clear();
  BOOST_CHECK(!store.ColdBuild(invalid_rebuild, &update, &error));
  BOOST_CHECK_EQUAL(store.topology_revision(), 6);
  HostBaGraphColdInput rebuild = MakeGraph();
  rebuild.topology_revision = 7;
  BOOST_REQUIRE_MESSAGE(store.ColdBuild(rebuild, &update, &error), error);
  BOOST_CHECK(update.full_rebuild);
  BOOST_CHECK_EQUAL(store.topology_revision(), 7);
  BOOST_CHECK_GT(store.generation(), generation2_value);

  CoalescedBaGraphMutation camera_delete;
  camera_delete.owner_epoch = 101;
  camera_delete.revision_before = 7;
  camera_delete.revision_after = 8;
  camera_delete.tombstone_camera_ids.push_back(7);
  BOOST_CHECK(!store.ApplyCoalescedMutation(camera_delete, &update, &error));
  BOOST_CHECK(update.full_rebuild_required);
  BOOST_CHECK(!store.AcquireReadLease().valid());
}

BOOST_AUTO_TEST_CASE(OwnerIsolationAndCascadingTombstone) {
  HostBaGraphStore first(101);
  HostBaGraphStore second(202);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE(first.ColdBuild(MakeGraph(), &update, &error));
  HostBaGraphColdInput other = MakeGraph();
  other.owner_epoch = 202;
  BOOST_REQUIRE(second.ColdBuild(other, &update, &error));
  {
    const CatalogReadLease first_lease = first.AcquireReadLease();
    const CatalogReadLease second_lease = second.AcquireReadLease();
    BOOST_REQUIRE(first_lease.valid());
    BOOST_REQUIRE(second_lease.valid());
    BOOST_CHECK(first.IsCurrent(first_lease));
    BOOST_CHECK(second.IsCurrent(second_lease));
    BOOST_CHECK(!second.IsCurrent(first_lease));
    BOOST_CHECK(!first.IsCurrent(second_lease));
  }

  CoalescedBaGraphMutation mutation;
  mutation.owner_epoch = 101;
  mutation.revision_before = 1;
  mutation.revision_after = 2;
  mutation.tombstone_point_ids.push_back(30);
  BOOST_REQUIRE_MESSAGE(first.ApplyCoalescedMutation(mutation, &update, &error),
                        error);
  const CatalogReadLease lease = first.AcquireReadLease();
  BOOST_CHECK(!lease.points()[0].header.alive);
  BOOST_CHECK(!lease.observations()[0].header.alive);
  BOOST_CHECK(!lease.observations()[1].header.alive);
  BOOST_CHECK_EQUAL(update.cascaded_observation_tombstones, 2);
}

BOOST_AUTO_TEST_CASE(DeltaComplexityTracksOnlyTouchedIncidence) {
  constexpr size_t kUnrelatedObservationCount = 4096;
  HostBaGraphStore store(303);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      store.ColdBuild(MakeLargeGraph(kUnrelatedObservationCount), &update,
                      &error),
      error);
  BOOST_CHECK_EQUAL(update.full_catalog_scans, 1);

  const auto apply = [&](CoalescedBaGraphMutation mutation) {
    mutation.owner_epoch = 303;
    BOOST_REQUIRE_MESSAGE(
        store.ApplyCoalescedMutation(mutation, &update, &error), error);
    BOOST_CHECK(update.in_place_delta);
    BOOST_CHECK_EQUAL(update.transaction_copy_bytes, 0);
    BOOST_CHECK_EQUAL(update.full_catalog_scans, 0);
    BOOST_CHECK_EQUAL(update.capacity_growth_events, 0);
    BOOST_CHECK_EQUAL(update.capacity_growth_copy_bytes, 0);
    BOOST_CHECK_EQUAL(update.hash_rehash_events, 0);
    BOOST_CHECK_EQUAL(update.hash_rehash_entries, 0);
  };

  CoalescedBaGraphMutation add;
  add.revision_before = 1;
  add.revision_after = 2;
  add.observation_upserts.push_back(
      {10000, 1, 20001, {{10.0, 20.0}}});
  apply(add);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_visited, 0);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_appended, 2);
  BOOST_CHECK_EQUAL(update.touched_entity_records, 2);

  CoalescedBaGraphMutation xy;
  xy.revision_before = 2;
  xy.revision_after = 3;
  xy.observation_upserts.push_back(
      {10000, 1, 20001, {{11.0, 21.0}}});
  apply(xy);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_visited, 0);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_appended, 0);
  BOOST_CHECK_EQUAL(update.touched_observation_records, 1);

  CoalescedBaGraphMutation rebind;
  rebind.revision_before = 3;
  rebind.revision_after = 4;
  rebind.observation_upserts.push_back(
      {10000, 1, 20002, {{11.0, 21.0}}});
  apply(rebind);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_visited, 0);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_appended, 2);

  CoalescedBaGraphMutation tombstone;
  tombstone.revision_before = 4;
  tombstone.revision_after = 5;
  tombstone.tombstone_observations.push_back({10000, 1});
  apply(tombstone);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_visited, 0);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_appended, 0);

  CoalescedBaGraphMutation revive;
  revive.revision_before = 5;
  revive.revision_after = 6;
  revive.observation_upserts.push_back(
      {10000, 1, 20001, {{11.0, 21.0}}});
  apply(revive);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_appended, 2);

  CoalescedBaGraphMutation original_to_b;
  original_to_b.revision_before = 6;
  original_to_b.revision_after = 7;
  original_to_b.observation_upserts.push_back(
      {10000, 0, 20001, {{0.0, 1.0}}});
  apply(original_to_b);
  CoalescedBaGraphMutation original_to_a;
  original_to_a.revision_before = 7;
  original_to_a.revision_after = 8;
  original_to_a.observation_upserts.push_back(
      {10000, 0, 20000, {{0.0, 1.0}}});
  apply(original_to_a);

  CoalescedBaGraphMutation delete_point;
  delete_point.revision_before = 8;
  delete_point.revision_after = 9;
  delete_point.tombstone_point_ids.push_back(20001);
  apply(delete_point);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_visited, 4);
  BOOST_CHECK_LT(update.adjacency_nodes_visited,
                 kUnrelatedObservationCount);
  {
    const CatalogReadLease lease = store.AcquireReadLease();
    BOOST_REQUIRE(lease.valid());
    BOOST_CHECK(lease.FindObservation(10000, 0)->header.alive);
    BOOST_CHECK(!lease.FindObservation(10000, 1)->header.alive);
    BOOST_CHECK_EQUAL(lease.points()[0].track_length, 1);
  }

  CoalescedBaGraphMutation delete_image;
  delete_image.revision_before = 9;
  delete_image.revision_after = 10;
  delete_image.tombstone_image_ids.push_back(10002);
  apply(delete_image);
  BOOST_CHECK_EQUAL(update.adjacency_nodes_visited, 1);
  BOOST_CHECK_LT(update.adjacency_nodes_visited,
                 kUnrelatedObservationCount);
}

BOOST_AUTO_TEST_CASE(GeometricCapacityGrowthIsAmortized) {
  HostBaGraphStore store(404);
  HostBaGraphColdInput cold = MakeGraph();
  cold.owner_epoch = 404;
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE_MESSAGE(store.ColdBuild(cold, &update, &error), error);

  uint64_t revision = 1;
  for (uint64_t i = 0; i < 8; ++i) {
    CoalescedBaGraphMutation append;
    append.owner_epoch = 404;
    append.revision_before = revision;
    append.revision_after = ++revision;
    append.point_upserts.push_back({1000 + i});
    BOOST_REQUIRE_MESSAGE(
        store.ApplyCoalescedMutation(append, &update, &error), error);
    BOOST_CHECK_EQUAL(update.capacity_growth_events, 0);
    BOOST_CHECK_EQUAL(update.hash_rehash_events, 0);
  }

  CoalescedBaGraphMutation cross_capacity;
  cross_capacity.owner_epoch = 404;
  cross_capacity.revision_before = revision;
  cross_capacity.revision_after = ++revision;
  for (uint64_t i = 0; i < 60; ++i)
    cross_capacity.point_upserts.push_back({2000 + i});
  BOOST_REQUIRE_MESSAGE(
      store.ApplyCoalescedMutation(cross_capacity, &update, &error), error);
  BOOST_CHECK_EQUAL(update.capacity_growth_events, 1);
  BOOST_CHECK_EQUAL(update.capacity_growth_copy_bytes,
                    9 * sizeof(HostBaPointSlot));
  BOOST_CHECK_EQUAL(update.hash_rehash_events, 1);
  BOOST_CHECK_EQUAL(update.hash_rehash_entries, 9);

  CoalescedBaGraphMutation within_headroom;
  within_headroom.owner_epoch = 404;
  within_headroom.revision_before = revision;
  within_headroom.revision_after = ++revision;
  within_headroom.point_upserts.push_back({3000});
  BOOST_REQUIRE_MESSAGE(
      store.ApplyCoalescedMutation(within_headroom, &update, &error), error);
  BOOST_CHECK_EQUAL(update.capacity_growth_events, 0);
  BOOST_CHECK_EQUAL(update.hash_rehash_events, 0);

  CoalescedBaGraphMutation update_only;
  update_only.owner_epoch = 404;
  update_only.revision_before = revision;
  update_only.revision_after = ++revision;
  update_only.point_upserts.push_back({30});
  BOOST_REQUIRE_MESSAGE(
      store.ApplyCoalescedMutation(update_only, &update, &error), error);
  BOOST_CHECK(update.semantic_noop);
  BOOST_CHECK_EQUAL(update.capacity_growth_events, 0);
  BOOST_CHECK_EQUAL(update.hash_rehash_events, 0);
}

BOOST_AUTO_TEST_CASE(GlobalWholeAndSelectionRevisionRemainExplicitIdentity) {
  HostBaGraphStore store(101);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE(store.ColdBuild(MakeGraph(), &update, &error));
  const CatalogReadLease lease = store.AcquireReadLease();
  DenseActiveState state = MakeState();
  state.images[0].quaternion = {{1.0, 0.0, 0.0, 0.0}};
  NativeHostSolveMaterializer materializer;
  for (const BaKind kind : {BaKind::kGlobal, BaKind::kWhole}) {
    BaSolveIntent intent = MakeIntent(lease);
    intent.kind = kind;
    intent.selection_revision += static_cast<uint8_t>(kind);
    intent.active_image_slots.push_back(1);
    intent.explicit_variable_point_slots.clear();
    intent.fixed_pose_slots.push_back(0);
    NativeHostSolveView view;
    ActiveStateBuffer active;
    NativeHostSolvePreparationRuntime runtime;
    BOOST_REQUIRE_MESSAGE(materializer.Materialize(
                              lease, intent, state, &view, &active, &runtime,
                              &error),
                          error);
    BOOST_CHECK(view.kind == kind);
    BOOST_CHECK_EQUAL(view.identity.selection_revision,
                      intent.selection_revision);
    BOOST_CHECK(view.boundary_image_slots.empty());
    BOOST_CHECK_EQUAL(view.active_image_slots.size(), 2);
    BOOST_CHECK_EQUAL(view.fixed.images[0].pose_constant, 1);
    BOOST_CHECK_EQUAL(view.fixed.points[0].constant, 0);
  }
}

BOOST_AUTO_TEST_CASE(ReadLeasePinsPublicationPastStoreLifetime) {
  CatalogReadLease lease;
  {
    HostBaGraphStore store(101);
    HostBaGraphUpdateResult update;
    std::string error;
    BOOST_REQUIRE(store.ColdBuild(MakeGraph(), &update, &error));
    lease = store.AcquireReadLease();
  }
  BOOST_REQUIRE(lease.valid());
  BOOST_CHECK_EQUAL(lease.owner_epoch(), 101);
  BOOST_CHECK_EQUAL(lease.observations().size, 2);
  BOOST_CHECK_EQUAL(lease.FindObservation(21, 1)->point_slot, 0);
}

BOOST_AUTO_TEST_CASE(StateGenerationAndVariableCameraFailClosed) {
  HostBaGraphStore store(101);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE(store.ColdBuild(MakeGraph(), &update, &error));
  const CatalogReadLease lease = store.AcquireReadLease();
  DenseActiveState state = MakeState();
  BaSolveIntent intent = MakeIntent(lease);
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer active;
  NativeHostSolvePreparationRuntime runtime;

  state.images[0].state_generation -= 1;
  BOOST_CHECK(!materializer.Materialize(lease, intent, state, &view, &active,
                                        &runtime, &error));
  BOOST_CHECK(error.find("state slot invalid") != std::string::npos);

  state = MakeState();
  intent.camera_policies[0].constant = false;
  error.clear();
  BOOST_CHECK(!materializer.Materialize(lease, intent, state, &view, &active,
                                        &runtime, &error));
  BOOST_CHECK(error.find("does not support variable cameras") !=
              std::string::npos);
}

BOOST_AUTO_TEST_CASE(NonemptyResidualSelectionRequiresExplicitSourceOrder) {
  HostBaGraphStore store(101);
  HostBaGraphUpdateResult update;
  std::string error;
  BOOST_REQUIRE(store.ColdBuild(MakeGraph(), &update, &error));
  const CatalogReadLease lease = store.AcquireReadLease();
  BaSolveIntent intent = MakeIntent(lease);
  intent.source_insertion_order.clear();
  DenseActiveState state = MakeState();
  NativeHostSolveMaterializer materializer;
  NativeHostSolveView view;
  ActiveStateBuffer active;
  NativeHostSolvePreparationRuntime runtime;
  BOOST_CHECK(!materializer.Materialize(lease, intent, state, &view, &active,
                                        &runtime, &error));
  BOOST_CHECK(error.find("explicitly cover every residual") !=
              std::string::npos);

  const auto rejected = [&](const std::vector<BaSolveIntent::ResidualSelection>&
                                order) {
    BaSolveIntent candidate = MakeIntent(lease);
    candidate.source_insertion_order = order;
    NativeHostSolveView candidate_view;
    ActiveStateBuffer candidate_state;
    NativeHostSolvePreparationRuntime candidate_runtime;
    error.clear();
    return !materializer.Materialize(
        lease, candidate, state, &candidate_view, &candidate_state,
        &candidate_runtime, &error);
  };
  BOOST_CHECK(rejected({{ResidualKind::kVisual, 0},
                        {ResidualKind::kVisual, 0},
                        {ResidualKind::kLidar, 3}}));
  BOOST_CHECK(rejected({{ResidualKind::kVisual, 0},
                        {ResidualKind::kVisual, 99},
                        {ResidualKind::kLidar, 3}}));
  BOOST_CHECK(rejected({{ResidualKind::kLidar, 0},
                        {ResidualKind::kVisual, 1},
                        {ResidualKind::kVisual, 0}}));
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap
