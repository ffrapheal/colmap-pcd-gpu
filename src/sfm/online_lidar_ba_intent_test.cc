#define TEST_NAME "sfm/online_lidar_ba_intent"
#include "util/testing.h"

#ifdef GPU_BA_CUDA_ENABLED

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include "base/camera_models.h"
#include "gpu_ba/native_graph_problem_store.h"
#include "optim/bundle_adjustment.h"
#include "sfm/online_lidar_ba_intent.h"

namespace colmap {
namespace {

using WorldPoint = std::array<float, 3>;

constexpr size_t kImagePointCount = 128;
constexpr uint64_t kOwnerEpoch = 9107;

class IntentTempPcd {
 public:
  IntentTempPcd() {
    static std::atomic<uint64_t> next_id{0};
    path_ = "/tmp/online_lidar_ba_intent_test_" +
            std::to_string(static_cast<uint64_t>(getpid())) + "_" +
            std::to_string(next_id.fetch_add(1)) + ".pcd";
  }

  ~IntentTempPcd() { std::remove(path_.c_str()); }

  IntentTempPcd(const IntentTempPcd&) = delete;
  IntentTempPcd& operator=(const IntentTempPcd&) = delete;

  const std::string& Path() const { return path_; }

  void Write(const std::vector<WorldPoint>& points) const {
    std::ofstream file(path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
      throw std::runtime_error("cannot create intent test PCD");
    }
    file << "VERSION 0.7\n"
         << "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
         << "SIZE 4 4 4 4 4 4 4 4\n"
         << "TYPE F F F F F F F F\n"
         << "COUNT 1 1 1 1 1 1 1 1\n"
         << "WIDTH " << points.size() << "\n"
         << "HEIGHT 1\n"
         << "POINTS " << points.size() << "\n"
         << "DATA ascii\n"
         << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const WorldPoint& point : points) {
      file << point[2] << ' ' << -point[0] << ' ' << -point[1]
           << " 1 0 0 0 0\n";
    }
    if (!file) {
      throw std::runtime_error("cannot write intent test PCD");
    }
  }

 private:
  std::string path_;
};

lidar::IncrementalCausalLidarMapDependencies IntentMapDependencies(
    const std::array<float, 4>& outer_normal,
    const std::array<float, 4>& inner_normal) {
  lidar::IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [outer_normal, inner_normal](
          const std::vector<float>& support_xyz,
          const std::vector<float>& query_xyz,
          const float outer_radius,
          const float inner_radius,
          std::vector<float>* outer_normals,
          std::vector<float>* inner_normals,
          lidar::CudaNormalEstimationTiming* timing,
          std::string* error) {
        (void)support_xyz;
        (void)outer_radius;
        (void)inner_radius;
        if (outer_normals == nullptr || inner_normals == nullptr ||
            timing == nullptr || error == nullptr || query_xyz.size() % 3 != 0) {
          return false;
        }
        outer_normals->clear();
        inner_normals->clear();
        for (size_t index = 0; index < query_xyz.size() / 3; ++index) {
          outer_normals->insert(
              outer_normals->end(), outer_normal.begin(), outer_normal.end());
          inner_normals->insert(
              inner_normals->end(), inner_normal.begin(), inner_normal.end());
        }
        *timing = lidar::CudaNormalEstimationTiming();
        error->clear();
        return true;
      };
  return dependencies;
}

std::shared_ptr<const lidar::LidarMapSnapshot> SnapshotFromPoints(
    const std::vector<WorldPoint>& points,
    const std::array<float, 4>& outer_normal =
        std::array<float, 4>{{2.0f, 0.0f, 0.0f, 0.125f}},
    const std::array<float, 4>& inner_normal =
        std::array<float, 4>{{0.0f, 2.0f, 0.0f, 0.25f}}) {
  IntentTempPcd scan;
  scan.Write(points);
  lidar::IncrementalCausalLidarMap map(
      IntentMapDependencies(outer_normal, inner_normal));
  lidar::ScanSource source;
  source.scan_index = 1;
  source.pcd_path = scan.Path();
  source.input_frame = lidar::LidarCoordinateFrame::FASTLIO_WORLD;
  std::string error;
  if (!map.AppendScan(source, nullptr, &error)) {
    throw std::runtime_error(error);
  }
  return map.GetSnapshot();
}

lidar::PcdProjectionOptions ProjectionOptions() {
  lidar::PcdProjectionOptions options;
  options.depth_image_scale = 1.0;
  options.max_proj_scale = 0;
  options.min_proj_scale = 0;
  options.min_proj_dist = 1.0;
  options.choose_meter = 10.0f;
  options.min_lidar_proj_dist = 0.0;
  return options;
}

BundleAdjustmentOptions StrictOptions() {
  BundleAdjustmentOptions options;
  options.ba_backend = "custom_cuda";
  options.ba_fallback_to_ceres = false;
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp64";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";
  options.ba_cuda_prepared_selection_cache =
      gpu_ba::CudaPreparedSelectionCacheMode::kEnabled;
  options.ba_lidar_residual = "legacy_exact";
  options.refine_extrinsics = true;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.if_add_lidar_constraint = true;
  options.proj_lidar_constraint_weight = 11.0;
  options.icp_lidar_constraint_weight = 23.0;
  options.icp_ground_lidar_constraint_weight = 37.0;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  options.loss_function_scale = 0.0;
  options.solver_options.max_num_iterations = 9;
  options.solver_options.max_num_consecutive_invalid_steps = 4;
  options.solver_options.function_tolerance = 1e-8;
  options.solver_options.gradient_tolerance = 2e-9;
  options.solver_options.parameter_tolerance = 3e-10;
  options.solver_options.max_solver_time_in_seconds = 17.0;
  options.solver_options.initial_trust_region_radius = 123.0;
  options.solver_options.min_trust_region_radius = 1e-20;
  options.solver_options.max_trust_region_radius = 1e12;
  options.solver_options.min_relative_decrease = 5e-4;
  options.solver_options.min_lm_diagonal = 7e-7;
  options.solver_options.max_lm_diagonal = 8e20;
  return options;
}

void SetSnapshotIdentity(
    const std::shared_ptr<const lidar::LidarMapSnapshot>& snapshot,
    OnlineLidarAssociationRequest* request) {
  request->expected_map_version = snapshot->Version();
  request->expected_max_scan_index = snapshot->MaxScanIndex();
  request->expected_snapshot_sha256 = snapshot->SnapshotSha256();
  request->expected_geometry_sha256 = snapshot->GeometrySha256();
  request->snapshot = snapshot;
}

class IntentFixture {
 public:
  explicit IntentFixture(
      std::vector<WorldPoint> snapshot_points = {
          WorldPoint{{1.0f, 2.0f, 3.0f}},
          WorldPoint{{4.0f, 5.0f, 6.0f}},
          WorldPoint{{7.0f, 8.0f, 9.0f}}},
      const std::array<float, 4>& outer_normal =
          std::array<float, 4>{{2.0f, 0.0f, 0.0f, 0.125f}},
      const std::array<float, 4>& inner_normal =
          std::array<float, 4>{{0.0f, 2.0f, 0.0f, 0.25f}})
      : snapshot(SnapshotFromPoints(
            snapshot_points, outer_normal, inner_normal)) {
    Camera camera;
    camera.SetCameraId(1);
    camera.InitializeWithId(OpenCVCameraModel::model_id, 100.0, 100, 80);
    camera.SetParams(
        {100.0, 100.0, 50.0, 40.0, 0.0, 0.0, 0.0, 0.0});
    reconstruction.AddCamera(camera);

    for (image_t image_id = 1; image_id <= 4; ++image_id) {
      Image image;
      image.SetImageId(image_id);
      image.SetCameraId(camera.CameraId());
      image.SetName("intent_image_" + std::to_string(image_id));
      image.SetQvec(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
      image.SetTvec(Eigen::Vector3d::Zero());
      image.SetPoints2D(std::vector<Eigen::Vector2d>(
          kImagePointCount, Eigen::Vector2d(50.0, 40.0)));
      reconstruction.AddImage(std::move(image));
      reconstruction.RegisterImage(image_id);
    }
  }

  ~IntentFixture() {
    if (journal_started) {
      reconstruction.EndStructureJournal();
    }
  }

  point3D_t AddPoint(const Eigen::Vector3d& xyz,
                     std::vector<TrackElement> elements,
                     const int global_opt_num = 0) {
    Track track;
    track.SetElements(std::move(elements));
    const point3D_t point3D_id = reconstruction.AddPoint3D(
        xyz, std::move(track), Eigen::Vector3ub(7, 8, 9));
    for (int index = 0; index < global_opt_num; ++index) {
      reconstruction.Point3D(point3D_id).AddGlobalOptNum();
    }
    return point3D_id;
  }

  void StartJournal() {
    reconstruction.BeginStructureJournal(kOwnerEpoch, 256);
    journal_started = true;
  }

  OnlineLidarAssociationRequest Request(
      std::vector<point3D_t> point3D_ids) const {
    OnlineLidarAssociationRequest request;
    request.attempt_id = 17;
    request.pass_index = 3;
    request.trigger_image_id = 2;
    request.ordered_frozen_image_ids = {1, 3, 2};
    request.point3D_ids = std::move(point3D_ids);
    request.options.local_lidar_kdtree_only = true;
    request.options.kdtree_max_search_range = 0.6;
    request.options.kdtree_min_search_range = 0.25;
    request.options.search_range_drop_speed = 0.1;
    request.projection_options = ProjectionOptions();
    SetSnapshotIdentity(snapshot, &request);
    return request;
  }

  Reconstruction reconstruction;
  std::shared_ptr<const lidar::LidarMapSnapshot> snapshot;
  bool journal_started = false;
};

class TransactionLocalNativeStore {
 public:
  TransactionLocalNativeStore(Reconstruction* reconstruction,
                              const uint64_t owner_epoch)
      : store_(reconstruction, owner_epoch) {
    binding_.store = &store_;
    binding_.owner_epoch = owner_epoch;
    binding_.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  }

  ~TransactionLocalNativeStore() {
    std::string ignored;
    store_.Shutdown(&ignored);
  }

  const gpu_ba::CudaHostStoreBinding& Binding() const { return binding_; }

 private:
  gpu_ba::GpuBaHostProblemStore store_;
  gpu_ba::CudaHostStoreBinding binding_;
};

std::array<uint8_t, 32> DecodeSha256(const std::string& encoded) {
  if (encoded.size() != 64) {
    throw std::runtime_error("test SHA256 has the wrong size");
  }
  std::array<uint8_t, 32> decoded{};
  for (size_t index = 0; index < decoded.size(); ++index) {
    const auto nibble = [](const char value) -> uint8_t {
      if (value >= '0' && value <= '9') {
        return static_cast<uint8_t>(value - '0');
      }
      if (value >= 'a' && value <= 'f') {
        return static_cast<uint8_t>(value - 'a' + 10);
      }
      throw std::runtime_error("test SHA256 is not lowercase hexadecimal");
    };
    decoded[index] = static_cast<uint8_t>(
        (nibble(encoded[2 * index]) << 4) | nibble(encoded[2 * index + 1]));
  }
  return decoded;
}

void CheckCudaOptionsEqual(const gpu_ba::CudaFullLmOptions& first,
                           const gpu_ba::CudaFullLmOptions& second) {
  BOOST_CHECK(first.arithmetic_precision == second.arithmetic_precision);
  BOOST_CHECK(first.device_context_mode == second.device_context_mode);
  BOOST_CHECK(first.hot_kernel_mode == second.hot_kernel_mode);
  BOOST_CHECK(first.execution_profile == second.execution_profile);
  BOOST_CHECK(first.audit_profile == second.audit_profile);
  BOOST_CHECK(first.prepared_selection_cache_mode ==
              second.prepared_selection_cache_mode);
  BOOST_CHECK_EQUAL(first.performance_mode, second.performance_mode);
  BOOST_CHECK(first.instrumentation_mode == second.instrumentation_mode);
  BOOST_CHECK(first.current_linearization_cache_mode ==
              second.current_linearization_cache_mode);
  BOOST_CHECK(first.fault_injection == second.fault_injection);
  BOOST_CHECK_EQUAL(first.capture_state_trace, second.capture_state_trace);
  BOOST_CHECK_EQUAL(first.pair_chunk_limit_bytes_for_testing,
                    second.pair_chunk_limit_bytes_for_testing);
  BOOST_CHECK_EQUAL(first.max_num_iterations, second.max_num_iterations);
  BOOST_CHECK_EQUAL(first.max_num_consecutive_invalid_steps,
                    second.max_num_consecutive_invalid_steps);
  BOOST_CHECK_EQUAL(first.max_solver_time_in_seconds,
                    second.max_solver_time_in_seconds);
  BOOST_CHECK_EQUAL(first.function_tolerance, second.function_tolerance);
  BOOST_CHECK_EQUAL(first.gradient_tolerance, second.gradient_tolerance);
  BOOST_CHECK_EQUAL(first.parameter_tolerance, second.parameter_tolerance);
  BOOST_CHECK_EQUAL(first.initial_trust_region_radius,
                    second.initial_trust_region_radius);
  BOOST_CHECK_EQUAL(first.min_trust_region_radius,
                    second.min_trust_region_radius);
  BOOST_CHECK_EQUAL(first.max_trust_region_radius,
                    second.max_trust_region_radius);
  BOOST_CHECK_EQUAL(first.min_relative_decrease,
                    second.min_relative_decrease);
  BOOST_CHECK_EQUAL(first.layer_c.lambda, second.layer_c.lambda);
  BOOST_CHECK(first.layer_c.fault_injection == second.layer_c.fault_injection);
  BOOST_CHECK_EQUAL(first.layer_c.pair_chunk_limit_bytes_for_testing,
                    second.layer_c.pair_chunk_limit_bytes_for_testing);
  BOOST_CHECK(first.layer_c.schur_contribution_backend ==
              second.layer_c.schur_contribution_backend);
  BOOST_CHECK_EQUAL(first.layer_c.schur_segment_size_for_testing,
                    second.layer_c.schur_segment_size_for_testing);
  BOOST_CHECK(first.layer_c.layer_b.loss_mode ==
              second.layer_c.layer_b.loss_mode);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.loss_scale,
                    second.layer_c.layer_b.loss_scale);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.cost_reduction_threads,
                    second.layer_c.layer_b.cost_reduction_threads);
  BOOST_CHECK(first.layer_c.layer_b.reduction_mode ==
              second.layer_c.layer_b.reduction_mode);
  BOOST_CHECK(first.layer_c.layer_b.fault_injection ==
              second.layer_c.layer_b.fault_injection);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.memory_budget_override_bytes,
                    second.layer_c.layer_b.memory_budget_override_bytes);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.min_lm_diagonal,
                    second.layer_c.layer_b.min_lm_diagonal);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.max_lm_diagonal,
                    second.layer_c.layer_b.max_lm_diagonal);
  BOOST_CHECK(first.layer_c.layer_b.hessian_assembly_backend ==
              second.layer_c.layer_b.hessian_assembly_backend);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.hessian_segment_size_for_testing,
                    second.layer_c.layer_b.hessian_segment_size_for_testing);
  BOOST_CHECK(first.layer_c.layer_b.frozen_pose_jacobi_scaling ==
              second.layer_c.layer_b.frozen_pose_jacobi_scaling);
  BOOST_CHECK(first.layer_c.layer_b.frozen_point_jacobi_scaling ==
              second.layer_c.layer_b.frozen_point_jacobi_scaling);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.device_frozen_jacobi_scaling,
                    second.layer_c.layer_b.device_frozen_jacobi_scaling);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.layer_a.device,
                    second.layer_c.layer_b.layer_a.device);
  BOOST_CHECK_EQUAL(first.layer_c.layer_b.layer_a.block_size,
                    second.layer_c.layer_b.layer_a.block_size);
  BOOST_CHECK(first.layer_c.layer_b.layer_a.memory_mode ==
              second.layer_c.layer_b.layer_a.memory_mode);
  BOOST_CHECK(first.layer_c.layer_b.layer_a.residual_order ==
              second.layer_c.layer_b.layer_a.residual_order);
  BOOST_CHECK_EQUAL(first.audit_fault_record_capacity_for_testing,
                    second.audit_fault_record_capacity_for_testing);
  BOOST_CHECK_EQUAL(first.audit_secondary_capacity_for_testing,
                    second.audit_secondary_capacity_for_testing);
  BOOST_CHECK_EQUAL(first.audit_resource_registry_capacity_for_testing,
                    second.audit_resource_registry_capacity_for_testing);
  BOOST_CHECK_EQUAL(first.audit_timing_interval_capacity_for_testing,
                    second.audit_timing_interval_capacity_for_testing);
  BOOST_CHECK_EQUAL(first.prepared_host_view, second.prepared_host_view);
  BOOST_CHECK_EQUAL(first.indexed_active_solve, second.indexed_active_solve);
  BOOST_CHECK_EQUAL(first.indexed_catalog_tables,
                    second.indexed_catalog_tables);
}

void CheckReconstructionEqual(const Reconstruction& first,
                              const Reconstruction& second) {
  BOOST_CHECK_EQUAL(first.StructureOwnerEpoch(), second.StructureOwnerEpoch());
  BOOST_CHECK_EQUAL(first.StructureRevision(), second.StructureRevision());
  BOOST_CHECK(first.RegImageIds() == second.RegImageIds());
  BOOST_REQUIRE_EQUAL(first.Cameras().size(), second.Cameras().size());
  BOOST_REQUIRE_EQUAL(first.Images().size(), second.Images().size());
  BOOST_REQUIRE_EQUAL(first.Points3D().size(), second.Points3D().size());
  for (const auto& item : first.Cameras()) {
    BOOST_REQUIRE(second.ExistsCamera(item.first));
    BOOST_CHECK(item.second.Params() == second.Camera(item.first).Params());
  }
  for (const auto& item : first.Images()) {
    BOOST_REQUIRE(second.ExistsImage(item.first));
    const Image& lhs = item.second;
    const Image& rhs = second.Image(item.first);
    BOOST_CHECK_EQUAL(lhs.CameraId(), rhs.CameraId());
    BOOST_CHECK_EQUAL(lhs.IsRegistered(), rhs.IsRegistered());
    BOOST_CHECK(lhs.Qvec() == rhs.Qvec());
    BOOST_CHECK(lhs.Tvec() == rhs.Tvec());
    BOOST_REQUIRE_EQUAL(lhs.NumPoints2D(), rhs.NumPoints2D());
    for (point2D_t index = 0; index < lhs.NumPoints2D(); ++index) {
      BOOST_CHECK(lhs.Point2D(index).XY() == rhs.Point2D(index).XY());
      BOOST_CHECK_EQUAL(lhs.Point2D(index).Point3DId(),
                        rhs.Point2D(index).Point3DId());
    }
  }
  for (const auto& item : first.Points3D()) {
    BOOST_REQUIRE(second.ExistsPoint3D(item.first));
    const Point3D& lhs = item.second;
    const Point3D& rhs = second.Point3D(item.first);
    BOOST_CHECK(lhs.XYZ() == rhs.XYZ());
    BOOST_CHECK_EQUAL(lhs.GlobalOptNum(), rhs.GlobalOptNum());
    BOOST_REQUIRE_EQUAL(lhs.Track().Length(), rhs.Track().Length());
    for (size_t index = 0; index < lhs.Track().Length(); ++index) {
      BOOST_CHECK_EQUAL(lhs.Track().Element(index).image_id,
                        rhs.Track().Element(index).image_id);
      BOOST_CHECK_EQUAL(lhs.Track().Element(index).point2D_idx,
                        rhs.Track().Element(index).point2D_idx);
    }
  }
}

void CheckBuildOutputCleared(
    const OnlineLidarNativeBaSolveIntentBuildOutput& output) {
  BOOST_CHECK(!output.built());
  BOOST_CHECK(output.association_output().associations.empty());
  BOOST_CHECK_EQUAL(output.association_audit().attempt_id, 0);
  BOOST_CHECK(output.association_audit().association_sha256.empty());
  BOOST_CHECK_EQUAL(output.intent().owner_epoch, 0);
  BOOST_CHECK(output.intent().active_image_ids.empty());
  BOOST_CHECK(!output.intent().config.resolved);
  CheckCudaOptionsEqual(output.resolved_cuda_options(),
                        gpu_ba::CudaFullLmOptions());
}

void PrimeSummary(OnlineLidarMaterializationSummary* summary) {
  summary->per_image_materialized_count.emplace(99, 99);
  summary->trigger_materialized_count = 99;
  summary->materialized_association_count = 99;
  summary->materialized_visual_residual_count = 99;
  summary->materialized_lidar_residual_count = 99;
  summary->materialized_association_ids.push_back(99);
  summary->materialized_online_lidar_identity.valid = true;
  summary->solver_evaluation_pending = true;
}

void CheckSummaryCleared(const OnlineLidarMaterializationSummary& summary) {
  BOOST_CHECK(summary.per_image_materialized_count.empty());
  BOOST_CHECK_EQUAL(summary.trigger_materialized_count, 0);
  BOOST_CHECK_EQUAL(summary.materialized_association_count, 0);
  BOOST_CHECK_EQUAL(summary.materialized_visual_residual_count, 0);
  BOOST_CHECK_EQUAL(summary.materialized_lidar_residual_count, 0);
  BOOST_CHECK(summary.materialized_association_ids.empty());
  BOOST_CHECK(!summary.materialized_online_lidar_identity.valid);
  BOOST_CHECK(!summary.solver_evaluation_pending);
}

bool BuildIntent(
    const IntentFixture& fixture,
    const BundleAdjustmentOptions& options,
    const OnlineLidarAssociationRequest& request,
    const uint64_t selection_revision,
    OnlineLidarNativeBaSolveIntentBuildOutput* output,
    std::string* error) {
  return BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction,
      options,
      request,
      kOwnerEpoch,
      fixture.reconstruction.StructureRevision(),
      selection_revision,
      gpu_ba::BaKind::kLocal,
      output,
      error);
}

struct DownstreamCallAudit {
  uint64_t native_prepare_calls = 0;
  uint64_t ceres_solve_calls = 0;
};

void CheckBuildFailure(
    const IntentFixture& fixture,
    const BundleAdjustmentOptions& options,
    const OnlineLidarAssociationRequest& request,
    const uint64_t owner_epoch,
    const uint64_t topology_revision,
    const uint64_t selection_revision,
    const gpu_ba::BaKind kind) {
  const Reconstruction before = fixture.reconstruction;
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  DownstreamCallAudit downstream;
  std::string error = "stale";
  const bool built = BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction,
      options,
      request,
      owner_epoch,
      topology_revision,
      selection_revision,
      kind,
      &output,
      &error);
  if (built) {
    ++downstream.native_prepare_calls;
  }
  BOOST_CHECK(!built);
  BOOST_CHECK(!error.empty());
  BOOST_CHECK_EQUAL(downstream.native_prepare_calls, 0);
  BOOST_CHECK_EQUAL(downstream.ceres_solve_calls, 0);
  CheckBuildOutputCleared(output);
  CheckReconstructionEqual(before, fixture.reconstruction);
}

void CheckIdentityEqual(const gpu_ba::NativeBaOnlineLidarIdentity& first,
                        const gpu_ba::NativeBaOnlineLidarIdentity& second) {
  BOOST_CHECK_EQUAL(first.valid, second.valid);
  BOOST_CHECK_EQUAL(first.trigger_image_id, second.trigger_image_id);
  BOOST_CHECK_EQUAL(first.map_version, second.map_version);
  BOOST_CHECK_EQUAL(first.max_scan_index, second.max_scan_index);
  BOOST_CHECK(first.snapshot_sha256 == second.snapshot_sha256);
  BOOST_CHECK(first.geometry_sha256 == second.geometry_sha256);
  BOOST_CHECK(first.association_sha256 == second.association_sha256);
}

void CheckCanonicalBuildSemantics(
    const OnlineLidarNativeBaSolveIntentBuildOutput& first,
    const OnlineLidarNativeBaSolveIntentBuildOutput& second) {
  BOOST_CHECK_EQUAL(first.association_audit().association_sha256,
                    second.association_audit().association_sha256);
  BOOST_CHECK_EQUAL(first.association_audit().selected_association_count,
                    second.association_audit().selected_association_count);
  BOOST_CHECK(
      first.association_audit().preliminary_selected_count_by_image ==
      second.association_audit().preliminary_selected_count_by_image);
  BOOST_CHECK(first.intent().active_image_ids ==
              second.intent().active_image_ids);
  BOOST_CHECK(first.intent().explicit_variable_point_ids ==
              second.intent().explicit_variable_point_ids);
  BOOST_CHECK_EQUAL(first.intent().lidar_constraints.size(),
                    second.intent().lidar_constraints.size());
  CheckIdentityEqual(first.intent().online_lidar_identity,
                     second.intent().online_lidar_identity);
  for (size_t index = 0; index < first.intent().lidar_constraints.size();
       ++index) {
    const gpu_ba::NativeBaLidarConstraint& lhs =
        first.intent().lidar_constraints[index];
    const gpu_ba::NativeBaLidarConstraint& rhs =
        second.intent().lidar_constraints[index];
    BOOST_CHECK_EQUAL(lhs.point3D_id, rhs.point3D_id);
    BOOST_CHECK_EQUAL(lhs.association_id, rhs.association_id);
    BOOST_CHECK_EQUAL(lhs.owner_image_id, rhs.owner_image_id);
    BOOST_CHECK_EQUAL(lhs.owner_point2D_idx, rhs.owner_point2D_idx);
    BOOST_CHECK_EQUAL(lhs.lidar_type, rhs.lidar_type);
    BOOST_CHECK(lhs.frozen_point3D_xyz == rhs.frozen_point3D_xyz);
    BOOST_CHECK(lhs.plane == rhs.plane);
    BOOST_CHECK(lhs.lidar_xyz == rhs.lidar_xyz);
    BOOST_CHECK_EQUAL(lhs.weight, rhs.weight);
    BOOST_CHECK_EQUAL(lhs.search_range, rhs.search_range);
  }
  CheckCudaOptionsEqual(first.resolved_cuda_options(),
                        second.resolved_cuda_options());
}

gpu_ba::PreparedNativeActiveSolve Prepare(
    IntentFixture* fixture,
    const OnlineLidarNativeBaSolveIntentBuildOutput& build_output,
    const gpu_ba::CudaHostStoreBinding& binding) {
  gpu_ba::PreparedNativeActiveSolve prepared;
  std::string error;
  if (!gpu_ba::PrepareCudaNativeBaSolve(build_output.intent(),
                                        &fixture->reconstruction,
                                        build_output.resolved_cuda_options(),
                                        binding,
                                        &prepared,
                                        &error)) {
    throw std::runtime_error(error);
  }
  return prepared;
}

void CheckSummaryFailure(
    const gpu_ba::NativeHostSolveView& view,
    const OnlineLidarNativeBaSolveIntentBuildOutput& build_output) {
  OnlineLidarMaterializationSummary summary;
  PrimeSummary(&summary);
  std::string error = "stale";
  BOOST_CHECK(!SummarizeOnlineLidarMaterialization(
      view, build_output, &summary, &error));
  BOOST_CHECK(!error.empty());
  CheckSummaryCleared(summary);
}

template <typename...>
using VoidT = void;

template <typename T, typename = void>
struct HasTriggerActualCount : std::false_type {};

template <typename T>
struct HasTriggerActualCount<
    T,
    VoidT<decltype(std::declval<T&>().trigger_actual_count)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasActualAssociationCount : std::false_type {};

template <typename T>
struct HasActualAssociationCount<
    T,
    VoidT<decltype(std::declval<T&>().actual_association_count)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSolverEvaluated : std::false_type {};

template <typename T>
struct HasSolverEvaluated<
    T,
    VoidT<decltype(std::declval<T&>().solver_evaluated)>> : std::true_type {};

static_assert(!HasTriggerActualCount<OnlineLidarMaterializationSummary>::value,
              "summary must name trigger counts as materialized");
static_assert(
    !HasActualAssociationCount<OnlineLidarMaterializationSummary>::value,
    "summary must name association counts as materialized");
static_assert(!HasSolverEvaluated<OnlineLidarMaterializationSummary>::value,
              "summary must not claim solver evaluation");
static_assert(
    std::is_same<
        decltype(std::declval<
                 OnlineLidarNativeBaSolveIntentBuildOutput&>()
                     .association_output()),
        const OnlineLidarAssociationOutput&>::value,
    "association output accessor must remain const");
static_assert(
    std::is_same<
        decltype(std::declval<
                 OnlineLidarNativeBaSolveIntentBuildOutput&>()
                     .association_audit()),
        const OnlineLidarAssociationAudit&>::value,
    "association audit accessor must remain const");
static_assert(
    std::is_same<
        decltype(std::declval<
                 OnlineLidarNativeBaSolveIntentBuildOutput&>()
                     .intent()),
        const gpu_ba::NativeBaSolveIntent&>::value,
    "native intent accessor must remain const");
static_assert(
    std::is_same<
        decltype(std::declval<
                 OnlineLidarNativeBaSolveIntentBuildOutput&>()
                     .resolved_cuda_options()),
        const gpu_ba::CudaFullLmOptions&>::value,
    "resolved CUDA options accessor must remain const");
static_assert(
    std::is_nothrow_move_constructible<
        OnlineLidarNativeBaSolveIntentBuildOutput>::value,
    "build output move construction must be noexcept");
static_assert(
    std::is_nothrow_move_assignable<
        OnlineLidarNativeBaSolveIntentBuildOutput>::value,
    "build output move assignment must be noexcept");

BOOST_AUTO_TEST_CASE(TriggerPreflightCannotPublishAnIncompleteIntent) {
  IntentFixture fixture;
  const point3D_t point_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}, {1, 0}});
  fixture.StartJournal();
  const auto request = fixture.Request({point_id});
  const auto options = StrictOptions();
  OnlineLidarNativeBaSolveIntentBuildOutput original;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildIntent(fixture, options, request, 11,
                                    &original, &error), error);
  OnlineLidarNativeBaSolveIntentBuildOutput gated = original;
  OnlineLidarAssociationTriggerGate gate;
  gate.minimum_count = 50;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction, options, request, kOwnerEpoch,
      fixture.reconstruction.StructureRevision(), 11, gpu_ba::BaKind::kLocal,
      &gated, &error, &gate), error);
  BOOST_CHECK(gate.checked);
  BOOST_CHECK(!gate.passed);
  BOOST_CHECK_EQUAL(gate.actual_count, 1);
  BOOST_CHECK(!gated.built());
  BOOST_CHECK(gated.intent().lidar_constraints.empty());

  gate.minimum_count = 1;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction, options, request, kOwnerEpoch,
      fixture.reconstruction.StructureRevision(), 11, gpu_ba::BaKind::kLocal,
      &gated, &error, &gate), error);
  BOOST_REQUIRE(gated.built());
  BOOST_CHECK(gate.passed);
  BOOST_CHECK_EQUAL(gated.association_audit().association_sha256,
                    original.association_audit().association_sha256);
  BOOST_CHECK_EQUAL(gated.intent().lidar_constraints.size(),
                    original.intent().lidar_constraints.size());
}

BOOST_AUTO_TEST_CASE(RejectedTriggerDefersFullWindowObservationValidation) {
  IntentFixture fixture;
  const point3D_t trigger_point = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}, {1, 0}});
  const point3D_t historical_point = fixture.AddPoint(
      Eigen::Vector3d(4.0, 5.0, 6.0), {{1, 1}, {3, 1}});
  fixture.reconstruction.Point3D(historical_point).Track().AddElement(1, 1);
  fixture.StartJournal();
  const auto request = fixture.Request({trigger_point, historical_point});
  const auto options = StrictOptions();
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  OnlineLidarAssociationTriggerGate gate;
  gate.minimum_count = 50;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction, options, request, kOwnerEpoch,
      fixture.reconstruction.StructureRevision(), 11, gpu_ba::BaKind::kLocal,
      &output, &error, &gate), error);
  BOOST_CHECK(gate.checked);
  BOOST_CHECK(!gate.passed);
  BOOST_CHECK(!output.built());
  BOOST_CHECK_EQUAL(gate.actual_count, 1);
  gate.minimum_count = 1;
  BOOST_CHECK(!BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction, options, request, kOwnerEpoch,
      fixture.reconstruction.StructureRevision(), 11, gpu_ba::BaKind::kLocal,
      &output, &error, &gate));
  BOOST_CHECK(gate.passed);
  BOOST_CHECK(!output.built());
}

BOOST_AUTO_TEST_CASE(BuildOutputMoveFullyResetsSource) {
  IntentFixture fixture;
  const point3D_t point3D_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0),
      {{3, 4}, {1, 5}, {2, 9}, {4, 10}},
      2);
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();
  const OnlineLidarAssociationRequest request = fixture.Request({point3D_id});

  OnlineLidarNativeBaSolveIntentBuildOutput source;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      BuildIntent(fixture, options, request, 73, &source, &error), error);
  BOOST_REQUIRE(source.built());
  BOOST_REQUIRE_EQUAL(source.association_output().associations.size(), 1);
  const uint64_t association_id =
      source.association_output().associations.front().association_id;

  OnlineLidarNativeBaSolveIntentBuildOutput moved(std::move(source));
  CheckBuildOutputCleared(source);
  BOOST_REQUIRE(moved.built());
  BOOST_REQUIRE_EQUAL(moved.association_output().associations.size(), 1);
  BOOST_CHECK_EQUAL(
      moved.association_output().associations.front().association_id,
      association_id);
  BOOST_CHECK(!moved.intent().lidar_constraints.empty());

  OnlineLidarNativeBaSolveIntentBuildOutput assigned;
  assigned = std::move(moved);
  CheckBuildOutputCleared(moved);
  BOOST_REQUIRE(assigned.built());
  BOOST_REQUIRE_EQUAL(assigned.association_output().associations.size(), 1);
  BOOST_CHECK_EQUAL(
      assigned.association_output().associations.front().association_id,
      association_id);
}

BOOST_AUTO_TEST_CASE(StrictKdBuildCarriesCompleteNativeContract) {
  IntentFixture fixture;
  const point3D_t point3D_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0),
      {{3, 4}, {1, 5}, {2, 9}, {4, 10}},
      2);
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();
  const OnlineLidarAssociationRequest request = fixture.Request({point3D_id});
  const Reconstruction before = fixture.reconstruction;

  OnlineLidarNativeBaSolveIntentBuildOutput output;
  std::string error;
  constexpr uint64_t kSelectionRevision = 73;
  BOOST_REQUIRE_MESSAGE(
      BuildIntent(
          fixture, options, request, kSelectionRevision, &output, &error),
      error);
  BOOST_CHECK(error.empty());
  CheckReconstructionEqual(before, fixture.reconstruction);

  BOOST_CHECK(output.built());
  BOOST_REQUIRE_EQUAL(output.association_output().associations.size(), 1);
  BOOST_CHECK_EQUAL(output.association_audit().selected_association_count, 1);
  BOOST_CHECK_EQUAL(
      output.association_audit().kdtree_selected_association_count, 1);
  BOOST_CHECK_EQUAL(output.association_audit().projection_call_count, 0);
  BOOST_CHECK(output.intent().active_image_ids ==
              request.ordered_frozen_image_ids);
  BOOST_CHECK(output.intent().visual_observation_scope ==
              gpu_ba::NativeBaVisualObservationScope::kActiveImagesOnly);
  BOOST_CHECK(output.intent().fixed_pose_ids.empty());
  BOOST_REQUIRE_EQUAL(output.intent().translation_policies.size(), 3);
  for (size_t index = 0; index < output.intent().translation_policies.size();
       ++index) {
    BOOST_CHECK_EQUAL(output.intent().translation_policies[index].image_id,
                      request.ordered_frozen_image_ids[index]);
    BOOST_CHECK_EQUAL(output.intent().translation_policies[index].constant_mask,
                      0);
  }
  BOOST_REQUIRE_EQUAL(output.intent().camera_policies.size(), 1);
  BOOST_CHECK_EQUAL(output.intent().camera_policies.front().camera_id, 1);
  BOOST_CHECK(output.intent().camera_policies.front().constant);
  BOOST_CHECK(output.intent().camera_policies.front().fixed_parameter_indices
                  .empty());
  BOOST_CHECK(output.intent().explicit_constant_point_ids.empty());
  BOOST_REQUIRE_EQUAL(output.intent().explicit_variable_point_ids.size(), 1);
  BOOST_CHECK_EQUAL(output.intent().explicit_variable_point_ids.front(),
                    point3D_id);
  BOOST_REQUIRE_EQUAL(output.intent().point_policies.size(), 1);
  BOOST_CHECK_EQUAL(output.intent().point_policies.front().point3D_id,
                    point3D_id);
  BOOST_CHECK(!output.intent().point_policies.front().constant);
  BOOST_CHECK_EQUAL(output.intent().point_policies.front().config_role, 1);
  BOOST_CHECK(output.intent().point_policies.front().has_search_range);
  BOOST_CHECK_SMALL(
      std::abs(output.intent().point_policies.front().search_range - 0.4),
      1e-12);

  const gpu_ba::NativeBaOnlineLidarIdentity& identity =
      output.intent().online_lidar_identity;
  BOOST_CHECK(identity.valid);
  BOOST_CHECK_EQUAL(identity.trigger_image_id, request.trigger_image_id);
  BOOST_CHECK(identity.snapshot_sha256 ==
              DecodeSha256(request.expected_snapshot_sha256));
  BOOST_CHECK(identity.geometry_sha256 ==
              DecodeSha256(request.expected_geometry_sha256));
  BOOST_CHECK(identity.association_sha256 ==
              DecodeSha256(output.association_audit().association_sha256));
  BOOST_CHECK_EQUAL(output.intent().lidar_map_generation,
                    request.expected_map_version);
  BOOST_CHECK_EQUAL(output.intent().lidar_match_config_generation,
                    kSelectionRevision);
  BOOST_CHECK_EQUAL(output.intent().config.config_generation,
                    kSelectionRevision);

  const OnlineLidarAssociation& association =
      output.association_output().associations.front();
  BOOST_CHECK(association.plane.key == association.plane_key);
  BOOST_CHECK(association.plane.frame ==
              lidar::LidarCoordinateFrame::COLMAP_WORLD);
  BOOST_CHECK(association.plane.scale == lidar::LidarNormalScale::BA);
  BOOST_CHECK((association.plane.point ==
               std::array<float, 3>{{1.0f, 2.0f, 3.0f}}));
  BOOST_CHECK((association.plane.normal ==
               std::array<float, 3>{{0.0f, 1.0f, 0.0f}}));
  BOOST_CHECK_EQUAL(association.plane.curvature, 0.25f);
  BOOST_CHECK_GT(association.plane.normal_revision, 0);
  BOOST_CHECK_GT(association.plane.voxel_count, 0);
  BOOST_REQUIRE_EQUAL(output.intent().lidar_constraints.size(), 1);
  const gpu_ba::NativeBaLidarConstraint& constraint =
      output.intent().lidar_constraints.front();
  BOOST_CHECK_EQUAL(constraint.point3D_id, association.point3D_id);
  BOOST_CHECK_EQUAL(constraint.constraint_slot, association.association_id);
  BOOST_CHECK_EQUAL(constraint.physical_identity,
                    association.association_id + 1);
  BOOST_CHECK_EQUAL(constraint.association_id, association.association_id);
  BOOST_CHECK_EQUAL(constraint.owner_image_id, association.owner_image_id);
  BOOST_CHECK_EQUAL(constraint.owner_point2D_idx,
                    association.owner_point2D_idx);
  BOOST_CHECK_EQUAL(constraint.lidar_type,
                    static_cast<uint8_t>(LidarPointType::IcpGround));
  BOOST_CHECK(constraint.frozen_point3D_xyz == association.point3D_xyz);
  BOOST_CHECK(constraint.plane == association.plane_abcd);
  BOOST_CHECK_EQUAL(constraint.plane[0], 0.0);
  BOOST_CHECK_EQUAL(constraint.plane[1], 1.0);
  BOOST_CHECK_EQUAL(constraint.plane[2], 0.0);
  BOOST_CHECK_EQUAL(constraint.plane[3], -2.0);
  BOOST_CHECK_EQUAL(constraint.lidar_xyz[0], association.plane.point[0]);
  BOOST_CHECK_EQUAL(constraint.lidar_xyz[1], association.plane.point[1]);
  BOOST_CHECK_EQUAL(constraint.lidar_xyz[2], association.plane.point[2]);
  BOOST_CHECK_EQUAL(constraint.weight,
                    options.icp_ground_lidar_constraint_weight);
  BOOST_CHECK_EQUAL(constraint.search_range, association.search_range);

  gpu_ba::CudaFullLmOptions expected_options;
  gpu_ba::NativeCudaResolvedConfig expected_config;
  BOOST_REQUIRE_MESSAGE(ResolveNativeBundleAdjustmentCudaConfiguration(
                            options,
                            options.solver_options,
                            kSelectionRevision,
                            &expected_options,
                            &expected_config,
                            &error),
                        error);
  CheckCudaOptionsEqual(output.resolved_cuda_options(), expected_options);
  BOOST_CHECK(output.resolved_cuda_options().prepared_selection_cache_mode ==
              gpu_ba::CudaPreparedSelectionCacheMode::kEnabled);
  BOOST_CHECK(output.resolved_cuda_options().layer_c.layer_b.layer_a
                  .residual_order ==
              gpu_ba::CudaResidualOrder::kSourceInsertion);
}

BOOST_AUTO_TEST_CASE(FailedRebuildResetsSealedOutput) {
  IntentFixture fixture;
  const point3D_t point3D_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0), {{1, 1}, {2, 2}});
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();
  const OnlineLidarAssociationRequest request = fixture.Request({point3D_id});
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      BuildIntent(fixture, options, request, 79, &output, &error), error);
  BOOST_REQUIRE(output.built());

  BOOST_CHECK(!BuildOnlineLidarNativeBaSolveIntent(
      fixture.reconstruction,
      options,
      request,
      0,
      fixture.reconstruction.StructureRevision(),
      80,
      gpu_ba::BaKind::kLocal,
      &output,
      &error));
  BOOST_CHECK(!error.empty());
  CheckBuildOutputCleared(output);
}

BOOST_AUTO_TEST_CASE(PointOrderIsCanonicalAndWindowOrderIsSemantic) {
  IntentFixture fixture;
  const point3D_t first_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0), {{1, 1}, {2, 1}});
  const point3D_t second_id = fixture.AddPoint(
      Eigen::Vector3d(4.0, 5.0, 6.0), {{2, 2}, {3, 2}});
  const point3D_t owner_order_id = fixture.AddPoint(
      Eigen::Vector3d(7.0, 8.0, 9.0), {{1, 3}, {3, 3}});
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();

  const OnlineLidarAssociationRequest sorted_request =
      fixture.Request({first_id, second_id});
  OnlineLidarAssociationRequest shuffled_request =
      fixture.Request({second_id, first_id});
  const std::vector<point3D_t> original_shuffle =
      shuffled_request.point3D_ids;
  OnlineLidarNativeBaSolveIntentBuildOutput sorted_output;
  OnlineLidarNativeBaSolveIntentBuildOutput shuffled_output;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      BuildIntent(
          fixture, options, sorted_request, 81, &sorted_output, &error),
      error);
  BOOST_REQUIRE_MESSAGE(
      BuildIntent(
          fixture, options, shuffled_request, 81, &shuffled_output, &error),
      error);
  BOOST_CHECK(shuffled_request.point3D_ids == original_shuffle);
  CheckCanonicalBuildSemantics(sorted_output, shuffled_output);
  BOOST_CHECK(std::is_sorted(
      shuffled_output.intent().explicit_variable_point_ids.begin(),
      shuffled_output.intent().explicit_variable_point_ids.end()));

  OnlineLidarAssociationRequest duplicate_request =
      fixture.Request({second_id, first_id, second_id});
  CheckBuildFailure(fixture,
                    options,
                    duplicate_request,
                    kOwnerEpoch,
                    fixture.reconstruction.StructureRevision(),
                    82,
                    gpu_ba::BaKind::kLocal);

  OnlineLidarAssociationRequest first_window =
      fixture.Request({owner_order_id});
  OnlineLidarAssociationRequest second_window = first_window;
  second_window.ordered_frozen_image_ids = {3, 1, 2};
  OnlineLidarNativeBaSolveIntentBuildOutput first_window_output;
  OnlineLidarNativeBaSolveIntentBuildOutput second_window_output;
  BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                    options,
                                    first_window,
                                    83,
                                    &first_window_output,
                                    &error),
                        error);
  BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                    options,
                                    second_window,
                                    83,
                                    &second_window_output,
                                    &error),
                        error);
  BOOST_CHECK(first_window_output.intent().active_image_ids ==
              first_window.ordered_frozen_image_ids);
  BOOST_CHECK(second_window_output.intent().active_image_ids ==
              second_window.ordered_frozen_image_ids);
  BOOST_REQUIRE_EQUAL(
      first_window_output.association_output().associations.size(), 1);
  BOOST_REQUIRE_EQUAL(
      second_window_output.association_output().associations.size(), 1);
  BOOST_CHECK_EQUAL(first_window_output.association_output()
                        .associations.front()
                        .owner_image_id,
                    1);
  BOOST_CHECK_EQUAL(second_window_output.association_output()
                        .associations.front()
                        .owner_image_id,
                    3);
  BOOST_CHECK_NE(first_window_output.association_audit().association_sha256,
                 second_window_output.association_audit().association_sha256);
  BOOST_CHECK(first_window_output.intent().online_lidar_identity
                  .association_sha256 !=
              second_window_output.intent().online_lidar_identity
                  .association_sha256);
}

BOOST_AUTO_TEST_CASE(UsedLidarTypeAloneSelectsAndValidatesWeight) {
  {
    IntentFixture fixture({WorldPoint{{0.0f, 0.0f, 2.0f}}});
    const point3D_t point3D_id =
        fixture.AddPoint(Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 7}});
    fixture.StartJournal();
    BundleAdjustmentOptions options = StrictOptions();
    options.proj_lidar_constraint_weight = 1.25;
    options.icp_lidar_constraint_weight = 0.0;
    options.icp_ground_lidar_constraint_weight =
        std::numeric_limits<double>::quiet_NaN();
    OnlineLidarAssociationRequest request = fixture.Request({point3D_id});
    request.options.local_lidar_kdtree_only = false;
    request.options.min_proj_num = 100;
    OnlineLidarNativeBaSolveIntentBuildOutput output;
    std::string error;
    BOOST_REQUIRE_MESSAGE(
        BuildIntent(fixture, options, request, 91, &output, &error), error);
    BOOST_REQUIRE_EQUAL(output.intent().lidar_constraints.size(), 1);
    BOOST_CHECK_EQUAL(output.intent().lidar_constraints.front().lidar_type,
                      static_cast<uint8_t>(LidarPointType::Proj));
    BOOST_CHECK_EQUAL(output.intent().lidar_constraints.front().weight, 1.25);
  }

  {
    IntentFixture fixture(
        {WorldPoint{{1.0f, 2.0f, 3.0f}}},
        std::array<float, 4>{{0.0f, 2.0f, 0.0f, 0.125f}},
        std::array<float, 4>{{2.0f, 0.0f, 0.0f, 0.25f}});
    const point3D_t point3D_id =
        fixture.AddPoint(Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 1}});
    fixture.StartJournal();
    BundleAdjustmentOptions options = StrictOptions();
    options.proj_lidar_constraint_weight =
        std::numeric_limits<double>::quiet_NaN();
    options.icp_lidar_constraint_weight = 2.5;
    options.icp_ground_lidar_constraint_weight = 0.0;
    OnlineLidarNativeBaSolveIntentBuildOutput output;
    std::string error;
    BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                      options,
                                      fixture.Request({point3D_id}),
                                      92,
                                      &output,
                                      &error),
                          error);
    BOOST_REQUIRE_EQUAL(output.intent().lidar_constraints.size(), 1);
    BOOST_CHECK_EQUAL(output.intent().lidar_constraints.front().lidar_type,
                      static_cast<uint8_t>(LidarPointType::Icp));
    BOOST_CHECK_EQUAL(output.intent().lidar_constraints.front().weight, 2.5);
  }

  {
    IntentFixture fixture;
    const point3D_t point3D_id =
        fixture.AddPoint(Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 1}});
    fixture.StartJournal();
    BundleAdjustmentOptions options = StrictOptions();
    options.proj_lidar_constraint_weight = 0.0;
    options.icp_lidar_constraint_weight =
        std::numeric_limits<double>::quiet_NaN();
    options.icp_ground_lidar_constraint_weight = 3.75;
    OnlineLidarNativeBaSolveIntentBuildOutput output;
    std::string error;
    BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                      options,
                                      fixture.Request({point3D_id}),
                                      93,
                                      &output,
                                      &error),
                          error);
    BOOST_REQUIRE_EQUAL(output.intent().lidar_constraints.size(), 1);
    BOOST_CHECK_EQUAL(output.intent().lidar_constraints.front().lidar_type,
                      static_cast<uint8_t>(LidarPointType::IcpGround));
    BOOST_CHECK_EQUAL(output.intent().lidar_constraints.front().weight, 3.75);
  }

  {
    IntentFixture fixture({WorldPoint{{1.0f, 2.0f, 3.0f}}});
    const point3D_t point3D_id =
        fixture.AddPoint(Eigen::Vector3d(30.0, 30.0, 30.0), {{2, 1}});
    fixture.StartJournal();
    BundleAdjustmentOptions options = StrictOptions();
    options.proj_lidar_constraint_weight =
        std::numeric_limits<double>::quiet_NaN();
    options.icp_lidar_constraint_weight =
        std::numeric_limits<double>::quiet_NaN();
    options.icp_ground_lidar_constraint_weight =
        std::numeric_limits<double>::quiet_NaN();
    OnlineLidarNativeBaSolveIntentBuildOutput output;
    std::string error;
    BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                      options,
                                      fixture.Request({point3D_id}),
                                      94,
                                      &output,
                                      &error),
                          error);
    BOOST_CHECK(output.association_output().associations.empty());
    BOOST_CHECK(output.intent().lidar_constraints.empty());
    BOOST_CHECK(output.intent().online_lidar_identity.valid);
    BOOST_CHECK_EQUAL(
        output.association_audit().trigger_preliminary_selected_count, 0);

    TransactionLocalNativeStore store(&fixture.reconstruction, kOwnerEpoch);
    gpu_ba::PreparedNativeActiveSolve prepared =
        Prepare(&fixture, output, store.Binding());
    BOOST_REQUIRE(prepared.valid());
    OnlineLidarMaterializationSummary summary;
    BOOST_REQUIRE_MESSAGE(SummarizeOnlineLidarMaterialization(
                              *prepared.view(), output, &summary, &error),
                          error);
    BOOST_CHECK_EQUAL(summary.trigger_materialized_count, 0);
    BOOST_CHECK_EQUAL(summary.materialized_association_count, 0);
    BOOST_CHECK(summary.materialized_online_lidar_identity.valid);
    BOOST_CHECK(summary.solver_evaluation_pending);
    prepared.Release();
  }
}

BOOST_AUTO_TEST_CASE(HardFailuresAreAtomicAndNeverReachASolver) {
  IntentFixture fixture;
  const point3D_t point3D_id =
      fixture.AddPoint(Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 1}});
  const point3D_t no_window_id =
      fixture.AddPoint(Eigen::Vector3d(4.0, 5.0, 6.0), {{4, 2}});
  fixture.StartJournal();
  const BundleAdjustmentOptions strict = StrictOptions();
  const OnlineLidarAssociationRequest request = fixture.Request({point3D_id});
  const uint64_t topology_revision =
      fixture.reconstruction.StructureRevision();

  CheckBuildFailure(fixture,
                    strict,
                    request,
                    kOwnerEpoch + 1,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  CheckBuildFailure(fixture,
                    strict,
                    request,
                    kOwnerEpoch,
                    topology_revision + 1,
                    101,
                    gpu_ba::BaKind::kLocal);
  CheckBuildFailure(fixture,
                    strict,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    0,
                    gpu_ba::BaKind::kLocal);
  CheckBuildFailure(fixture,
                    strict,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kGlobal);

  BundleAdjustmentOptions invalid = strict;
  invalid.ba_backend = "ceres_cpu";
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid = strict;
  invalid.ba_fallback_to_ceres = true;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid = strict;
  invalid.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kActiveSpec;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);

  invalid = strict;
  invalid.refine_extrinsics = false;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid = strict;
  invalid.refine_focal_length = true;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid = strict;
  invalid.refine_principal_point = true;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid = strict;
  invalid.refine_extra_params = true;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid = strict;
  invalid.if_add_lidar_constraint = false;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);

  invalid = strict;
  invalid.icp_ground_lidar_constraint_weight = 0.0;
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  invalid.icp_ground_lidar_constraint_weight =
      std::numeric_limits<double>::quiet_NaN();
  CheckBuildFailure(fixture,
                    invalid,
                    request,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);

  CheckBuildFailure(fixture,
                    strict,
                    fixture.Request({no_window_id}),
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  OnlineLidarAssociationRequest mismatched_hash = request;
  mismatched_hash.expected_snapshot_sha256[0] =
      mismatched_hash.expected_snapshot_sha256[0] == '1' ? '2' : '1';
  CheckBuildFailure(fixture,
                    strict,
                    mismatched_hash,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
  OnlineLidarAssociationRequest malformed_hash = request;
  malformed_hash.expected_geometry_sha256 = "not-a-sha256";
  CheckBuildFailure(fixture,
                    strict,
                    malformed_hash,
                    kOwnerEpoch,
                    topology_revision,
                    101,
                    gpu_ba::BaKind::kLocal);
}

BOOST_AUTO_TEST_CASE(ResolvedOptionsPrepareAndMaterializedSummaryAreExact) {
  IntentFixture fixture;
  const point3D_t point3D_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0),
      {{1, 1}, {3, 2}, {2, 3}, {4, 4}},
      1);
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();
  const OnlineLidarAssociationRequest request = fixture.Request({point3D_id});
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      BuildIntent(fixture, options, request, 111, &output, &error), error);

  TransactionLocalNativeStore store(&fixture.reconstruction, kOwnerEpoch);
  gpu_ba::PreparedNativeActiveSolve prepared =
      Prepare(&fixture, output, store.Binding());
  BOOST_REQUIRE(prepared.valid());
  const gpu_ba::NativeHostSolveView* view = prepared.view();
  BOOST_REQUIRE(view != nullptr);
  BOOST_CHECK_EQUAL(view->identity.catalog_generation,
                    view->catalog.generation());
  BOOST_CHECK_GT(view->identity.catalog_generation, 0);
  BOOST_CHECK(view->ActiveImageSlots().size() ==
              request.ordered_frozen_image_ids.size());
  BOOST_CHECK(view->BoundaryImageSlots().empty());
  BOOST_REQUIRE_EQUAL(view->Fixed().images.size(),
                      request.ordered_frozen_image_ids.size());
  for (const gpu_ba::ImageFixedPolicyResult& fixed : view->Fixed().images) {
    BOOST_CHECK_EQUAL(fixed.pose_constant, 0);
    BOOST_CHECK_EQUAL(fixed.boundary_pose, 0);
    BOOST_CHECK_EQUAL(fixed.translation_subset_mask, 0);
    BOOST_CHECK_EQUAL(fixed.tangent_size, 6);
  }
  BOOST_REQUIRE_EQUAL(view->Fixed().cameras.size(), 1);
  BOOST_CHECK_EQUAL(view->Fixed().cameras.front().constant, 1);

  std::set<image_t> active_image_ids(
      request.ordered_frozen_image_ids.begin(),
      request.ordered_frozen_image_ids.end());
  const auto observations = view->catalog.observations();
  const auto images = view->catalog.images();
  BOOST_REQUIRE_EQUAL(view->VisualObservationSlots().size(), 3);
  for (const uint32_t observation_slot : view->VisualObservationSlots()) {
    BOOST_REQUIRE_LT(observation_slot, observations.size);
    const gpu_ba::HostBaObservationSlot& observation =
        observations[observation_slot];
    BOOST_REQUIRE_LT(observation.image_slot, images.size);
    BOOST_CHECK_EQUAL(active_image_ids.count(
                          images[observation.image_slot].image_id),
                      1);
    BOOST_CHECK_NE(images[observation.image_slot].image_id, 4);
  }

  OnlineLidarMaterializationSummary summary;
  BOOST_REQUIRE_MESSAGE(SummarizeOnlineLidarMaterialization(
                            *view, output, &summary, &error),
                        error);
  BOOST_CHECK(summary.per_image_materialized_count ==
              output.association_audit().preliminary_selected_count_by_image);
  BOOST_CHECK_EQUAL(summary.trigger_materialized_count,
                    output.association_audit()
                        .trigger_preliminary_selected_count);
  BOOST_CHECK_EQUAL(summary.materialized_association_count,
                    output.association_audit().selected_association_count);
  BOOST_CHECK_EQUAL(summary.materialized_lidar_residual_count, 1);
  BOOST_CHECK_EQUAL(summary.materialized_visual_residual_count, 3);
  BOOST_CHECK(summary.materialized_association_ids ==
              std::vector<uint64_t>{0});
  CheckIdentityEqual(summary.materialized_online_lidar_identity,
                     output.intent().online_lidar_identity);
  BOOST_CHECK(summary.solver_evaluation_pending);
  prepared.Release();
}

BOOST_AUTO_TEST_CASE(DistinctFeaturesInOneImageRetainAllVisualResiduals) {
  IntentFixture fixture;
  const point3D_t point3D_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 7}, {1, 1}, {2, 2}, {3, 3}});
  fixture.StartJournal();
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                    StrictOptions(),
                                    fixture.Request({point3D_id}),
                                    122,
                                    &output,
                                    &error), error);
  BOOST_REQUIRE_EQUAL(output.association_output().associations.size(), 1);
  BOOST_CHECK_EQUAL(output.association_output().associations.front()
                        .owner_image_id, 2);
  BOOST_CHECK_EQUAL(output.association_output().associations.front()
                        .owner_point2D_idx, 2);
  TransactionLocalNativeStore store(&fixture.reconstruction, kOwnerEpoch);
  gpu_ba::PreparedNativeActiveSolve prepared =
      Prepare(&fixture, output, store.Binding());
  BOOST_REQUIRE(prepared.valid());
  BOOST_CHECK_EQUAL(prepared.view()->VisualObservationSlots().size(), 4);
  OnlineLidarMaterializationSummary summary;
  BOOST_REQUIRE_MESSAGE(SummarizeOnlineLidarMaterialization(
                            *prepared.view(), output, &summary, &error), error);
  BOOST_CHECK_EQUAL(summary.materialized_visual_residual_count, 4);
  BOOST_CHECK_EQUAL(summary.materialized_lidar_residual_count, 1);
  BOOST_CHECK_EQUAL(summary.trigger_materialized_count, 1);
  BOOST_CHECK_EQUAL(fixture.reconstruction.Point3D(point3D_id).Track().Length(),
                    4);
  prepared.Release();
}

BOOST_AUTO_TEST_CASE(SummaryRejectsTamperingAndCacheHitsRemainValid) {
  IntentFixture fixture;
  const point3D_t point3D_id = fixture.AddPoint(
      Eigen::Vector3d(1.0, 2.0, 3.0), {{1, 1}, {2, 2}, {4, 3}}, 1);
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                    options,
                                    fixture.Request({point3D_id}),
                                    121,
                                    &output,
                                    &error),
                        error);
  TransactionLocalNativeStore store(&fixture.reconstruction, kOwnerEpoch);

  gpu_ba::PreparedNativeActiveSolve first =
      Prepare(&fixture, output, store.Binding());
  BOOST_CHECK_EQUAL(first.runtime().host_plan_hits, 0);
  BOOST_CHECK_EQUAL(first.runtime().host_plan_misses, 1);
  first.Release();

  gpu_ba::PreparedNativeActiveSolve cached =
      Prepare(&fixture, output, store.Binding());
  BOOST_REQUIRE(cached.valid());
  BOOST_CHECK_EQUAL(cached.runtime().host_plan_hits, 1);
  BOOST_CHECK_EQUAL(cached.runtime().host_plan_build_calls, 0);
  OnlineLidarMaterializationSummary cached_summary;
  BOOST_REQUIRE_MESSAGE(SummarizeOnlineLidarMaterialization(
                            *cached.view(), output, &cached_summary, &error),
                        error);
  BOOST_CHECK_EQUAL(cached_summary.materialized_association_count, 1);
  BOOST_CHECK(cached_summary.solver_evaluation_pending);

  gpu_ba::NativeHostSolveView wrong_view = *cached.view();
  ++wrong_view.identity.selection_revision;
  CheckSummaryFailure(wrong_view, output);
  cached.Release();

  gpu_ba::NativeBaSolveIntent tampered_intent = output.intent();
  tampered_intent.lidar_constraints.front().weight += 1.0;
  gpu_ba::PreparedNativeActiveSolve tampered_prepared;
  BOOST_REQUIRE_MESSAGE(gpu_ba::PrepareCudaNativeBaSolve(
                            tampered_intent,
                            &fixture.reconstruction,
                            output.resolved_cuda_options(),
                            store.Binding(),
                            &tampered_prepared,
                            &error),
                        error);
  BOOST_REQUIRE(tampered_prepared.view() != nullptr);
  CheckSummaryFailure(*tampered_prepared.view(), output);
  BOOST_CHECK_NE(tampered_prepared.view()->LidarConstraints().front().weight,
                 output.intent().lidar_constraints.front().weight);
  tampered_prepared.Release();
}

BOOST_AUTO_TEST_CASE(DefaultUnbuiltOutputIsRejectedBySummary) {
  OnlineLidarNativeBaSolveIntentBuildOutput output;
  BOOST_CHECK(!output.built());
  CheckSummaryFailure(gpu_ba::NativeHostSolveView(), output);
}

BOOST_AUTO_TEST_CASE(MaterializedTriggerBoundaryCountsAre49And50) {
  std::vector<WorldPoint> world_points;
  world_points.reserve(50);
  for (size_t index = 0; index < 50; ++index) {
    world_points.push_back(WorldPoint{{
        static_cast<float>(1.0 + 0.05 * index), 2.0f, 3.0f}});
  }
  IntentFixture fixture(world_points);
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(world_points.size());
  for (size_t index = 0; index < world_points.size(); ++index) {
    const WorldPoint& xyz = world_points[index];
    point3D_ids.push_back(fixture.AddPoint(
        Eigen::Vector3d(xyz[0], xyz[1], xyz[2]),
        {{2, static_cast<point2D_t>(index)}}));
  }
  fixture.StartJournal();
  const BundleAdjustmentOptions options = StrictOptions();
  TransactionLocalNativeStore store(&fixture.reconstruction, kOwnerEpoch);

  const auto build_and_summarize =
      [&](const size_t point_count,
          const uint64_t selection_revision) {
        OnlineLidarAssociationRequest request = fixture.Request(
            std::vector<point3D_t>(point3D_ids.begin(),
                                   point3D_ids.begin() + point_count));
        OnlineLidarNativeBaSolveIntentBuildOutput output;
        std::string error;
        BOOST_REQUIRE_MESSAGE(BuildIntent(fixture,
                                          options,
                                          request,
                                          selection_revision,
                                          &output,
                                          &error),
                              error);
        BOOST_REQUIRE_EQUAL(output.association_output().associations.size(),
                            point_count);
        gpu_ba::PreparedNativeActiveSolve prepared =
            Prepare(&fixture, output, store.Binding());
        OnlineLidarMaterializationSummary summary;
        BOOST_REQUIRE_MESSAGE(SummarizeOnlineLidarMaterialization(
                                  *prepared.view(), output, &summary, &error),
                              error);
        BOOST_CHECK_EQUAL(summary.trigger_materialized_count, point_count);
        BOOST_CHECK_EQUAL(summary.materialized_association_count, point_count);
        BOOST_CHECK_EQUAL(summary.materialized_lidar_residual_count,
                          point_count);
        BOOST_CHECK(summary.solver_evaluation_pending);
        prepared.Release();
        return summary.trigger_materialized_count;
      };

  BOOST_CHECK_EQUAL(build_and_summarize(49, 149), 49);
  BOOST_CHECK_EQUAL(build_and_summarize(50, 150), 50);
}

}  // namespace
}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED
