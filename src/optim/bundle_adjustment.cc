// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "optim/bundle_adjustment.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/projection.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#ifdef GPU_BA_ENABLED
#include "gpu_ba/ceres_fidelity.h"
#include "gpu_ba/snapshot_recorder.h"
#endif
#ifdef GPU_BA_CUDA_ENABLED
#include "gpu_ba/custom_cuda.h"
#include "gpu_ba/native_graph_problem_store.h"
#endif

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentOptions::CudaExecutionProfile
ParseBundleAdjustmentCudaExecutionProfile(const std::string& value) {
  if (value == "baseline")
    return BundleAdjustmentOptions::CudaExecutionProfile::BASELINE;
  if (value == "runtime_pool")
    return BundleAdjustmentOptions::CudaExecutionProfile::RUNTIME_POOL;
  if (value == "arena")
    return BundleAdjustmentOptions::CudaExecutionProfile::ARENA;
  if (value == "device_scaling")
    return BundleAdjustmentOptions::CudaExecutionProfile::DEVICE_SCALING;
  if (value == "fast_identity")
    return BundleAdjustmentOptions::CudaExecutionProfile::FAST_IDENTITY;
  if (value == "unified_builder")
    return BundleAdjustmentOptions::CudaExecutionProfile::UNIFIED_BUILDER;
  if (value == "compact_layer_a")
    return BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_LAYER_A;
  if (value == "compact_control")
    return BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  return BundleAdjustmentOptions::CudaExecutionProfile::INVALID;
}

const char* BundleAdjustmentCudaExecutionProfileName(
    const BundleAdjustmentOptions::CudaExecutionProfile profile) {
  switch (profile) {
    case BundleAdjustmentOptions::CudaExecutionProfile::BASELINE:
      return "baseline";
    case BundleAdjustmentOptions::CudaExecutionProfile::RUNTIME_POOL:
      return "runtime_pool";
    case BundleAdjustmentOptions::CudaExecutionProfile::ARENA:
      return "arena";
    case BundleAdjustmentOptions::CudaExecutionProfile::DEVICE_SCALING:
      return "device_scaling";
    case BundleAdjustmentOptions::CudaExecutionProfile::FAST_IDENTITY:
      return "fast_identity";
    case BundleAdjustmentOptions::CudaExecutionProfile::UNIFIED_BUILDER:
      return "unified_builder";
    case BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_LAYER_A:
      return "compact_layer_a";
    case BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL:
      return "compact_control";
    case BundleAdjustmentOptions::CudaExecutionProfile::INVALID:
      return "invalid";
  }
  return "invalid";
}

ceres::LossFunction* BundleAdjustmentOptions::CreateLossFunction() const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::SOFT_L1:
      loss_function = new ceres::SoftLOneLoss(loss_function_scale);
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  CHECK_NOTNULL(loss_function);
  return loss_function;
}

bool BundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  CHECK_OPTION_GE(ba_ceres_oracle_repeat_count, 1);
  CHECK_OPTION_LE(ba_ceres_oracle_repeat_count, 8);
  CHECK(ba_backend == "ceres_cpu" || ba_backend == "custom_cpu" ||
        ba_backend == "custom_cuda" || ba_backend == "compare");
  CHECK(ba_cuda_execution_profile != CudaExecutionProfile::INVALID);
  CHECK(ba_cuda_audit_profile == "compatibility_default" ||
        ba_cuda_audit_profile == "production" ||
        ba_cuda_audit_profile == "correctness" ||
        ba_cuda_audit_profile == "forensic");
  CHECK(ba_cuda_arithmetic_precision == "compatibility_default" ||
        ba_cuda_arithmetic_precision == "fp64" ||
        ba_cuda_arithmetic_precision == "fp32_core" ||
        ba_cuda_arithmetic_precision == "fp32_state_quantized" ||
        ba_cuda_arithmetic_precision == "fp32_mixed");
  CHECK(ba_cuda_hessian_assembly_backend == "compatibility_default" ||
        ba_cuda_hessian_assembly_backend == "pose_owned" ||
        ba_cuda_hessian_assembly_backend == "observation_segmented");
  CHECK(ba_cuda_hot_kernel_mode == "compatibility_default" ||
        ba_cuda_hot_kernel_mode == "reference" ||
        ba_cuda_hot_kernel_mode == "optimized" ||
        ba_cuda_hot_kernel_mode == "transformed");
  CHECK(ba_cuda_schur_contribution_backend == "compatibility_default" ||
        ba_cuda_schur_contribution_backend == "direct" ||
        ba_cuda_schur_contribution_backend == "segmented");
  CHECK(ba_snapshot_capture == "none" || ba_snapshot_capture == "local" ||
        ba_snapshot_capture == "global" || ba_snapshot_capture == "whole" ||
        ba_snapshot_capture == "all");
  CHECK(ba_cuda_schur_mode == "deterministic" ||
        ba_cuda_schur_mode == "atomic" ||
        ba_cuda_schur_mode == "block_reduce");
  CHECK(ba_cuda_host_problem_store == "disabled" ||
        ba_cuda_host_problem_store == "host_prepared_store");
  CHECK(ba_cuda_problem_source == gpu_ba::CudaProblemSource::kLegacySnapshot ||
        ba_cuda_problem_source == gpu_ba::CudaProblemSource::kActiveSpec ||
        ba_cuda_problem_source == gpu_ba::CudaProblemSource::kIndexedCatalog ||
        ba_cuda_problem_source == gpu_ba::CudaProblemSource::kNativeGraph);
  CHECK(ba_lidar_residual == "legacy_exact" ||
        ba_lidar_residual == "legacy_guarded" ||
        ba_lidar_residual == "signed");
  return true;
}

ceres::Solver::Options CreateEffectiveBundleAdjustmentSolverOptions(
    const BundleAdjustmentOptions& options,
    size_t config_num_images,
    int problem_num_residuals) {
  ceres::Solver::Options solver_options = options.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  if (config_num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (config_num_images <= kMaxNumImagesDirectSparseSolver &&
             has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  if (problem_num_residuals <
      options.min_num_residuals_for_multi_threading) {
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif
  } else {
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif
  }
  return solver_options;
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {
  lidar_searched_image_ids_.clear();
  lidar_maps_.clear();
}

// image_ids_是一个包含image_id的set
size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;

  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.Track().Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        num_observations_for_point += 1;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

void BundleAdjustmentConfig::AddLidarPoint(const point3D_t& point3D_id, const class LidarPoint& lidar_point) {
  lidar_maps_.insert({point3D_id,lidar_point});
}

void BundleAdjustmentConfig::AddPointcloud(std::shared_ptr<lidar::PointCloudProcess> ptr) {
  point_cloud_process_ = ptr;
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamera(const camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamera(const camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool BundleAdjustmentConfig::IsConstantCamera(const camera_t camera_id) const {
  return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void BundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariablePose(const image_t image_id) {
  constant_poses_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantPose(const image_t image_id) const {
  return constant_poses_.find(image_id) != constant_poses_.end();
}

void BundleAdjustmentConfig::SetConstantTvec(const image_t image_id,
                                             const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantTvec(const image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantTvec(const image_t image_id) const {
  return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
  return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

const std::unordered_map<point3D_t, double>&
BundleAdjustmentConfig::LidarSearchRanges() const {
  return lidar_search_ranges_;
}

void BundleAdjustmentConfig::SetLidarSearchRange(
    const point3D_t point3D_id, double search_range) {
  lidar_search_ranges_[point3D_id] = search_range;
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::Project2Image(Reconstruction* reconstruction,const point3D_t& point3D_id, const image_t& image_id, const int& match_features_threshold) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  
  const auto& track_els = point3D.Track().Elements();

  for (auto& track_el : track_els){
    image_t image_id2 = track_el.image_id;
    if (image_id2 != image_id) {
      if (reconstruction->ExistsImagePair(image_id,image_id2)){
        size_t corrs = reconstruction->ImagePair(image_id, image_id2).num_total_corrs;
        if (corrs <= match_features_threshold) {
          continue;
        }
      }
    }
    
    auto ptr = lidar_searched_image_ids_.find(track_el.image_id);
    if (ptr == lidar_searched_image_ids_.end()){
      // The image has not been projected into pcd_projection
      // Read all feature points in the image and find corresponding lidar points
      Image& image = reconstruction->Image(track_el.image_id);
      Camera& camera = reconstruction->Camera(image.CameraId());

      std::map<point3D_t,Eigen::Matrix<double,6,1>> map;
      point_cloud_process_ -> pcd_proj_ -> SetNewImage(image,camera,map);
      lidar_searched_image_ids_.insert({track_el.image_id,map});
    }
  }
}

void BundleAdjustmentConfig::MatchVariablePoint2LidarPoint(Reconstruction* reconstruction,const point3D_t point3D_id){
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  Eigen::Vector3d pt_xyz = point3D.XYZ();
  double angle = 360;
  Eigen::Vector6d lidar_pt;

  const auto& track_els = point3D.Track().Elements(); 
  for (auto& track_el : track_els){
    auto iter = lidar_searched_image_ids_.find(track_el.image_id);
    if (iter == lidar_searched_image_ids_.end()) continue;
    auto lidar_pt_ptr = iter->second.find(point3D_id);
    if (lidar_pt_ptr != iter->second.end()){
      Eigen::Matrix<double, 6, 1> lidar_pt_temp = lidar_pt_ptr->second;
      Eigen::Vector3d norm = lidar_pt_temp.block(3,0,3,1);
      Eigen::Vector3d vec = pt_xyz - lidar_pt_temp.block(0,0,3,1);
      double angle_temp = std::abs(vec.dot(norm)/(norm.norm()*vec.norm()));
      if (angle_temp < angle){
        angle = angle_temp;
        lidar_pt = lidar_pt_temp;
      }
    }
  }
  if (angle != 360){
    Eigen::Vector3d norm = lidar_pt.block(3,0,3,1);
    Eigen::Vector3d l_pt = lidar_pt.block(0,0,3,1);
    double d = 0 - l_pt.dot(norm);
    Eigen::Vector4d plane;
    plane << norm(0),norm(1),norm(2),d;

    LidarPoint lidar_point(LidarPointType::Proj,l_pt,plane);
    lidar_point.SetDist( lidar_point.ComputeDist(pt_xyz));
    lidar_point.SetAngle( lidar_point.ComputeAngle(pt_xyz));
    Eigen::Vector3ub color;
    color << 255,0,0;
    lidar_point.SetColor(color);
    AddLidarPoint(point3D_id,lidar_point);
    reconstruction -> AddLidarPoint(point3D_id,lidar_point);
  }
}

void BundleAdjustmentConfig::MatchClosestLidarPoint(Reconstruction* reconstruction, 
                                                    const point3D_t& point3D_id, 
                                                    double& max_search_range){
  SetLidarSearchRange(point3D_id, max_search_range);
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  Eigen::Vector3d pt_xyz = point3D.XYZ();
  Eigen::Vector6d lidar_pt;
  if (point_cloud_process_->SearchNearestNeiborByKdtree(pt_xyz,lidar_pt)){
    Eigen::Vector3d norm = lidar_pt.block(3,0,3,1);
    Eigen::Vector3d l_pt = lidar_pt.block(0,0,3,1);
    double d = 0 - l_pt.dot(norm);
    Eigen::Vector4d plane;
    plane << norm(0),norm(1),norm(2),d;

    LidarPoint lidar_point(l_pt,plane);
    if (std::abs(norm(1)/norm(0))>10 && std::abs(norm(1)/norm(2))>10) {
      lidar_point.SetType(LidarPointType::IcpGround);
      Eigen::Vector3ub color;
      color << 255,255,0;
      lidar_point.SetColor(color);
    } else {
      lidar_point.SetType(LidarPointType::Icp);
      Eigen::Vector3ub color;
      color << 0,255,0;
      lidar_point.SetColor(color);
    }
    double dist = lidar_point.ComputePointToPointDist(pt_xyz);
    if (dist > max_search_range) return;
    lidar_point.SetDist(dist);
    lidar_point.SetAngle( lidar_point.ComputeAngle(pt_xyz));
    AddLidarPoint(point3D_id,lidar_point);
    reconstruction -> AddLidarPoint(point3D_id,lidar_point);
  }
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

namespace {

std::atomic<uint64_t> g_bundle_adjustment_ceres_solve_calls{0};
std::atomic<int> g_bundle_adjustment_failure_mode_for_testing{0};

enum class StableBaError {
  kNone,
  kUnsupportedBackend,
  kUnsupportedConfiguration,
  kSnapshotCaptureConfiguration,
  kSnapshotDirectoryRequired,
  kEmptyProblem,
  kTransactionCheckpointFailed,
  kTransactionRestoreFailed,
  kSnapshotFinalizeFailed,
  kCustomCudaFailed,
  kCommitIntegrityError,
  kCeresFallbackFailed,
  kCeresSolveFailed,
};

const char* StableBaErrorName(const StableBaError error) {
  switch (error) {
    case StableBaError::kNone: return "";
    case StableBaError::kUnsupportedBackend: return "UNSUPPORTED_BA_BACKEND";
    case StableBaError::kUnsupportedConfiguration:
      return "UNSUPPORTED_CONFIGURATION";
    case StableBaError::kSnapshotCaptureConfiguration:
      return "SNAPSHOT_CAPTURE_CONFIGURATION";
    case StableBaError::kSnapshotDirectoryRequired:
      return "SNAPSHOT_DIRECTORY_REQUIRED";
    case StableBaError::kEmptyProblem: return "EMPTY_BA_PROBLEM";
    case StableBaError::kTransactionCheckpointFailed:
      return "TRANSACTION_CHECKPOINT_FAILED";
    case StableBaError::kTransactionRestoreFailed:
      return "TRANSACTION_RESTORE_FAILED";
    case StableBaError::kSnapshotFinalizeFailed:
      return "SNAPSHOT_FINALIZE_FAILED";
    case StableBaError::kCustomCudaFailed: return "CUSTOM_CUDA_FAILED";
    case StableBaError::kCommitIntegrityError:
      return "COMMIT_INTEGRITY_ERROR";
    case StableBaError::kCeresFallbackFailed: return "CERES_FALLBACK_FAILED";
    case StableBaError::kCeresSolveFailed: return "CERES_SOLVE_FAILED";
  }
  return "INTERNAL_STABLE_ERROR_ENUM";
}

void SetStableError(BundleAdjustmentExecutionResult* result,
                    const StableBaError error) {
  result->stable_error = StableBaErrorName(error);
}

enum class CheckpointEntityKind : uint8_t {
  kQuaternion,
  kTranslation,
  kPoint3D,
  kCamera,
};

struct CheckpointIdentity {
  CheckpointEntityKind kind;
  uint64_t entity_id;

  bool operator==(const CheckpointIdentity& other) const {
    return kind == other.kind && entity_id == other.entity_id;
  }
};

struct CheckpointIdentityHash {
  size_t operator()(const CheckpointIdentity& value) const {
    return std::hash<uint64_t>()(value.entity_id) ^
           (static_cast<size_t>(value.kind) * 0x9e3779b9u);
  }
};

#ifdef GPU_BA_CUDA_ENABLED
struct SnapshotParameterIdentity {
  gpu_ba::ParameterKind kind;
  uint64_t entity_id;

  bool operator==(const SnapshotParameterIdentity& other) const {
    return kind == other.kind && entity_id == other.entity_id;
  }
};

struct SnapshotParameterIdentityHash {
  size_t operator()(const SnapshotParameterIdentity& value) const {
    return std::hash<uint64_t>()(value.entity_id) ^
           (static_cast<size_t>(value.kind) * 0x9e3779b9u);
  }
};
#endif

struct SelectedQvecState {
  image_t image_id = kInvalidImageId;
  Eigen::Vector4d qvec;
};

struct ParameterBlockCheckpoint {
  CheckpointEntityKind kind = CheckpointEntityKind::kPoint3D;
  uint64_t entity_id = 0;
  double* values = nullptr;
  int ambient_size = 0;
  std::vector<double> entry_values;
  std::vector<double> ceres_pre_solve_values;
};

struct BaParameterCheckpoint {
  struct Qvec {
    image_t image_id = kInvalidImageId;
    double* values = nullptr;
    Eigen::Vector4d entry_value;
    Eigen::Vector4d ceres_pre_solve_value;
  };
  std::vector<Qvec> selected_qvecs;
  std::vector<ParameterBlockCheckpoint> blocks;
  uint64_t entity_count = 0;
  uint64_t backup_bytes = 0;
  uint64_t identity_index_entries = 0;
  uint64_t parameter_lookup_count = 0;
};

std::vector<SelectedQvecState> CaptureSelectedQvecs(
    const BundleAdjustmentConfig& config,
    const Reconstruction& reconstruction) {
  std::vector<SelectedQvecState> state;
  state.reserve(config.NumImages());
  for (const image_t image_id : config.Images()) {
    if (reconstruction.ExistsImage(image_id)) {
      state.push_back({image_id, reconstruction.Image(image_id).Qvec()});
    }
  }
  return state;
}

bool RestoreSelectedQvecs(const std::vector<SelectedQvecState>& state,
                          Reconstruction* reconstruction,
                          std::string* error) {
  for (const auto& image : state) {
    if (!reconstruction->ExistsImage(image.image_id)) {
      *error = "selected image is missing during qvec restore";
      return false;
    }
  }
  for (const auto& image : state) {
    reconstruction->Image(image.image_id).SetQvec(image.qvec);
  }
  return true;
}

bool CheckpointEntityExists(const ParameterBlockCheckpoint& block,
                            const Reconstruction& reconstruction) {
  switch (block.kind) {
    case CheckpointEntityKind::kQuaternion:
    case CheckpointEntityKind::kTranslation:
      return reconstruction.ExistsImage(static_cast<image_t>(block.entity_id));
    case CheckpointEntityKind::kPoint3D:
      return reconstruction.ExistsPoint3D(
          static_cast<point3D_t>(block.entity_id));
    case CheckpointEntityKind::kCamera:
      return reconstruction.ExistsCamera(static_cast<camera_t>(block.entity_id));
  }
  return false;
}

double* CurrentEntityDataPointer(const ParameterBlockCheckpoint& block,
                                 Reconstruction* reconstruction) {
  switch (block.kind) {
    case CheckpointEntityKind::kQuaternion:
      return reconstruction->Image(static_cast<image_t>(block.entity_id))
          .Qvec().data();
    case CheckpointEntityKind::kTranslation:
      return reconstruction->Image(static_cast<image_t>(block.entity_id))
          .Tvec().data();
    case CheckpointEntityKind::kPoint3D:
      return reconstruction->Point3D(static_cast<point3D_t>(block.entity_id))
          .XYZ().data();
    case CheckpointEntityKind::kCamera:
      return reconstruction->Camera(static_cast<camera_t>(block.entity_id))
          .Params().data();
  }
  return nullptr;
}

bool RestoreParameterCheckpoint(const BaParameterCheckpoint& checkpoint,
                                bool restore_entry,
                                const ceres::Problem* problem,
                                Reconstruction* reconstruction,
                                std::string* error) {
  if (g_bundle_adjustment_failure_mode_for_testing.load() ==
      static_cast<int>(
          BundleAdjuster::FailureModeForTesting::kRestoreValidationFailure)) {
    *error = "forced restore validation failure";
    return false;
  }
  // Validate all identities before changing the first parameter.
  for (const auto& qvec : checkpoint.selected_qvecs) {
    if (!reconstruction->ExistsImage(qvec.image_id) || qvec.values == nullptr ||
        reconstruction->Image(qvec.image_id).Qvec().data() != qvec.values) {
      *error = "selected qvec identity changed during restore";
      return false;
    }
  }
  for (const auto& block : checkpoint.blocks) {
    const auto& source = restore_entry ? block.entry_values
                                       : block.ceres_pre_solve_values;
    if (!CheckpointEntityExists(block, *reconstruction) ||
        CurrentEntityDataPointer(block, reconstruction) != block.values ||
        block.values == nullptr ||
        (problem != nullptr && !problem->HasParameterBlock(block.values)) ||
        (problem != nullptr &&
         problem->ParameterBlockSize(block.values) != block.ambient_size) ||
        source.size() != static_cast<size_t>(block.ambient_size)) {
      *error = "parameter checkpoint identity changed during restore";
      return false;
    }
  }
  for (const auto& block : checkpoint.blocks) {
    const auto& source = restore_entry ? block.entry_values
                                       : block.ceres_pre_solve_values;
    std::copy(source.begin(), source.end(), block.values);
  }
  for (const auto& qvec : checkpoint.selected_qvecs) {
    reconstruction->Image(qvec.image_id).SetQvec(
        restore_entry ? qvec.entry_value : qvec.ceres_pre_solve_value);
  }
  return true;
}

#ifdef GPU_BA_ENABLED
bool BuildActiveSpecParameterCheckpoint(
    const std::vector<SelectedQvecState>& selected_qvecs,
    const gpu_ba::ActiveBaSolveSpec& spec,
    Reconstruction* reconstruction,
    BaParameterCheckpoint* checkpoint,
    std::string* error) {
  if (reconstruction == nullptr || checkpoint == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  std::unordered_map<image_t, Eigen::Vector4d> entry_qvecs;
  entry_qvecs.reserve(selected_qvecs.size());
  checkpoint->selected_qvecs.clear();
  checkpoint->selected_qvecs.reserve(selected_qvecs.size());
  for (const SelectedQvecState& selected : selected_qvecs) {
    if (!reconstruction->ExistsImage(selected.image_id)) {
      *error = "selected image disappeared while building active checkpoint";
      return false;
    }
    Image& image = reconstruction->Image(selected.image_id);
    entry_qvecs.emplace(selected.image_id, selected.qvec);
    checkpoint->selected_qvecs.push_back(
        {selected.image_id, image.Qvec().data(), selected.qvec, image.Qvec()});
  }

  checkpoint->blocks.clear();
  checkpoint->blocks.reserve(spec.parameter_blocks.size());
  checkpoint->backup_bytes = 0;
  checkpoint->identity_index_entries = spec.parameter_blocks.size();
  checkpoint->parameter_lookup_count = 0;
  std::unordered_set<CheckpointIdentity, CheckpointIdentityHash> unique;
  for (const gpu_ba::ActiveBaParameterBlockSpec& parameter :
       spec.parameter_blocks) {
    ParameterBlockCheckpoint block;
    block.entity_id = parameter.entity_id;
    block.ambient_size = static_cast<int>(parameter.ambient_size);
    switch (parameter.kind) {
      case gpu_ba::ParameterKind::kQuaternion:
        block.kind = CheckpointEntityKind::kQuaternion;
        break;
      case gpu_ba::ParameterKind::kTranslation:
        block.kind = CheckpointEntityKind::kTranslation;
        break;
      case gpu_ba::ParameterKind::kPoint3D:
        block.kind = CheckpointEntityKind::kPoint3D;
        break;
      case gpu_ba::ParameterKind::kCamera:
        block.kind = CheckpointEntityKind::kCamera;
        break;
    }
    ++checkpoint->parameter_lookup_count;
    if (!CheckpointEntityExists(block, *reconstruction)) {
      *error = "active checkpoint references a missing entity";
      return false;
    }
    block.values = CurrentEntityDataPointer(block, reconstruction);
    if (block.values == nullptr || block.ambient_size <= 0) {
      *error = "active checkpoint has an invalid parameter block";
      return false;
    }
    block.ceres_pre_solve_values.assign(
        block.values, block.values + block.ambient_size);
    block.entry_values = block.ceres_pre_solve_values;
    if (block.kind == CheckpointEntityKind::kQuaternion) {
      const auto entry = entry_qvecs.find(
          static_cast<image_t>(block.entity_id));
      if (entry != entry_qvecs.end()) {
        std::copy(entry->second.data(), entry->second.data() + 4,
                  block.entry_values.begin());
      }
    }
    checkpoint->backup_bytes +=
        2 * block.entry_values.size() * sizeof(double);
    unique.emplace(CheckpointIdentity{block.kind, block.entity_id});
    checkpoint->blocks.push_back(std::move(block));
  }
  checkpoint->entity_count = unique.size();
  return true;
}
#endif

bool BuildParameterCheckpoint(
    const std::vector<SelectedQvecState>& selected_qvecs,
    const BundleAdjustmentConfig& config,
    const std::unordered_set<camera_t>& camera_ids,
    const std::unordered_map<point3D_t, size_t>& point_observations,
    const ceres::Problem& problem,
    Reconstruction* reconstruction,
    BaParameterCheckpoint* checkpoint,
    std::string* error) {
  struct Identity {
    CheckpointEntityKind kind;
    uint64_t entity_id;
  };
  std::unordered_map<double*, Identity> identities;
  identities.reserve(config.NumImages() * 2 + camera_ids.size() +
                     point_observations.size() + config.NumPoints());
  for (const image_t image_id : config.Images()) {
    if (!reconstruction->ExistsImage(image_id)) continue;
    Image& image = reconstruction->Image(image_id);
    identities.emplace(image.Qvec().data(),
                       Identity{CheckpointEntityKind::kQuaternion, image_id});
    identities.emplace(image.Tvec().data(),
                       Identity{CheckpointEntityKind::kTranslation, image_id});
  }
  for (const camera_t camera_id : camera_ids) {
    if (!reconstruction->ExistsCamera(camera_id)) continue;
    identities.emplace(reconstruction->Camera(camera_id).Params().data(),
                       Identity{CheckpointEntityKind::kCamera, camera_id});
  }
  const auto add_point = [&](point3D_t point3D_id) {
    if (!reconstruction->ExistsPoint3D(point3D_id)) return;
    identities.emplace(reconstruction->Point3D(point3D_id).XYZ().data(),
                       Identity{CheckpointEntityKind::kPoint3D, point3D_id});
  };
  for (const auto& item : point_observations) add_point(item.first);
  for (const point3D_t id : config.VariablePoints()) add_point(id);
  for (const point3D_t id : config.ConstantPoints()) add_point(id);

  std::unordered_map<image_t, Eigen::Vector4d> original_qvecs;
  original_qvecs.reserve(selected_qvecs.size());
  for (const auto& value : selected_qvecs) {
    original_qvecs.emplace(value.image_id, value.qvec);
  }

  checkpoint->selected_qvecs.clear();
  checkpoint->selected_qvecs.reserve(selected_qvecs.size());
  for (const auto& value : selected_qvecs) {
    if (!reconstruction->ExistsImage(value.image_id)) {
      *error = "selected image disappeared while building checkpoint";
      return false;
    }
    Image& image = reconstruction->Image(value.image_id);
    checkpoint->selected_qvecs.push_back(
        {value.image_id, image.Qvec().data(), value.qvec, image.Qvec()});
  }

  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  checkpoint->blocks.clear();
  checkpoint->blocks.reserve(parameter_blocks.size());
  checkpoint->backup_bytes = 0;
  checkpoint->identity_index_entries = identities.size();
  checkpoint->parameter_lookup_count = 0;
  std::unordered_set<CheckpointIdentity, CheckpointIdentityHash>
      unique_entities;
  for (double* values : parameter_blocks) {
    ++checkpoint->parameter_lookup_count;
    const auto identity = identities.find(values);
    if (identity == identities.end()) {
      *error = "Ceres parameter block has no scoped reconstruction identity";
      return false;
    }
    ParameterBlockCheckpoint block;
    block.kind = identity->second.kind;
    block.entity_id = identity->second.entity_id;
    block.values = values;
    block.ambient_size = problem.ParameterBlockSize(values);
    block.ceres_pre_solve_values.assign(values, values + block.ambient_size);
    block.entry_values = block.ceres_pre_solve_values;
    if (block.kind == CheckpointEntityKind::kQuaternion) {
      const auto original = original_qvecs.find(
          static_cast<image_t>(block.entity_id));
      if (original != original_qvecs.end()) {
        std::copy(original->second.data(), original->second.data() + 4,
                  block.entry_values.begin());
      }
    }
    checkpoint->backup_bytes +=
        2 * block.entry_values.size() * sizeof(double);
    unique_entities.emplace(CheckpointIdentity{block.kind, block.entity_id});
    checkpoint->blocks.push_back(std::move(block));
  }
  checkpoint->entity_count = unique_entities.size();
  return true;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream stream;
  for (const char ch : value) {
    switch (ch) {
      case '\\': stream << "\\\\"; break;
      case '"': stream << "\\\""; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(static_cast<unsigned char>(ch))
                 << std::dec << std::setfill(' ');
        } else {
          stream << ch;
        }
        break;
    }
  }
  return stream.str();
}

void WriteJsonNumber(std::ostream& stream, const double value) {
  if (std::isfinite(value)) {
    stream << value;
  } else {
    stream << "null";
  }
}

void WriteExecutionTelemetry(const std::string& path,
                             const BundleAdjustmentExecutionResult& value) {
  if (path.empty()) return;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::ofstream file(path, std::ios::app);
  if (!file) {
    LOG(ERROR) << "Cannot append BA telemetry: " << path;
    return;
  }
  file << std::setprecision(17)
       << "{\"call_index\":" << value.call_index
       << ",\"ba_kind\":\"" << JsonEscape(value.ba_kind) << "\""
       << ",\"registered_images\":" << value.registered_images
       << ",\"requested_backend\":\"" << JsonEscape(value.requested_backend)
       << "\",\"executed_backend\":\"" << JsonEscape(value.executed_backend)
       << "\",\"problem_source_requested\":\""
       << JsonEscape(value.problem_source_requested)
       << "\",\"problem_source_effective\":\""
       << JsonEscape(value.problem_source_effective)
       << "\",\"problem_source_fallback_reason\":\""
       << JsonEscape(value.problem_source_fallback_reason)
       << "\",\"selected_schur\":\"" << JsonEscape(value.selected_schur)
       << "\",\"execution_profile\":\""
       << JsonEscape(value.execution_profile)
       << "\",\"audit_profile_requested\":\""
       << JsonEscape(value.audit_profile_requested)
       << "\",\"audit_profile_effective\":\""
       << JsonEscape(value.audit_profile_effective)
       << "\",\"capture_state_trace_effective\":"
       << (value.capture_state_trace_effective ? "true" : "false")
       << ",\"instrumentation_effective\":"
       << (value.instrumentation_effective ? "true" : "false")
       << ",\"production_audit_invariants_checked\":"
       << (value.production_audit_invariants_checked ? "true" : "false")
       << ",\"production_audit_invariants_pass\":"
       << (value.production_audit_invariants_pass ? "true" : "false")
       << ",\"production_audit_violation_count\":"
       << value.production_audit_violation_count
       << ",\"first_production_audit_violation\":\""
       << JsonEscape(value.first_production_audit_violation)
       << "\",\"state_hash_computations\":"
       << value.state_hash_computations
       << ",\"topology_fingerprint_computations\":"
       << value.topology_fingerprint_computations
       << ",\"audit_mirror_bytes\":" << value.audit_mirror_bytes
       << ",\"optional_full_array_d2h_bytes\":"
       << value.optional_full_array_d2h_bytes
       << ",\"mixed_double_edge_materialization_bytes\":"
       << value.mixed_double_edge_materialization_bytes
       << ",\"arithmetic_precision_requested\":\""
       << JsonEscape(value.arithmetic_precision_requested)
       << "\",\"arithmetic_precision_effective\":\""
       << JsonEscape(value.arithmetic_precision_effective)
       << "\",\"hessian_backend_requested\":\""
       << JsonEscape(value.hessian_backend_requested)
       << "\",\"hessian_backend_effective\":\""
       << JsonEscape(value.hessian_backend_effective)
       << "\",\"hot_kernel_requested\":\""
       << JsonEscape(value.hot_kernel_requested)
       << "\",\"hot_kernel_effective\":\""
       << JsonEscape(value.hot_kernel_effective)
       << "\",\"schur_contribution_backend_requested\":\""
       << JsonEscape(value.schur_contribution_backend_requested)
       << "\",\"schur_contribution_backend_effective\":\""
       << JsonEscape(value.schur_contribution_backend_effective)
       << "\",\"state_storage_precision\":\""
       << JsonEscape(value.state_storage_precision)
       << "\",\"residual_jacobian_precision\":\""
       << JsonEscape(value.residual_jacobian_precision)
       << "\",\"hessian_schur_precision\":\""
       << JsonEscape(value.hessian_schur_precision)
       << "\",\"factorization_routine\":\""
       << JsonEscape(value.factorization_routine)
       << "\",\"delta_precision\":\""
       << JsonEscape(value.delta_precision)
       << "\",\"quaternion_plus_precision\":\""
       << JsonEscape(value.quaternion_plus_precision)
       << "\",\"cost_precision\":\""
       << JsonEscape(value.cost_precision)
       << "\",\"controller_precision\":\""
       << JsonEscape(value.controller_precision)
       << "\",\"summary_source\":\"" << JsonEscape(value.summary_source)
       << "\",\"success\":" << (value.success ? "true" : "false")
       << ",\"termination\":\"" << JsonEscape(value.termination)
       << "\",\"stable_error\":\"" << JsonEscape(value.stable_error)
       << "\",\"diagnostic_message\":\""
       << JsonEscape(value.diagnostic_message)
       << "\",\"fallback_used\":"
       << (value.fallback_used ? "true" : "false")
       << ",\"fallback_reason\":\"" << JsonEscape(value.fallback_reason)
       << "\",\"wall_seconds\":";
  WriteJsonNumber(file, value.wall_seconds);
  file << ",\"trial_steps\":" << value.trial_steps
       << ",\"accepted_steps\":" << value.accepted_steps
       << ",\"accepted_commits\":" << value.accepted_commits
       << ",\"rejected_steps\":" << value.rejected_steps
       << ",\"invalid_steps\":" << value.invalid_steps
       << ",\"initial_cost\":";
  WriteJsonNumber(file, value.initial_cost);
  file << ",\"final_cost\":";
  WriteJsonNumber(file, value.final_cost);
  file << ",\"max_backward_error\":";
  WriteJsonNumber(file, value.max_backward_error);
  file
       << ",\"residuals\":" << value.residuals
       << ",\"residual_blocks\":" << value.residual_blocks
       << ",\"parameter_blocks\":" << value.parameter_blocks
       << ",\"parameters\":" << value.parameters
       << ",\"effective_parameters\":" << value.effective_parameters
       << ",\"ceres_problem_created\":"
       << (value.ceres_problem_created ? "true" : "false")
       << ",\"ceres_cost_function_creations\":"
       << value.ceres_cost_function_creations
       << ",\"ceres_add_residual_calls\":"
       << value.ceres_add_residual_calls
       << ",\"snapshot_recorder_created\":"
       << (value.snapshot_recorder_created ? "true" : "false")
       << ",\"snapshot_materialization_calls\":"
       << value.snapshot_materialization_calls
       << ",\"active_spec_build_calls\":"
       << value.active_spec_build_calls
       << ",\"residual_enumerator_passes\":"
       << value.residual_enumerator_passes
       << ",\"residual_enumerator_items\":"
       << value.residual_enumerator_items
       << ",\"peak_cuda_bytes\":" << value.peak_cuda_bytes
       << ",\"ceres_solve_calls\":" << value.ceres_solve_calls
       << ",\"ceres_solve_calls_scope\":\""
       << JsonEscape(value.ceres_solve_calls_scope) << "\""
       << ",\"transaction_qvec_count\":" << value.transaction_qvec_count
       << ",\"transaction_parameter_block_count\":"
       << value.transaction_parameter_block_count
       << ",\"transaction_backup_entity_count\":"
       << value.transaction_backup_entity_count
       << ",\"transaction_backup_bytes\":" << value.transaction_backup_bytes
       << ",\"transaction_identity_index_entries\":"
       << value.transaction_identity_index_entries
       << ",\"transaction_parameter_lookup_count\":"
       << value.transaction_parameter_lookup_count
       << ",\"transaction_checkpoint_milliseconds\":";
  WriteJsonNumber(file, value.transaction_checkpoint_milliseconds);
  file << ",\"transaction_restore_milliseconds\":";
  WriteJsonNumber(file, value.transaction_restore_milliseconds);
  file << ",\"host_store_mode_requested\":\""
       << JsonEscape(value.host_store_mode_requested)
       << "\",\"host_store_mode_effective\":\""
       << JsonEscape(value.host_store_mode_effective)
       << "\",\"host_store_view_action\":\""
       << JsonEscape(value.host_store_view_action)
       << "\",\"host_store_rebuild_reason\":\""
       << JsonEscape(value.host_store_rebuild_reason)
       << "\",\"host_store_owner_epoch\":" << value.host_store_owner_epoch
       << ",\"host_store_catalog_generation\":"
       << value.host_store_catalog_generation
       << ",\"host_store_view_generation\":"
       << value.host_store_view_generation
       << ",\"host_store_lookup_calls\":" << value.host_store_lookup_calls
       << ",\"host_store_hits\":" << value.host_store_hits
       << ",\"host_store_misses\":" << value.host_store_misses
       << ",\"host_store_journal_cursor_before\":"
       << value.host_store_journal_cursor_before
       << ",\"host_store_journal_cursor_after\":"
       << value.host_store_journal_cursor_after
       << ",\"host_store_catalog_cold_builds\":"
       << value.host_store_catalog_cold_builds
       << ",\"host_store_catalog_delta_updates\":"
       << value.host_store_catalog_delta_updates
       << ",\"host_store_catalog_full_rebuilds\":"
       << value.host_store_catalog_full_rebuilds
       << ",\"host_store_journal_events\":"
       << value.host_store_journal_events
       << ",\"host_store_journal_gaps\":"
       << value.host_store_journal_gaps
       << ",\"host_store_journal_overflows\":"
       << value.host_store_journal_overflows
       << ",\"host_store_journal_unknown_events\":"
       << value.host_store_journal_unknown_events
       << ",\"host_store_exact_view_reuses\":"
       << value.host_store_exact_view_reuses
       << ",\"host_store_view_patches\":"
       << value.host_store_view_patches
       << ",\"host_store_view_rebuilds\":"
       << value.host_store_view_rebuilds
       << ",\"host_store_fallback_rebuilds\":"
       << value.host_store_fallback_rebuilds
       << ",\"host_store_builder_calls_executed\":"
       << value.host_store_builder_calls_executed
       << ",\"host_store_builder_calls_saved\":"
       << value.host_store_builder_calls_saved
       << ",\"host_store_builder_traversals_executed\":"
       << value.host_store_builder_traversals_executed
       << ",\"host_store_builder_traversals_saved\":"
       << value.host_store_builder_traversals_saved
       << ",\"host_store_estimated_builder_bytes_executed\":"
       << value.host_store_estimated_builder_bytes_executed
       << ",\"host_store_estimated_builder_bytes_saved\":"
       << value.host_store_estimated_builder_bytes_saved
       << ",\"host_store_static_binding_builder_calls\":"
       << value.host_store_static_binding_builder_calls
       << ",\"host_store_cost_layout_builder_calls\":"
       << value.host_store_cost_layout_builder_calls
       << ",\"host_store_hessian_topology_builder_calls\":"
       << value.host_store_hessian_topology_builder_calls
       << ",\"host_store_hessian_segment_plan_builder_calls\":"
       << value.host_store_hessian_segment_plan_builder_calls
       << ",\"host_store_schur_topology_builder_calls\":"
       << value.host_store_schur_topology_builder_calls
       << ",\"host_store_schur_segment_plan_builder_calls\":"
       << value.host_store_schur_segment_plan_builder_calls
       << ",\"host_store_dynamic_state_refresh_calls\":"
       << value.host_store_dynamic_state_refresh_calls
       << ",\"host_store_indexed_catalog_full_graph_build_calls\":"
       << value.host_store_indexed_catalog_full_graph_build_calls
       << ",\"host_store_indexed_catalog_journal_apply_calls\":"
       << value.host_store_indexed_catalog_journal_apply_calls
       << ",\"host_store_indexed_catalog_flatten_calls\":"
       << value.host_store_indexed_catalog_flatten_calls
       << ",\"host_store_indexed_catalog_reconcile_calls\":"
       << value.host_store_indexed_catalog_reconcile_calls
       << ",\"host_store_indexed_catalog_export_calls\":"
       << value.host_store_indexed_catalog_export_calls
       << ",\"host_store_indexed_active_materializer_calls\":"
       << value.host_store_indexed_active_materializer_calls
       << ",\"host_store_indexed_full_graph_records_scanned\":"
       << value.host_store_indexed_full_graph_records_scanned
       << ",\"host_store_indexed_estimated_impl_copy_bytes\":"
       << value.host_store_indexed_estimated_impl_copy_bytes
       << ",\"host_store_indexed_estimated_export_bytes\":"
       << value.host_store_indexed_estimated_export_bytes
       << ",\"host_store_descriptor_identity\":"
       << value.host_store_descriptor_identity
       << ",\"host_store_descriptor_items\":"
       << value.host_store_descriptor_items
       << ",\"host_store_descriptor_hash_updates\":"
       << value.host_store_descriptor_hash_updates
       << ",\"host_store_device_static_h2d_saved_calls\":"
       << value.host_store_device_static_h2d_saved_calls
       << ",\"host_store_device_static_h2d_saved_bytes\":"
       << value.host_store_device_static_h2d_saved_bytes
       << ",\"host_store_dynamic_h2d_calls\":"
       << value.host_store_dynamic_h2d_calls
       << ",\"host_store_dynamic_h2d_bytes\":"
       << value.host_store_dynamic_h2d_bytes
       << ",\"cuda_host_build_cuda_layer_a_inputs_calls\":"
       << value.cuda_host_build_cuda_layer_a_inputs_calls
       << ",\"cuda_host_build_static_layout_calls\":"
       << value.cuda_host_build_static_layout_calls
       << ",\"cuda_host_build_cost_layout_calls\":"
       << value.cuda_host_build_cost_layout_calls
       << ",\"cuda_host_build_layer_b_topology_calls\":"
       << value.cuda_host_build_layer_b_topology_calls
       << ",\"cuda_host_build_layer_c_topology_calls\":"
       << value.cuda_host_build_layer_c_topology_calls
       << ",\"indexed_device_catalog_lookup_calls\":"
       << value.indexed_device_catalog_lookup_calls
       << ",\"indexed_device_catalog_reuse_calls\":"
       << value.indexed_device_catalog_reuse_calls
       << ",\"indexed_device_catalog_revision_rebinds\":"
       << value.indexed_device_catalog_revision_rebinds
       << ",\"indexed_device_catalog_full_upload_calls\":"
       << value.indexed_device_catalog_full_upload_calls
       << ",\"indexed_device_catalog_full_upload_bytes\":"
       << value.indexed_device_catalog_full_upload_bytes
       << ",\"indexed_device_catalog_patch_upload_calls\":"
       << value.indexed_device_catalog_patch_upload_calls
       << ",\"indexed_device_catalog_patch_upload_bytes\":"
       << value.indexed_device_catalog_patch_upload_bytes
       << ",\"indexed_device_catalog_invalidations\":"
       << value.indexed_device_catalog_invalidations
       << ",\"indexed_device_catalog_prefix_bytes\":"
       << value.indexed_device_catalog_prefix_bytes
       << ",\"indexed_device_catalog_arena_generation\":"
       << value.indexed_device_catalog_arena_generation
       << ",\"host_store_host_resident_bytes\":"
       << value.host_store_host_resident_bytes
       << ",\"host_store_host_peak_bytes\":"
       << value.host_store_host_peak_bytes
       << ",\"host_store_owner_identity_violations\":"
       << value.host_store_owner_identity_violations
       << ",\"host_store_coverage_violations\":"
       << value.host_store_coverage_violations
       << ",\"host_store_busy_failures\":"
       << value.host_store_busy_failures
       << ",\"host_store_catalog_valid\":"
       << (value.host_store_catalog_valid ? "true" : "false")
       << ",\"host_store_device_prepared_context_reuse\":"
       << (value.host_store_device_prepared_context_reuse ? "true" : "false")
       << ",\"host_store_full_cross_solve_initialization_removed\":"
       << (value.host_store_full_cross_solve_initialization_removed
               ? "true" : "false")
       << ",\"host_store_descriptor_milliseconds\":";
  WriteJsonNumber(file, value.host_store_descriptor_milliseconds);
  file << ",\"host_store_lookup_milliseconds\":";
  WriteJsonNumber(file, value.host_store_lookup_milliseconds);
  file << ",\"host_store_catalog_delta_milliseconds\":";
  WriteJsonNumber(file, value.host_store_catalog_delta_milliseconds);
  file << ",\"host_store_view_build_or_patch_milliseconds\":";
  WriteJsonNumber(file, value.host_store_view_build_or_patch_milliseconds);
  file << ",\"host_store_dynamic_refresh_milliseconds\":";
  WriteJsonNumber(file, value.host_store_dynamic_refresh_milliseconds);
  file << ",\"host_store_preparation_total_milliseconds\":";
  WriteJsonNumber(file, value.host_store_preparation_total_milliseconds);
  file << ",\"host_store_prepare_and_cuda_wall_seconds\":";
  WriteJsonNumber(file, value.host_store_prepare_and_cuda_wall_seconds);
  file << ",\"problem_source_prepare_and_cuda_wall_seconds\":";
  WriteJsonNumber(file, value.problem_source_prepare_and_cuda_wall_seconds);
  file << ",\"custom_cuda_caller_wall_seconds\":";
  WriteJsonNumber(file, value.custom_cuda_caller_wall_seconds);
  file << ",\"bundle_adjuster_setup_milliseconds\":";
  WriteJsonNumber(file, value.bundle_adjuster_setup_milliseconds);
  file << ",\"snapshot_materialization_milliseconds\":";
  WriteJsonNumber(file, value.snapshot_materialization_milliseconds);
  file << ",\"active_spec_build_milliseconds\":";
  WriteJsonNumber(file, value.active_spec_build_milliseconds);
  file << ",\"active_spec_checkpoint_milliseconds\":";
  WriteJsonNumber(file, value.active_spec_checkpoint_milliseconds);
  file << ",\"legacy_cpu_preparation_milliseconds\":";
  WriteJsonNumber(file, value.legacy_cpu_preparation_milliseconds);
  file << ",\"fast_cpu_preparation_milliseconds\":";
  WriteJsonNumber(file, value.fast_cpu_preparation_milliseconds);
  file << ",\"trigger_image_id\":" << value.trigger_image_id
       << ",\"refinement_index\":" << value.refinement_index;
  file << "}\n";
}

void CopyHostStoreRuntimeToExecution(
    const gpu_ba::CudaHostProblemStoreRuntimeInfo& source,
    BundleAdjustmentExecutionResult* target) {
  target->host_store_mode_requested = source.mode_requested;
  target->host_store_mode_effective = source.mode_effective;
  target->host_store_view_action = source.view_action;
  target->host_store_rebuild_reason = source.rebuild_reason;
  target->host_store_owner_epoch = source.owner_epoch;
  target->host_store_catalog_generation = source.catalog_generation;
  target->host_store_view_generation = source.view_generation;
  target->host_store_lookup_calls = source.store_lookup_calls;
  target->host_store_hits = source.store_hits;
  target->host_store_misses = source.store_misses;
  target->host_store_journal_cursor_before = source.journal_cursor_before;
  target->host_store_journal_cursor_after = source.journal_cursor_after;
  target->host_store_catalog_cold_builds = source.catalog_cold_builds;
  target->host_store_catalog_delta_updates = source.catalog_delta_updates;
  target->host_store_catalog_full_rebuilds = source.catalog_full_rebuilds;
  target->host_store_journal_events = source.journal_events;
  target->host_store_journal_gaps = source.journal_gaps;
  target->host_store_journal_overflows = source.journal_overflows;
  target->host_store_journal_unknown_events = source.journal_unknown_events;
  target->host_store_exact_view_reuses = source.exact_view_reuses;
  target->host_store_view_patches = source.solve_view_patches;
  target->host_store_view_rebuilds = source.solve_view_rebuilds;
  target->host_store_fallback_rebuilds = source.fallback_rebuilds;
  target->host_store_builder_calls_executed =
      source.host_builder_calls_executed;
  target->host_store_builder_calls_saved = source.host_builder_calls_saved;
  target->host_store_builder_traversals_executed =
      source.host_builder_traversals_executed;
  target->host_store_builder_traversals_saved =
      source.host_builder_traversals_saved;
  target->host_store_estimated_builder_bytes_executed =
      source.estimated_host_builder_bytes_executed;
  target->host_store_estimated_builder_bytes_saved =
      source.estimated_host_builder_bytes_saved;
  target->host_store_static_binding_builder_calls =
      source.static_binding_builder_calls;
  target->host_store_cost_layout_builder_calls =
      source.cost_layout_builder_calls;
  target->host_store_hessian_topology_builder_calls =
      source.hessian_topology_builder_calls;
  target->host_store_hessian_segment_plan_builder_calls =
      source.hessian_segment_plan_builder_calls;
  target->host_store_schur_topology_builder_calls =
      source.schur_topology_builder_calls;
  target->host_store_schur_segment_plan_builder_calls =
      source.schur_segment_plan_builder_calls;
  target->host_store_dynamic_state_refresh_calls =
      source.dynamic_state_refresh_calls;
  target->host_store_indexed_catalog_full_graph_build_calls =
      source.indexed_catalog_full_graph_build_calls;
  target->host_store_indexed_catalog_journal_apply_calls =
      source.indexed_catalog_journal_apply_calls;
  target->host_store_indexed_catalog_flatten_calls =
      source.indexed_catalog_flatten_calls;
  target->host_store_indexed_catalog_reconcile_calls =
      source.indexed_catalog_reconcile_calls;
  target->host_store_indexed_catalog_export_calls =
      source.indexed_catalog_export_calls;
  target->host_store_indexed_active_materializer_calls =
      source.indexed_active_materializer_calls;
  target->host_store_indexed_full_graph_records_scanned =
      source.indexed_full_graph_records_scanned;
  target->host_store_indexed_estimated_impl_copy_bytes =
      source.indexed_estimated_impl_copy_bytes;
  target->host_store_indexed_estimated_export_bytes =
      source.indexed_estimated_export_bytes;
  target->host_store_device_static_h2d_saved_calls =
      source.device_static_h2d_saved_calls;
  target->host_store_device_static_h2d_saved_bytes =
      source.device_static_h2d_saved_bytes;
  target->host_store_dynamic_h2d_calls = source.dynamic_h2d_calls;
  target->host_store_dynamic_h2d_bytes = source.dynamic_h2d_bytes;
  target->host_store_host_resident_bytes = source.host_resident_bytes;
  target->host_store_host_peak_bytes = source.host_peak_bytes;
  target->host_store_owner_identity_violations =
      source.owner_identity_violations;
  target->host_store_coverage_violations = source.coverage_violations;
  target->host_store_busy_failures = source.store_busy_failures;
  target->host_store_catalog_valid = source.host_catalog_valid;
  target->host_store_device_prepared_context_reuse =
      source.device_prepared_context_reuse;
  target->host_store_full_cross_solve_initialization_removed =
      source.full_cross_solve_initialization_removed;
  target->host_store_descriptor_milliseconds =
      source.snapshot_descriptor_wall_milliseconds;
  target->host_store_lookup_milliseconds =
      source.store_lookup_wall_milliseconds;
  target->host_store_catalog_delta_milliseconds =
      source.catalog_delta_update_wall_milliseconds;
  target->host_store_view_build_or_patch_milliseconds =
      source.solve_view_build_or_patch_wall_milliseconds;
  target->host_store_dynamic_refresh_milliseconds =
      source.dynamic_state_refresh_wall_milliseconds;
  target->host_store_preparation_total_milliseconds =
      source.host_preparation_total_wall_milliseconds;
}

void CopyHostStoreCommitStateToExecution(
    const gpu_ba::CudaHostProblemStoreRuntimeInfo& source,
    BundleAdjustmentExecutionResult* target) {
  target->host_store_journal_cursor_after = source.journal_cursor_after;
  target->host_store_catalog_generation = source.catalog_generation;
  target->host_store_catalog_valid = source.host_catalog_valid;
  target->host_store_host_resident_bytes = source.host_resident_bytes;
  target->host_store_host_peak_bytes = source.host_peak_bytes;
}

#ifdef GPU_BA_CUDA_ENABLED
bool ValidateCustomCudaProductionSupport(
    const BundleAdjustmentOptions& options,
    const ceres::Solver::Options& effective_options,
    const gpu_ba::CudaSolveProblem& snapshot,
    std::string* error) {
  if (options.ba_cuda_schur_mode != "deterministic") {
    *error = "UNSUPPORTED_CONFIGURATION: deterministic Schur is required";
    return false;
  }
  if (options.refine_focal_length || options.refine_principal_point ||
      options.refine_extra_params) {
    *error = "UNSUPPORTED_CONFIGURATION: variable intrinsics";
    return false;
  }
  if (options.loss_function_type ==
      BundleAdjustmentOptions::LossFunctionType::CAUCHY) {
    *error = "UNSUPPORTED_CONFIGURATION: CAUCHY loss";
    return false;
  }
  if (!effective_options.jacobi_scaling ||
      effective_options.use_nonmonotonic_steps) {
    *error = "UNSUPPORTED_CONFIGURATION: Ceres scaling/step policy";
    return false;
  }
  if (effective_options.minimizer_type != ceres::TRUST_REGION ||
      effective_options.trust_region_strategy_type !=
          ceres::LEVENBERG_MARQUARDT ||
      effective_options.use_inner_iterations) {
    *error = "UNSUPPORTED_CONFIGURATION: LM trust-region policy required";
    return false;
  }
  const bool valid_lm_numbers =
      std::isfinite(effective_options.initial_trust_region_radius) &&
      std::isfinite(effective_options.min_trust_region_radius) &&
      std::isfinite(effective_options.max_trust_region_radius) &&
      effective_options.initial_trust_region_radius > 0.0 &&
      effective_options.min_trust_region_radius > 0.0 &&
      effective_options.max_trust_region_radius >=
          effective_options.initial_trust_region_radius &&
      effective_options.initial_trust_region_radius >=
          effective_options.min_trust_region_radius &&
      std::isfinite(effective_options.min_lm_diagonal) &&
      std::isfinite(effective_options.max_lm_diagonal) &&
      effective_options.min_lm_diagonal > 0.0 &&
      effective_options.max_lm_diagonal >=
          effective_options.min_lm_diagonal &&
      std::isfinite(effective_options.function_tolerance) &&
      std::isfinite(effective_options.gradient_tolerance) &&
      std::isfinite(effective_options.parameter_tolerance) &&
      effective_options.function_tolerance >= 0.0 &&
      effective_options.gradient_tolerance >= 0.0 &&
      effective_options.parameter_tolerance >= 0.0;
  if (!valid_lm_numbers) {
    *error = "UNSUPPORTED_CONFIGURATION: invalid LM radius/diagonal/tolerance";
    return false;
  }
  for (const auto& camera : snapshot.cameras) {
    if (camera.model_id != OpenCVCameraModel::kModelId ||
        camera.params.size() != 8 || !camera.constant) {
      *error = "UNSUPPORTED_CONFIGURATION: fixed 8-parameter OPENCV required";
      return false;
    }
  }
  for (const auto& parameter : snapshot.parameter_blocks_source_order) {
    if (parameter.kind == gpu_ba::ParameterKind::kCamera &&
        !parameter.constant) {
      *error = "UNSUPPORTED_CONFIGURATION: variable camera block";
      return false;
    }
  }
  return true;
}

gpu_ba::CudaFullLmOptions CreateProductionCudaOptions(
    const BundleAdjustmentOptions& options,
    const ceres::Solver::Options& effective_options) {
  gpu_ba::CudaFullLmOptions cuda;
  cuda.device_context_mode = gpu_ba::CudaDeviceContextMode::kDeviceControl;
  if (options.ba_cuda_audit_profile == "production") {
    cuda.audit_profile = gpu_ba::CudaAuditProfile::kProduction;
  } else if (options.ba_cuda_audit_profile == "correctness") {
    cuda.audit_profile = gpu_ba::CudaAuditProfile::kCorrectness;
  } else if (options.ba_cuda_audit_profile == "forensic") {
    cuda.audit_profile = gpu_ba::CudaAuditProfile::kForensic;
  }
  if (options.ba_cuda_arithmetic_precision == "fp64") {
    cuda.arithmetic_precision = gpu_ba::CudaArithmeticPrecision::kFp64;
  } else if (options.ba_cuda_arithmetic_precision == "fp32_core") {
    cuda.arithmetic_precision = gpu_ba::CudaArithmeticPrecision::kFp32Core;
  } else if (options.ba_cuda_arithmetic_precision ==
             "fp32_state_quantized") {
    cuda.arithmetic_precision =
        gpu_ba::CudaArithmeticPrecision::kFp32StateQuantizedMixed;
  } else if (options.ba_cuda_arithmetic_precision == "fp32_mixed") {
    cuda.arithmetic_precision =
        gpu_ba::CudaArithmeticPrecision::kFp32MixedStable;
  }
  if (options.ba_cuda_hessian_assembly_backend == "pose_owned") {
    cuda.layer_c.layer_b.hessian_assembly_backend =
        gpu_ba::CudaHessianAssemblyBackend::kPoseOwnedOptimizedReference;
  } else if (options.ba_cuda_hessian_assembly_backend ==
             "observation_segmented") {
    cuda.layer_c.layer_b.hessian_assembly_backend =
        gpu_ba::CudaHessianAssemblyBackend::kObservationSegmented;
  }
  if (options.ba_cuda_hot_kernel_mode == "reference") {
    cuda.hot_kernel_mode = gpu_ba::CudaHotKernelMode::kReference;
  } else if (options.ba_cuda_hot_kernel_mode == "optimized") {
    cuda.hot_kernel_mode = gpu_ba::CudaHotKernelMode::kOptimized;
  } else if (options.ba_cuda_hot_kernel_mode == "compatibility_default") {
    cuda.hot_kernel_mode = gpu_ba::CudaHotKernelMode::kCompatibilityDefault;
  } else {
    cuda.hot_kernel_mode = gpu_ba::CudaHotKernelMode::kTransformed;
  }
  if (options.ba_cuda_schur_contribution_backend == "segmented") {
    cuda.layer_c.schur_contribution_backend =
        gpu_ba::CudaSchurContributionBackend::kSegmentedTransformed;
  } else if (options.ba_cuda_schur_contribution_backend ==
             "compatibility_default") {
    cuda.layer_c.schur_contribution_backend =
        gpu_ba::CudaSchurContributionBackend::kCompatibilityDefault;
  } else {
    cuda.layer_c.schur_contribution_backend =
        gpu_ba::CudaSchurContributionBackend::kDirectTransformed;
  }
  switch (options.ba_cuda_execution_profile) {
    case BundleAdjustmentOptions::CudaExecutionProfile::BASELINE:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kBaseline;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::RUNTIME_POOL:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kRuntimePool;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::ARENA:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kArena;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::DEVICE_SCALING:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kDeviceScaling;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::FAST_IDENTITY:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kFastIdentity;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::UNIFIED_BUILDER:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kUnifiedBuilder;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_LAYER_A:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kCompactLayerA;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kCompactControl;
      break;
    case BundleAdjustmentOptions::CudaExecutionProfile::INVALID:
      cuda.execution_profile = gpu_ba::CudaExecutionProfile::kBaseline;
      break;
  }
  cuda.layer_c.layer_b.layer_a.device = options.ba_cuda_device;
  cuda.layer_c.layer_b.layer_a.residual_order =
      gpu_ba::CudaResidualOrder::kSourceInsertion;
  cuda.layer_c.layer_b.loss_mode =
      options.loss_function_type == BundleAdjustmentOptions::LossFunctionType::SOFT_L1
          ? gpu_ba::CudaLossMode::kSoftL1
          : gpu_ba::CudaLossMode::kTrivial;
  cuda.layer_c.layer_b.loss_scale = options.loss_function_scale;
  cuda.layer_c.layer_b.reduction_mode =
      gpu_ba::CudaReductionMode::kParallelDeterministic;
  cuda.layer_c.layer_b.cost_reduction_threads = 128;
  cuda.performance_mode = true;
  cuda.current_linearization_cache_mode =
      gpu_ba::CudaCurrentLinearizationCacheMode::kEnabled;
  cuda.max_num_iterations = effective_options.max_num_iterations;
  cuda.max_num_consecutive_invalid_steps =
      effective_options.max_num_consecutive_invalid_steps;
  cuda.max_solver_time_in_seconds =
      effective_options.max_solver_time_in_seconds;
  cuda.function_tolerance = effective_options.function_tolerance;
  cuda.gradient_tolerance = effective_options.gradient_tolerance;
  cuda.parameter_tolerance = effective_options.parameter_tolerance;
  cuda.initial_trust_region_radius =
      effective_options.initial_trust_region_radius;
  cuda.min_trust_region_radius = effective_options.min_trust_region_radius;
  cuda.max_trust_region_radius = effective_options.max_trust_region_radius;
  cuda.min_relative_decrease = effective_options.min_relative_decrease;
  cuda.layer_c.layer_b.min_lm_diagonal = effective_options.min_lm_diagonal;
  cuda.layer_c.layer_b.max_lm_diagonal = effective_options.max_lm_diagonal;
  return cuda;
}

bool SameSnapshotTopology(const gpu_ba::Snapshot& initial,
                          const gpu_ba::Snapshot& candidate,
                          std::string* error) {
  if (initial.images.size() != candidate.images.size() ||
      initial.points.size() != candidate.points.size() ||
      initial.cameras.size() != candidate.cameras.size() ||
      initial.observations.size() != candidate.observations.size() ||
      initial.lidar.size() != candidate.lidar.size() ||
      initial.parameter_blocks_source_order.size() !=
          candidate.parameter_blocks_source_order.size() ||
      initial.parameter_blocks_canonical_order.size() !=
          candidate.parameter_blocks_canonical_order.size() ||
      initial.source_insertion_order.size() !=
          candidate.source_insertion_order.size() ||
      initial.canonical_order.size() != candidate.canonical_order.size() ||
      initial.tracks.size() != candidate.tracks.size()) {
    *error = "CUDA_COMMIT_TOPOLOGY_SIZE_MISMATCH";
    return false;
  }
  for (size_t i = 0; i < initial.cameras.size(); ++i) {
    const auto& a = initial.cameras[i];
    const auto& b = candidate.cameras[i];
    if (a.camera_id != b.camera_id || a.model_id != b.model_id ||
        a.width != b.width || a.height != b.height ||
        a.constant != b.constant || a.params.size() != b.params.size()) {
      *error = "CUDA_COMMIT_CAMERA_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.images.size(); ++i) {
    const auto& a = initial.images[i];
    const auto& b = candidate.images[i];
    if (a.image_id != b.image_id || a.camera_id != b.camera_id ||
        a.selected != b.selected || a.pose_constant != b.pose_constant ||
        a.has_pose_parameter_blocks != b.has_pose_parameter_blocks ||
        a.constant_tvec_mask != b.constant_tvec_mask) {
      *error = "CUDA_COMMIT_IMAGE_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.points.size(); ++i) {
    const auto& a = initial.points[i];
    const auto& b = candidate.points[i];
    if (a.point3D_id != b.point3D_id || a.constant != b.constant ||
        a.config_role != b.config_role ||
        a.has_search_range != b.has_search_range ||
        a.search_range != b.search_range) {
      *error = "CUDA_COMMIT_POINT_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.observations.size(); ++i) {
    const auto& a = initial.observations[i];
    const auto& b = candidate.observations[i];
    if (a.source_index != b.source_index || a.image_id != b.image_id ||
        a.point2D_idx != b.point2D_idx || a.point3D_id != b.point3D_id ||
        a.pose_constant != b.pose_constant || a.xy != b.xy) {
      *error = "CUDA_COMMIT_OBSERVATION_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.tracks.size(); ++i) {
    const auto& a = initial.tracks[i];
    const auto& b = candidate.tracks[i];
    if (a.point3D_id != b.point3D_id || a.image_id != b.image_id ||
        a.point2D_idx != b.point2D_idx) {
      *error = "CUDA_COMMIT_TRACK_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.lidar.size(); ++i) {
    const auto& a = initial.lidar[i];
    const auto& b = candidate.lidar[i];
    if (a.source_index != b.source_index || a.point3D_id != b.point3D_id ||
        a.lidar_type != b.lidar_type ||
        a.has_search_range != b.has_search_range ||
        a.search_range != b.search_range || a.weight != b.weight ||
        a.lidar_xyz != b.lidar_xyz || a.plane != b.plane) {
      *error = "CUDA_COMMIT_LIDAR_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.parameter_blocks_source_order.size(); ++i) {
    const auto& a = initial.parameter_blocks_source_order[i];
    const auto& b = candidate.parameter_blocks_source_order[i];
    if (a.kind != b.kind || a.entity_id != b.entity_id ||
        a.ambient_size != b.ambient_size || a.tangent_size != b.tangent_size ||
        a.constant != b.constant) {
      *error = "CUDA_COMMIT_PARAMETER_BLOCK_MISMATCH";
      return false;
    }
  }
  if (initial.parameter_blocks_canonical_order !=
      candidate.parameter_blocks_canonical_order) {
    *error = "CUDA_COMMIT_PARAMETER_CANONICAL_ORDER_MISMATCH";
    return false;
  }
  for (size_t i = 0; i < initial.source_insertion_order.size(); ++i) {
    const auto& a = initial.source_insertion_order[i];
    const auto& b = candidate.source_insertion_order[i];
    if (a.source_index != b.source_index ||
        a.residual_kind != b.residual_kind || a.image_id != b.image_id ||
        a.point2D_idx != b.point2D_idx || a.point3D_id != b.point3D_id) {
      *error = "CUDA_COMMIT_RESIDUAL_ORDER_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.canonical_order.size(); ++i) {
    const auto& a = initial.canonical_order[i];
    const auto& b = candidate.canonical_order[i];
    if (a.source_index != b.source_index ||
        a.residual_kind != b.residual_kind || a.image_id != b.image_id ||
        a.point2D_idx != b.point2D_idx || a.point3D_id != b.point3D_id) {
      *error = "CUDA_COMMIT_CANONICAL_ORDER_MISMATCH";
      return false;
    }
  }
  return true;
}

bool PrepareAndCommitCudaState(const gpu_ba::Snapshot& initial,
                               const gpu_ba::Snapshot& candidate,
                               Reconstruction* reconstruction,
                               std::string* error) {
  if (!SameSnapshotTopology(initial, candidate, error)) return false;

  struct QUpdate { image_t id; Eigen::Vector4d value; };
  struct TUpdate { image_t id; Eigen::Vector3d value; };
  struct PUpdate { point3D_t id; Eigen::Vector3d value; };
  std::vector<QUpdate> q_updates;
  std::vector<TUpdate> t_updates;
  std::vector<PUpdate> p_updates;
  q_updates.reserve(initial.images.size());
  t_updates.reserve(initial.images.size());
  p_updates.reserve(initial.points.size());
  std::unordered_set<SnapshotParameterIdentity,
                     SnapshotParameterIdentityHash> unique_blocks;

  std::unordered_map<camera_t, const gpu_ba::CameraSnapshot*> final_cameras;
  std::unordered_map<image_t, const gpu_ba::ImageSnapshot*> before_images;
  std::unordered_map<image_t, const gpu_ba::ImageSnapshot*> after_images;
  std::unordered_map<point3D_t, const gpu_ba::PointSnapshot*> before_points;
  std::unordered_map<point3D_t, const gpu_ba::PointSnapshot*> after_points;
  for (const auto& value : candidate.cameras) {
    if (!final_cameras.emplace(value.camera_id, &value).second) {
      *error = "CUDA_COMMIT_DUPLICATE_CAMERA_ID";
      return false;
    }
  }
  for (const auto& value : initial.images) {
    if (!before_images.emplace(value.image_id, &value).second) {
      *error = "CUDA_COMMIT_DUPLICATE_IMAGE_ID";
      return false;
    }
  }
  for (const auto& value : candidate.images) {
    if (!after_images.emplace(value.image_id, &value).second) {
      *error = "CUDA_COMMIT_DUPLICATE_IMAGE_ID";
      return false;
    }
  }
  for (const auto& value : initial.points) {
    if (!before_points.emplace(value.point3D_id, &value).second) {
      *error = "CUDA_COMMIT_DUPLICATE_POINT_ID";
      return false;
    }
  }
  for (const auto& value : candidate.points) {
    if (!after_points.emplace(value.point3D_id, &value).second) {
      *error = "CUDA_COMMIT_DUPLICATE_POINT_ID";
      return false;
    }
  }

  for (const auto& camera : initial.cameras) {
    const auto final = final_cameras.find(camera.camera_id);
    if (final == final_cameras.end() ||
        final->second->model_id != camera.model_id ||
        final->second->params != camera.params || !camera.constant) {
      *error = "CUDA_COMMIT_FIXED_CAMERA_CHANGED";
      return false;
    }
  }
  for (const auto& image : initial.images) {
    const auto after = after_images.find(image.image_id);
    if (after == after_images.end()) {
      *error = "CUDA_COMMIT_IMAGE_MISSING";
      return false;
    }
    if ((image.pose_constant || !image.has_pose_parameter_blocks) &&
        (after->second->qvec != image.qvec ||
         after->second->tvec != image.tvec)) {
      *error = "CUDA_COMMIT_FIXED_POSE_CHANGED";
      return false;
    }
  }
  for (const auto& point : initial.points) {
    const auto after = after_points.find(point.point3D_id);
    if (after == after_points.end()) {
      *error = "CUDA_COMMIT_POINT_MISSING";
      return false;
    }
    if (point.constant && after->second->xyz != point.xyz) {
      *error = "CUDA_COMMIT_FIXED_POINT_CHANGED";
      return false;
    }
  }

  for (const auto& parameter : initial.parameter_blocks_source_order) {
    if (!unique_blocks.emplace(
            SnapshotParameterIdentity{parameter.kind, parameter.entity_id})
             .second) {
      *error = "CUDA_COMMIT_DUPLICATE_PARAMETER_BLOCK";
      return false;
    }
    if (parameter.kind == gpu_ba::ParameterKind::kQuaternion ||
        parameter.kind == gpu_ba::ParameterKind::kTranslation) {
      const image_t id = static_cast<image_t>(parameter.entity_id);
      const auto before_it = before_images.find(id);
      const auto after_it = after_images.find(id);
      if (before_it == before_images.end() || after_it == after_images.end() ||
          !reconstruction->ExistsImage(id)) {
        *error = "CUDA_COMMIT_IMAGE_MISSING";
        return false;
      }
      const auto* before = before_it->second;
      const auto* after = after_it->second;
      if (parameter.kind == gpu_ba::ParameterKind::kQuaternion) {
        Eigen::Vector4d value(after->qvec.data());
        if (!value.allFinite()) {
          *error = "CUDA_COMMIT_NONFINITE_QUATERNION";
          return false;
        }
        if (parameter.constant) {
          if (after->qvec != before->qvec) {
            *error = "CUDA_COMMIT_FIXED_QUATERNION_CHANGED";
            return false;
          }
        } else {
          const double norm_error = std::abs(value.norm() - 1.0);
          if (norm_error > 1e-10) {
            *error = "CUDA_COMMIT_QUATERNION_NORM";
            return false;
          }
          value.normalize();
          if (std::abs(value.norm() - 1.0) > 2e-15) {
            *error = "CUDA_COMMIT_NORMALIZED_QUATERNION_NORM";
            return false;
          }
          q_updates.push_back({id, value});
        }
      } else {
        Eigen::Vector3d value = reconstruction->Image(id).Tvec();
        for (int k = 0; k < 3; ++k) {
          const double final_value = after->tvec[static_cast<size_t>(k)];
          if (!std::isfinite(final_value)) {
            *error = "CUDA_COMMIT_NONFINITE_TRANSLATION";
            return false;
          }
          const bool fixed = parameter.constant ||
              ((before->constant_tvec_mask & (1u << k)) != 0);
          if (fixed) {
            if (final_value != before->tvec[static_cast<size_t>(k)]) {
              *error = "CUDA_COMMIT_FIXED_TRANSLATION_CHANGED";
              return false;
            }
          } else {
            value[k] = final_value;
          }
        }
        if (!parameter.constant) t_updates.push_back({id, value});
      }
    } else if (parameter.kind == gpu_ba::ParameterKind::kPoint3D) {
      const point3D_t id = static_cast<point3D_t>(parameter.entity_id);
      const auto before_it = before_points.find(id);
      const auto after_it = after_points.find(id);
      if (before_it == before_points.end() || after_it == after_points.end() ||
          !reconstruction->ExistsPoint3D(id)) {
        *error = "CUDA_COMMIT_POINT_MISSING";
        return false;
      }
      const auto* before = before_it->second;
      const auto* after = after_it->second;
      Eigen::Vector3d value(after->xyz.data());
      if (!value.allFinite()) {
        *error = "CUDA_COMMIT_NONFINITE_POINT";
        return false;
      }
      if (parameter.constant) {
        if (after->xyz != before->xyz) {
          *error = "CUDA_COMMIT_FIXED_POINT_CHANGED";
          return false;
        }
      } else {
        p_updates.push_back({id, value});
      }
    }
  }

  // All validation and allocation has completed. The commit itself only
  // assigns fixed-size Eigen values into existing reconstruction entities.
  for (const auto& update : q_updates) {
    reconstruction->Image(update.id).SetQvec(update.value);
  }
  for (const auto& update : t_updates) {
    reconstruction->Image(update.id).SetTvec(update.value);
  }
  for (const auto& update : p_updates) {
    reconstruction->Point3D(update.id).SetXYZ(update.value);
  }
  return true;
}

void ProjectCudaSummary(const gpu_ba::CudaSolveProblem& snapshot,
                        const int parameter_blocks,
                        const int parameters,
                        const int residuals,
                        const gpu_ba::CudaFullLmResult& cuda,
                        double wall_seconds,
                        ceres::Solver::Summary* summary) {
  *summary = ceres::Solver::Summary();
  summary->num_parameter_blocks = parameter_blocks;
  summary->num_parameters = parameters;
  summary->num_residual_blocks =
      static_cast<int>(snapshot.observations.size() + snapshot.lidar.size());
  summary->num_residuals = residuals;
  summary->num_parameter_blocks_reduced = 0;
  summary->num_parameters_reduced = 0;
  summary->num_effective_parameters_reduced = 0;
  for (const auto& parameter : snapshot.parameter_blocks_source_order) {
    if (!parameter.constant) {
      ++summary->num_parameter_blocks_reduced;
      summary->num_parameters_reduced += parameter.ambient_size;
      summary->num_effective_parameters_reduced += parameter.tangent_size;
    }
  }
  summary->num_residual_blocks_reduced = summary->num_residual_blocks;
  summary->num_residuals_reduced = summary->num_residuals;
  summary->initial_cost = cuda.initial_cost;
  summary->final_cost = cuda.final_cost;
  summary->num_successful_steps = cuda.accepted_steps;
  summary->num_unsuccessful_steps = cuda.rejected_steps + cuda.invalid_steps;
  summary->total_time_in_seconds = wall_seconds;
  summary->minimizer_time_in_seconds = wall_seconds;
  summary->termination_type =
      cuda.termination_type == gpu_ba::CudaTerminationType::kConvergence
          ? ceres::CONVERGENCE
          : (cuda.termination_type == gpu_ba::CudaTerminationType::kNoConvergence
                 ? ceres::NO_CONVERGENCE
                 : ceres::FAILURE);
  summary->message = cuda.termination_reason;
}
#endif  // GPU_BA_CUDA_ENABLED

#ifdef GPU_BA_ENABLED
std::string FidelityLossName(
    BundleAdjustmentOptions::LossFunctionType type) {
  switch (type) {
    case BundleAdjustmentOptions::LossFunctionType::TRIVIAL:
      return "TRIVIAL";
    case BundleAdjustmentOptions::LossFunctionType::SOFT_L1:
      return "SOFT_L1";
    case BundleAdjustmentOptions::LossFunctionType::CAUCHY:
      return "CAUCHY";
  }
  return "UNSUPPORTED";
}

std::vector<gpu_ba::ParameterConstraintSnapshot>
BuildFidelityParameterConstraints(
    const gpu_ba::Snapshot& snapshot,
    const BundleAdjustmentOptions& options,
    const Reconstruction& reconstruction) {
  std::unordered_map<uint32_t, const gpu_ba::ImageSnapshot*> images;
  for (const auto& image : snapshot.images) images.emplace(image.image_id, &image);
  std::vector<gpu_ba::ParameterConstraintSnapshot> constraints;
  constraints.reserve(snapshot.parameter_blocks_source_order.size());
  for (const auto& parameter : snapshot.parameter_blocks_source_order) {
    gpu_ba::ParameterConstraintSnapshot constraint;
    constraint.kind = parameter.kind;
    constraint.entity_id = parameter.entity_id;
    constraint.constant = parameter.constant;
    if (parameter.kind == gpu_ba::ParameterKind::kQuaternion &&
        !parameter.constant) {
      constraint.local_parameterization = "quaternion";
    } else if (parameter.kind == gpu_ba::ParameterKind::kTranslation &&
               !parameter.constant) {
      const auto image_it = images.find(static_cast<uint32_t>(parameter.entity_id));
      if (image_it != images.end()) {
        for (int index = 0; index < 3; ++index) {
          if ((image_it->second->constant_tvec_mask & (1u << index)) != 0) {
            constraint.constant_indices.push_back(index);
          }
        }
      }
      constraint.local_parameterization =
          constraint.constant_indices.empty() ? "euclidean" : "subset";
    } else if (parameter.kind == gpu_ba::ParameterKind::kCamera &&
               !parameter.constant) {
      const Camera& camera =
          reconstruction.Camera(static_cast<camera_t>(parameter.entity_id));
      if (!options.refine_focal_length) {
        for (const size_t index : camera.FocalLengthIdxs()) {
          constraint.constant_indices.push_back(static_cast<int32_t>(index));
        }
      }
      if (!options.refine_principal_point) {
        for (const size_t index : camera.PrincipalPointIdxs()) {
          constraint.constant_indices.push_back(static_cast<int32_t>(index));
        }
      }
      if (!options.refine_extra_params) {
        for (const size_t index : camera.ExtraParamsIdxs()) {
          constraint.constant_indices.push_back(static_cast<int32_t>(index));
        }
      }
      std::sort(constraint.constant_indices.begin(),
                constraint.constant_indices.end());
      constraint.constant_indices.erase(
          std::unique(constraint.constant_indices.begin(),
                      constraint.constant_indices.end()),
          constraint.constant_indices.end());
      constraint.local_parameterization =
          constraint.constant_indices.empty() ? "euclidean" : "subset";
    } else {
      constraint.local_parameterization = "euclidean";
    }
    constraints.push_back(std::move(constraint));
  }
  return constraints;
}

#endif  // GPU_BA_ENABLED
}  // namespace

BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions& options,
                               const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

BundleAdjuster::~BundleAdjuster() = default;

void BundleAdjuster::SetOptimazePhrase(const OptimazePhrase& phrase) {
  optimize_phrase_ = phrase;
}

void BundleAdjuster::SetCudaHostStoreBinding(
    const gpu_ba::CudaHostStoreBinding& binding) noexcept {
  cuda_host_store_binding_ = binding;
}

#ifdef GPU_BA_ENABLED
bool BundleAdjuster::CompareProblemSourcesForTesting(
    Reconstruction* reconstruction,
    ActiveBaProblemSourceComparisonForTesting* comparison,
    std::string* error) {
  if (reconstruction == nullptr || comparison == nullptr || error == nullptr ||
      solve_called_) {
    if (error != nullptr) *error = "invalid active-spec comparison request";
    return false;
  }
  solve_called_ = true;
  error->clear();
  gpu_ba::BaKind ba_kind = gpu_ba::BaKind::kGlobal;
  if (optimize_phrase_ == OptimazePhrase::Local) {
    ba_kind = gpu_ba::BaKind::kLocal;
  } else if (optimize_phrase_ == OptimazePhrase::WholeMap) {
    ba_kind = gpu_ba::BaKind::kWhole;
  }
  const auto enumerate = [&](ceres::LossFunction* loss) {
    if (options_.if_add_lidar_constraint &&
        optimize_phrase_ == OptimazePhrase::Local) {
      SetUpLocalByLidar(reconstruction, loss);
    } else if (options_.if_add_lidar_constraint &&
               optimize_phrase_ == OptimazePhrase::Global) {
      SetUpGlobalByLidar(reconstruction, loss);
    } else if (options_.if_add_lidar_constraint &&
               optimize_phrase_ == OptimazePhrase::WholeMap) {
      SetUpAdjustWholeMapByLidar(reconstruction, loss);
    } else {
      SetUp(reconstruction, loss);
    }
  };

  problem_ = std::make_unique<ceres::Problem>();
  snapshot_recorder_.reset(new gpu_ba::SnapshotRecorder(false));
  active_spec_builder_.reset();
  camera_ids_.clear();
  point3D_num_observations_.clear();
  enumerate(options_.CreateLossFunction());
  const ceres::Solver::Options effective =
      CreateEffectiveBundleAdjustmentSolverOptions(
          options_, config_.NumImages(), problem_->NumResiduals());
  gpu_ba::SnapshotWriteResult ignored_write;
  if (!CaptureSnapshotIfRequested(reconstruction, effective, false, 1,
                                  &comparison->legacy, &ignored_write)) {
    *error = "legacy comparison snapshot construction failed";
    return false;
  }

  problem_.reset();
  snapshot_recorder_.reset();
  active_spec_builder_.reset(new gpu_ba::ActiveBaSolveSpecBuilder());
  camera_ids_.clear();
  point3D_num_observations_.clear();
  enumerate(nullptr);
  if (!active_spec_builder_->Finalize(
          options_, effective, config_, *reconstruction, camera_ids_,
          point3D_num_observations_, ba_kind, 1, 1,
          reconstruction->StructureRevision(), &comparison->active, error)) {
    return false;
  }
  return gpu_ba::CompareActiveBaSolveSpecToSnapshot(
      comparison->active, comparison->legacy, error);
}
#endif

bool BundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!solve_called_) << "Cannot use the same BundleAdjuster multiple times";
  solve_called_ = true;
  problem_.reset();
#ifdef GPU_BA_ENABLED
  snapshot_recorder_.reset();
  active_spec_builder_.reset();
#endif
  ceres_cost_function_creations_ = 0;
  ceres_add_residual_calls_ = 0;
  summary_ = ceres::Solver::Summary();
  execution_result_ = BundleAdjustmentExecutionResult();
  const auto solve_start = std::chrono::steady_clock::now();
  const bool custom_cuda_requested = options_.ba_backend == "custom_cuda";
  const bool active_spec_requested =
      options_.ba_cuda_problem_source ==
      gpu_ba::CudaProblemSource::kActiveSpec;
  const bool indexed_catalog_requested =
      options_.ba_cuda_problem_source ==
      gpu_ba::CudaProblemSource::kIndexedCatalog;
  const bool native_graph_requested =
      options_.ba_cuda_problem_source ==
      gpu_ba::CudaProblemSource::kNativeGraph;
  const bool typed_problem_requested =
      active_spec_requested || indexed_catalog_requested ||
      native_graph_requested;
#ifdef GPU_BA_CUDA_ENABLED
  const bool custom_cuda_compiled = true;
#else
  const bool custom_cuda_compiled = false;
#endif
  const std::vector<SelectedQvecState> pre_setup_qvecs =
      custom_cuda_requested ? CaptureSelectedQvecs(config_, *reconstruction)
                            : std::vector<SelectedQvecState>();
  execution_result_.transaction_qvec_count = pre_setup_qvecs.size();
  BaParameterCheckpoint parameter_checkpoint;
  bool checkpoint_ready = false;
#ifdef GPU_BA_CUDA_ENABLED
  gpu_ba::PreparedHostSolveView prepared_host_view;
  gpu_ba::PreparedIndexedActiveSolve prepared_indexed_solve;
  gpu_ba::PreparedNativeActiveSolve prepared_native_solve;
  gpu_ba::NativeActiveSolveInputs native_inputs;
  bool host_store_device_cleanup_failed = false;
#endif
#ifdef GPU_BA_ENABLED
  gpu_ba::ActiveBaSolveSpec active_spec;
#endif
  const auto restore_transaction = [&](const bool entry,
                                       std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    const bool ok = checkpoint_ready
        ? RestoreParameterCheckpoint(parameter_checkpoint, entry, problem_.get(),
                                     reconstruction, error)
        : RestoreSelectedQvecs(pre_setup_qvecs, reconstruction, error);
    execution_result_.transaction_restore_milliseconds +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    return ok;
  };
  const auto finish = [&](const bool success) {
#ifdef GPU_BA_CUDA_ENABLED
    if (prepared_native_solve.valid()) {
      std::string complete_error;
      if (!prepared_native_solve.Complete(&complete_error) &&
          execution_result_.diagnostic_message.empty()) {
        execution_result_.diagnostic_message = complete_error;
      }
    }
    if (prepared_indexed_solve.valid()) {
      std::string complete_error;
      if (!prepared_indexed_solve.Complete(
              false, host_store_device_cleanup_failed, &complete_error) &&
          execution_result_.diagnostic_message.empty()) {
        execution_result_.diagnostic_message = complete_error;
      }
      CopyHostStoreCommitStateToExecution(
          prepared_indexed_solve.runtime_info(), &execution_result_);
    }
    if (prepared_host_view.valid()) {
      std::string complete_error;
      if (!prepared_host_view.Complete(
              false, host_store_device_cleanup_failed, &complete_error) &&
          execution_result_.diagnostic_message.empty()) {
        execution_result_.diagnostic_message = complete_error;
      }
      CopyHostStoreCommitStateToExecution(prepared_host_view.runtime_info(),
                                          &execution_result_);
    }
#endif
    execution_result_.success = success;
    execution_result_.ceres_problem_created = problem_ != nullptr;
    execution_result_.ceres_cost_function_creations =
        ceres_cost_function_creations_;
    execution_result_.ceres_add_residual_calls = ceres_add_residual_calls_;
    execution_result_.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      solve_start).count();
    WriteExecutionTelemetry(options_.ba_telemetry_path, execution_result_);
    return success;
  };

  static std::atomic<uint64_t> next_ba_call_index{0};
  const uint64_t ba_call_index = ++next_ba_call_index;
  execution_result_.call_index = ba_call_index;
  execution_result_.registered_images = reconstruction->NumRegImages();
  execution_result_.requested_backend = options_.ba_backend;
  execution_result_.execution_profile =
      BundleAdjustmentCudaExecutionProfileName(
          options_.ba_cuda_execution_profile);
  execution_result_.audit_profile_requested = options_.ba_cuda_audit_profile;
  execution_result_.arithmetic_precision_requested =
      options_.ba_cuda_arithmetic_precision;
  execution_result_.hessian_backend_requested =
      options_.ba_cuda_hessian_assembly_backend;
  execution_result_.hot_kernel_requested = options_.ba_cuda_hot_kernel_mode;
  execution_result_.schur_contribution_backend_requested =
      options_.ba_cuda_schur_contribution_backend;
  execution_result_.host_store_mode_requested =
      options_.ba_cuda_host_problem_store;
  execution_result_.problem_source_requested =
      gpu_ba::CudaProblemSourceName(options_.ba_cuda_problem_source);
  execution_result_.problem_source_effective =
      execution_result_.problem_source_requested;
  execution_result_.trigger_image_id = options_.ba_trigger_image_id;
  execution_result_.refinement_index = options_.ba_refinement_index;
  if (optimize_phrase_ == OptimazePhrase::Local) {
    execution_result_.ba_kind = "local";
  } else if (optimize_phrase_ == OptimazePhrase::WholeMap) {
    execution_result_.ba_kind = "whole";
  } else {
    execution_result_.ba_kind = "global";
  }

  const bool unsupported_backend = options_.ba_backend != "ceres_cpu" &&
                                   !custom_cuda_requested;
  if (unsupported_backend && !options_.ba_fallback_to_ceres) {
    SetStableError(&execution_result_, StableBaError::kUnsupportedBackend);
    execution_result_.termination = "failure";
    LOG(ERROR) << "BA backend " << options_.ba_backend
               << " is not implemented and fallback is disabled";
    return finish(false);
  }

  if (custom_cuda_requested && !custom_cuda_compiled &&
      !options_.ba_fallback_to_ceres) {
    execution_result_.executed_backend = "none";
    SetStableError(&execution_result_, StableBaError::kUnsupportedConfiguration);
    execution_result_.diagnostic_message =
        "custom_cuda is not compiled into this binary";
    execution_result_.termination = "failure";
    return finish(false);
  }

#ifdef GPU_BA_ENABLED
  gpu_ba::BaKind ba_kind = gpu_ba::BaKind::kGlobal;
  if (optimize_phrase_ == OptimazePhrase::Local) {
    ba_kind = gpu_ba::BaKind::kLocal;
  } else if (optimize_phrase_ == OptimazePhrase::WholeMap) {
    ba_kind = gpu_ba::BaKind::kWhole;
  }
  execution_result_.ba_kind = gpu_ba::BaKindName(ba_kind);
  std::string capture_error;
  const bool capture_snapshot = gpu_ba::ShouldCaptureSnapshot(
      options_.ba_snapshot_capture, options_.ba_snapshot_registered_images,
      ba_kind, reconstruction->NumRegImages(), &capture_error);
  if (!capture_error.empty()) {
    LOG(ERROR) << capture_error;
    SetStableError(&execution_result_,
                   StableBaError::kSnapshotCaptureConfiguration);
    execution_result_.termination = "failure";
    return finish(false);
  }
  if (typed_problem_requested &&
      (!custom_cuda_requested || !custom_cuda_compiled || capture_snapshot ||
       options_.ba_snapshot_capture != "none" ||
       !options_.ba_ceres_oracle_dir.empty() ||
       !options_.ba_compare_dir.empty())) {
    execution_result_.problem_source_effective = "invalid";
    execution_result_.problem_source_fallback_reason =
        (indexed_catalog_requested || native_graph_requested)
            ? "INDEXED_CATALOG_REQUIRES_CUSTOM_CUDA_NO_CAPTURE_NO_ORACLE"
            : "ACTIVE_SPEC_REQUIRES_CUSTOM_CUDA_NO_CAPTURE_NO_ORACLE";
    execution_result_.executed_backend = "none";
    SetStableError(&execution_result_, StableBaError::kUnsupportedConfiguration);
    execution_result_.diagnostic_message =
        std::string(gpu_ba::CudaProblemSourceName(
            options_.ba_cuda_problem_source)) +
        " requires custom_cuda, snapshot_capture=none, and oracle outputs "
        "disabled";
    execution_result_.termination = "failure";
    return finish(false);
  }
  if ((indexed_catalog_requested || native_graph_requested) &&
      cuda_host_store_binding_.mode !=
          gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore) {
    execution_result_.problem_source_effective = "invalid";
    execution_result_.problem_source_fallback_reason =
        native_graph_requested
            ? "NATIVE_GRAPH_REQUIRES_MAPPER_HOST_STORE"
            : "INDEXED_CATALOG_REQUIRES_MAPPER_HOST_STORE";
    execution_result_.executed_backend = "none";
    SetStableError(&execution_result_, StableBaError::kUnsupportedConfiguration);
    execution_result_.diagnostic_message =
        std::string(gpu_ba::CudaProblemSourceName(
            options_.ba_cuda_problem_source)) +
        " requires a Mapper-owned host_prepared_store binding";
    execution_result_.termination = "failure";
    return finish(false);
  }
  if (!typed_problem_requested) {
    problem_ = std::make_unique<ceres::Problem>();
    execution_result_.ceres_problem_created = true;
  }
  if (!typed_problem_requested &&
      (capture_snapshot || (custom_cuda_requested && custom_cuda_compiled))) {
    if (capture_snapshot && options_.ba_snapshot_dir.empty()) {
      LOG(ERROR) << "ba_snapshot_dir is required when snapshot capture is enabled";
      SetStableError(&execution_result_,
                     StableBaError::kSnapshotDirectoryRequired);
      execution_result_.termination = "failure";
      return finish(false);
    }
    snapshot_recorder_.reset(new gpu_ba::SnapshotRecorder(
        cuda_host_store_binding_.mode ==
        gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore));
    execution_result_.snapshot_recorder_created = true;
  }
  if (typed_problem_requested) {
    active_spec_builder_.reset(new gpu_ba::ActiveBaSolveSpecBuilder());
  }
#else
  if (typed_problem_requested) {
    execution_result_.problem_source_effective = "invalid";
    execution_result_.problem_source_fallback_reason =
        indexed_catalog_requested
            ? "INDEXED_CATALOG_NOT_COMPILED"
            : (native_graph_requested ? "NATIVE_GRAPH_NOT_COMPILED"
                                      : "ACTIVE_SPEC_NOT_COMPILED");
    SetStableError(&execution_result_, StableBaError::kUnsupportedConfiguration);
    execution_result_.termination = "failure";
    return finish(false);
  }
  problem_ = std::make_unique<ceres::Problem>();
  execution_result_.ceres_problem_created = true;
#endif

  ceres::LossFunction* loss_function =
      typed_problem_requested ? nullptr : options_.CreateLossFunction();
  const auto problem_source_prepare_start = std::chrono::steady_clock::now();
  const auto setup_start = std::chrono::steady_clock::now();
  if(options_.if_add_lidar_constraint && optimize_phrase_ == OptimazePhrase::Local) {
    SetUpLocalByLidar(reconstruction, loss_function);
  } else if (options_.if_add_lidar_constraint && optimize_phrase_ == OptimazePhrase::Global) {
    SetUpGlobalByLidar(reconstruction, loss_function);
  } else if (options_.if_add_lidar_constraint && optimize_phrase_ == OptimazePhrase::WholeMap) {
    SetUpAdjustWholeMapByLidar(reconstruction, loss_function);
  } else if (!options_.if_add_lidar_constraint) {
    SetUp(reconstruction, loss_function);
  } else {
    std::cout<<"The correct optimization type is missing, error occurred."<<std::endl;
  }
  execution_result_.bundle_adjuster_setup_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - setup_start).count();

  const int scalar_residual_count = typed_problem_requested
#ifdef GPU_BA_ENABLED
      ? static_cast<int>(active_spec_builder_->ScalarResidualCount())
#else
      ? 0
#endif
      : problem_->NumResiduals();
  if (scalar_residual_count == 0) {
    SetStableError(&execution_result_, StableBaError::kEmptyProblem);
    execution_result_.termination = "failure";
    std::string restore_error;
    if (custom_cuda_requested &&
        !restore_transaction(true, &restore_error)) {
      SetStableError(&execution_result_,
                     StableBaError::kTransactionRestoreFailed);
      execution_result_.diagnostic_message = restore_error;
    }
    return finish(false);
  }

  ceres::Solver::Options solver_options =
      CreateEffectiveBundleAdjustmentSolverOptions(
          options_, config_.NumImages(), scalar_residual_count);
#ifdef GPU_BA_ENABLED
  if (typed_problem_requested) {
    const auto active_spec_start = std::chrono::steady_clock::now();
    std::string active_spec_error;
    const bool forced_active_spec_failure =
        g_bundle_adjustment_failure_mode_for_testing.load() ==
        static_cast<int>(
            BundleAdjuster::FailureModeForTesting::kActiveSpecBuildFailure);
    if (forced_active_spec_failure ||
        !active_spec_builder_->Finalize(
            options_, solver_options, config_, *reconstruction, camera_ids_,
            point3D_num_observations_, ba_kind, ba_call_index,
            cuda_host_store_binding_.owner_epoch,
            reconstruction->StructureRevision(), &active_spec,
            &active_spec_error)) {
      if (forced_active_spec_failure) {
        active_spec_error = "forced active spec build failure for testing";
      }
      SetStableError(&execution_result_, StableBaError::kSnapshotFinalizeFailed);
      execution_result_.diagnostic_message = active_spec_error;
      execution_result_.termination = "failure";
      std::string restore_error;
      if (!restore_transaction(true, &restore_error)) {
        SetStableError(&execution_result_,
                       StableBaError::kTransactionRestoreFailed);
        execution_result_.diagnostic_message = restore_error;
      }
      return finish(false);
    }
    execution_result_.active_spec_build_calls = 1;
    execution_result_.active_spec_build_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - active_spec_start).count();
    execution_result_.residual_enumerator_passes =
        active_spec.residual_enumerator_passes;
    execution_result_.residual_enumerator_items =
        active_spec.residual_enumerator_items;
    execution_result_.residuals = active_spec.scalar_residual_count;
    execution_result_.residual_blocks = active_spec.residual_block_count;
    execution_result_.parameter_blocks = active_spec.parameter_blocks.size();
    execution_result_.parameters = active_spec.ambient_parameter_count;
    execution_result_.effective_parameters =
        active_spec.effective_parameter_count;
    execution_result_.host_store_descriptor_identity =
        active_spec.problem.prepared_host_topology_identity;
    execution_result_.host_store_descriptor_items =
        active_spec.problem.prepared_host_descriptor_items;
    execution_result_.host_store_descriptor_hash_updates =
        active_spec.problem.prepared_host_descriptor_hash_updates;
  } else
#endif
  {
    execution_result_.residuals = problem_->NumResiduals();
    execution_result_.residual_blocks = problem_->NumResidualBlocks();
    execution_result_.parameter_blocks = problem_->NumParameterBlocks();
    execution_result_.parameters = problem_->NumParameters();
    std::vector<double*> legacy_parameter_blocks;
    problem_->GetParameterBlocks(&legacy_parameter_blocks);
    for (double* block : legacy_parameter_blocks) {
      if (!problem_->IsParameterBlockConstant(block)) {
        execution_result_.effective_parameters +=
            problem_->ParameterBlockLocalSize(block);
      }
    }
  }

  if (custom_cuda_requested) {
    const auto checkpoint_start = std::chrono::steady_clock::now();
    std::string checkpoint_error;
    const bool checkpoint_ok = typed_problem_requested
#ifdef GPU_BA_ENABLED
        ? BuildActiveSpecParameterCheckpoint(
              pre_setup_qvecs, active_spec, reconstruction,
              &parameter_checkpoint, &checkpoint_error)
#else
        ? false
#endif
        : BuildParameterCheckpoint(pre_setup_qvecs, config_, camera_ids_,
                                   point3D_num_observations_, *problem_,
                                   reconstruction, &parameter_checkpoint,
                                   &checkpoint_error);
    if (!checkpoint_ok) {
      SetStableError(&execution_result_,
                     StableBaError::kTransactionCheckpointFailed);
      execution_result_.diagnostic_message = checkpoint_error;
      execution_result_.termination = "failure";
      std::string restore_error;
      if (!RestoreSelectedQvecs(pre_setup_qvecs, reconstruction,
                                &restore_error)) {
        SetStableError(&execution_result_,
                       StableBaError::kTransactionRestoreFailed);
        execution_result_.diagnostic_message = restore_error;
      }
      return finish(false);
    }
    checkpoint_ready = true;
    execution_result_.transaction_parameter_block_count =
        parameter_checkpoint.blocks.size();
    execution_result_.transaction_backup_entity_count =
        parameter_checkpoint.entity_count;
    execution_result_.transaction_backup_bytes =
        parameter_checkpoint.backup_bytes +
        pre_setup_qvecs.size() * 3 * 4 * sizeof(double);
    execution_result_.transaction_identity_index_entries =
        parameter_checkpoint.identity_index_entries;
    execution_result_.transaction_parameter_lookup_count =
        parameter_checkpoint.parameter_lookup_count;
    execution_result_.transaction_checkpoint_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - checkpoint_start).count();
    if (typed_problem_requested) {
      execution_result_.active_spec_checkpoint_milliseconds =
          execution_result_.transaction_checkpoint_milliseconds;
    }
  }

#ifdef GPU_BA_ENABLED
  gpu_ba::Snapshot captured_snapshot;
  gpu_ba::SnapshotWriteResult captured_snapshot_result;
  gpu_ba::CeresFidelityRecord fidelity_record;
  bool capture_fidelity_oracle = false;
  if (snapshot_recorder_ != nullptr) {
    ++execution_result_.snapshot_materialization_calls;
    const auto snapshot_materialization_start =
        std::chrono::steady_clock::now();
    if (!CaptureSnapshotIfRequested(reconstruction, solver_options,
                                    capture_snapshot, ba_call_index,
                                    &captured_snapshot,
                                    &captured_snapshot_result)) {
      SetStableError(&execution_result_, StableBaError::kSnapshotFinalizeFailed);
      execution_result_.termination = "failure";
      std::string restore_error;
      if (!restore_transaction(true, &restore_error)) {
        SetStableError(&execution_result_,
                       StableBaError::kTransactionRestoreFailed);
        execution_result_.diagnostic_message = restore_error;
      }
      return finish(false);
    }
    execution_result_.snapshot_materialization_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            snapshot_materialization_start).count();
    execution_result_.host_store_descriptor_identity =
        captured_snapshot.prepared_host_topology_identity;
    execution_result_.host_store_descriptor_items =
        captured_snapshot.prepared_host_descriptor_items;
    execution_result_.host_store_descriptor_hash_updates =
        captured_snapshot.prepared_host_descriptor_hash_updates;
    capture_fidelity_oracle = capture_snapshot &&
                              !options_.ba_ceres_oracle_dir.empty();
    if (capture_fidelity_oracle) {
      fidelity_record.record_kind = "original_bundle_adjuster";
      fidelity_record.run_id = options_.ba_ceres_oracle_run_id;
      if (!gpu_ba::CaptureFidelityProvenance(&fidelity_record.provenance,
                                             &capture_error)) {
        LOG(ERROR) << "Ceres fidelity provenance failed: " << capture_error;
        return false;
      }
      fidelity_record.snapshot_id = captured_snapshot.metadata.snapshot_id;
      fidelity_record.snapshot_manifest_path =
          captured_snapshot_result.manifest_path;
      fidelity_record.snapshot_payload_path =
          captured_snapshot_result.payload_path;
      fidelity_record.snapshot_payload_sha256 =
          captured_snapshot_result.integrity.payload_sha256;
      fidelity_record.snapshot_manifest_sha256 =
          captured_snapshot_result.integrity.manifest_sha256;
      fidelity_record.effective_options = gpu_ba::CaptureEffectiveCeresOptions(
          options_.solver_options, solver_options,
          options_.min_num_residuals_for_multi_threading);
      const auto constraints = BuildFidelityParameterConstraints(
          captured_snapshot, options_, *reconstruction);
      const std::string loss_name = FidelityLossName(options_.loss_function_type);
      std::vector<gpu_ba::LossSpecificationSnapshot> losses;
      gpu_ba::LossSpecificationSnapshot visual_loss;
      visual_loss.residual_class = "visual";
      visual_loss.type = loss_name;
      visual_loss.scale = options_.loss_function_scale;
      visual_loss.residual_block_count = captured_snapshot.observations.size();
      losses.push_back(visual_loss);
      gpu_ba::LossSpecificationSnapshot lidar_loss;
      lidar_loss.residual_class = "lidar";
      lidar_loss.type = loss_name;
      lidar_loss.scale = options_.loss_function_scale;
      lidar_loss.residual_block_count = captured_snapshot.lidar.size();
      losses.push_back(lidar_loss);
      fidelity_record.problem = gpu_ba::BuildFidelityProblemSnapshot(
          captured_snapshot, captured_snapshot_result.integrity,
          fidelity_record.effective_options, config_.NumImages(),
          problem_->NumResiduals(), problem_->NumParameterBlocks(),
          problem_->NumParameters(), constraints, losses, false);
    }
  }
#endif

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

#ifdef GPU_BA_CUDA_ENABLED
  if (custom_cuda_requested) {
    execution_result_.selected_schur = "transformed";
    execution_result_.summary_source = "custom_cuda_projection";
    std::string cuda_error;
    const gpu_ba::CudaSolveProblem& cuda_problem =
        typed_problem_requested ? active_spec.problem
                              : static_cast<const gpu_ba::CudaSolveProblem&>(
                                    captured_snapshot);
    bool preflight_ok = ValidateCustomCudaProductionSupport(
        options_, solver_options, cuda_problem, &cuda_error);
    gpu_ba::CudaFullLmResult cuda_result;
    gpu_ba::BaSolveResult native_result;
    bool cuda_ok = false;
    double cuda_wall = 0.0;
    if (preflight_ok) {
      const auto host_store_and_cuda_start = std::chrono::steady_clock::now();
      gpu_ba::CudaFullLmOptions cuda_options =
          CreateProductionCudaOptions(options_, solver_options);
      gpu_ba::CudaHostProblemStoreRuntimeInfo host_store_runtime;
      if (native_graph_requested) {
        if (!gpu_ba::MakeNativeActiveSolveInputs(
                active_spec, &native_inputs, &cuda_error) ||
            !gpu_ba::PrepareCudaNativeActiveSolve(
                native_inputs, cuda_options, cuda_host_store_binding_,
                &prepared_native_solve, &cuda_error)) {
          preflight_ok = false;
        }
      } else if (indexed_catalog_requested) {
        if (!gpu_ba::PrepareCudaIndexedActiveSolve(
                cuda_problem, cuda_options, cuda_host_store_binding_,
                &prepared_indexed_solve, &cuda_error)) {
          preflight_ok = false;
        } else {
          cuda_options.indexed_active_solve =
              prepared_indexed_solve.descriptor();
          cuda_options.indexed_catalog_tables =
              prepared_indexed_solve.catalog_tables();
        }
        CopyHostStoreRuntimeToExecution(prepared_indexed_solve.runtime_info(),
                                        &execution_result_);
      } else if (cuda_host_store_binding_.mode !=
                 gpu_ba::CudaHostProblemStoreMode::kDisabled) {
        if (!gpu_ba::PrepareCudaHostSolveView(
                cuda_problem, cuda_options, cuda_host_store_binding_,
                &prepared_host_view, &host_store_runtime, &cuda_error)) {
          preflight_ok = false;
          CopyHostStoreRuntimeToExecution(host_store_runtime,
                                          &execution_result_);
        } else {
          cuda_options.prepared_host_view = &prepared_host_view;
          CopyHostStoreRuntimeToExecution(host_store_runtime,
                                          &execution_result_);
        }
      }
      if (preflight_ok) {
        const auto cuda_start = std::chrono::steady_clock::now();
        const int failure_mode =
            g_bundle_adjustment_failure_mode_for_testing.load();
        if (failure_mode ==
                static_cast<int>(FailureModeForTesting::kCustomCudaFailure) ||
            failure_mode == static_cast<int>(
                                FailureModeForTesting::
                                    kCustomCudaThenCeresFailure) ||
            failure_mode == static_cast<int>(
                                FailureModeForTesting::
                                    kRestoreValidationFailure)) {
          cuda_error = "forced \"custom CUDA\" failure\nfor testing";
          cuda_result.initial_cost = std::numeric_limits<double>::quiet_NaN();
          cuda_result.final_cost = std::numeric_limits<double>::infinity();
        } else {
          if (native_graph_requested) {
            cuda_ok = gpu_ba::RunCustomCudaSolve(
                prepared_native_solve.request(), &native_result, &cuda_error);
            cuda_result.success = native_result.success;
            cuda_result.error = native_result.error;
            cuda_result.termination_reason = native_result.termination_reason;
            cuda_result.initial_cost = native_result.initial_cost;
            cuda_result.final_cost = native_result.final_cost;
            cuda_result.trial_iterations = native_result.trial_iterations;
            cuda_result.accepted_steps = native_result.accepted_steps;
            cuda_result.accepted_decisions = native_result.accepted_decisions;
            cuda_result.accepted_commits = native_result.accepted_commits;
            cuda_result.rejected_steps = native_result.rejected_steps;
            cuda_result.invalid_steps = native_result.invalid_steps;
            cuda_result.runtime.final_internal_state_epoch =
                native_result.final_internal_state_epoch;
          } else {
            cuda_ok = typed_problem_requested
                ? gpu_ba::RunCustomCudaSolve(active_spec.problem, cuda_options,
                                             &cuda_result, &cuda_error)
                : gpu_ba::RunCustomCudaSolve(captured_snapshot, cuda_options,
                                             &cuda_result, &cuda_error);
          }
        }
        cuda_wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cuda_start).count();
        execution_result_.custom_cuda_caller_wall_seconds = cuda_wall;
      }
      execution_result_.host_store_prepare_and_cuda_wall_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() -
              host_store_and_cuda_start).count();
    }
    execution_result_.problem_source_prepare_and_cuda_wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      problem_source_prepare_start).count();
    if (typed_problem_requested) {
      execution_result_.fast_cpu_preparation_milliseconds =
          execution_result_.bundle_adjuster_setup_milliseconds +
          execution_result_.active_spec_build_milliseconds +
          execution_result_.host_store_preparation_total_milliseconds +
          execution_result_.active_spec_checkpoint_milliseconds;
    } else {
      execution_result_.legacy_cpu_preparation_milliseconds =
          execution_result_.bundle_adjuster_setup_milliseconds +
          execution_result_.snapshot_materialization_milliseconds +
          execution_result_.host_store_preparation_total_milliseconds +
          execution_result_.transaction_checkpoint_milliseconds;
    }
    execution_result_.executed_backend = "custom_cuda";
    execution_result_.trial_steps = cuda_result.trial_iterations;
    execution_result_.accepted_steps = cuda_result.accepted_steps;
    execution_result_.accepted_commits = cuda_result.accepted_commits;
    execution_result_.rejected_steps = cuda_result.rejected_steps;
    execution_result_.invalid_steps = cuda_result.invalid_steps;
    execution_result_.initial_cost = cuda_result.initial_cost;
    execution_result_.final_cost = cuda_result.final_cost;
    execution_result_.peak_cuda_bytes =
        cuda_result.runtime.persistent_device.peak_resident_bytes;
    host_store_device_cleanup_failed = native_graph_requested
        ? native_result.resource_cleanup_failed
        : (cuda_result.error_classification ==
               gpu_ba::CudaSolveErrorClass::kResourceCleanup ||
           cuda_result.runtime.resource_cleanup_failures != 0);
    if (cuda_host_store_binding_.mode ==
            gpu_ba::CudaHostProblemStoreMode::kDisabled ||
        cuda_result.runtime.host_problem_store.mode_requested != "disabled") {
      CopyHostStoreRuntimeToExecution(
          cuda_result.runtime.host_problem_store, &execution_result_);
    }
    const auto& precision_runtime = cuda_result.runtime.persistent_device;
    execution_result_.audit_profile_effective =
        cuda_result.runtime.audit_profile_effective;
    execution_result_.capture_state_trace_effective =
        cuda_result.runtime.capture_state_trace_effective;
    execution_result_.instrumentation_effective =
        cuda_result.runtime.instrumentation_effective;
    execution_result_.production_audit_invariants_checked =
        cuda_result.runtime.production_audit_invariants_checked;
    execution_result_.production_audit_invariants_pass =
        cuda_result.runtime.production_audit_invariants_pass;
    execution_result_.production_audit_violation_count =
        cuda_result.runtime.production_audit_violation_count;
    execution_result_.first_production_audit_violation =
        cuda_result.runtime.first_production_audit_violation;
    execution_result_.state_hash_computations =
        cuda_result.runtime.state_hash_audit.current.computations +
        cuda_result.runtime.state_hash_audit.trial.computations;
    execution_result_.topology_fingerprint_computations =
        cuda_result.runtime.topology_fingerprint_computations;
    execution_result_.audit_mirror_bytes =
        precision_runtime.audit_mirror_b_bytes +
        precision_runtime.audit_mirror_state_bytes +
        precision_runtime.audit_mirror_schur_bytes +
        precision_runtime.audit_mirror_delta_bytes;
    execution_result_.optional_full_array_d2h_bytes =
        precision_runtime.host_diagnostics_d2h_bytes +
        precision_runtime.bootstrap_scaling_full_d2h_bytes +
        precision_runtime.steady_full_b_d2h_bytes +
        precision_runtime.steady_schur_d2h_bytes +
        precision_runtime.steady_delta_d2h_bytes +
        precision_runtime.steady_trial_state_d2h_bytes +
        precision_runtime.mixed_full_array_d2h_bytes;
    execution_result_.mixed_double_edge_materialization_bytes =
        precision_runtime.mixed_double_edge_materialization_bytes;
    execution_result_.cuda_host_build_cuda_layer_a_inputs_calls =
        precision_runtime.host_build_cuda_layer_a_inputs_calls;
    execution_result_.cuda_host_build_static_layout_calls =
        precision_runtime.host_build_static_layout_calls;
    execution_result_.cuda_host_build_cost_layout_calls =
        precision_runtime.host_build_cost_layout_calls;
    execution_result_.cuda_host_build_layer_b_topology_calls =
        precision_runtime.host_build_layer_b_topology_calls;
    execution_result_.cuda_host_build_layer_c_topology_calls =
        precision_runtime.host_build_layer_c_topology_calls;
    execution_result_.indexed_device_catalog_lookup_calls =
        precision_runtime.indexed_device_catalog_lookup_calls;
    execution_result_.indexed_device_catalog_reuse_calls =
        precision_runtime.indexed_device_catalog_reuse_calls;
    execution_result_.indexed_device_catalog_revision_rebinds =
        precision_runtime.indexed_device_catalog_revision_rebinds;
    execution_result_.indexed_device_catalog_full_upload_calls =
        precision_runtime.indexed_device_catalog_full_upload_calls;
    execution_result_.indexed_device_catalog_full_upload_bytes =
        precision_runtime.indexed_device_catalog_full_upload_bytes;
    execution_result_.indexed_device_catalog_patch_upload_calls =
        precision_runtime.indexed_device_catalog_patch_upload_calls;
    execution_result_.indexed_device_catalog_patch_upload_bytes =
        precision_runtime.indexed_device_catalog_patch_upload_bytes;
    execution_result_.indexed_device_catalog_invalidations =
        precision_runtime.indexed_device_catalog_invalidations;
    execution_result_.indexed_device_catalog_prefix_bytes =
        precision_runtime.indexed_device_catalog_prefix_bytes;
    execution_result_.indexed_device_catalog_arena_generation =
        precision_runtime.indexed_device_catalog_arena_generation;
    execution_result_.arithmetic_precision_effective =
        precision_runtime.arithmetic_precision_effective;
    execution_result_.hessian_backend_effective =
        precision_runtime.hessian_assembly_backend_effective;
    execution_result_.hot_kernel_effective =
        precision_runtime.hot_kernel_transformed
            ? "transformed"
            : (precision_runtime.hot_kernel_optimized ? "optimized"
                                                       : "reference");
    execution_result_.schur_contribution_backend_effective =
        precision_runtime.schur_contribution_backend_effective;
    execution_result_.state_storage_precision =
        precision_runtime.state_storage_precision;
    execution_result_.residual_jacobian_precision =
        precision_runtime.residual_jacobian_precision;
    execution_result_.hessian_schur_precision =
        precision_runtime.hessian_schur_precision;
    execution_result_.factorization_routine =
        precision_runtime.factorization_routine;
    execution_result_.delta_precision = precision_runtime.delta_precision;
    execution_result_.quaternion_plus_precision =
        precision_runtime.quaternion_plus_precision;
    execution_result_.cost_precision = precision_runtime.cost_precision;
    execution_result_.controller_precision =
        precision_runtime.controller_precision;
    for (const auto& iteration : cuda_result.trace) {
      execution_result_.max_backward_error = std::max(
          execution_result_.max_backward_error, iteration.backward_error);
    }
    if (native_graph_requested) {
      execution_result_.max_backward_error =
          native_result.max_backward_error;
    }

    const double cost_tolerance =
        std::max(1e-8, 1e-10 * std::max(1.0, cuda_result.initial_cost));
    bool numerical_contract =
        cuda_ok && cuda_result.success &&
        std::isfinite(cuda_result.initial_cost) &&
        std::isfinite(cuda_result.final_cost) &&
        cuda_result.initial_cost >= 0.0 && cuda_result.final_cost >= 0.0 &&
        cuda_result.final_cost <= cuda_result.initial_cost + cost_tolerance &&
        (cuda_result.accepted_commits == 0 ||
         cuda_result.final_cost < cuda_result.initial_cost) &&
        cuda_result.accepted_decisions == cuda_result.accepted_commits &&
        (!native_graph_requested ||
         native_result.backward_error_samples != 0) &&
        std::isfinite(execution_result_.max_backward_error);
    if (!numerical_contract && cuda_error.empty()) {
      cuda_error = "CUDA_NUMERICAL_ACCEPTANCE_CONTRACT";
    }

    if (numerical_contract) {
      if (!native_graph_requested &&
          g_bundle_adjustment_failure_mode_for_testing.load() ==
              static_cast<int>(FailureModeForTesting::kCudaTopologyMismatch) &&
          !cuda_result.final_state.images.empty()) {
        ++cuda_result.final_state.images.front().camera_id;
      }
      std::string commit_error;
      const bool commit_ok = native_graph_requested
          ? gpu_ba::ValidateAndCommitNativeBaState(
                native_inputs, prepared_native_solve,
                native_result.final_state, reconstruction, &commit_error)
          : (typed_problem_requested
                 ? gpu_ba::ValidateAndCommitActiveBaState(
                       active_spec, cuda_result.final_state, reconstruction,
                       &commit_error)
                 : PrepareAndCommitCudaState(captured_snapshot,
                                             cuda_result.final_state,
                                             reconstruction, &commit_error));
      if (!commit_ok) {
        SetStableError(&execution_result_, StableBaError::kCommitIntegrityError);
        execution_result_.diagnostic_message = commit_error;
        execution_result_.termination = "failure";
        std::string restore_error;
        if (!restore_transaction(true, &restore_error)) {
          SetStableError(&execution_result_,
                         StableBaError::kTransactionRestoreFailed);
          execution_result_.diagnostic_message = restore_error;
        }
        LOG(ERROR) << "Custom CUDA BA commit failed: " << commit_error;
        return finish(false);
      }
      if (prepared_indexed_solve.valid()) {
        std::string complete_error;
        if (!prepared_indexed_solve.Complete(
                true, host_store_device_cleanup_failed, &complete_error)) {
          SetStableError(&execution_result_,
                         StableBaError::kCommitIntegrityError);
          execution_result_.diagnostic_message = complete_error;
          execution_result_.termination = "failure";
          std::string restore_error;
          if (!restore_transaction(true, &restore_error)) {
            SetStableError(&execution_result_,
                           StableBaError::kTransactionRestoreFailed);
            execution_result_.diagnostic_message = restore_error;
          }
          return finish(false);
        }
        CopyHostStoreCommitStateToExecution(
            prepared_indexed_solve.runtime_info(), &execution_result_);
      }
      if (prepared_native_solve.valid()) {
        std::string complete_error;
        if (!prepared_native_solve.Complete(&complete_error)) {
          SetStableError(&execution_result_,
                         StableBaError::kCommitIntegrityError);
          execution_result_.diagnostic_message = complete_error;
          execution_result_.termination = "failure";
          std::string restore_error;
          if (!restore_transaction(true, &restore_error)) {
            SetStableError(&execution_result_,
                           StableBaError::kTransactionRestoreFailed);
            execution_result_.diagnostic_message = restore_error;
          }
          return finish(false);
        }
      }
      if (prepared_host_view.valid()) {
        std::string complete_error;
        if (!prepared_host_view.Complete(
                true, host_store_device_cleanup_failed, &complete_error)) {
          SetStableError(&execution_result_,
                         StableBaError::kCommitIntegrityError);
          execution_result_.diagnostic_message = complete_error;
          execution_result_.termination = "failure";
          std::string restore_error;
          if (!restore_transaction(true, &restore_error)) {
            SetStableError(&execution_result_,
                           StableBaError::kTransactionRestoreFailed);
            execution_result_.diagnostic_message = restore_error;
          }
          return finish(false);
        }
        CopyHostStoreCommitStateToExecution(prepared_host_view.runtime_info(),
                                            &execution_result_);
      }
      ProjectCudaSummary(cuda_problem,
                         static_cast<int>(execution_result_.parameter_blocks),
                         static_cast<int>(execution_result_.parameters),
                         static_cast<int>(execution_result_.residuals),
                         cuda_result, cuda_wall, &summary_);
      execution_result_.termination = cuda_result.termination_reason;
      if (solver_options.minimizer_progress_to_stdout) std::cout << std::endl;
      if (options_.print_summary) {
        PrintHeading2("Bundle adjustment report (custom CUDA)");
        PrintSolverSummary(summary_);
      }
      TearDown(reconstruction);
      return finish(true);
    }

    if (!options_.ba_fallback_to_ceres) {
      SetStableError(&execution_result_,
                     preflight_ok ? StableBaError::kCustomCudaFailed
                                  : StableBaError::kUnsupportedConfiguration);
      execution_result_.diagnostic_message = cuda_error;
      execution_result_.termination = "failure";
      std::string restore_error;
      if (!restore_transaction(true, &restore_error)) {
        SetStableError(&execution_result_,
                       StableBaError::kTransactionRestoreFailed);
        execution_result_.diagnostic_message = restore_error;
      }
      LOG(ERROR) << "Custom CUDA BA failed without fallback: "
                 << execution_result_.stable_error;
      return finish(false);
    }

    if (typed_problem_requested && !preflight_ok) {
      SetStableError(&execution_result_,
                     StableBaError::kUnsupportedConfiguration);
      execution_result_.diagnostic_message = cuda_error;
      execution_result_.termination = "failure";
      std::string restore_error;
      if (!restore_transaction(true, &restore_error)) {
        SetStableError(&execution_result_,
                       StableBaError::kTransactionRestoreFailed);
        execution_result_.diagnostic_message = restore_error;
      }
      return finish(false);
    }

    // CUDA has not committed. Restore the exact post-setup state bound by the
    // existing Ceres Problem and invoke Ceres at most once.
    std::string restore_error;
    if (!restore_transaction(false, &restore_error)) {
      SetStableError(&execution_result_,
                     StableBaError::kTransactionRestoreFailed);
      execution_result_.diagnostic_message = restore_error;
      execution_result_.termination = "failure";
      return finish(false);
    }
    if (typed_problem_requested) {
      problem_ = std::make_unique<ceres::Problem>();
      execution_result_.ceres_problem_created = true;
      active_spec_builder_.reset();
      camera_ids_.clear();
      point3D_num_observations_.clear();
      ceres::LossFunction* fallback_loss = options_.CreateLossFunction();
      const auto fallback_setup_start = std::chrono::steady_clock::now();
      if (options_.if_add_lidar_constraint &&
          optimize_phrase_ == OptimazePhrase::Local) {
        SetUpLocalByLidar(reconstruction, fallback_loss);
      } else if (options_.if_add_lidar_constraint &&
                 optimize_phrase_ == OptimazePhrase::Global) {
        SetUpGlobalByLidar(reconstruction, fallback_loss);
      } else if (options_.if_add_lidar_constraint &&
                 optimize_phrase_ == OptimazePhrase::WholeMap) {
        SetUpAdjustWholeMapByLidar(reconstruction, fallback_loss);
      } else if (!options_.if_add_lidar_constraint) {
        SetUp(reconstruction, fallback_loss);
      }
      execution_result_.bundle_adjuster_setup_milliseconds +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - fallback_setup_start).count();
      execution_result_.problem_source_effective = "legacy_snapshot";
      execution_result_.problem_source_fallback_reason =
          "CUSTOM_CUDA_FAILURE_LAZY_CERES";
    }
    execution_result_.fallback_used = true;
    execution_result_.fallback_reason =
        preflight_ok ? "CUSTOM_CUDA_FAILED" : "UNSUPPORTED_CONFIGURATION";
    execution_result_.diagnostic_message = cuda_error;
    execution_result_.executed_backend = "ceres_cpu";
    execution_result_.summary_source = "ceres";
  } else {
    execution_result_.executed_backend = "ceres_cpu";
    execution_result_.selected_schur =
        ceres::LinearSolverTypeToString(solver_options.linear_solver_type);
    execution_result_.summary_source = "ceres";
    if (unsupported_backend) {
      execution_result_.fallback_used = true;
      execution_result_.fallback_reason = "UNSUPPORTED_BA_BACKEND";
    }
  }
#else
  if (custom_cuda_requested) {
    execution_result_.fallback_used = true;
    execution_result_.fallback_reason = "UNSUPPORTED_CONFIGURATION";
    execution_result_.diagnostic_message =
        "custom_cuda is not compiled into this binary";
  }
  execution_result_.executed_backend = "ceres_cpu";
  execution_result_.selected_schur =
      ceres::LinearSolverTypeToString(solver_options.linear_solver_type);
  execution_result_.summary_source = "ceres";
  if (unsupported_backend) {
    execution_result_.fallback_used = true;
    execution_result_.fallback_reason = "UNSUPPORTED_BA_BACKEND";
  }
#endif

  ++g_bundle_adjustment_ceres_solve_calls;
  ++execution_result_.ceres_solve_calls;
  if (g_bundle_adjustment_failure_mode_for_testing.load() ==
          static_cast<int>(FailureModeForTesting::kCeresFailure) ||
      g_bundle_adjustment_failure_mode_for_testing.load() ==
          static_cast<int>(
              FailureModeForTesting::kCustomCudaThenCeresFailure)) {
    ceres::Solve(solver_options, problem_.get(), &summary_);
    // Ensure the rollback test covers a non-qvec block even if this tiny
    // problem converges before Ceres changes one on its own.
    for (auto& block : parameter_checkpoint.blocks) {
      if (block.kind != CheckpointEntityKind::kQuaternion &&
          block.ambient_size > 0) {
        block.values[0] += 0.125;
        execution_result_.diagnostic_message =
            "forced Ceres failure after non-qvec parameter mutation";
        break;
      }
    }
    summary_.termination_type = ceres::FAILURE;
    summary_.message = "forced Ceres failure for testing";
  } else {
    ceres::Solve(solver_options, problem_.get(), &summary_);
  }

#ifdef GPU_BA_ENABLED
  if (capture_fidelity_oracle) {
    fidelity_record.summary = gpu_ba::CaptureCeresSummary(summary_);
    gpu_ba::Snapshot post_state = captured_snapshot;
    if (!gpu_ba::UpdateSnapshotStateFromReconstruction(
            *reconstruction, &post_state, &capture_error)) {
      LOG(ERROR) << "Ceres fidelity post-state capture failed: "
                 << capture_error;
      return false;
    }
    post_state.metadata.snapshot_id =
        captured_snapshot.metadata.snapshot_id + "-original-" +
        options_.ba_ceres_oracle_run_id + "-post";
    gpu_ba::SnapshotWriteResult post_state_result;
    const std::string post_state_dir =
        JoinPaths(options_.ba_ceres_oracle_dir, "post_states");
    if (!gpu_ba::WriteSnapshot(post_state, post_state_dir, &post_state_result,
                               &capture_error)) {
      LOG(ERROR) << "Ceres fidelity post-state write failed: " << capture_error;
      return false;
    }
    fidelity_record.post_state_manifest_path = post_state_result.manifest_path;
    fidelity_record.post_state_payload_path = post_state_result.payload_path;
    fidelity_record.post_state_payload_sha256 =
        post_state_result.integrity.payload_sha256;
    fidelity_record.final_state_sha256 =
        gpu_ba::CanonicalStateSha256(post_state);
    gpu_ba::CeresFidelityWriteResult oracle_result;
    if (!gpu_ba::WriteCeresFidelityRecord(
            fidelity_record, options_.ba_ceres_oracle_dir, &oracle_result,
            &capture_error)) {
      LOG(ERROR) << "Ceres fidelity oracle write failed: " << capture_error;
      return false;
    }
    std::cout << "Ceres BundleAdjuster oracle: " << oracle_result.prefix_path
              << std::endl
              << "  binary_sha256: " << oracle_result.binary_sha256
              << std::endl
              << "  post_state_sha256: "
              << fidelity_record.final_state_sha256 << std::endl;

    // Optional fidelity-only repeated runs. The first ceres::Solve above
    // remains the production oracle. For every additional sample, restore the
    // exact solve-pre state, reuse the original Problem/effective options, and
    // finally restore A so Mapper observes exactly the first solve's result.
    for (int repeat_index = 1;
         repeat_index < options_.ba_ceres_oracle_repeat_count;
         ++repeat_index) {
      const std::string repeat_name(
          1, static_cast<char>('a' + repeat_index));
      bool repeat_ok = gpu_ba::RestoreReconstructionStateFromSnapshot(
          captured_snapshot, reconstruction, &capture_error);
      ceres::Solver::Summary repeat_summary;
      gpu_ba::CeresFidelityRecord repeat_record = fidelity_record;
      gpu_ba::Snapshot repeat_post_state = captured_snapshot;
      gpu_ba::SnapshotWriteResult repeat_post_state_result;
      gpu_ba::CeresFidelityWriteResult repeat_oracle_result;
      if (repeat_ok) {
        ++g_bundle_adjustment_ceres_solve_calls;
        ++execution_result_.ceres_solve_calls;
        ceres::Solve(solver_options, problem_.get(), &repeat_summary);
        repeat_record.record_kind = "original_bundle_adjuster_repeat";
        repeat_record.run_id = options_.ba_ceres_oracle_run_id +
                               "-repeat-" + repeat_name;
        repeat_record.summary = gpu_ba::CaptureCeresSummary(repeat_summary);
        repeat_ok = gpu_ba::UpdateSnapshotStateFromReconstruction(
            *reconstruction, &repeat_post_state, &capture_error);
      }
      if (repeat_ok) {
        repeat_post_state.metadata.snapshot_id =
            captured_snapshot.metadata.snapshot_id + "-original-" +
            repeat_record.run_id + "-post";
        repeat_ok = gpu_ba::WriteSnapshot(
            repeat_post_state, post_state_dir, &repeat_post_state_result,
            &capture_error);
      }
      if (repeat_ok) {
        repeat_record.post_state_manifest_path =
            repeat_post_state_result.manifest_path;
        repeat_record.post_state_payload_path =
            repeat_post_state_result.payload_path;
        repeat_record.post_state_payload_sha256 =
            repeat_post_state_result.integrity.payload_sha256;
        repeat_record.final_state_sha256 =
            gpu_ba::CanonicalStateSha256(repeat_post_state);
        repeat_ok = gpu_ba::WriteCeresFidelityRecord(
            repeat_record, options_.ba_ceres_oracle_dir,
            &repeat_oracle_result, &capture_error);
      }
      std::string restore_error;
      if (!gpu_ba::RestoreReconstructionStateFromSnapshot(
              post_state, reconstruction, &restore_error)) {
        LOG(ERROR) << "Failed to restore oracle A after fidelity repeat-"
                   << repeat_name << ": " << restore_error;
        return false;
      }
      if (!repeat_ok) {
        LOG(ERROR) << "Ceres fidelity repeat-" << repeat_name
                   << " capture failed: " << capture_error;
        return false;
      }
      std::cout << "Ceres BundleAdjuster repeat-" << repeat_name
                << " oracle: " << repeat_oracle_result.prefix_path
                << std::endl
                << "  binary_sha256: "
                << repeat_oracle_result.binary_sha256 << std::endl
                << "  post_state_sha256: "
                << repeat_record.final_state_sha256 << std::endl;
    }
  }
#endif

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction);
  execution_result_.trial_steps =
      summary_.num_successful_steps + summary_.num_unsuccessful_steps;
  execution_result_.accepted_steps = summary_.num_successful_steps;
  execution_result_.rejected_steps = summary_.num_unsuccessful_steps;
  execution_result_.initial_cost = summary_.initial_cost;
  execution_result_.final_cost = summary_.final_cost;
  execution_result_.termination = ceres::TerminationTypeToString(
      summary_.termination_type);
  const bool usable = summary_.IsSolutionUsable();
  if (!usable && custom_cuda_requested) {
    std::string restore_error;
    if (!restore_transaction(true, &restore_error)) {
      SetStableError(&execution_result_,
                     StableBaError::kTransactionRestoreFailed);
      execution_result_.diagnostic_message = restore_error;
    } else {
      SetStableError(&execution_result_, StableBaError::kCeresFallbackFailed);
      if (!execution_result_.diagnostic_message.empty()) {
        execution_result_.diagnostic_message += "; ";
      }
      execution_result_.diagnostic_message += summary_.message;
    }
    return finish(false);
  }
  if (!usable) {
    SetStableError(&execution_result_, StableBaError::kCeresSolveFailed);
    execution_result_.diagnostic_message = summary_.message;
  }
  return finish(usable);
}

const ceres::Solver::Summary& BundleAdjuster::Summary() const {
  return summary_;
}

const BundleAdjustmentExecutionResult& BundleAdjuster::ExecutionResult() const {
  return execution_result_;
}

uint64_t BundleAdjuster::CeresSolveCallCountForTesting() {
  return g_bundle_adjustment_ceres_solve_calls.load();
}

void BundleAdjuster::ResetCeresSolveCallCountForTesting() {
  g_bundle_adjustment_ceres_solve_calls.store(0);
}

void BundleAdjuster::SetFailureModeForTesting(
    const FailureModeForTesting mode) {
  g_bundle_adjustment_failure_mode_for_testing.store(static_cast<int>(mode));
}

#ifdef GPU_BA_ENABLED
bool BundleAdjuster::CaptureSnapshotIfRequested(
    Reconstruction* reconstruction,
    const ceres::Solver::Options& solver_options,
    bool write_snapshot,
    uint64_t ba_call_index,
    gpu_ba::Snapshot* snapshot,
    gpu_ba::SnapshotWriteResult* result) {
  gpu_ba::BaKind ba_kind = gpu_ba::BaKind::kGlobal;
  if (optimize_phrase_ == OptimazePhrase::Local) {
    ba_kind = gpu_ba::BaKind::kLocal;
  } else if (optimize_phrase_ == OptimazePhrase::WholeMap) {
    ba_kind = gpu_ba::BaKind::kWhole;
  }
  std::string error;
  const bool success = write_snapshot
      ? snapshot_recorder_->FinalizeAndWrite(
            options_, solver_options, config_, *reconstruction, *problem_,
            ba_kind, ba_call_index, snapshot, result, &error)
      : snapshot_recorder_->Finalize(
            options_, solver_options, config_, *reconstruction, *problem_,
            ba_kind, ba_call_index, snapshot, &error);
  if (!success) {
    LOG(ERROR) << "GPU BA snapshot capture failed: " << error;
    return false;
  }
  if (write_snapshot) {
    std::cout << "GPU BA snapshot: " << result->prefix_path << std::endl
              << "  payload_sha256: " << result->integrity.payload_sha256
              << std::endl
              << "  manifest_sha256: " << result->integrity.manifest_sha256
              << std::endl;
  }
  return true;
}
#endif

void BundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::SetUpLocalByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  for (auto iter = config_.lidar_maps_.begin(); iter != config_.lidar_maps_.end(); iter++){
    const auto point3D_id = iter->first;
    AddLidarToProblem(point3D_id, reconstruction, loss_function);
  }

  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::SetUpGlobalByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {

  for (const image_t image_id : config_.Images()) {
      AddImageInSphereToProblem(image_id, reconstruction, loss_function);
  }
  
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  for (auto iter = config_.lidar_maps_.begin(); iter != config_.lidar_maps_.end(); iter++){
    const auto point3D_id = iter->first;
    AddLidarToProblem(point3D_id, reconstruction, loss_function);
  }

  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::SetUpAdjustWholeMapByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {

  for (const image_t image_id : config_.Images()) {
      AddImageToProblem(image_id, reconstruction, loss_function);
  }
  
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  for (auto iter = config_.lidar_maps_.begin(); iter != config_.lidar_maps_.end(); iter++){
    const auto point3D_id = iter->first;
    AddLidarToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::TearDown(Reconstruction*) {
  // Nothing to do
}

void BundleAdjuster::AddImageInSphereToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {

  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  // a ptr to the first state of the vector
  double* camera_params_data = camera.ParamsData();
  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  // Add residuals to bundle adjustment problem.
  size_t num_observations = 0;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    if (!point2D.HasPoint3D()) {
      continue;
    }
    num_observations += 1;
    point3D_num_observations_[point2D.Point3DId()] += 1;
    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    if (!point3D.IfInSphere()) {
      continue;
    }

    assert(point3D.Track().Length() > 1);

    if (constant_pose) {
      if (problem_ != nullptr) {
        ceres::CostFunction* cost_function = nullptr;
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
        ++ceres_cost_function_creations_;
        problem_->AddResidualBlock(cost_function, loss_function,
                                   point3D.XYZ().data(), camera_params_data);
        ++ceres_add_residual_calls_;
      }

#ifdef GPU_BA_ENABLED
      if (snapshot_recorder_ != nullptr) {
        snapshot_recorder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, true);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId(),
            point3D.XYZ().data());
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId(),
            camera_params_data);
      }
      if (active_spec_builder_ != nullptr) {
        active_spec_builder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, true);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId());
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId());
      }
#endif
    } 
    else {
      if (problem_ != nullptr) {
        ceres::CostFunction* cost_function = nullptr;
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
        ++ceres_cost_function_creations_;
        problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                   tvec_data, point3D.XYZ().data(),
                                   camera_params_data);
        ++ceres_add_residual_calls_;
      }

#ifdef GPU_BA_ENABLED
      if (snapshot_recorder_ != nullptr) {
        snapshot_recorder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, false);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kQuaternion, image_id, qvec_data);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kTranslation, image_id, tvec_data);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId(),
            point3D.XYZ().data());
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId(),
            camera_params_data);
      }
      if (active_spec_builder_ != nullptr) {
        active_spec_builder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, false);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kQuaternion, image_id);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kTranslation, image_id);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId());
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId());
      }
#endif
    }
  }

  if (num_observations > 0) {
    camera_ids_.insert(image.CameraId());

    // Set pose parameterization.
    if (!constant_pose && problem_ != nullptr) {

      SetQuaternionManifold(problem_.get(), qvec_data);
      if (config_.HasConstantTvec(image_id)) {
        const std::vector<int>& constant_tvec_idxs =
            config_.ConstantTvec(image_id);
        SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
      }
    }
  }
}

void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();
  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  // Add residuals to bundle adjustment problem.
  size_t num_observations = 0;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    if (!point2D.HasPoint3D()) {
      continue;
    }
    num_observations += 1;
    point3D_num_observations_[point2D.Point3DId()] += 1;
    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);
    if (constant_pose) {
      if (problem_ != nullptr) {
        ceres::CostFunction* cost_function = nullptr;
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
        ++ceres_cost_function_creations_;
        problem_->AddResidualBlock(cost_function, loss_function,
                                   point3D.XYZ().data(), camera_params_data);
        ++ceres_add_residual_calls_;
      }

#ifdef GPU_BA_ENABLED
      if (snapshot_recorder_ != nullptr) {
        snapshot_recorder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, true);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId(),
            point3D.XYZ().data());
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId(),
            camera_params_data);
      }
      if (active_spec_builder_ != nullptr) {
        active_spec_builder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, true);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId());
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId());
      }
#endif
    } 
    else {
      if (problem_ != nullptr) {
        ceres::CostFunction* cost_function = nullptr;
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
        ++ceres_cost_function_creations_;
        problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                   tvec_data, point3D.XYZ().data(),
                                   camera_params_data);
        ++ceres_add_residual_calls_;
      }

#ifdef GPU_BA_ENABLED
      if (snapshot_recorder_ != nullptr) {
        snapshot_recorder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, false);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kQuaternion, image_id, qvec_data);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kTranslation, image_id, tvec_data);
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId(),
            point3D.XYZ().data());
        snapshot_recorder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId(),
            camera_params_data);
      }
      if (active_spec_builder_ != nullptr) {
        active_spec_builder_->RecordVisualResidual(
            image_id, point2D_idx, point2D.Point3DId(),
            {{point2D.X(), point2D.Y()}}, false);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kQuaternion, image_id);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kTranslation, image_id);
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kPoint3D, point2D.Point3DId());
        active_spec_builder_->RecordParameterBlock(
            gpu_ba::ParameterKind::kCamera, image.CameraId());
      }
#endif
    }
  }

  if (num_observations > 0) {
    camera_ids_.insert(image.CameraId());

    // Set pose parameterization.
    if (!constant_pose && problem_ != nullptr) {

      SetQuaternionManifold(problem_.get(), qvec_data);
      if (config_.HasConstantTvec(image_id)) {
        const std::vector<int>& constant_tvec_idxs =
            config_.ConstantTvec(image_id);
        SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
      }
    }
  }
}

void BundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  // track_el is a struct including image_id and point2d_id
  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `FillImages`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_observations_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    if (problem_ != nullptr) {
      ceres::CostFunction* cost_function = nullptr;
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
      ++ceres_cost_function_creations_;
      problem_->AddResidualBlock(cost_function, loss_function,
                                 point3D.XYZ().data(), camera.ParamsData());
      ++ceres_add_residual_calls_;
    }
#ifdef GPU_BA_ENABLED
    if (snapshot_recorder_ != nullptr) {
      snapshot_recorder_->RecordVisualResidual(
          track_el.image_id, track_el.point2D_idx, point3D_id,
          {{point2D.X(), point2D.Y()}}, true);
      snapshot_recorder_->RecordParameterBlock(
          gpu_ba::ParameterKind::kPoint3D, point3D_id,
          point3D.XYZ().data());
      snapshot_recorder_->RecordParameterBlock(
          gpu_ba::ParameterKind::kCamera, image.CameraId(),
          camera.ParamsData());
    }
    if (active_spec_builder_ != nullptr) {
      active_spec_builder_->RecordVisualResidual(
          track_el.image_id, track_el.point2D_idx, point3D_id,
          {{point2D.X(), point2D.Y()}}, true);
      active_spec_builder_->RecordParameterBlock(
          gpu_ba::ParameterKind::kPoint3D, point3D_id);
      active_spec_builder_->RecordParameterBlock(
          gpu_ba::ParameterKind::kCamera, image.CameraId());
    }
#endif
  }
}

void BundleAdjuster::AddLidarToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
                        
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  auto ptr = config_.lidar_maps_.find(point3D_id);
  if (ptr != config_.lidar_maps_.end()){
    Eigen::Matrix<double,4,1> abcd = ptr->second.LidarABCD();

    for (int i = 0; i < 4; i++){
      if(std::isnan(abcd(i))){
        return;
      } 
    }

    double w;
    LidarPointType type = ptr->second.Type();
    if (type == LidarPointType::Proj){
        w = options_.proj_lidar_constraint_weight;
    } else if (type == LidarPointType::Icp){
      w = options_.icp_lidar_constraint_weight;
    } else if (type == LidarPointType::IcpGround){
      w = options_.icp_ground_lidar_constraint_weight;
    } else {
      std::cout<<"This lidar point type is missing"<<std::endl;
      return;
    }
    if (problem_ != nullptr) {
      ceres::CostFunction* cost_function =
          BundleAdjustmentLidarCostFunction::Create(abcd, w);
      ++ceres_cost_function_creations_;
      problem_->AddResidualBlock(cost_function, loss_function,
                                 point3D.XYZ().data());
      ++ceres_add_residual_calls_;
    }
#ifdef GPU_BA_ENABLED
    if (snapshot_recorder_ != nullptr) {
      const Eigen::Vector3d lidar_xyz = ptr->second.LidarXYZ();
      std::array<double, 3> xyz{{lidar_xyz[0], lidar_xyz[1], lidar_xyz[2]}};
      std::array<double, 4> plane{{abcd[0], abcd[1], abcd[2], abcd[3]}};
      const auto range = config_.LidarSearchRanges().find(point3D_id);
      const bool has_range = range != config_.LidarSearchRanges().end();
      snapshot_recorder_->RecordLidarResidual(
          point3D_id, static_cast<uint8_t>(type), xyz, plane, w, has_range,
          has_range ? range->second : 0.0);
      snapshot_recorder_->RecordParameterBlock(
          gpu_ba::ParameterKind::kPoint3D, point3D_id,
          point3D.XYZ().data());
    }
    if (active_spec_builder_ != nullptr) {
      const Eigen::Vector3d lidar_xyz = ptr->second.LidarXYZ();
      std::array<double, 3> xyz{{lidar_xyz[0], lidar_xyz[1], lidar_xyz[2]}};
      std::array<double, 4> plane{{abcd[0], abcd[1], abcd[2], abcd[3]}};
      const auto range = config_.LidarSearchRanges().find(point3D_id);
      const bool has_range = range != config_.LidarSearchRanges().end();
      active_spec_builder_->RecordLidarResidual(
          point3D_id, static_cast<uint8_t>(type), xyz, plane, w, has_range,
          has_range ? range->second : 0.0);
      active_spec_builder_->RecordParameterBlock(
          gpu_ba::ParameterKind::kPoint3D, point3D_id);
    }
#endif
  }
  
}

void BundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
  if (problem_ == nullptr) return;
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
  for (const camera_t camera_id : camera_ids_) {
    Camera& camera = reconstruction->Camera(camera_id);

    if (constant_camera || config_.IsConstantCamera(camera_id)) {
      problem_->SetParameterBlockConstant(camera.ParamsData());
      continue;
    } else {
      std::vector<int> const_camera_params;

      if (!options_.refine_focal_length) {
        const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_principal_point) {
        const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_extra_params) {
        const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        SetSubsetManifold(static_cast<int>(camera.NumParams()),
                          const_camera_params, problem_.get(),
                          camera.ParamsData());
      }
    }
  }
}

void BundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
  if (problem_ == nullptr) return;
  for (const auto elem : point3D_num_observations_) {
    Point3D& point3D = reconstruction->Point3D(elem.first);
    if (point3D.Track().Length() > elem.second) {
      problem_->SetParameterBlockConstant(point3D.XYZ().data());
    }
  }

  for (const point3D_t point3D_id : config_.ConstantPoints()) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

////////////////////////////////////////////////////////////////////////////////
// ParallelBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

bool ParallelBundleAdjuster::Options::Check() const {
  CHECK_OPTION_GE(max_num_iterations, 0);
  return true;
}

ParallelBundleAdjuster::ParallelBundleAdjuster(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const BundleAdjustmentConfig& config)
    : options_(options),
      ba_options_(ba_options),
      config_(config),
      num_measurements_(0) {
  CHECK(options_.Check());
  CHECK(ba_options_.Check());
  CHECK_EQ(config_.NumConstantCameras(), 0)
      << "PBA does not allow to set individual cameras constant";
  CHECK_EQ(config_.NumConstantPoses(), 0)
      << "PBA does not allow to set individual translational elements constant";
  CHECK_EQ(config_.NumConstantTvecs(), 0)
      << "PBA does not allow to set individual translational elements constant";
  CHECK(config_.NumVariablePoints() == 0 && config_.NumConstantPoints() == 0)
      << "PBA does not allow to parameterize individual 3D points";
}

bool ParallelBundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK_EQ(num_measurements_, 0)
      << "Cannot use the same ParallelBundleAdjuster multiple times";
  CHECK(!ba_options_.refine_principal_point);
  CHECK_EQ(ba_options_.refine_focal_length, ba_options_.refine_extra_params);

  SetUp(reconstruction);

  const int num_residuals = static_cast<int>(2 * measurements_.size());

  size_t num_threads = options_.num_threads;
  if (num_residuals < options_.min_num_residuals_for_multi_threading) {
    num_threads = 1;
  }

  pba::ParallelBA::DeviceT device;
  const int kMaxNumResidualsFloat = 100 * 1000;
  if (num_residuals > kMaxNumResidualsFloat) {
    // The threshold for using double precision is empirically chosen and
    // ensures that the system can be reliable solved.
    device = pba::ParallelBA::PBA_CPU_DOUBLE;
  } else {
    if (options_.gpu_index < 0) {
      device = pba::ParallelBA::PBA_CUDA_DEVICE_DEFAULT;
    } else {
      device = static_cast<pba::ParallelBA::DeviceT>(
          pba::ParallelBA::PBA_CUDA_DEVICE0 + options_.gpu_index);
    }
  }

  pba::ParallelBA pba(device, num_threads);

  pba.SetNextBundleMode(pba::ParallelBA::BUNDLE_FULL);
  pba.EnableRadialDistortion(pba::ParallelBA::PBA_PROJECTION_DISTORTION);
  pba.SetFixedIntrinsics(!ba_options_.refine_focal_length &&
                         !ba_options_.refine_extra_params);

  pba::ConfigBA* pba_config = pba.GetInternalConfig();
  pba_config->__lm_delta_threshold /= 100.0f;
  pba_config->__lm_gradient_threshold /= 100.0f;
  pba_config->__lm_mse_threshold = 0.0f;
  pba_config->__cg_min_iteration = 10;
  pba_config->__verbose_level = 2;
  pba_config->__lm_max_iteration = options_.max_num_iterations;

  pba.SetCameraData(cameras_.size(), cameras_.data());
  pba.SetPointData(points3D_.size(), points3D_.data());
  pba.SetProjection(measurements_.size(), measurements_.data(),
                    point3D_idxs_.data(), camera_idxs_.data());

  Timer timer;
  timer.Start();
  pba.RunBundleAdjustment();
  timer.Pause();

  // Compose Ceres solver summary from PBA options.
  summary_.num_residuals_reduced = num_residuals;
  summary_.num_effective_parameters_reduced =
      static_cast<int>(8 * config_.NumImages() -
                       2 * config_.NumConstantCameras() + 3 * points3D_.size());
  summary_.num_successful_steps = pba_config->GetIterationsLM() + 1;
  summary_.termination_type = ceres::TerminationType::USER_SUCCESS;
  summary_.initial_cost =
      pba_config->GetInitialMSE() * summary_.num_residuals_reduced / 4;
  summary_.final_cost =
      pba_config->GetFinalMSE() * summary_.num_residuals_reduced / 4;
  summary_.total_time_in_seconds = timer.ElapsedSeconds();

  TearDown(reconstruction);

  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  return true;
}

const ceres::Solver::Summary& ParallelBundleAdjuster::Summary() const {
  return summary_;
}

bool ParallelBundleAdjuster::IsSupported(const BundleAdjustmentOptions& options,
                                         const Reconstruction& reconstruction) {
  if (options.refine_principal_point ||
      options.refine_focal_length != options.refine_extra_params) {
    return false;
  }

  // Check that all cameras are SIMPLE_RADIAL and that no intrinsics are shared.
  std::set<camera_t> camera_ids;
  for (const auto& image : reconstruction.Images()) {
    if (image.second.IsRegistered()) {
      if (camera_ids.count(image.second.CameraId()) != 0 ||
          reconstruction.Camera(image.second.CameraId()).ModelId() !=
              SimpleRadialCameraModel::model_id) {
        return false;
      }
      camera_ids.insert(image.second.CameraId());
    }
  }
  return true;
}

void ParallelBundleAdjuster::SetUp(Reconstruction* reconstruction) {
  // Important: PBA requires the track of 3D points to be stored
  // contiguously, i.e. the point3D_idxs_ vector contains consecutive indices.
  cameras_.reserve(config_.NumImages());
  camera_ids_.reserve(config_.NumImages());
  ordered_image_ids_.reserve(config_.NumImages());
  image_id_to_camera_idx_.reserve(config_.NumImages());
  AddImagesToProblem(reconstruction);
  AddPointsToProblem(reconstruction);
}

void ParallelBundleAdjuster::TearDown(Reconstruction* reconstruction) {
  for (size_t i = 0; i < cameras_.size(); ++i) {
    const image_t image_id = ordered_image_ids_[i];
    const pba::CameraT& pba_camera = cameras_[i];

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    Image& image = reconstruction->Image(image_id);
    Eigen::Matrix3d rotation_matrix;
    pba_camera.GetMatrixRotation(rotation_matrix.data());
    pba_camera.GetTranslation(image.Tvec().data());
    image.Qvec() = RotationMatrixToQuaternion(rotation_matrix.transpose());

    Camera& camera = reconstruction->Camera(image.CameraId());
    camera.Params(0) = pba_camera.GetFocalLength();
    camera.Params(3) = pba_camera.GetProjectionDistortion();
  }

  for (size_t i = 0; i < points3D_.size(); ++i) {
    Point3D& point3D = reconstruction->Point3D(ordered_point3D_ids_[i]);
    points3D_[i].GetPoint(point3D.XYZ().data());
  }
}

void ParallelBundleAdjuster::AddImagesToProblem(
    Reconstruction* reconstruction) {
  for (const image_t image_id : config_.Images()) {
    const Image& image = reconstruction->Image(image_id);
    CHECK_EQ(camera_ids_.count(image.CameraId()), 0)
        << "PBA does not support shared intrinsics";

    const Camera& camera = reconstruction->Camera(image.CameraId());
    CHECK_EQ(camera.ModelId(), SimpleRadialCameraModel::model_id)
        << "PBA only supports the SIMPLE_RADIAL camera model";

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    const Eigen::Matrix3d rotation_matrix =
        QuaternionToRotationMatrix(image.Qvec()).transpose();

    pba::CameraT pba_camera;
    pba_camera.SetFocalLength(camera.Params(0));
    pba_camera.SetProjectionDistortion(camera.Params(3));
    pba_camera.SetMatrixRotation(rotation_matrix.data());
    pba_camera.SetTranslation(image.Tvec().data());

    CHECK(!config_.HasConstantTvec(image_id))
        << "PBA cannot fix partial extrinsics";
    if (!ba_options_.refine_extrinsics || config_.HasConstantPose(image_id)) {
      CHECK(config_.IsConstantCamera(image.CameraId()))
          << "PBA cannot fix extrinsics only";
      pba_camera.SetConstantCamera();
    } else if (config_.IsConstantCamera(image.CameraId())) {
      pba_camera.SetFixedIntrinsic();
    } else {
      pba_camera.SetVariableCamera();
    }

    num_measurements_ += image.NumPoints3D();
    cameras_.push_back(pba_camera);
    camera_ids_.insert(image.CameraId());
    ordered_image_ids_.push_back(image_id);
    image_id_to_camera_idx_.emplace(image_id,
                                    static_cast<int>(cameras_.size()) - 1);

    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids_.insert(point2D.Point3DId());
      }
    }
  }
}

void ParallelBundleAdjuster::AddPointsToProblem(
    Reconstruction* reconstruction) {
  points3D_.resize(point3D_ids_.size());
  ordered_point3D_ids_.resize(point3D_ids_.size());
  measurements_.resize(num_measurements_);
  camera_idxs_.resize(num_measurements_);
  point3D_idxs_.resize(num_measurements_);

  int point3D_idx = 0;
  size_t measurement_idx = 0;

  for (const auto point3D_id : point3D_ids_) {
    const Point3D& point3D = reconstruction->Point3D(point3D_id);
    points3D_[point3D_idx].SetPoint(point3D.XYZ().data());
    ordered_point3D_ids_[point3D_idx] = point3D_id;

    for (const auto track_el : point3D.Track().Elements()) {
      if (image_id_to_camera_idx_.count(track_el.image_id) > 0) {
        const Image& image = reconstruction->Image(track_el.image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        measurements_[measurement_idx].SetPoint2D(
            point2D.X() - camera.Params(1), point2D.Y() - camera.Params(2));
        camera_idxs_[measurement_idx] =
            image_id_to_camera_idx_.at(track_el.image_id);
        point3D_idxs_[measurement_idx] = point3D_idx;
        measurement_idx += 1;
      }
    }
    point3D_idx += 1;
  }

  CHECK_EQ(point3D_idx, points3D_.size());
  CHECK_EQ(measurement_idx, measurements_.size());
}

////////////////////////////////////////////////////////////////////////////////
// RigBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

RigBundleAdjuster::RigBundleAdjuster(const BundleAdjustmentOptions& options,
                                     const Options& rig_options,
                                     const BundleAdjustmentConfig& config)
    : BundleAdjuster(options, config), rig_options_(rig_options) {}

bool RigBundleAdjuster::Solve(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs) {
  CHECK_NOTNULL(reconstruction);
  CHECK_NOTNULL(camera_rigs);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  // Check the validity of the provided camera rigs.
  std::unordered_set<camera_t> rig_camera_ids;
  for (auto& camera_rig : *camera_rigs) {
    camera_rig.Check(*reconstruction);
    for (const auto& camera_id : camera_rig.GetCameraIds()) {
      CHECK_EQ(rig_camera_ids.count(camera_id), 0)
          << "Camera must not be part of multiple camera rigs";
      rig_camera_ids.insert(camera_id);
    }

    for (const auto& snapshot : camera_rig.Snapshots()) {
      for (const auto& image_id : snapshot) {
        CHECK_EQ(image_id_to_camera_rig_.count(image_id), 0)
            << "Image must not be part of multiple camera rigs";
        image_id_to_camera_rig_.emplace(image_id, &camera_rig);
      }
    }
  }

  problem_ = std::make_unique<ceres::Problem>();

  ceres::LossFunction* loss_function = options_.CreateLossFunction();
  SetUp(reconstruction, camera_rigs, loss_function);

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Rig Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction, *camera_rigs);

  return true;
}

void RigBundleAdjuster::SetUp(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs,
                              ceres::LossFunction* loss_function) {
  ComputeCameraRigPoses(*reconstruction, *camera_rigs);

  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, camera_rigs, loss_function);
  }

  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
  ParameterizeCameraRigs(reconstruction);
}

void RigBundleAdjuster::TearDown(Reconstruction* reconstruction,
                                 const std::vector<CameraRig>& camera_rigs) {
  for (const auto& elem : image_id_to_camera_rig_) {
    const auto image_id = elem.first;
    const auto& camera_rig = *elem.second;
    auto& image = reconstruction->Image(image_id);
    ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                     *image_id_to_rig_tvec_.at(image_id),
                     camera_rig.RelativeQvec(image.CameraId()),
                     camera_rig.RelativeTvec(image.CameraId()), &image.Qvec(),
                     &image.Tvec());
  }
}

void RigBundleAdjuster::AddImageToProblem(const image_t image_id,
                                          Reconstruction* reconstruction,
                                          std::vector<CameraRig>* camera_rigs,
                                          ceres::LossFunction* loss_function) {
  const double max_squared_reproj_error =
      rig_options_.max_reproj_error * rig_options_.max_reproj_error;

  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  const bool constant_pose = config_.HasConstantPose(image_id);
  const bool constant_tvec = config_.HasConstantTvec(image_id);

  double* qvec_data = nullptr;
  double* tvec_data = nullptr;
  double* rig_qvec_data = nullptr;
  double* rig_tvec_data = nullptr;
  double* camera_params_data = camera.ParamsData();
  CameraRig* camera_rig = nullptr;
  Eigen::Matrix3x4d rig_proj_matrix = Eigen::Matrix3x4d::Zero();

  if (image_id_to_camera_rig_.count(image_id) > 0) {
    CHECK(!constant_pose)
        << "Images contained in a camera rig must not have constant pose";
    CHECK(!constant_tvec)
        << "Images contained in a camera rig must not have constant tvec";
    camera_rig = image_id_to_camera_rig_.at(image_id);
    rig_qvec_data = image_id_to_rig_qvec_.at(image_id)->data();
    rig_tvec_data = image_id_to_rig_tvec_.at(image_id)->data();
    qvec_data = camera_rig->RelativeQvec(image.CameraId()).data();
    tvec_data = camera_rig->RelativeTvec(image.CameraId()).data();

    // Concatenate the absolute pose of the rig and the relative pose the camera
    // within the rig to detect outlier observations.
    Eigen::Vector4d rig_concat_qvec;
    Eigen::Vector3d rig_concat_tvec;
    ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                     *image_id_to_rig_tvec_.at(image_id),
                     camera_rig->RelativeQvec(image.CameraId()),
                     camera_rig->RelativeTvec(image.CameraId()),
                     &rig_concat_qvec, &rig_concat_tvec);
    rig_proj_matrix = ComposeProjectionMatrix(rig_concat_qvec, rig_concat_tvec);
  } else {
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();
    qvec_data = image.Qvec().data();
    tvec_data = image.Tvec().data();
  }

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());
  camera_ids_.insert(image.CameraId());

  // The number of added observations for the current image.
  size_t num_observations = 0;

  // Add residuals to bundle adjustment problem.
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    if (camera_rig != nullptr &&
        CalculateSquaredReprojectionError(point2D.XY(), point3D.XYZ(),
                                          rig_proj_matrix,
                                          camera) > max_squared_reproj_error) {
      continue;
    }

    num_observations += 1;
    point3D_num_observations_[point2D.Point3DId()] += 1;

    ceres::CostFunction* cost_function = nullptr;

    if (camera_rig == nullptr) {
      if (constant_pose) {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem_->AddResidualBlock(cost_function, loss_function,
                                   point3D.XYZ().data(), camera_params_data);
      } else {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                   tvec_data, point3D.XYZ().data(),
                                   camera_params_data);
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::kModelId:                                               \
    cost_function =                                                         \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
                                                                            \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
      problem_->AddResidualBlock(cost_function, loss_function, rig_qvec_data,
                                 rig_tvec_data, qvec_data, tvec_data,
                                 point3D.XYZ().data(), camera_params_data);
    }
  }

  if (num_observations > 0) {
    parameterized_qvec_data_.insert(qvec_data);

    if (camera_rig != nullptr) {
      parameterized_qvec_data_.insert(rig_qvec_data);

      // Set the relative pose of the camera constant if relative pose
      // refinement is disabled or if it is the reference camera to avoid over-
      // parameterization of the camera pose.
      if (!rig_options_.refine_relative_poses ||
          image.CameraId() == camera_rig->RefCameraId()) {
        problem_->SetParameterBlockConstant(qvec_data);
        problem_->SetParameterBlockConstant(tvec_data);
      }
    }

    // Set pose parameterization.
    if (!constant_pose && constant_tvec) {
      const std::vector<int>& constant_tvec_idxs =
          config_.ConstantTvec(image_id);
      SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
    }
  }
}

void RigBundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                          Reconstruction* reconstruction,
                                          ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `AddImageToProblem`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_observations_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::kModelId:                                              \
    cost_function =                                                        \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(     \
            image.Qvec(), image.Tvec(), point2D.XY());                     \
    problem_->AddResidualBlock(cost_function, loss_function,               \
                               point3D.XYZ().data(), camera.ParamsData()); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
  }
}

void RigBundleAdjuster::ComputeCameraRigPoses(
    const Reconstruction& reconstruction,
    const std::vector<CameraRig>& camera_rigs) {
  camera_rig_qvecs_.reserve(camera_rigs.size());
  camera_rig_tvecs_.reserve(camera_rigs.size());
  for (const auto& camera_rig : camera_rigs) {
    camera_rig_qvecs_.emplace_back();
    camera_rig_tvecs_.emplace_back();
    auto& rig_qvecs = camera_rig_qvecs_.back();
    auto& rig_tvecs = camera_rig_tvecs_.back();
    rig_qvecs.resize(camera_rig.NumSnapshots());
    rig_tvecs.resize(camera_rig.NumSnapshots());
    for (size_t snapshot_idx = 0; snapshot_idx < camera_rig.NumSnapshots();
         ++snapshot_idx) {
      camera_rig.ComputeAbsolutePose(snapshot_idx, reconstruction,
                                     &rig_qvecs[snapshot_idx],
                                     &rig_tvecs[snapshot_idx]);
      for (const auto image_id : camera_rig.Snapshots()[snapshot_idx]) {
        image_id_to_rig_qvec_.emplace(image_id, &rig_qvecs[snapshot_idx]);
        image_id_to_rig_tvec_.emplace(image_id, &rig_tvecs[snapshot_idx]);
      }
    }
  }
}

void RigBundleAdjuster::ParameterizeCameraRigs(Reconstruction* reconstruction) {
  for (double* qvec_data : parameterized_qvec_data_) {
    SetQuaternionManifold(problem_.get(), qvec_data);
  }
}

void PrintSolverSummary(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(16) << "Residuals : ";
  std::cout << std::left << summary.num_residuals_reduced << std::endl;

  std::cout << std::right << std::setw(16) << "Parameters : ";
  std::cout << std::left << summary.num_effective_parameters_reduced
            << std::endl;

  std::cout << std::right << std::setw(16) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;

  std::cout << std::right << std::setw(16) << "Time : ";
  std::cout << std::left << summary.total_time_in_seconds << " [s]"
            << std::endl;

  std::cout << std::right << std::setw(16) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Final cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  std::cout << std::right << termination << std::endl;
  std::cout << std::endl;
}

}  // namespace colmap
