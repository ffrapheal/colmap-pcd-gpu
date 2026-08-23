#include "gpu_ba/custom_cuda.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

#include "gpu_ba/fixed_linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

std::string JsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (char character : value) {
    if (character == '"') output << "\\\"";
    else if (character == '\\') output << "\\\\";
    else if (character == '\n') output << "\\n";
    else output << character;
  }
  output << '"';
  return output.str();
}

std::string NumberOrNull(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

void WriteStableDiagnostic(std::ostream& output,
                           const CudaStableDiagnosticV2& value) {
  output << "{\"error_classification\":"
         << static_cast<int>(value.error_classification)
         << ",\"failure_site\":" << static_cast<int>(value.failure_site)
         << ",\"status_domain\":" << static_cast<int>(value.status_domain)
         << ",\"raw_status_code\":" << value.raw_status_code << '}';
}

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = std::min(
      values.size() - 1,
      static_cast<size_t>(std::ceil(fraction * values.size()) - 1));
  return values[index];
}

double Rms(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  long double sum = 0.0;
  for (double value : values) sum += value * value;
  return std::sqrt(static_cast<double>(sum / values.size()));
}

void WriteTimingPhase(std::ostream& output,
                      const char* name,
                      const CudaPhaseTiming& phase,
                      bool* first) {
  if (!*first) output << ',';
  *first = false;
  output << JsonString(name) << ":{\"host_wall_ms\":"
         << NumberOrNull(phase.host_wall_milliseconds)
         << ",\"cuda_event_ms\":"
         << NumberOrNull(phase.cuda_event_milliseconds)
         << ",\"calls\":" << phase.calls << ",\"bytes\":"
         << phase.bytes << '}';
}

void WriteTimingLedger(std::ostream& output, const CudaTimingLedger& ledger) {
  output << "{\"inclusive_calls\":{";
  bool inclusive_first = true;
  WriteTimingPhase(output, "layer_a_call", ledger.layer_a_call,
                   &inclusive_first);
  WriteTimingPhase(output, "layer_b_call", ledger.layer_b_call,
                   &inclusive_first);
  WriteTimingPhase(output, "layer_c_call", ledger.layer_c_call,
                   &inclusive_first);
  output << "},\"exclusive_phases\":{";
  bool first = true;
#define WRITE_PHASE(name) WriteTimingPhase(output, #name, ledger.name, &first)
  WRITE_PHASE(build_cuda_layer_a_inputs);
  WRITE_PHASE(topology_lookup);
  WRITE_PHASE(topology_build);
  WRITE_PHASE(topology_refresh);
  WRITE_PHASE(topology_cache);
  WRITE_PHASE(build_cost_order);
  WRITE_PHASE(allocation);
  WRITE_PHASE(free);
  WRITE_PHASE(h2d_memcpy);
  WRITE_PHASE(d2h_memcpy);
  WRITE_PHASE(d2d_memcpy);
  WRITE_PHASE(managed_prefetch);
  WRITE_PHASE(layer_a_kernel);
  WRITE_PHASE(layer_b_pose_kernel);
  WRITE_PHASE(layer_b_point_kernel);
  WRITE_PHASE(layer_b_edge_kernel);
  WRITE_PHASE(layer_b_gradient_kernel);
  WRITE_PHASE(layer_c_point_factor);
  WRITE_PHASE(schur);
  WRITE_PHASE(rhs);
  WRITE_PHASE(cusolver);
  WRITE_PHASE(back_substitution);
  WRITE_PHASE(back_substitution_host_self);
  WRITE_PHASE(trial_cost);
  WRITE_PHASE(apply_layer_c_step);
  WRITE_PHASE(compute_layer_c_diagnostics);
  WRITE_PHASE(lm_decision);
  WRITE_PHASE(synchronization);
  WRITE_PHASE(state_hash);
  WRITE_PHASE(topology_fingerprint_audit);
  WRITE_PHASE(controller_self);
  WRITE_PHASE(resource_release);
  WRITE_PHASE(other);
#undef WRITE_PHASE
  output << "},\"counters\":{\"topology_epoch\":"
         << ledger.topology_epoch
         << ",\"topology_build_count\":" << ledger.topology_build_count
         << ",\"topology_refresh_count\":"
         << ledger.topology_refresh_count
         << ",\"cache_lookup_count\":" << ledger.cache_lookup_count
         << ",\"cache_hit_count\":" << ledger.cache_hit_count
         << ",\"cache_miss_count\":" << ledger.cache_miss_count
         << ",\"cache_invalidation_count\":"
         << ledger.cache_invalidation_count
         << ",\"BuildCudaLayerAInputs_count\":"
         << ledger.build_cuda_layer_a_inputs_count
         << ",\"BuildCostOrder_count\":" << ledger.build_cost_order_count
         << ",\"layer_a_calls\":" << ledger.layer_a_calls
         << ",\"layer_b_calls\":" << ledger.layer_b_calls
         << ",\"layer_c_calls\":" << ledger.layer_c_calls
         << ",\"cost_calls\":" << ledger.cost_calls
         << ",\"topology_fingerprint_computations\":"
         << ledger.topology_fingerprint_computations
         << ",\"topology_generation\":" << ledger.topology_generation
         << "},\"topology_fingerprint\":"
         << (ledger.topology_fingerprint.empty()
                 ? "null"
                 : JsonString(ledger.topology_fingerprint))
         << ",\"transfer_memory_kind\":{\"h2d_host\":"
         << JsonString(ledger.h2d_host_memory_kind)
         << ",\"d2h_host\":" << JsonString(ledger.d2h_host_memory_kind)
         << ",\"d2d\":" << JsonString(ledger.d2d_memory_kind)
         << "},\"synchronization_audit\":{\"stream_calls\":"
         << ledger.synchronization_audit.stream_synchronize_calls
         << ",\"event_calls\":"
         << ledger.synchronization_audit.event_synchronize_calls
         << ",\"device_calls\":"
         << ledger.synchronization_audit.device_synchronize_calls
         << ",\"stream_host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.stream_host_wait_milliseconds)
         << ",\"event_host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.event_host_wait_milliseconds)
         << ",\"device_host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.device_host_wait_milliseconds)
         << ",\"sites\":{\"layer_a_final\":{"
         << "\"calls\":" << ledger.synchronization_audit.layer_a_final.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.layer_a_final.host_wait_milliseconds)
         << "},\"layer_b_final\":{\"calls\":"
         << ledger.synchronization_audit.layer_b_final.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.layer_b_final.host_wait_milliseconds)
         << "},\"layer_c_factor\":{\"calls\":"
         << ledger.synchronization_audit.layer_c_factor.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.layer_c_factor.host_wait_milliseconds)
         << "},\"layer_c_solver\":{\"calls\":"
         << ledger.synchronization_audit.layer_c_solver.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.layer_c_solver.host_wait_milliseconds)
         << "},\"layer_c_final\":{\"calls\":"
         << ledger.synchronization_audit.layer_c_final.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(
                ledger.synchronization_audit.layer_c_final.host_wait_milliseconds)
         << "},\"trial_cost_final\":{\"calls\":"
         << ledger.synchronization_audit.trial_cost_final.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(ledger.synchronization_audit.trial_cost_final
                             .host_wait_milliseconds)
         << "},\"resource_release\":{\"calls\":"
         << ledger.synchronization_audit.resource_release.calls
         << ",\"host_wait_ms\":"
         << NumberOrNull(ledger.synchronization_audit.resource_release
                             .host_wait_milliseconds)
         << "}}},\"accounted_interval_wall_ms\":"
         << NumberOrNull(ledger.accounted_interval_wall_milliseconds)
         << ",\"explicitly_measured_host_wall_ms\":"
         << NumberOrNull(ledger.explicitly_measured_host_wall_milliseconds)
         << ",\"uninstrumented_remainder_ms\":"
         << NumberOrNull(ledger.uninstrumented_remainder_milliseconds)
         << ",\"known_host_wall_ms\":"
         << NumberOrNull(ledger.known_host_wall_milliseconds)
         << ",\"other_host_wall_ms\":"
         << NumberOrNull(ledger.other_host_wall_milliseconds)
         << ",\"host_partition_error_percent\":"
         << NumberOrNull(ledger.host_partition_error_percent)
         << ",\"host_partition_pass\":"
         << (ledger.host_partition_pass ? "true" : "false") << '}';
}

struct StateMetrics {
  bool topology_pass = true;
  bool finite_pass = true;
  bool invariant_pass = true;
  std::string topology_error;
  std::string finite_error;
  std::string invariant_error;
  uint64_t reference_image_count = 0;
  uint64_t candidate_image_count = 0;
  uint64_t reference_point_count = 0;
  uint64_t candidate_point_count = 0;
  uint64_t reference_camera_count = 0;
  uint64_t candidate_camera_count = 0;
  double camera_max = 0.0;
  double camera_p95 = 0.0;
  double camera_rms = 0.0;
  double rotation_max_degrees = 0.0;
  double rotation_p95_degrees = 0.0;
  double rotation_rms_degrees = 0.0;
  double translation_max = 0.0;
  double translation_p95 = 0.0;
  double translation_rms = 0.0;
  double point_max = 0.0;
  double point_p95 = 0.0;
  double point_rms = 0.0;
  std::string worst_rotation_id;
  std::string worst_translation_id;
  std::string worst_point_id;
  std::string worst_camera_id;
};

StateMetrics CompareStates(const Snapshot& reference,
                           const Snapshot& candidate) {
  StateMetrics result;
  result.reference_image_count = reference.images.size();
  result.candidate_image_count = candidate.images.size();
  result.reference_point_count = reference.points.size();
  result.candidate_point_count = candidate.points.size();
  result.reference_camera_count = reference.cameras.size();
  result.candidate_camera_count = candidate.cameras.size();
  auto fail = [&](const std::string& message) {
    if (result.topology_pass) {
      result.topology_pass = false;
      result.topology_error = message;
    }
  };
  auto finite_fail = [&](const std::string& message) {
    result.finite_pass = false;
    if (result.finite_error.empty()) result.finite_error = message;
    fail(message);
  };
  auto invariant_fail = [&](const std::string& message) {
    result.invariant_pass = false;
    if (result.invariant_error.empty()) result.invariant_error = message;
    fail(message);
  };
  auto finite_double = [&](double value, const std::string& name) {
    if (!std::isfinite(value)) finite_fail("non-finite " + name);
  };
  auto finite_snapshot = [&](const Snapshot& state, const char* label) {
    for (const CameraSnapshot& camera : state.cameras) {
      for (size_t i = 0; i < camera.params.size(); ++i)
        if (!std::isfinite(camera.params[i]))
          finite_fail(std::string(label) + " camera=" +
                      std::to_string(camera.camera_id) + " param=" +
                      std::to_string(i));
    }
    for (const ImageSnapshot& image : state.images) {
      for (size_t i = 0; i < image.qvec.size(); ++i)
        if (!std::isfinite(image.qvec[i]))
          finite_fail(std::string(label) + " image=" +
                      std::to_string(image.image_id) + " qvec=" +
                      std::to_string(i));
      for (size_t i = 0; i < image.tvec.size(); ++i)
        if (!std::isfinite(image.tvec[i]))
          finite_fail(std::string(label) + " image=" +
                      std::to_string(image.image_id) + " tvec=" +
                      std::to_string(i));
    }
    for (const PointSnapshot& point : state.points) {
      for (size_t i = 0; i < point.xyz.size(); ++i)
        if (!std::isfinite(point.xyz[i]))
          finite_fail(std::string(label) + " point=" +
                      std::to_string(point.point3D_id) + " xyz=" +
                      std::to_string(i));
      finite_double(point.search_range,
                    std::string(label) + " point search_range");
    }
    for (const ObservationSnapshot& observation : state.observations) {
      for (size_t i = 0; i < observation.xy.size(); ++i)
        if (!std::isfinite(observation.xy[i]))
          finite_fail(std::string(label) + " observation source=" +
                      std::to_string(observation.source_index));
    }
    for (const LidarSnapshot& lidar : state.lidar) {
      for (size_t i = 0; i < lidar.lidar_xyz.size(); ++i)
        if (!std::isfinite(lidar.lidar_xyz[i]))
          finite_fail(std::string(label) + " lidar source=" +
                      std::to_string(lidar.source_index));
      for (size_t i = 0; i < lidar.plane.size(); ++i)
        if (!std::isfinite(lidar.plane[i]))
          finite_fail(std::string(label) + " lidar plane source=" +
                      std::to_string(lidar.source_index));
      finite_double(lidar.search_range,
                    std::string(label) + " lidar search_range");
      finite_double(lidar.weight,
                    std::string(label) + " lidar weight");
    }
  };
  finite_snapshot(reference, "reference");
  finite_snapshot(candidate, "candidate");
  auto check_unique = [&](const Snapshot& state, const char* label) {
    std::set<uint64_t> observations;
    for (const ObservationSnapshot& value : state.observations) {
      if (!observations.insert(value.source_index).second)
        invariant_fail(std::string("duplicate ") + label +
                       " observation source=" +
                       std::to_string(value.source_index));
    }
    std::set<uint64_t> lidar;
    for (const LidarSnapshot& value : state.lidar) {
      if (!lidar.insert(value.source_index).second)
        invariant_fail(std::string("duplicate ") + label +
                       " lidar source=" + std::to_string(value.source_index));
    }
  };
  check_unique(reference, "reference");
  check_unique(candidate, "candidate");
  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  std::unordered_map<uint32_t, const CameraSnapshot*> cameras;
  std::unordered_map<uint32_t, const ImageSnapshot*> reference_images;
  std::unordered_map<uint64_t, const PointSnapshot*> reference_points;
  std::unordered_map<uint32_t, const CameraSnapshot*> reference_cameras;
  for (const ImageSnapshot& image : candidate.images) {
    if (!images.emplace(image.image_id, &image).second)
      fail("duplicate candidate image=" + std::to_string(image.image_id));
  }
  for (const PointSnapshot& point : candidate.points) {
    if (!points.emplace(point.point3D_id, &point).second)
      fail("duplicate candidate point3D=" +
           std::to_string(point.point3D_id));
  }
  for (const CameraSnapshot& camera : candidate.cameras) {
    if (!cameras.emplace(camera.camera_id, &camera).second)
      fail("duplicate candidate camera=" + std::to_string(camera.camera_id));
  }
  for (const ImageSnapshot& image : reference.images) {
    if (!reference_images.emplace(image.image_id, &image).second)
      fail("duplicate reference image=" + std::to_string(image.image_id));
  }
  for (const PointSnapshot& point : reference.points) {
    if (!reference_points.emplace(point.point3D_id, &point).second)
      fail("duplicate reference point3D=" +
           std::to_string(point.point3D_id));
  }
  for (const CameraSnapshot& camera : reference.cameras) {
    if (!reference_cameras.emplace(camera.camera_id, &camera).second)
      fail("duplicate reference camera=" +
           std::to_string(camera.camera_id));
  }
  if (reference_images.size() != images.size())
    fail("image_count reference=" + std::to_string(reference_images.size()) +
         " candidate=" + std::to_string(images.size()));
  if (reference_points.size() != points.size())
    fail("point_count reference=" + std::to_string(reference_points.size()) +
         " candidate=" + std::to_string(points.size()));
  if (reference_cameras.size() != cameras.size())
    fail("camera_count reference=" +
         std::to_string(reference_cameras.size()) + " candidate=" +
         std::to_string(cameras.size()));
  for (const auto& item : images) {
    if (reference_images.find(item.first) == reference_images.end())
      fail("extra candidate image=" + std::to_string(item.first));
  }
  for (const auto& item : points) {
    if (reference_points.find(item.first) == reference_points.end())
      fail("extra candidate point3D=" + std::to_string(item.first));
  }
  for (const auto& item : cameras) {
    if (reference_cameras.find(item.first) == reference_cameras.end())
      fail("extra candidate camera=" + std::to_string(item.first));
  }
  std::vector<double> rotations;
  std::vector<double> translations;
  std::vector<double> point_errors;
  std::vector<double> camera_errors;
  for (const ImageSnapshot& image : reference.images) {
    const auto it = images.find(image.image_id);
    if (it == images.end()) {
      fail("missing image=" + std::to_string(image.image_id));
      continue;
    }
    if (!result.finite_pass) continue;
    if (image.camera_id != it->second->camera_id ||
        image.selected != it->second->selected ||
        image.pose_constant != it->second->pose_constant ||
        image.has_pose_parameter_blocks != it->second->has_pose_parameter_blocks ||
        image.constant_tvec_mask != it->second->constant_tvec_mask) {
      invariant_fail("image topology/constant flags differ image=" +
                     std::to_string(image.image_id));
    }
    if (!result.finite_pass) continue;
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t i = 0; i < 4; ++i) {
      dot += image.qvec[i] * it->second->qvec[i];
      lhs_norm += image.qvec[i] * image.qvec[i];
      rhs_norm += it->second->qvec[i] * it->second->qvec[i];
    }
    dot = std::abs(dot) / std::sqrt(lhs_norm * rhs_norm);
    dot = std::min(1.0, std::max(-1.0, dot));
    const double rotation =
        2.0 * std::acos(dot) * 180.0 / 3.14159265358979323846;
    double translation_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      const double difference = image.tvec[i] - it->second->tvec[i];
      translation_squared += difference * difference;
    }
    const double translation = std::sqrt(translation_squared);
    rotations.push_back(rotation);
    translations.push_back(translation);
    if (rotation > result.rotation_max_degrees) {
      result.rotation_max_degrees = rotation;
      result.worst_rotation_id = std::to_string(image.image_id);
    }
    if (translation > result.translation_max) {
      result.translation_max = translation;
      result.worst_translation_id = std::to_string(image.image_id);
    }
  }
  for (const PointSnapshot& point : reference.points) {
    const auto it = points.find(point.point3D_id);
    if (it == points.end()) {
      fail("missing point3D=" + std::to_string(point.point3D_id));
      continue;
    }
    if (point.constant != it->second->constant ||
        point.config_role != it->second->config_role ||
        point.has_search_range != it->second->has_search_range) {
      invariant_fail("point topology/constant flags differ point3D=" +
                     std::to_string(point.point3D_id));
    }
    if (point.has_search_range &&
        point.search_range != it->second->search_range) {
      invariant_fail("point search range differs point3D=" +
                     std::to_string(point.point3D_id));
    }
    if (!result.finite_pass) continue;
    double squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      const double difference = point.xyz[i] - it->second->xyz[i];
      squared += difference * difference;
    }
    const double distance = std::sqrt(squared);
    point_errors.push_back(distance);
    if (distance > result.point_max) {
      result.point_max = distance;
      result.worst_point_id = std::to_string(point.point3D_id);
    }
  }
  for (const CameraSnapshot& camera : reference.cameras) {
    const auto it = cameras.find(camera.camera_id);
    if (it == cameras.end()) {
      fail("missing camera=" + std::to_string(camera.camera_id));
      continue;
    }
    if (camera.model_id != it->second->model_id ||
        camera.width != it->second->width ||
        camera.height != it->second->height ||
        camera.constant != it->second->constant ||
        camera.params.size() != it->second->params.size()) {
      invariant_fail("camera topology differs camera=" +
                     std::to_string(camera.camera_id));
      continue;
    }
    for (size_t i = 0; i < camera.params.size(); ++i) {
      const double error = std::abs(camera.params[i] - it->second->params[i]);
      camera_errors.push_back(error);
      if (error > result.camera_max) {
        result.camera_max = error;
        result.worst_camera_id = std::to_string(camera.camera_id) +
                                 ":param=" + std::to_string(i);
      }
    }
  }
  // Check every topology collection in both directions. This catches a
  // candidate that silently drops an observation/track while keeping the
  // image and point IDs unchanged.
  if (reference.observations.size() != candidate.observations.size() ||
      reference.tracks.size() != candidate.tracks.size() ||
      reference.lidar.size() != candidate.lidar.size() ||
      reference.parameter_blocks_source_order.size() !=
          candidate.parameter_blocks_source_order.size() ||
      reference.source_insertion_order.size() != candidate.source_insertion_order.size() ||
      reference.canonical_order.size() != candidate.canonical_order.size() ||
      reference.parameter_blocks_canonical_order.size() !=
          candidate.parameter_blocks_canonical_order.size()) {
    invariant_fail("topology collection size differs");
  }
  auto compare_order = [&](const std::vector<OrderEntrySnapshot>& lhs,
                           const std::vector<OrderEntrySnapshot>& rhs,
                           const char* name) {
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < count; ++i) {
      if (lhs[i].source_index != rhs[i].source_index ||
          lhs[i].residual_kind != rhs[i].residual_kind ||
          lhs[i].image_id != rhs[i].image_id ||
          lhs[i].point2D_idx != rhs[i].point2D_idx ||
          lhs[i].point3D_id != rhs[i].point3D_id) {
        invariant_fail(std::string(name) + " differs at index=" +
                       std::to_string(i));
        return;
      }
    }
  };
  compare_order(reference.source_insertion_order,
                candidate.source_insertion_order, "source insertion order");
  compare_order(reference.canonical_order, candidate.canonical_order,
                "canonical order");
  auto compare_observations = [&]() {
    const size_t count = std::min(reference.observations.size(),
                                  candidate.observations.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& a = reference.observations[i];
      const auto& b = candidate.observations[i];
      if (a.source_index != b.source_index || a.image_id != b.image_id ||
          a.point2D_idx != b.point2D_idx || a.point3D_id != b.point3D_id ||
          a.pose_constant != b.pose_constant || a.xy != b.xy) {
        invariant_fail("observation differs at index=" + std::to_string(i));
        return;
      }
    }
  };
  auto compare_tracks = [&]() {
    const size_t count = std::min(reference.tracks.size(), candidate.tracks.size());
    for (size_t i = 0; i < count; ++i) {
      if (reference.tracks[i].point3D_id != candidate.tracks[i].point3D_id ||
          reference.tracks[i].image_id != candidate.tracks[i].image_id ||
          reference.tracks[i].point2D_idx != candidate.tracks[i].point2D_idx) {
        invariant_fail("track differs at index=" + std::to_string(i));
        return;
      }
    }
  };
  auto compare_lidar = [&]() {
    const size_t count = std::min(reference.lidar.size(), candidate.lidar.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& a = reference.lidar[i];
      const auto& b = candidate.lidar[i];
      if (a.source_index != b.source_index || a.point3D_id != b.point3D_id ||
          a.lidar_type != b.lidar_type || a.has_search_range != b.has_search_range ||
          a.search_range != b.search_range || a.weight != b.weight ||
          a.lidar_xyz != b.lidar_xyz || a.plane != b.plane) {
        invariant_fail("lidar correspondence differs at index=" +
                       std::to_string(i));
        return;
      }
    }
  };
  auto compare_parameters = [&]() {
    const size_t count = std::min(reference.parameter_blocks_source_order.size(),
                                  candidate.parameter_blocks_source_order.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& a = reference.parameter_blocks_source_order[i];
      const auto& b = candidate.parameter_blocks_source_order[i];
      if (a.source_index != b.source_index || a.kind != b.kind ||
          a.entity_id != b.entity_id || a.ambient_size != b.ambient_size ||
          a.tangent_size != b.tangent_size || a.constant != b.constant) {
        invariant_fail("parameter block differs at index=" + std::to_string(i));
        return;
      }
    }
    if (reference.parameter_blocks_canonical_order !=
        candidate.parameter_blocks_canonical_order) {
      invariant_fail("parameter canonical order differs");
    }
  };
  compare_observations();
  compare_tracks();
  compare_lidar();
  compare_parameters();
  if (reference.metadata.snapshot_id != candidate.metadata.snapshot_id ||
      reference.metadata.ba_kind != candidate.metadata.ba_kind ||
      reference.metadata.registered_image_count != candidate.metadata.registered_image_count ||
      reference.metadata.ba_call_index != candidate.metadata.ba_call_index ||
      reference.metadata.refinement_index != candidate.metadata.refinement_index ||
      reference.metadata.trigger_image_id != candidate.metadata.trigger_image_id ||
      reference.metadata.optimize_phrase != candidate.metadata.optimize_phrase ||
      reference.metadata.loss_function != candidate.metadata.loss_function ||
      reference.metadata.lidar_residual_mode != candidate.metadata.lidar_residual_mode ||
      reference.metadata.lidar_correspondence_version != candidate.metadata.lidar_correspondence_version ||
      reference.metadata.schur_mode != candidate.metadata.schur_mode ||
      reference.metadata.refine_focal_length != candidate.metadata.refine_focal_length ||
      reference.metadata.refine_principal_point != candidate.metadata.refine_principal_point ||
      reference.metadata.refine_extra_params != candidate.metadata.refine_extra_params ||
      reference.metadata.refine_extrinsics != candidate.metadata.refine_extrinsics ||
      reference.metadata.proj_lidar_weight != candidate.metadata.proj_lidar_weight ||
      reference.metadata.icp_lidar_weight != candidate.metadata.icp_lidar_weight ||
      reference.metadata.icp_ground_lidar_weight != candidate.metadata.icp_ground_lidar_weight ||
      reference.metadata.function_tolerance != candidate.metadata.function_tolerance ||
      reference.metadata.gradient_tolerance != candidate.metadata.gradient_tolerance ||
      reference.metadata.parameter_tolerance != candidate.metadata.parameter_tolerance ||
      reference.metadata.max_num_iterations != candidate.metadata.max_num_iterations ||
      reference.metadata.max_linear_solver_iterations != candidate.metadata.max_linear_solver_iterations ||
      reference.metadata.max_consecutive_invalid_steps != candidate.metadata.max_consecutive_invalid_steps) {
    invariant_fail("snapshot metadata/config differs");
  }
  result.rotation_p95_degrees = Percentile(rotations, 0.95);
  result.rotation_rms_degrees = Rms(rotations);
  result.translation_p95 = Percentile(translations, 0.95);
  result.translation_rms = Rms(translations);
  result.point_p95 = Percentile(point_errors, 0.95);
  result.point_rms = Rms(point_errors);
  result.camera_p95 = Percentile(camera_errors, 0.95);
  result.camera_rms = Rms(camera_errors);
  return result;
}

struct TraceComparison {
  bool structure_pass = true;
  bool numeric_pass = true;
  int32_t first_structure_iteration = -1;
  std::string first_structure_field;
  int32_t first_numeric_iteration = -1;
  std::string first_numeric_field;
  double first_reference = 0.0;
  double first_candidate = 0.0;
  double first_absolute_error = 0.0;
  double first_relative_error = 0.0;
};

bool Within(double reference, double candidate, double atol, double rtol) {
  return std::isfinite(reference) && std::isfinite(candidate) &&
         std::abs(reference - candidate) <=
             atol + rtol * std::max(std::abs(reference), std::abs(candidate));
}

TraceComparison CompareTrace(const std::vector<CustomCpuIteration>& reference,
                             const std::vector<CudaLmIteration>& candidate) {
  TraceComparison result;
  const size_t count = std::min(reference.size(), candidate.size());
  for (size_t i = 0; i < count; ++i) {
    if (reference[i].iteration != candidate[i].iteration ||
        reference[i].accepted != candidate[i].accepted ||
        reference[i].invalid != candidate[i].invalid ||
        reference[i].factorization_success != candidate[i].factorization_success ||
        reference[i].step_valid != candidate[i].step_valid ||
        reference[i].trial_finite != candidate[i].trial_finite) {
      result.structure_pass = false;
      result.first_structure_iteration = static_cast<int32_t>(i);
      result.first_structure_field =
          "iteration/accepted/invalid/factorization_success/step_valid/trial_finite";
      break;
    }
  }
  if (result.structure_pass && reference.size() != candidate.size()) {
    result.structure_pass = false;
    result.first_structure_iteration = static_cast<int32_t>(count);
    result.first_structure_field = "trace_length";
  }
  auto compare = [&](size_t iteration, const char* field, double lhs,
                     double rhs, double atol, double rtol) {
    if (!result.numeric_pass || Within(lhs, rhs, atol, rtol)) return;
    result.numeric_pass = false;
    result.first_numeric_iteration = static_cast<int32_t>(iteration);
    result.first_numeric_field = field;
    result.first_reference = lhs;
    result.first_candidate = rhs;
    result.first_absolute_error = std::abs(lhs - rhs);
    result.first_relative_error = result.first_absolute_error /
        std::max({std::abs(lhs), std::abs(rhs), 1e-300});
  };
  auto compare_bool = [&](size_t iteration, const char* field, bool lhs,
                          bool rhs) {
    if (!result.structure_pass || lhs == rhs) return;
    result.structure_pass = false;
    result.first_structure_iteration = static_cast<int32_t>(iteration);
    result.first_structure_field = field;
  };
  for (size_t i = 0; i < count && result.numeric_pass; ++i) {
    compare(i, "cost_before", reference[i].cost_before,
            candidate[i].cost_before, 1e-8, 1e-7);
    compare(i, "trial_cost", reference[i].trial_cost,
            candidate[i].trial_cost, 1e-8, 1e-7);
    compare(i, "cost_after", reference[i].cost_after,
            candidate[i].cost_after, 1e-8, 1e-7);
    compare(i, "radius_before", reference[i].radius_before,
            candidate[i].radius_before, 1e-12, 1e-10);
    compare(i, "radius_after", reference[i].radius_after,
            candidate[i].radius_after, 1e-12, 1e-10);
    compare(i, "predicted_reduction", reference[i].predicted_reduction,
            candidate[i].predicted_reduction, 1e-8, 1e-7);
    compare(i, "projected_gradient", reference[i].projected_gradient_max_norm,
            candidate[i].projected_gradient_max_norm, 1e-8, 1e-7);
    compare(i, "scaled_gradient", reference[i].scaled_gradient_norm,
            candidate[i].scaled_gradient_norm, 1e-8, 1e-7);
    compare(i, "lambda_before", reference[i].lambda_before,
            candidate[i].lambda_before, 1e-12, 1e-10);
    compare(i, "lambda_after", reference[i].lambda_after,
            candidate[i].lambda_after, 1e-12, 1e-10);
    compare(i, "lm_diagonal_min", reference[i].lm_diagonal_min,
            candidate[i].lm_diagonal_min, 5.1e-7, 1e-6);
    compare(i, "lm_diagonal_max", reference[i].lm_diagonal_max,
            candidate[i].lm_diagonal_max, 5.1e-7, 1e-6);
    compare(i, "actual_reduction", reference[i].actual_reduction,
            candidate[i].actual_reduction, 1e-8, 1e-7);
    compare(i, "rho", reference[i].rho, candidate[i].rho, 1e-9, 1e-7);
    compare(i, "function_metric", reference[i].function_metric,
            candidate[i].function_metric, 1e-12, 1e-8);
    compare(i, "parameter_metric", reference[i].parameter_metric,
            candidate[i].parameter_metric, 1e-12, 1e-8);
    compare(i, "step_norm", reference[i].step_norm, candidate[i].step_norm,
            1e-12, 1e-8);
    compare(i, "backward_error", reference[i].backward_error,
            candidate[i].backward_error, 1e-12, 1e-8);
    compare_bool(i, "factorization_success", reference[i].factorization_success,
                 candidate[i].factorization_success);
    compare_bool(i, "step_valid", reference[i].step_valid,
                 candidate[i].step_valid);
    compare_bool(i, "trial_finite", reference[i].trial_finite,
                 candidate[i].trial_finite);
    if (result.structure_pass && i + 1 == count &&
        reference[i].termination_reason != candidate[i].termination_reason) {
      // Only the final trace item carries a termination reason in both
      // implementations; comparing it here avoids treating empty per-trial
      // strings as a solver-path mismatch.
      result.structure_pass = false;
      result.first_structure_iteration = static_cast<int32_t>(i);
      result.first_structure_field = "termination_reason";
    }
  }
  return result;
}

const char* CudaTerminationName(CudaTerminationType type) {
  if (type == CudaTerminationType::kConvergence) return "CONVERGENCE";
  if (type == CudaTerminationType::kNoConvergence) return "NO_CONVERGENCE";
  return "FAILURE";
}

Snapshot ComparatorTestSnapshot() {
  Snapshot snapshot;
  snapshot.metadata.snapshot_id = "comparator-self-test";
  snapshot.metadata.loss_function = "TRIVIAL";
  CameraSnapshot camera;
  camera.camera_id = 1;
  camera.model_id = 4;
  camera.width = 640;
  camera.height = 480;
  camera.constant = true;
  camera.params = {500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0};
  snapshot.cameras.push_back(camera);
  ImageSnapshot image;
  image.image_id = 2;
  image.camera_id = 1;
  image.selected = true;
  image.pose_constant = false;
  image.has_pose_parameter_blocks = true;
  snapshot.images.push_back(image);
  PointSnapshot point;
  point.point3D_id = 3;
  point.xyz = {{0.0, 0.0, 4.0}};
  snapshot.points.push_back(point);
  ObservationSnapshot observation;
  observation.source_index = 4;
  observation.image_id = 2;
  observation.point2D_idx = 0;
  observation.point3D_id = 3;
  observation.xy = {{320.0, 240.0}};
  snapshot.observations.push_back(observation);
  TrackElementSnapshot track;
  track.point3D_id = 3;
  track.image_id = 2;
  track.point2D_idx = 0;
  snapshot.tracks.push_back(track);
  LidarSnapshot lidar;
  lidar.source_index = 5;
  lidar.point3D_id = 3;
  lidar.weight = 1.0;
  lidar.plane = {{0.0, 0.0, 1.0, -4.0}};
  snapshot.lidar.push_back(lidar);
  ParameterBlockSnapshot parameter;
  parameter.source_index = 0;
  parameter.kind = ParameterKind::kPoint3D;
  parameter.entity_id = 3;
  parameter.ambient_size = 3;
  parameter.tangent_size = 3;
  snapshot.parameter_blocks_source_order.push_back(parameter);
  snapshot.parameter_blocks_canonical_order.push_back(0);
  OrderEntrySnapshot visual_order;
  visual_order.source_index = 4;
  visual_order.residual_kind = ResidualKind::kVisual;
  visual_order.image_id = 2;
  visual_order.point3D_id = 3;
  snapshot.source_insertion_order.push_back(visual_order);
  snapshot.canonical_order = snapshot.source_insertion_order;
  return snapshot;
}

int RunStateComparatorSelfTest() {
  const Snapshot baseline = ComparatorTestSnapshot();
  if (!CompareStates(baseline, baseline).topology_pass) return 1;
  auto must_fail = [&](const char* name, const Snapshot& candidate) {
    const StateMetrics result = CompareStates(baseline, candidate);
    if (result.topology_pass) {
      std::cerr << "comparator self-test did not reject " << name << '\n';
      return false;
    }
    return true;
  };
  Snapshot candidate = baseline;
  candidate.images[0].qvec[0] = std::numeric_limits<double>::quiet_NaN();
  if (!must_fail("nan_pose", candidate)) return 1;
  candidate = baseline;
  candidate.points[0].xyz[0] = std::numeric_limits<double>::quiet_NaN();
  if (!must_fail("nan_point", candidate)) return 1;
  candidate = baseline;
  candidate.cameras[0].params[0] = std::numeric_limits<double>::infinity();
  if (!must_fail("nan_camera", candidate)) return 1;
  candidate = baseline;
  candidate.images.push_back(baseline.images[0]);
  if (!must_fail("duplicate_image", candidate)) return 1;
  candidate = baseline;
  candidate.points.push_back(baseline.points[0]);
  if (!must_fail("duplicate_point", candidate)) return 1;
  candidate = baseline;
  candidate.cameras.push_back(baseline.cameras[0]);
  if (!must_fail("duplicate_camera", candidate)) return 1;
  candidate = baseline;
  candidate.images.push_back(baseline.images[0]);
  candidate.images.back().image_id = 99;
  if (!must_fail("extra_candidate_image", candidate)) return 1;
  candidate = baseline;
  candidate.points.push_back(baseline.points[0]);
  candidate.points.back().point3D_id = 99;
  if (!must_fail("extra_candidate_point", candidate)) return 1;
  candidate = baseline;
  candidate.cameras.push_back(baseline.cameras[0]);
  candidate.cameras.back().camera_id = 99;
  if (!must_fail("extra_candidate_camera", candidate)) return 1;
  candidate = baseline;
  candidate.observations[0].xy[0] += 1.0;
  if (!must_fail("observation_change", candidate)) return 1;
  candidate = baseline;
  candidate.tracks[0].point2D_idx = 1;
  if (!must_fail("track_change", candidate)) return 1;
  candidate = baseline;
  candidate.lidar[0].weight = 2.0;
  if (!must_fail("lidar_change", candidate)) return 1;
  candidate = baseline;
  candidate.parameter_blocks_source_order[0].constant = true;
  if (!must_fail("parameter_change", candidate)) return 1;
  std::cout << "STATE_COMPARATOR_TEST_PASS\n";
  return 0;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  using namespace colmap::gpu_ba;
  if (argc == 2 && std::string(argv[1]) == "--state-comparator-self-test") {
    return RunStateComparatorSelfTest();
  }
  if (argc < 3) {
    std::cerr << "usage: gpu_ba_custom_cuda_full_lm_replay "
                 "SNAPSHOT REPORT [COST_REDUCTION_THREADS] "
                 "[capture] [parallel] [forced_reject_probe] "
                 "[instrumentation_ab] [sequence1|sequence20|warmup_measured] "
                 "[--hessian_assembly_backend pose_owned|observation_atomic|observation_segmented] "
                 "[--hessian_segment_size 16|32|64] "
                 "[--schur_contribution_backend direct|segmented] "
                 "[--schur_segment_size 16|32|64] "
                 "[--cuda_arithmetic_precision fp64|fp32_core|fp32_state_quantized|fp32_mixed] "
                 "[--device_context_mode legacy|persistent|device_state|device_control] "
                 "[--execution_profile baseline|runtime_pool|arena|device_scaling|fast_identity|unified_builder|compact_layer_a|compact_control] "
                 "[--audit_profile production|correctness|forensic] "
                 "[--current_linearization_cache off|on] "
                 "[--final_state_output DIRECTORY]\n";
    return 2;
  }
  const char* instrumentation_environment =
      std::getenv("COLMAP_PCD_GPU_BA_INSTRUMENTATION");
  const std::string instrumentation_requested =
      instrumentation_environment == nullptr ? "0" : instrumentation_environment;
  const std::string instrumentation_source =
      instrumentation_environment == nullptr ? "default" : "environment";
  if (instrumentation_requested != "0" && instrumentation_requested != "1") {
    std::cerr << "COLMAP_PCD_GPU_BA_INSTRUMENTATION must be exactly 0 or 1\n";
    return 2;
  }
  const bool instrumentation_enabled = instrumentation_requested == "1";
  const char* cache_environment =
      std::getenv("COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE");
  std::string cache_requested =
      cache_environment == nullptr ? "0" : cache_environment;
  std::string cache_source =
      cache_environment == nullptr ? "default" : "environment";
  if (cache_requested != "0" && cache_requested != "1") {
    std::cerr << "COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE must be "
                 "exactly 0 or 1\n";
    return 2;
  }
  bool cache_enabled = cache_requested == "1";
  const char* device_context_environment =
      std::getenv("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT");
  std::string device_context_requested =
      device_context_environment == nullptr ? "legacy"
                                            : device_context_environment;
  const char* execution_profile_environment =
      std::getenv("COLMAP_PCD_GPU_BA_EXECUTION_PROFILE");
  std::string execution_profile_requested =
      execution_profile_environment == nullptr ? "baseline"
                                                : execution_profile_environment;
  const std::vector<std::string> valid_execution_profiles = {
      "baseline", "runtime_pool", "arena", "device_scaling",
      "fast_identity", "unified_builder", "compact_layer_a",
      "compact_control"};
  const auto execution_profile_it = std::find(
      valid_execution_profiles.begin(), valid_execution_profiles.end(),
      execution_profile_requested);
  if (execution_profile_it == valid_execution_profiles.end()) {
    std::cerr << "COLMAP_PCD_GPU_BA_EXECUTION_PROFILE is invalid\n";
    return 2;
  }
  CudaExecutionProfile execution_profile = static_cast<CudaExecutionProfile>(
      std::distance(valid_execution_profiles.begin(), execution_profile_it));
  CudaDeviceContextMode device_context_mode =
      CudaDeviceContextMode::kCompatibilityDefault;
  const int threads = argc >= 4 ? std::max(1, std::stoi(argv[3])) : 8;
  bool capture_states = false;
  bool performance_mode = false;
  bool forced_reject_probe = false;
  bool instrumentation_ab = false;
  size_t sequence_repeats = 0;
  bool sequence_same_input = false;
  CudaSchurContributionBackend schur_contribution_backend =
      CudaSchurContributionBackend::kCompatibilityDefault;
  std::string schur_contribution_backend_requested =
      "compatibility_default";
  uint32_t schur_segment_size = 0;
  CudaHessianAssemblyBackend hessian_assembly_backend =
      CudaHessianAssemblyBackend::kCompatibilityDefault;
  std::string hessian_assembly_backend_requested = "compatibility_default";
  uint32_t hessian_segment_size = 0;
  CudaArithmeticPrecision arithmetic_precision =
      CudaArithmeticPrecision::kCompatibilityDefault;
  std::string arithmetic_precision_requested = "compatibility_default";
  CudaAuditProfile audit_profile = CudaAuditProfile::kCompatibilityDefault;
  std::string audit_profile_requested = "compatibility_default";
  bool audit_profile_explicit = false;
  std::string final_state_output;
  for (int i = 4; i < argc; ++i) {
    capture_states = capture_states || std::string(argv[i]) == "capture";
    performance_mode = performance_mode || std::string(argv[i]) == "parallel";
    forced_reject_probe =
        forced_reject_probe || std::string(argv[i]) == "forced_reject_probe";
    instrumentation_ab =
        instrumentation_ab || std::string(argv[i]) == "instrumentation_ab";
    if (std::string(argv[i]) == "sequence1") sequence_repeats = 1;
    if (std::string(argv[i]) == "sequence20") sequence_repeats = 20;
    if (std::string(argv[i]) == "warmup_measured") {
      sequence_repeats = 1;
      sequence_same_input = true;
    }
    if (std::string(argv[i]) == "--cuda_arithmetic_precision") {
      if (++i >= argc) {
        std::cerr << "--cuda_arithmetic_precision requires fp64, fp32_core, "
                     "fp32_state_quantized, or fp32_mixed\n";
        return 2;
      }
      arithmetic_precision_requested = argv[i];
      if (arithmetic_precision_requested == "fp64") {
        arithmetic_precision = CudaArithmeticPrecision::kFp64;
      } else if (arithmetic_precision_requested == "fp32_core") {
        arithmetic_precision = CudaArithmeticPrecision::kFp32Core;
      } else if (arithmetic_precision_requested ==
                 "fp32_state_quantized") {
        arithmetic_precision =
            CudaArithmeticPrecision::kFp32StateQuantizedMixed;
      } else if (arithmetic_precision_requested == "fp32_mixed") {
        arithmetic_precision = CudaArithmeticPrecision::kFp32MixedStable;
      } else {
        std::cerr << "--cuda_arithmetic_precision requires fp64, fp32_core, "
                     "fp32_state_quantized, or fp32_mixed\n";
        return 2;
      }
    } else if (std::string(argv[i]) == "--device_context_mode") {
      if (++i >= argc) {
        std::cerr << "--device_context_mode requires legacy, persistent, "
                     "device_state, or device_control\n";
        return 2;
      }
      device_context_requested = argv[i];
      if (device_context_requested == "legacy") {
        device_context_mode = CudaDeviceContextMode::kLegacy;
      } else if (device_context_requested == "persistent") {
        device_context_mode = CudaDeviceContextMode::kPersistent;
      } else if (device_context_requested == "device_state") {
        device_context_mode = CudaDeviceContextMode::kDeviceState;
      } else if (device_context_requested == "device_control") {
        device_context_mode = CudaDeviceContextMode::kDeviceControl;
      } else {
        std::cerr << "--device_context_mode requires legacy, persistent, "
                     "device_state, or device_control\n";
        return 2;
      }
    } else if (std::string(argv[i]) == "--execution_profile") {
      if (++i >= argc) {
        std::cerr << "--execution_profile requires a cumulative profile\n";
        return 2;
      }
      execution_profile_requested = argv[i];
      const auto profile = std::find(valid_execution_profiles.begin(),
                                     valid_execution_profiles.end(),
                                     execution_profile_requested);
      if (profile == valid_execution_profiles.end()) {
        std::cerr << "--execution_profile requires a cumulative profile\n";
        return 2;
      }
      execution_profile = static_cast<CudaExecutionProfile>(
          std::distance(valid_execution_profiles.begin(), profile));
    } else if (std::string(argv[i]) == "--audit_profile") {
      if (++i >= argc) {
        std::cerr << "--audit_profile requires production, correctness, or "
                     "forensic\n";
        return 2;
      }
      audit_profile_requested = argv[i];
      audit_profile_explicit = true;
      if (audit_profile_requested == "production") {
        audit_profile = CudaAuditProfile::kProduction;
      } else if (audit_profile_requested == "correctness") {
        audit_profile = CudaAuditProfile::kCorrectness;
      } else if (audit_profile_requested == "forensic") {
        audit_profile = CudaAuditProfile::kForensic;
      } else {
        std::cerr << "--audit_profile requires production, correctness, or "
                     "forensic\n";
        return 2;
      }
    } else if (std::string(argv[i]) == "--current_linearization_cache") {
      if (++i >= argc) {
        std::cerr << "--current_linearization_cache requires off or on\n";
        return 2;
      }
      const std::string value = argv[i];
      if (value != "off" && value != "on") {
        std::cerr << "--current_linearization_cache requires off or on\n";
        return 2;
      }
      cache_enabled = value == "on";
      cache_requested = cache_enabled ? "1" : "0";
      cache_source = "typed_cli";
    } else if (std::string(argv[i]) == "--final_state_output") {
      if (++i >= argc) {
        std::cerr << "--final_state_output requires a directory\n";
        return 2;
      }
      final_state_output = argv[i];
    } else if (std::string(argv[i]) == "--hessian_assembly_backend") {
      if (++i >= argc) {
        std::cerr << "--hessian_assembly_backend requires pose_owned, "
                     "observation_atomic, or observation_segmented\n";
        return 2;
      }
      hessian_assembly_backend_requested = argv[i];
      if (hessian_assembly_backend_requested == "pose_owned") {
        hessian_assembly_backend =
            CudaHessianAssemblyBackend::kPoseOwnedOptimizedReference;
      } else if (hessian_assembly_backend_requested ==
                 "observation_atomic") {
        hessian_assembly_backend =
            CudaHessianAssemblyBackend::kObservationAtomic;
      } else if (hessian_assembly_backend_requested ==
                 "observation_segmented") {
        hessian_assembly_backend =
            CudaHessianAssemblyBackend::kObservationSegmented;
      } else {
        std::cerr << "--hessian_assembly_backend requires pose_owned, "
                     "observation_atomic, or observation_segmented\n";
        return 2;
      }
    } else if (std::string(argv[i]) == "--hessian_segment_size") {
      if (++i >= argc) {
        std::cerr << "--hessian_segment_size requires 16, 32, or 64\n";
        return 2;
      }
      hessian_segment_size = static_cast<uint32_t>(std::stoul(argv[i]));
      if (hessian_segment_size != 16 && hessian_segment_size != 32 &&
          hessian_segment_size != 64) {
        std::cerr << "--hessian_segment_size requires 16, 32, or 64\n";
        return 2;
      }
    } else if (std::string(argv[i]) == "--schur_contribution_backend") {
      if (++i >= argc) {
        std::cerr << "--schur_contribution_backend requires direct or segmented\n";
        return 2;
      }
      schur_contribution_backend_requested = argv[i];
      if (schur_contribution_backend_requested == "direct") {
        schur_contribution_backend =
            CudaSchurContributionBackend::kDirectTransformed;
      } else if (schur_contribution_backend_requested == "segmented") {
        schur_contribution_backend =
            CudaSchurContributionBackend::kSegmentedTransformed;
      } else {
        std::cerr << "--schur_contribution_backend requires direct or segmented\n";
        return 2;
      }
    } else if (std::string(argv[i]) == "--schur_segment_size") {
      if (++i >= argc) {
        std::cerr << "--schur_segment_size requires 16, 32, or 64\n";
        return 2;
      }
      schur_segment_size = static_cast<uint32_t>(std::stoul(argv[i]));
      if (schur_segment_size != 16 && schur_segment_size != 32 &&
          schur_segment_size != 64) {
        std::cerr << "--schur_segment_size requires 16, 32, or 64\n";
        return 2;
      }
    }
  }
  if (instrumentation_ab && audit_profile_explicit) {
    std::cerr << "instrumentation_ab cannot be combined with "
                 "--audit_profile\n";
    return 2;
  }
  if (capture_states && audit_profile == CudaAuditProfile::kProduction) {
    std::cerr << "debug state capture is forbidden by the production audit "
                 "profile\n";
    return 2;
  }
  Snapshot snapshot;
  SnapshotReadResult read;
  std::string error;
  if (!ReadSnapshot(argv[1], &snapshot, &read, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  CudaSchurContributionHistogram schur_histogram;
  if (!BuildCudaSchurContributionHistogramForTesting(
          snapshot, &schur_histogram, &error)) {
    std::cerr << "Schur contribution histogram failed: " << error << '\n';
    return 1;
  }

  CustomCpuSolverOptions cpu_options;
  cpu_options.loss_type = snapshot.metadata.loss_function.empty()
                              ? "TRIVIAL"
                              : snapshot.metadata.loss_function;
  cpu_options.requested_num_threads = threads;
  cpu_options.effective_num_threads = threads;
  cpu_options.max_num_iterations = snapshot.metadata.max_num_iterations;
  cpu_options.max_linear_solver_iterations =
      snapshot.metadata.max_linear_solver_iterations;
  cpu_options.max_num_consecutive_invalid_steps =
      snapshot.metadata.max_consecutive_invalid_steps > 0
          ? snapshot.metadata.max_consecutive_invalid_steps
          : 10;
  cpu_options.function_tolerance = snapshot.metadata.function_tolerance;
  cpu_options.gradient_tolerance = snapshot.metadata.gradient_tolerance;
  cpu_options.parameter_tolerance = snapshot.metadata.parameter_tolerance;
  const bool requested_state_trace =
      audit_profile == CudaAuditProfile::kForensic ||
      (audit_profile == CudaAuditProfile::kCompatibilityDefault &&
       !instrumentation_ab);
  cpu_options.capture_state_trace = requested_state_trace;
  if (forced_reject_probe) {
    cpu_options.max_num_iterations = 4;
    cpu_options.min_relative_decrease = 2.0;
    cpu_options.function_tolerance = 0.0;
    cpu_options.gradient_tolerance = 0.0;
    cpu_options.parameter_tolerance = 0.0;
  }
  CustomCpuSolveResult cpu;
  Snapshot cpu_state;
  const auto cpu_start = std::chrono::steady_clock::now();
  if (!RunCustomCpuSolve(snapshot, cpu_options, CustomCpuSolveMode::kFull,
                         &cpu, &cpu_state, &error)) {
    std::cerr << "custom_cpu oracle failed: " << error << '\n';
    return 1;
  }
  const double cpu_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cpu_start)
          .count();

  CudaFullLmOptions cuda_options;
  cuda_options.arithmetic_precision = arithmetic_precision;
  cuda_options.audit_profile = audit_profile;
  cuda_options.device_context_mode = device_context_mode;
  cuda_options.execution_profile = execution_profile;
  cuda_options.layer_c.schur_contribution_backend =
      schur_contribution_backend;
  cuda_options.layer_c.layer_b.hessian_assembly_backend =
      hessian_assembly_backend;
  cuda_options.layer_c.layer_b.hessian_segment_size_for_testing =
      hessian_segment_size;
  cuda_options.layer_c.schur_segment_size_for_testing = schur_segment_size;
  if (schur_contribution_backend !=
      CudaSchurContributionBackend::kCompatibilityDefault) {
    cuda_options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  }
  cuda_options.layer_c.layer_b.layer_a.residual_order =
      CudaResidualOrder::kSourceInsertion;
  cuda_options.layer_c.layer_b.cost_reduction_threads = threads;
  cuda_options.performance_mode = performance_mode;
  cuda_options.capture_state_trace = requested_state_trace;
  cuda_options.max_solver_time_in_seconds = 1e9;
  cuda_options.instrumentation_mode =
      instrumentation_enabled ? CudaInstrumentationMode::kEnabled
                              : CudaInstrumentationMode::kDisabled;
  cuda_options.current_linearization_cache_mode =
      cache_enabled ? CudaCurrentLinearizationCacheMode::kEnabled
                    : CudaCurrentLinearizationCacheMode::kDisabled;
  if (forced_reject_probe) {
    cuda_options.max_num_iterations = 4;
    cuda_options.min_relative_decrease = 2.0;
    cuda_options.function_tolerance = 0.0;
    cuda_options.gradient_tolerance = 0.0;
    cuda_options.parameter_tolerance = 0.0;
  }
  CudaFullLmResult cuda;
  const auto cuda_call_start = std::chrono::steady_clock::now();
  const bool cuda_ran = RunCustomCudaSolve(snapshot, cuda_options, &cuda, &error);
  cuda.runtime.cuda_solve_call_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - cuda_call_start)
          .count();
  struct SequenceSample {
    bool success = false;
    bool finite = false;
    double wall_milliseconds = 0.0;
    double final_cost = 0.0;
    double context_initialization_wall_milliseconds = 0.0;
    double topology_preparation_wall_milliseconds = 0.0;
    double segment_plan_host_milliseconds = 0.0;
    uint64_t pool_cold_creates = 0;
    uint64_t pool_hot_leases = 0;
    uint64_t pool_returns = 0;
    uint64_t arena_grow_calls = 0;
    uint64_t init_allocations = 0;
    uint64_t post_init_allocations = 0;
    uint64_t arena_reserved_bytes = 0;
    uint64_t arena_retained_bytes = 0;
    uint64_t init_stream_creates = 0;
    uint64_t init_event_creates = 0;
    uint64_t init_solver_creates = 0;
    uint64_t init_blas_creates = 0;
    uint64_t close_stream_destroys = 0;
    uint64_t close_event_destroys = 0;
    uint64_t close_solver_destroys = 0;
    uint64_t close_blas_destroys = 0;
    uint64_t mathematical_cache_hits = 0;
    uint64_t stale_handle_accepts = 0;
    uint64_t schur_contribution_calls = 0;
    uint64_t schur_rhs_calls = 0;
    uint64_t factorization_calls = 0;
    uint64_t pair_contribution_chunk_launches = 0;
    uint64_t hessian_assembly_calls = 0;
    uint64_t pose_block_calls = 0;
    uint64_t point_block_calls = 0;
    uint64_t edge_block_calls = 0;
    uint64_t jacobi_finalize_calls = 0;
    uint64_t gradient_summary_calls = 0;
    uint64_t hessian_workspace_bytes = 0;
    uint64_t layer_a_calls = 0;
    uint64_t layer_b_calls = 0;
    uint64_t layer_c_calls = 0;
    uint64_t cost_calls = 0;
    uint64_t dpotrf_calls = 0;
    uint64_t dpotrs_calls = 0;
    uint64_t spotrf_calls = 0;
    uint64_t spotrs_calls = 0;
    uint64_t mixed_scalar_packet_calls = 0;
    uint64_t mixed_scalar_packet_bytes = 0;
    uint64_t mixed_full_array_d2h_bytes = 0;
    uint64_t mixed_serial_full_scan_kernel_count = 0;
    uint64_t mixed_float_buffer_bytes = 0;
    uint64_t mixed_double_buffer_bytes = 0;
    int32_t accepted_commits = 0;
    int32_t rejected_steps = 0;
    int32_t invalid_steps = 0;
    int32_t factorization_failures = 0;
    std::string backend_requested;
    std::string backend_effective;
    std::string hessian_backend_requested;
    std::string hessian_backend_effective;
    std::string precision_requested;
    std::string precision_effective;
    std::string termination;
    double final_projected_gradient = 0.0;
  };
  std::vector<SequenceSample> sequence_samples;
  sequence_samples.reserve(sequence_repeats + (sequence_repeats == 0 ? 0 : 1));
  const auto append_sequence_sample = [&](const CudaFullLmResult& value,
                                          const bool success,
                                          const double wall_ms) {
    SequenceSample sample;
    const CudaPersistentDeviceRuntimeInfo& persistent =
        value.runtime.persistent_device;
    sample.success = success;
    sample.finite = std::isfinite(value.final_cost);
    sample.wall_milliseconds = wall_ms;
    sample.final_cost = value.final_cost;
    sample.context_initialization_wall_milliseconds =
        persistent.solve_context_initialization_wall_milliseconds;
    sample.topology_preparation_wall_milliseconds =
        persistent.problem_topology_preparation_wall_milliseconds;
    sample.segment_plan_host_milliseconds =
        persistent.schur_segment_plan_host_milliseconds;
    sample.pool_cold_creates = persistent.runtime_pool_cold_creates;
    sample.pool_hot_leases = persistent.runtime_pool_hot_leases;
    sample.pool_returns = persistent.runtime_pool_returns;
    sample.arena_grow_calls = persistent.arena_grow_calls;
    sample.init_allocations = persistent.initialization_allocation_calls;
    sample.post_init_allocations =
        persistent.post_initialize_allocation_calls;
    sample.arena_reserved_bytes = persistent.arena_required_bytes;
    sample.arena_retained_bytes = persistent.arena_retained_bytes;
    sample.init_stream_creates = persistent.init_stream_create_count;
    sample.init_event_creates = persistent.init_event_create_count;
    sample.init_solver_creates = persistent.init_solver_create_count;
    sample.init_blas_creates = persistent.init_blas_create_count;
    sample.close_stream_destroys = persistent.close_stream_destroy_count;
    sample.close_event_destroys = persistent.close_event_destroy_count;
    sample.close_solver_destroys = persistent.close_solver_destroy_count;
    sample.close_blas_destroys = persistent.close_blas_destroy_count;
    sample.mathematical_cache_hits = value.runtime.cross_solve_cache_hits;
    sample.stale_handle_accepts =
        persistent.cross_solve_state_handle_hits +
        persistent.state_lineage_violations +
        persistent.state_b_pair_violations;
    sample.schur_contribution_calls = persistent.schur_contribution_calls;
    sample.schur_rhs_calls = persistent.schur_rhs_calls;
    sample.factorization_calls = persistent.factorization_calls;
    sample.pair_contribution_chunk_launches =
        persistent.schur_pair_contribution_chunk_launches;
    sample.hessian_assembly_calls =
        persistent.hessian_gradient_assembly_calls;
    sample.pose_block_calls = persistent.pose_block_assembly_calls;
    sample.point_block_calls = persistent.point_block_assembly_calls;
    sample.edge_block_calls = persistent.edge_block_assembly_calls;
    sample.jacobi_finalize_calls =
        persistent.jacobi_damping_finalize_calls;
    sample.gradient_summary_calls = persistent.gradient_summary_calls;
    sample.hessian_workspace_bytes =
        persistent.hessian_partial_workspace_bytes;
    sample.layer_a_calls = value.runtime.layer_a_calls;
    sample.layer_b_calls = value.runtime.layer_b_calls;
    sample.layer_c_calls = value.runtime.layer_c_calls;
    sample.cost_calls = value.runtime.cost_calls;
    sample.dpotrf_calls = persistent.dpotrf_calls;
    sample.dpotrs_calls = persistent.dpotrs_calls;
    sample.spotrf_calls = persistent.spotrf_calls;
    sample.spotrs_calls = persistent.spotrs_calls;
    sample.mixed_scalar_packet_calls = persistent.mixed_scalar_packet_calls;
    sample.mixed_scalar_packet_bytes = persistent.mixed_scalar_packet_bytes;
    sample.mixed_full_array_d2h_bytes = persistent.mixed_full_array_d2h_bytes;
    sample.mixed_serial_full_scan_kernel_count =
        persistent.mixed_serial_full_scan_kernel_count;
    sample.mixed_float_buffer_bytes = persistent.mixed_float_buffer_bytes;
    sample.mixed_double_buffer_bytes = persistent.mixed_double_buffer_bytes;
    sample.accepted_commits = value.accepted_commits;
    sample.rejected_steps = value.rejected_steps;
    sample.invalid_steps = value.invalid_steps;
    sample.factorization_failures = value.factorization_failures;
    sample.backend_requested =
        persistent.schur_contribution_backend_requested;
    sample.backend_effective =
        persistent.schur_contribution_backend_effective;
    sample.hessian_backend_requested =
        persistent.hessian_assembly_backend_requested;
    sample.hessian_backend_effective =
        persistent.hessian_assembly_backend_effective;
    sample.precision_requested = persistent.arithmetic_precision_requested;
    sample.precision_effective = persistent.arithmetic_precision_effective;
    sample.termination = CudaTerminationName(value.termination_type);
    sample.final_projected_gradient =
        value.final_projected_gradient_max_norm;
    sequence_samples.push_back(sample);
  };
  if (sequence_repeats != 0) {
    append_sequence_sample(cuda, cuda_ran,
                           cuda.runtime.cuda_solve_call_wall_milliseconds);
    CudaFullLmOptions sequence_options = cuda_options;
    sequence_options.audit_profile = CudaAuditProfile::kCorrectness;
    sequence_options.capture_state_trace = false;
    sequence_options.instrumentation_mode = CudaInstrumentationMode::kDisabled;
    Snapshot sequence_state = cuda_ran ? cuda.final_state : snapshot;
    for (size_t repeat = 0; repeat < sequence_repeats; ++repeat) {
      CudaFullLmResult repeated;
      std::string repeated_error;
      const auto repeated_start = std::chrono::steady_clock::now();
      const Snapshot& repeated_input =
          sequence_same_input ? snapshot : sequence_state;
      const bool repeated_success = RunCustomCudaSolve(
          repeated_input, sequence_options, &repeated, &repeated_error);
      const double repeated_wall =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - repeated_start)
              .count();
      repeated.runtime.cuda_solve_call_wall_milliseconds = repeated_wall;
      append_sequence_sample(repeated, repeated_success, repeated_wall);
      if (!repeated_success || !std::isfinite(repeated.final_cost)) break;
      if (!sequence_same_input) sequence_state = repeated.final_state;
    }
  }
  const StateMetrics state = CompareStates(cpu_state, cuda.final_state);
  const TraceComparison trace = CompareTrace(cpu.trace, cuda.trace);
  const size_t accepted_state_count = std::min(
      cpu.accepted_state_trace.size(), cuda.accepted_state_trace.size());
  bool accepted_state_trace_structure_pass =
      cpu.accepted_state_trace.size() == cuda.accepted_state_trace.size();
  std::vector<StateMetrics> accepted_state_metrics;
  accepted_state_metrics.reserve(accepted_state_count);
  bool accepted_state_values_pass = accepted_state_trace_structure_pass;
  for (size_t i = 0; i < accepted_state_count; ++i) {
    accepted_state_metrics.push_back(CompareStates(
        cpu.accepted_state_trace[i], cuda.accepted_state_trace[i]));
    const StateMetrics& accepted = accepted_state_metrics.back();
    accepted_state_values_pass =
        accepted_state_values_pass && accepted.topology_pass &&
        accepted.finite_pass && accepted.invariant_pass &&
        accepted.rotation_max_degrees <= 1e-4 &&
        accepted.translation_max <= 1e-6 && accepted.point_max <= 1e-5 &&
        accepted.camera_max <= 1e-10;
  }
  const bool cost_pass = Within(cpu.final_cost, cuda.final_cost, 1e-8, 1e-7);
  const bool gradient_pass = Within(
      cpu.final_projected_gradient_max_norm,
      cuda.final_projected_gradient_max_norm, 1e-8, 1e-6);
  const bool termination_pass =
      CpuTerminationTypeName(cpu.termination_type) ==
          CudaTerminationName(cuda.termination_type) &&
      cpu.termination_reason == cuda.termination_reason;
  const bool state_pass = state.topology_pass && state.finite_pass &&
      state.invariant_pass &&
      state.rotation_max_degrees <= 1e-4 &&
      state.translation_max <= 1e-6 && state.point_max <= 1e-5 &&
      state.camera_max <= 1e-10;
  const bool pass = cuda_ran && cost_pass && gradient_pass &&
      termination_pass && trace.structure_pass && trace.numeric_pass &&
      state_pass && accepted_state_values_pass;
  cuda.runtime.forced_reject_probe_requested = forced_reject_probe;
  bool probe_trace_valid = true;
  if (forced_reject_probe) {
    for (size_t i = 1; i < cuda.trace.size(); ++i) {
      probe_trace_valid = probe_trace_valid && !cuda.trace[i].accepted &&
          cuda.trace[i].state_epoch == 0;
      if (instrumentation_enabled) {
        probe_trace_valid = probe_trace_valid &&
            cuda.trace[i].current_state_hash == cuda.runtime.initial_state_hash;
      }
    }
  }
  cuda.runtime.probe_valid =
      forced_reject_probe && cuda.runtime.actual_trials > 0 &&
      cuda.runtime.accepted_trials == 0 &&
      cuda.runtime.rejected_trials == cuda.runtime.actual_trials &&
      cuda.termination_type != CudaTerminationType::kFailure &&
      probe_trace_valid &&
      (!instrumentation_enabled ||
       cuda.runtime.state_hash_audit.current.computations == 1);

  const std::string report_path = argv[2];
  const size_t slash = report_path.find_last_of('/');
  const std::string report_dir =
      slash == std::string::npos ? "." : report_path.substr(0, slash);
  const std::string state_root = report_dir + "/debug-states";
  SnapshotWriteResult final_state_write;
  if (!final_state_output.empty() && cuda_ran && cuda.success) {
    Snapshot final_state_copy = cuda.final_state;
    final_state_copy.metadata.snapshot_id =
        snapshot.metadata.snapshot_id + "-" + arithmetic_precision_requested +
        "-final";
    std::string write_error;
    if (!WriteSnapshot(final_state_copy, final_state_output,
                       &final_state_write, &write_error)) {
      std::cerr << "final state output failed: " << write_error << '\n';
      return 1;
    }
  }
  if (capture_states) {
    const size_t count = std::min<size_t>(
        5, std::min(cpu.accepted_state_trace.size(),
                     cuda.accepted_state_trace.size()));
    for (size_t i = 0; i < count; ++i) {
      Snapshot cpu_state_copy = cpu.accepted_state_trace[i];
      Snapshot cuda_state_copy = cuda.accepted_state_trace[i];
      cpu_state_copy.metadata.snapshot_id =
          snapshot.metadata.snapshot_id + "-accepted" + std::to_string(i) +
          "-cpu";
      cuda_state_copy.metadata.snapshot_id =
          snapshot.metadata.snapshot_id + "-accepted" + std::to_string(i) +
          "-cuda";
      SnapshotWriteResult ignored;
      std::string write_error;
      if (!WriteSnapshot(cpu_state_copy, state_root + "/cpu", &ignored,
                         &write_error) ||
          !WriteSnapshot(cuda_state_copy, state_root + "/cuda", &ignored,
                         &write_error)) {
        std::cerr << "debug state capture failed: " << write_error << '\n';
        return 1;
      }
    }
  }

  std::ofstream output(argv[2]);
  if (!output) return 1;
  output << std::setprecision(17)
         << "{\n  \"phase\":\"phase6p3-custom-cuda-audit-v1\",\n"
         << "  \"instrumentation_phase\":\"phase7p0-lm-lifecycle-timing-v1\",\n"
         << "  \"instrumentation_ab_mode\":"
         << (instrumentation_ab ? "true" : "false")
         << ",\n  \"instrumentation\":{\"requested\":"
         << JsonString(instrumentation_requested)
         << ",\"effective\":"
         << (cuda.runtime.instrumentation_effective ? "true" : "false")
         << ",\"source\":" << JsonString(instrumentation_source) << "},\n"
         << "  \"current_linearization_cache\":{\"requested\":"
         << JsonString(cache_requested) << ",\"effective\":"
         << (cuda.runtime.current_linearization_cache_effective ? "true"
                                                                 : "false")
         << ",\"source\":" << JsonString(cache_source) << "},\n"
         << "  \"device_context\":{\"requested\":"
         << JsonString(device_context_requested) << ",\"effective\":"
         << JsonString(cuda.runtime.device_context_backend) << "},\n"
         << "  \"execution_profile\":"
         << JsonString(execution_profile_requested) << ",\n"
         << "  \"audit_profile\":{\"requested\":"
         << JsonString(audit_profile_requested)
         << ",\"effective\":"
         << JsonString(cuda.runtime.audit_profile_effective)
         << ",\"production_invariants_checked\":"
         << (cuda.runtime.production_audit_invariants_checked ? "true" : "false")
         << ",\"production_invariants_pass\":"
         << (cuda.runtime.production_audit_invariants_pass ? "true" : "false")
         << ",\"violation_count\":"
         << cuda.runtime.production_audit_violation_count
         << ",\"first_violation\":"
         << (cuda.runtime.first_production_audit_violation.empty()
                 ? "null"
                 : JsonString(cuda.runtime.first_production_audit_violation))
         << "},\n"
         << "  \"arithmetic_precision\":{\"requested\":"
         << JsonString(arithmetic_precision_requested)
         << ",\"effective\":"
         << JsonString(cuda.runtime.persistent_device
                           .arithmetic_precision_effective)
         << ",\"state_storage\":"
         << JsonString(cuda.runtime.persistent_device.state_storage_precision)
         << ",\"residual_jacobian\":"
         << JsonString(cuda.runtime.persistent_device
                           .residual_jacobian_precision)
         << ",\"hessian_schur\":"
         << JsonString(cuda.runtime.persistent_device.hessian_schur_precision)
         << ",\"factorization\":"
         << JsonString(cuda.runtime.persistent_device.factorization_routine)
         << ",\"delta\":"
         << JsonString(cuda.runtime.persistent_device.delta_precision)
         << ",\"quaternion_plus\":"
         << JsonString(cuda.runtime.persistent_device
                           .quaternion_plus_precision)
         << ",\"cost\":"
         << JsonString(cuda.runtime.persistent_device.cost_precision)
         << ",\"controller\":"
         << JsonString(cuda.runtime.persistent_device.controller_precision)
         << "},\n"
         << "  \"final_state_output\":{\"requested\":"
         << (final_state_output.empty() ? "false" : "true")
         << ",\"directory\":"
         << (final_state_output.empty() ? "null"
                                        : JsonString(final_state_output))
         << ",\"manifest\":"
         << (final_state_write.manifest_path.empty()
                 ? "null" : JsonString(final_state_write.manifest_path))
         << "},\n"
         << "  \"hessian_assembly_backend\":{\"requested\":"
         << JsonString(hessian_assembly_backend_requested)
         << ",\"effective\":"
         << JsonString(cuda.runtime.persistent_device
                           .hessian_assembly_backend_effective)
         << ",\"segment_size\":"
         << cuda.runtime.persistent_device.hessian_segment_size << "},\n"
         << "  \"schur_contribution_backend\":{\"requested\":"
         << JsonString(schur_contribution_backend_requested)
         << ",\"effective\":"
         << JsonString(cuda.runtime.persistent_device
                           .schur_contribution_backend_effective)
         << ",\"segment_size\":"
         << cuda.runtime.persistent_device.schur_segment_size << "},\n"
         << "  \"layer\":\"E_full_lm\",\n"
         << "  \"snapshot_id\":" << JsonString(snapshot.metadata.snapshot_id)
         << ",\n  \"schur_contribution_histogram\":{\"pose_count\":"
         << schur_histogram.pose_count << ",\"pair_count\":"
         << schur_histogram.pair_count << ",\"contribution_count\":"
         << schur_histogram.contribution_count
         << ",\"whole_pair_chunk_count\":"
         << schur_histogram.whole_pair_chunk_count
         << ",\"empty_pair_count\":" << schur_histogram.empty_pair_count
         << ",\"single_contribution_pair_count\":"
         << schur_histogram.single_contribution_pair_count
         << ",\"minimum\":" << schur_histogram.minimum
         << ",\"p50\":" << schur_histogram.p50
         << ",\"p90\":" << schur_histogram.p90
         << ",\"p95\":" << schur_histogram.p95
         << ",\"p99\":" << schur_histogram.p99
         << ",\"maximum\":" << schur_histogram.maximum
         << ",\"chunks\":[";
  for (size_t i = 0; i < schur_histogram.chunk_pair_begin.size(); ++i) {
    output << (i == 0 ? "" : ",") << "{\"pair_begin\":"
           << schur_histogram.chunk_pair_begin[i] << ",\"pair_end\":"
           << schur_histogram.chunk_pair_end[i]
           << ",\"contribution_begin\":"
           << schur_histogram.chunk_contribution_begin[i]
           << ",\"contribution_end\":"
           << schur_histogram.chunk_contribution_end[i] << "}";
  }
  output << "]},\n  \"pass\":" << (pass ? "true" : "false")
         << ",\n  \"cuda_ran\":" << (cuda_ran ? "true" : "false")
         << ",\n  \"cuda_error\":"
         << (error.empty() ? "null" : JsonString(error))
         << ",\n  \"same_process_sequence\":{\"requested_repeats\":"
         << sequence_repeats << ",\"same_input\":"
         << (sequence_same_input ? "true" : "false")
         << ",\"samples\":[";
  for (size_t i = 0; i < sequence_samples.size(); ++i) {
    const SequenceSample& sample = sequence_samples[i];
    output << (i == 0 ? "" : ",")
           << "{\"ordinal\":" << i
           << ",\"cold\":" << (i == 0 ? "true" : "false")
           << ",\"success\":" << (sample.success ? "true" : "false")
           << ",\"finite\":" << (sample.finite ? "true" : "false")
           << ",\"wall_milliseconds\":"
           << NumberOrNull(sample.wall_milliseconds)
           << ",\"final_cost\":" << NumberOrNull(sample.final_cost)
           << ",\"context_initialization_wall_milliseconds\":"
           << NumberOrNull(sample.context_initialization_wall_milliseconds)
           << ",\"topology_preparation_wall_milliseconds\":"
           << NumberOrNull(sample.topology_preparation_wall_milliseconds)
           << ",\"segment_plan_host_milliseconds\":"
           << NumberOrNull(sample.segment_plan_host_milliseconds)
           << ",\"pool_cold_creates\":" << sample.pool_cold_creates
           << ",\"pool_hot_leases\":" << sample.pool_hot_leases
           << ",\"pool_returns\":" << sample.pool_returns
           << ",\"arena_grow_calls\":" << sample.arena_grow_calls
           << ",\"initialization_allocation_calls\":"
           << sample.init_allocations
           << ",\"post_initialize_allocation_calls\":"
           << sample.post_init_allocations
           << ",\"arena_reserved_bytes\":" << sample.arena_reserved_bytes
           << ",\"arena_retained_bytes\":" << sample.arena_retained_bytes
           << ",\"init_resource_creates\":"
           << (sample.init_stream_creates + sample.init_event_creates +
               sample.init_solver_creates + sample.init_blas_creates)
           << ",\"close_resource_destroys\":"
           << (sample.close_stream_destroys + sample.close_event_destroys +
               sample.close_solver_destroys + sample.close_blas_destroys)
           << ",\"cross_solve_mathematical_cache_hits\":"
           << sample.mathematical_cache_hits
           << ",\"stale_handle_accepts\":" << sample.stale_handle_accepts
           << ",\"hessian_assembly_backend_requested\":"
           << JsonString(sample.hessian_backend_requested)
           << ",\"hessian_assembly_backend_effective\":"
           << JsonString(sample.hessian_backend_effective)
           << ",\"arithmetic_precision_requested\":"
           << JsonString(sample.precision_requested)
           << ",\"arithmetic_precision_effective\":"
           << JsonString(sample.precision_effective)
           << ",\"hessian_gradient_assembly_calls\":"
           << sample.hessian_assembly_calls
           << ",\"pose_block_assembly_calls\":" << sample.pose_block_calls
           << ",\"point_block_assembly_calls\":" << sample.point_block_calls
           << ",\"edge_block_assembly_calls\":" << sample.edge_block_calls
           << ",\"jacobi_damping_finalize_calls\":"
           << sample.jacobi_finalize_calls
           << ",\"gradient_summary_calls\":"
           << sample.gradient_summary_calls
           << ",\"hessian_partial_workspace_bytes\":"
           << sample.hessian_workspace_bytes
           << ",\"schur_contribution_backend_requested\":"
           << JsonString(sample.backend_requested)
           << ",\"schur_contribution_backend_effective\":"
           << JsonString(sample.backend_effective)
           << ",\"schur_contribution_calls\":"
           << sample.schur_contribution_calls
           << ",\"schur_rhs_calls\":" << sample.schur_rhs_calls
           << ",\"factorization_calls\":" << sample.factorization_calls
           << ",\"pair_contribution_chunk_launches\":"
           << sample.pair_contribution_chunk_launches
           << ",\"layer_a_calls\":" << sample.layer_a_calls
           << ",\"layer_b_calls\":" << sample.layer_b_calls
           << ",\"layer_c_calls\":" << sample.layer_c_calls
           << ",\"cost_calls\":" << sample.cost_calls
           << ",\"dpotrf_calls\":" << sample.dpotrf_calls
           << ",\"dpotrs_calls\":" << sample.dpotrs_calls
           << ",\"spotrf_calls\":" << sample.spotrf_calls
           << ",\"spotrs_calls\":" << sample.spotrs_calls
           << ",\"mixed_scalar_packet_calls\":"
           << sample.mixed_scalar_packet_calls
           << ",\"mixed_scalar_packet_bytes\":"
           << sample.mixed_scalar_packet_bytes
           << ",\"mixed_full_array_d2h_bytes\":"
           << sample.mixed_full_array_d2h_bytes
           << ",\"mixed_serial_full_scan_kernel_count\":"
           << sample.mixed_serial_full_scan_kernel_count
           << ",\"mixed_float_buffer_bytes\":"
           << sample.mixed_float_buffer_bytes
           << ",\"mixed_double_buffer_bytes\":"
           << sample.mixed_double_buffer_bytes
           << ",\"accepted_commits\":" << sample.accepted_commits
           << ",\"rejected_steps\":" << sample.rejected_steps
           << ",\"invalid_steps\":" << sample.invalid_steps
           << ",\"factorization_failures\":"
           << sample.factorization_failures
           << ",\"final_projected_gradient\":"
           << NumberOrNull(sample.final_projected_gradient)
           << ",\"termination\":" << JsonString(sample.termination)
           << "}";
  }
  output << "]}"
         << ",\n  \"cost_reduction_threads\":" << threads
         << ",\n  \"performance_mode\":"
         << (performance_mode ? "true" : "false")
         << ",\n  \"capture_state_trace\":"
         << (cuda.runtime.capture_state_trace_effective ? "true" : "false")
         << ",\n  \"cuda_memory_mode\":\"explicit_device_copy\""
         << ",\n  \"max_solver_time_in_seconds\":"
         << NumberOrNull(cuda_options.max_solver_time_in_seconds)
         << ",\n  \"loss\":"
         << JsonString(snapshot.metadata.loss_function.empty()
                           ? "TRIVIAL"
                           : snapshot.metadata.loss_function)
         << ",\n  \"reference_backend\":\"custom_cpu\""
         << ",\n  \"candidate_backend\":\"custom_cuda\""
         << ",\n  \"gates\":{\"cost_pass\":"
         << (cost_pass ? "true" : "false")
         << ",\"gradient_pass\":" << (gradient_pass ? "true" : "false")
         << ",\"termination_pass\":"
         << (termination_pass ? "true" : "false")
         << ",\"trace_structure_pass\":"
         << (trace.structure_pass ? "true" : "false")
         << ",\"trace_numeric_pass\":"
         << (trace.numeric_pass ? "true" : "false")
         << ",\"state_pass\":" << (state_pass ? "true" : "false")
         << ",\"accepted_state_trace_pass\":"
         << (accepted_state_values_pass ? "true" : "false")
         << "},\n  \"cpu\":{\"wall_milliseconds\":"
         << NumberOrNull(cpu_wall_milliseconds)
         << ",\"final_cost\":"
         << NumberOrNull(cpu.final_cost)
         << ",\"final_projected_gradient_max_norm\":"
         << NumberOrNull(cpu.final_projected_gradient_max_norm)
         << ",\"termination_type\":"
         << JsonString(CpuTerminationTypeName(cpu.termination_type))
         << ",\"termination_reason\":"
         << JsonString(cpu.termination_reason)
         << ",\"trial_iterations\":" << cpu.trial_iterations
         << ",\"accepted_steps\":" << cpu.accepted_steps
         << ",\"rejected_steps\":" << cpu.rejected_steps
         << "},\n  \"cuda\":{\"success\":"
         << (cuda.success ? "true" : "false")
         << ",\"initial_cost\":" << NumberOrNull(cuda.initial_cost)
         << ",\"final_cost\":" << NumberOrNull(cuda.final_cost)
         << ",\"final_projected_gradient_max_norm\":"
         << NumberOrNull(cuda.final_projected_gradient_max_norm)
         << ",\"final_scaled_gradient_norm\":"
         << NumberOrNull(cuda.final_scaled_gradient_norm)
         << ",\"final_lambda\":" << NumberOrNull(cuda.final_lambda)
         << ",\"final_radius\":" << NumberOrNull(cuda.final_radius)
         << ",\"termination_type\":"
         << JsonString(CudaTerminationName(cuda.termination_type))
         << ",\"termination_reason\":"
         << JsonString(cuda.termination_reason)
         << ",\"trial_iterations\":" << cuda.trial_iterations
         << ",\"accepted_steps\":" << cuda.accepted_steps
         << ",\"accepted_decisions\":" << cuda.accepted_decisions
         << ",\"accepted_commits\":" << cuda.accepted_commits
         << ",\"accepted_pending_preparation_failures\":"
         << cuda.accepted_pending_preparation_failures
         << ",\"accepted_pending_linearization_failures\":"
         << cuda.accepted_pending_linearization_failures
         << ",\"error_classification\":"
         << static_cast<int>(cuda.error_classification)
         << ",\"rejected_steps\":" << cuda.rejected_steps
         << ",\"invalid_steps\":" << cuda.invalid_steps
         << ",\"factorization_failures\":"
         << cuda.factorization_failures
         << "},\n  \"trace_first_divergence\":{"
         << "\"structure_iteration\":" << trace.first_structure_iteration
         << ",\"structure_field\":"
         << (trace.first_structure_field.empty()
                 ? "null"
                 : JsonString(trace.first_structure_field))
         << ",\"numeric_iteration\":" << trace.first_numeric_iteration
         << ",\"numeric_field\":"
         << (trace.first_numeric_field.empty()
                 ? "null"
                 : JsonString(trace.first_numeric_field))
         << ",\"reference\":" << NumberOrNull(trace.first_reference)
         << ",\"candidate\":" << NumberOrNull(trace.first_candidate)
         << ",\"absolute_error\":"
         << NumberOrNull(trace.first_absolute_error)
         << ",\"relative_error\":"
         << NumberOrNull(trace.first_relative_error)
         << "},\n  \"iteration_trace\":{\n    \"cpu\":[";
  for (size_t i = 0; i < cpu.trace.size(); ++i) {
    const CustomCpuIteration& value = cpu.trace[i];
    output << (i == 0 ? "" : ",")
           << "{\"iteration\":" << value.iteration
           << ",\"cost_before\":" << NumberOrNull(value.cost_before)
           << ",\"trial_cost\":" << NumberOrNull(value.trial_cost)
           << ",\"cost_after\":" << NumberOrNull(value.cost_after)
           << ",\"projected_gradient\":"
           << NumberOrNull(value.projected_gradient_max_norm)
           << ",\"scaled_gradient\":"
           << NumberOrNull(value.scaled_gradient_norm)
           << ",\"radius_before\":" << NumberOrNull(value.radius_before)
           << ",\"radius_after\":" << NumberOrNull(value.radius_after)
           << ",\"lambda_before\":" << NumberOrNull(value.lambda_before)
           << ",\"lambda_after\":" << NumberOrNull(value.lambda_after)
           << ",\"lm_diagonal_min\":"
           << NumberOrNull(value.lm_diagonal_min)
           << ",\"lm_diagonal_max\":"
           << NumberOrNull(value.lm_diagonal_max)
           << ",\"predicted_reduction\":"
           << NumberOrNull(value.predicted_reduction)
           << ",\"actual_reduction\":"
           << NumberOrNull(value.actual_reduction)
           << ",\"rho\":" << NumberOrNull(value.rho)
           << ",\"function_metric\":" << NumberOrNull(value.function_metric)
           << ",\"parameter_metric\":" << NumberOrNull(value.parameter_metric)
           << ",\"step_norm\":" << NumberOrNull(value.step_norm)
           << ",\"backward_error\":" << NumberOrNull(value.backward_error)
           << ",\"factorization_success\":"
           << (value.factorization_success ? "true" : "false")
           << ",\"step_valid\":" << (value.step_valid ? "true" : "false")
           << ",\"trial_finite\":"
           << (value.trial_finite ? "true" : "false")
           << ",\"accepted\":" << (value.accepted ? "true" : "false")
           << ",\"invalid\":" << (value.invalid ? "true" : "false")
           << ",\"termination_reason\":"
           << (value.termination_reason.empty()
                   ? "null"
                   : JsonString(value.termination_reason))
           << "}";
  }
  output << "],\n    \"cuda\":[";
  for (size_t i = 0; i < cuda.trace.size(); ++i) {
    const CudaLmIteration& value = cuda.trace[i];
    output << (i == 0 ? "" : ",")
           << "{\"iteration\":" << value.iteration
           << ",\"cost_before\":" << NumberOrNull(value.cost_before)
           << ",\"trial_cost\":" << NumberOrNull(value.trial_cost)
           << ",\"cost_after\":" << NumberOrNull(value.cost_after)
           << ",\"projected_gradient\":"
           << NumberOrNull(value.projected_gradient_max_norm)
           << ",\"scaled_gradient\":"
           << NumberOrNull(value.scaled_gradient_norm)
           << ",\"radius_before\":" << NumberOrNull(value.radius_before)
           << ",\"radius_after\":" << NumberOrNull(value.radius_after)
           << ",\"lambda_before\":" << NumberOrNull(value.lambda_before)
           << ",\"lambda_after\":" << NumberOrNull(value.lambda_after)
           << ",\"lm_diagonal_min\":"
           << NumberOrNull(value.lm_diagonal_min)
           << ",\"lm_diagonal_max\":"
           << NumberOrNull(value.lm_diagonal_max)
           << ",\"predicted_reduction\":"
           << NumberOrNull(value.predicted_reduction)
           << ",\"actual_reduction\":"
           << NumberOrNull(value.actual_reduction)
           << ",\"rho\":" << NumberOrNull(value.rho)
           << ",\"function_metric\":" << NumberOrNull(value.function_metric)
           << ",\"parameter_metric\":" << NumberOrNull(value.parameter_metric)
           << ",\"step_norm\":" << NumberOrNull(value.step_norm)
           << ",\"backward_error\":" << NumberOrNull(value.backward_error)
           << ",\"factorization_success\":"
           << (value.factorization_success ? "true" : "false")
           << ",\"step_valid\":" << (value.step_valid ? "true" : "false")
           << ",\"trial_finite\":"
           << (value.trial_finite ? "true" : "false")
           << ",\"accepted_decision\":"
           << (value.accepted_decision ? "true" : "false")
           << ",\"accepted_commit_success\":"
           << (value.accepted_commit_success ? "true" : "false")
           << ",\"accepted\":" << (value.accepted ? "true" : "false")
           << ",\"invalid\":" << (value.invalid ? "true" : "false")
           << ",\"termination_reason\":"
           << (value.termination_reason.empty()
                   ? "null"
                   : JsonString(value.termination_reason))
           << ",\"current_state_hash\":"
           << (value.current_state_hash.empty()
                   ? "null"
                   : JsonString(value.current_state_hash))
           << ",\"trial_state_hash\":"
           << (value.trial_state_hash.empty()
                   ? "null"
                   : JsonString(value.trial_state_hash))
           << ",\"state_epoch\":" << value.state_epoch
           << ",\"linearization_id\":" << value.linearization_id
           << ",\"linearization_reason\":"
           << (value.linearization_reason.empty()
                   ? "null"
                   : JsonString(value.linearization_reason))
           << ",\"delta_layer_a_calls\":" << value.delta_layer_a_calls
           << ",\"delta_layer_b_calls\":" << value.delta_layer_b_calls
           << ",\"delta_layer_c_calls\":" << value.delta_layer_c_calls
           << ",\"delta_cost_calls\":" << value.delta_cost_calls
           << ",\"delta_topology_builds\":"
           << value.delta_topology_builds
           << ",\"delta_topology_refreshes\":"
           << value.delta_topology_refreshes
           << ",\"delta_cache_hits\":" << value.delta_cache_hits
           << ",\"delta_cache_lookups\":" << value.delta_cache_lookups
           << ",\"delta_cache_misses\":" << value.delta_cache_misses
           << ",\"delta_cache_invalidations\":"
           << value.delta_cache_invalidations
           << ",\"cumulative_layer_a_calls\":"
           << value.cumulative_layer_a_calls
           << ",\"cumulative_layer_b_calls\":"
           << value.cumulative_layer_b_calls
           << ",\"cumulative_layer_c_calls\":"
           << value.cumulative_layer_c_calls
           << ",\"cumulative_cost_calls\":"
           << value.cumulative_cost_calls
           << ",\"cumulative_topology_builds\":"
           << value.cumulative_topology_builds
           << ",\"cumulative_topology_refreshes\":"
           << value.cumulative_topology_refreshes
           << ",\"cumulative_cache_hits\":"
           << value.cumulative_cache_hits
           << "}";
  }
  output << "]\n  },\n  \"accepted_state_trace\":{"
         << "\"structure_pass\":"
         << (accepted_state_trace_structure_pass ? "true" : "false")
         << ",\"values_pass\":"
         << (accepted_state_values_pass ? "true" : "false")
         << ",\"cpu_count\":" << cpu.accepted_state_trace.size()
         << ",\"cuda_count\":" << cuda.accepted_state_trace.size()
         << ",\"states\":[";
  for (size_t i = 0; i < accepted_state_metrics.size(); ++i) {
    const StateMetrics& value = accepted_state_metrics[i];
    output << (i == 0 ? "" : ",")
           << "{\"accepted_state\":" << i
           << ",\"rotation_max_degrees\":"
           << NumberOrNull(value.rotation_max_degrees)
           << ",\"translation_max\":"
           << NumberOrNull(value.translation_max)
           << ",\"point_max\":" << NumberOrNull(value.point_max)
           << ",\"camera_max\":" << NumberOrNull(value.camera_max)
           << ",\"camera_p95\":" << NumberOrNull(value.camera_p95)
           << ",\"camera_rms\":" << NumberOrNull(value.camera_rms)
           << ",\"topology_pass\":"
           << (value.topology_pass ? "true" : "false")
           << ",\"finite_pass\":"
           << (value.finite_pass ? "true" : "false")
           << ",\"invariant_pass\":"
           << (value.invariant_pass ? "true" : "false")
           << ",\"finite_error\":"
           << (value.finite_error.empty() ? "null"
                                           : JsonString(value.finite_error))
           << ",\"invariant_error\":"
           << (value.invariant_error.empty() ? "null"
                                              : JsonString(value.invariant_error))
           << ",\"topology_error\":"
           << (value.topology_error.empty()
                   ? "null"
                   : JsonString(value.topology_error))
           << ",\"worst_rotation_id\":"
           << (value.worst_rotation_id.empty()
                   ? "null"
                   : JsonString(value.worst_rotation_id))
           << ",\"worst_translation_id\":"
           << (value.worst_translation_id.empty()
                   ? "null"
                   : JsonString(value.worst_translation_id))
           << ",\"worst_point_id\":"
           << (value.worst_point_id.empty()
                   ? "null"
                   : JsonString(value.worst_point_id))
           << "}";
  }
  output << "]},\n  \"state\":{\"topology_pass\":"
         << (state.topology_pass ? "true" : "false")
         << ",\"finite_pass\":" << (state.finite_pass ? "true" : "false")
         << ",\"invariant_pass\":"
         << (state.invariant_pass ? "true" : "false")
         << ",\"topology_error\":"
         << (state.topology_error.empty() ? "null"
                                          : JsonString(state.topology_error))
         << ",\"finite_error\":"
         << (state.finite_error.empty() ? "null" : JsonString(state.finite_error))
         << ",\"invariant_error\":"
         << (state.invariant_error.empty()
                 ? "null"
                 : JsonString(state.invariant_error))
         << ",\"rotation_max_degrees\":"
         << NumberOrNull(state.rotation_max_degrees)
         << ",\"rotation_p95_degrees\":"
         << NumberOrNull(state.rotation_p95_degrees)
         << ",\"translation_max\":"
         << NumberOrNull(state.translation_max)
         << ",\"translation_p95\":"
         << NumberOrNull(state.translation_p95)
         << ",\"point_max\":" << NumberOrNull(state.point_max)
         << ",\"point_p95\":" << NumberOrNull(state.point_p95)
         << ",\"point_rms\":" << NumberOrNull(state.point_rms)
         << ",\"camera_max\":" << NumberOrNull(state.camera_max)
         << ",\"camera_p95\":" << NumberOrNull(state.camera_p95)
         << ",\"camera_rms\":" << NumberOrNull(state.camera_rms)
         << ",\"reference_image_count\":"
         << state.reference_image_count
         << ",\"candidate_image_count\":"
         << state.candidate_image_count
         << ",\"reference_point_count\":"
         << state.reference_point_count
         << ",\"candidate_point_count\":"
         << state.candidate_point_count
         << ",\"reference_camera_count\":"
         << state.reference_camera_count
         << ",\"candidate_camera_count\":"
         << state.candidate_camera_count
         << ",\"worst_rotation_id\":"
         << (state.worst_rotation_id.empty()
                 ? "null"
                 : JsonString(state.worst_rotation_id))
         << ",\"worst_translation_id\":"
         << (state.worst_translation_id.empty()
                 ? "null"
                 : JsonString(state.worst_translation_id))
         << ",\"worst_point_id\":"
         << (state.worst_point_id.empty() ? "null"
                                          : JsonString(state.worst_point_id))
         << "},\n  \"runtime\":{\"single_stream_synchronous\":"
         << (cuda.runtime.single_stream_synchronous ? "true" : "false")
         << ",\"buffers_reused_across_iterations\":"
         << (cuda.runtime.buffers_reused_across_iterations ? "true" : "false")
         << ",\"topology_reused_across_iterations\":"
         << (cuda.runtime.topology_reused_across_iterations ? "true" : "false")
         << ",\"buffer_reuse_hits\":" << cuda.runtime.buffer_reuse_hits
         << ",\"topology_reuse_hits\":" << cuda.runtime.topology_reuse_hits
         << ",\"stream_reuse_hits\":" << cuda.runtime.stream_reuse_hits
         << ",\"event_reuse_hits\":" << cuda.runtime.event_reuse_hits
         << ",\"solver_handle_reuse_hits\":"
         << cuda.runtime.solver_handle_reuse_hits
         << ",\"blas_handle_reuse_hits\":"
         << cuda.runtime.blas_handle_reuse_hits
         << ",\"cache_access_serialized\":"
         << (cuda.runtime.cache_access_serialized ? "true" : "false")
         << ",\"cache_context_isolated\":"
         << (cuda.runtime.cache_context_isolated ? "true" : "false")
         << ","
         << "\"host_wall_milliseconds\":"
         << NumberOrNull(cuda.runtime.host_wall_milliseconds)
         << ",\"legacy_runtime_host_wall_milliseconds\":"
         << NumberOrNull(cuda.runtime.host_wall_milliseconds)
         << ",\"cuda_solve_call_wall_milliseconds\":"
         << NumberOrNull(cuda.runtime.cuda_solve_call_wall_milliseconds)
         << ",\"timing_boundaries\":{\"legacy_runtime\":"
            "\"solver body through final accepted LM state, before final "
            "audit hash and solve-cache teardown\",\"audit_accounted\":"
            "\"solver entry through final audit hash/report preparation and "
            "explicit solve-cache teardown\",\"solve_call\":"
            "\"caller interval around RunCustomCudaSolve including all RAII "
            "teardown\"}"
         << ",\"layer_a_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.layer_a_kernel_milliseconds)
         << ",\"layer_b_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.layer_b_kernel_milliseconds)
         << ",\"schur_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.schur_kernel_milliseconds)
         << ",\"factorization_milliseconds\":"
         << NumberOrNull(cuda.runtime.factorization_milliseconds)
         << ",\"cost_reduction_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.cost_reduction_kernel_milliseconds)
         << ",\"gradient_reduction_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.gradient_reduction_kernel_milliseconds)
         << ",\"allocation_milliseconds\":"
         << NumberOrNull(cuda.runtime.allocation_milliseconds)
         << ",\"copy_milliseconds\":"
         << NumberOrNull(cuda.runtime.copy_milliseconds)
         << ",\"synchronization_milliseconds\":"
         << NumberOrNull(cuda.runtime.synchronization_milliseconds)
         << ",\"point_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.point_kernel_milliseconds)
         << ",\"back_substitution_milliseconds\":"
         << NumberOrNull(cuda.runtime.back_substitution_milliseconds)
         << ",\"trial_cost_kernel_milliseconds\":"
         << NumberOrNull(cuda.runtime.trial_cost_kernel_milliseconds)
         << ",\"peak_predicted_bytes\":"
         << cuda.runtime.peak_predicted_bytes
         << ",\"peak_active_bytes\":" << cuda.runtime.peak_active_bytes
         << ",\"peak_cached_bytes\":" << cuda.runtime.peak_cached_bytes
         << ",\"peak_resident_bytes\":"
         << cuda.runtime.peak_resident_bytes
         << ",\"minimum_free_bytes\":"
         << cuda.runtime.minimum_free_bytes
         << ",\"decision_bitwise_sha256\":"
         << JsonString(CudaFullLmDecisionBitwiseSha256(cuda))
         << ",\"semantic_bitwise_sha256_v1\":"
         << JsonString(CudaFullLmSemanticBitwiseSha256V1(cuda))
         << ",\"execution_structure_sha256_v1\":"
         << JsonString(CudaFullLmExecutionStructureSha256V1(cuda))
         << ",\"semantic_sha256_v2\":"
         << JsonString(CudaFullLmSemanticSha256V2(cuda))
         << ",\"execution_sha256_v2\":"
         << JsonString(CudaFullLmExecutionSha256V2(cuda))
         << ",\"diagnostic_sha256_v2\":"
         << JsonString(CudaFullLmDiagnosticSha256V2(cuda))
         << ",\"legacy_v1_comparison_available\":"
         << (CudaLegacyV1ComparisonAvailable(cuda) ? "true" : "false")
         << ",\"legacy_error_classification\":"
         << static_cast<int>(CudaLegacyV1ErrorClassification(cuda))
         << ",\"legacy_termination_reason\":"
         << JsonString(CudaLegacyV1TerminationReason(cuda))
         << ",\"final_parameters_bitwise_sha256\":"
         << JsonString(CudaFinalParametersBitwiseSha256(cuda.final_state))
         << ",\"final_topology_bitwise_sha256\":"
         << JsonString(CudaFinalTopologyBitwiseSha256(cuda.final_state))
         << "},\n"
         << "  \"phase7_instrumentation\":";
  WriteTimingLedger(output, cuda.runtime.timing);
  output << ",\"lifecycle\":{\"initial_state_hash\":"
         << (cuda.runtime.initial_state_hash.empty()
                 ? "null"
                 : JsonString(cuda.runtime.initial_state_hash))
         << ",\"final_state_hash\":"
         << (cuda.runtime.final_state_hash.empty()
                 ? "null"
                 : JsonString(cuda.runtime.final_state_hash))
         << ",\"final_state_epoch\":"
         << cuda.runtime.final_state_epoch
         << ",\"final_internal_state_epoch\":"
         << cuda.runtime.final_internal_state_epoch
         << ",\"final_linearization_id\":"
         << cuda.runtime.final_linearization_id
         << ",\"final_linearization_reason\":"
         << (cuda.runtime.final_linearization_reason.empty()
                 ? "null"
                 : JsonString(cuda.runtime.final_linearization_reason))
         << ",\"topology_fingerprint\":"
         << (cuda.runtime.topology_fingerprint.empty()
                 ? "null"
                 : JsonString(cuda.runtime.topology_fingerprint))
         << ",\"topology_epoch\":" << cuda.runtime.topology_epoch
         << ",\"layer_a_calls\":" << cuda.runtime.layer_a_calls
         << ",\"layer_b_calls\":" << cuda.runtime.layer_b_calls
         << ",\"layer_c_calls\":" << cuda.runtime.layer_c_calls
         << ",\"cost_calls\":" << cuda.runtime.cost_calls
         << ",\"topology_build_count\":"
         << cuda.runtime.topology_build_count
         << ",\"topology_refresh_count\":"
         << cuda.runtime.topology_refresh_count
         << ",\"cache_lookup_count\":"
         << cuda.runtime.cache_lookup_count
         << ",\"cache_hit_count\":" << cuda.runtime.cache_hit_count
         << ",\"cache_miss_count\":" << cuda.runtime.cache_miss_count
         << ",\"cache_invalidation_count\":"
         << cuda.runtime.cache_invalidation_count
         << ",\"BuildCudaLayerAInputs_count\":"
         << cuda.runtime.build_cuda_layer_a_inputs_count
         << ",\"BuildCostOrder_count\":"
         << cuda.runtime.build_cost_order_count
         << ",\"current_linearization\":{\"logical_requests\":"
         << cuda.runtime.current_linearization_logical_requests
         << ",\"lookup_aborts\":"
         << cuda.runtime.current_linearization_lookup_aborts
         << ",\"cache_lookups\":"
         << cuda.runtime.current_linearization_cache_lookups
         << ",\"cache_hits\":"
         << cuda.runtime.current_linearization_cache_hits
         << ",\"cache_misses\":"
         << cuda.runtime.current_linearization_cache_misses
         << ",\"build_attempts\":"
         << cuda.runtime.current_linearization_build_attempts
         << ",\"build_successes\":"
         << cuda.runtime.current_linearization_build_successes
         << ",\"build_failures\":"
         << cuda.runtime.current_linearization_build_failures
         << ",\"temporary_builds\":"
         << cuda.runtime.current_linearization_temporary_builds
         << ",\"publishes\":"
         << cuda.runtime.current_linearization_publishes
         << ",\"replacements\":"
         << cuda.runtime.current_linearization_replacements
         << ",\"invalidations\":"
         << cuda.runtime.current_linearization_invalidations
         << ",\"borrows\":"
         << cuda.runtime.current_linearization_borrows
         << ",\"copies\":"
         << cuda.runtime.current_linearization_copies
         << ",\"teardowns\":"
         << cuda.runtime.current_linearization_teardowns
         << ",\"last_borrowed_layer_b_address\":"
         << cuda.runtime.last_borrowed_layer_b_address
         << ",\"final_config_generation\":"
         << cuda.runtime.final_config_generation
         << ",\"final_topology_generation\":"
         << cuda.runtime.final_topology_generation
         << ",\"final_solve_generation\":"
         << cuda.runtime.final_solve_generation << "}"
         << ",\"resource_health\":{\"initial\":"
         << static_cast<int>(cuda.runtime.initial_resource_health)
         << ",\"final\":"
         << static_cast<int>(cuda.runtime.final_resource_health)
         << ",\"generation\":" << cuda.runtime.resource_generation
         << ",\"taint_count\":" << cuda.runtime.resource_taint_count
         << ",\"cleanup_attempts\":"
         << cuda.runtime.resource_cleanup_attempts
         << ",\"cleanup_successes\":"
         << cuda.runtime.resource_cleanup_successes
         << ",\"cleanup_failures\":"
         << cuda.runtime.resource_cleanup_failures
         << ",\"lookup_blocked_count\":"
         << cuda.runtime.resource_lookup_blocked_count
         << ",\"cross_solve_cache_hits\":"
         << cuda.runtime.cross_solve_cache_hits << "}"
         << ",\"topology_fingerprint_computations\":"
         << cuda.runtime.topology_fingerprint_computations
         << ",\"topology_generation\":"
         << cuda.runtime.topology_generation
         << ",\"state_hash_audit\":{\"layout_builds\":"
         << cuda.runtime.state_hash_audit.layout_builds
         << ",\"layout_blocks\":"
         << cuda.runtime.state_hash_audit.layout_blocks
         << ",\"layout_host_wall_ms\":"
         << NumberOrNull(
                cuda.runtime.state_hash_audit.layout_host_wall_milliseconds)
         << ",\"current\":{\"requests\":"
         << cuda.runtime.state_hash_audit.current.requests
         << ",\"computations\":"
         << cuda.runtime.state_hash_audit.current.computations
         << ",\"cache_hits\":"
         << cuda.runtime.state_hash_audit.current.cache_hits
         << ",\"blocks_visited\":"
         << cuda.runtime.state_hash_audit.current.blocks_visited
         << ",\"scalars_hashed\":"
         << cuda.runtime.state_hash_audit.current.scalars_hashed
         << ",\"host_wall_ms\":"
         << NumberOrNull(
                cuda.runtime.state_hash_audit.current.host_wall_milliseconds)
         << "},\"trial\":{\"requests\":"
         << cuda.runtime.state_hash_audit.trial.requests
         << ",\"computations\":"
         << cuda.runtime.state_hash_audit.trial.computations
         << ",\"cache_hits\":"
         << cuda.runtime.state_hash_audit.trial.cache_hits
         << ",\"blocks_visited\":"
         << cuda.runtime.state_hash_audit.trial.blocks_visited
         << ",\"scalars_hashed\":"
         << cuda.runtime.state_hash_audit.trial.scalars_hashed
         << ",\"host_wall_ms\":"
         << NumberOrNull(
                cuda.runtime.state_hash_audit.trial.host_wall_milliseconds)
         << "}},\"current_linearization_builds_by_state_epoch\":[";
  for (size_t i = 0;
       i < cuda.runtime.current_linearization_builds_by_state_epoch.size();
       ++i) {
    output << (i == 0 ? "" : ",")
           << cuda.runtime.current_linearization_builds_by_state_epoch[i];
  }
  const CudaPersistentDeviceRuntimeInfo& persistent =
      cuda.runtime.persistent_device;
  output << "]},\"persistent_device\":{\"requested\":"
         << (persistent.requested ? "true" : "false")
         << ",\"effective\":" << (persistent.effective ? "true" : "false")
         << ",\"close_succeeded\":"
         << (persistent.close_succeeded ? "true" : "false")
         << ",\"context_identity\":" << persistent.context_identity
         << ",\"initialization_allocation_calls\":"
         << persistent.initialization_allocation_calls
         << ",\"initialization_allocation_bytes\":"
         << persistent.initialization_allocation_bytes
         << ",\"post_initialize_allocation_calls\":"
         << persistent.post_initialize_allocation_calls
         << ",\"post_initialize_allocation_bytes\":"
         << persistent.post_initialize_allocation_bytes
         << ",\"resource_create\":{\"init_stream\":"
         << persistent.init_stream_create_count
         << ",\"init_event\":" << persistent.init_event_create_count
         << ",\"init_solver\":" << persistent.init_solver_create_count
         << ",\"init_blas\":" << persistent.init_blas_create_count
         << ",\"steady_stream\":" << persistent.steady_stream_create_count
         << ",\"steady_event\":" << persistent.steady_event_create_count
         << ",\"steady_solver\":" << persistent.steady_solver_create_count
         << ",\"steady_blas\":" << persistent.steady_blas_create_count
         << "},\"resource_destroy\":{\"steady_stream\":"
         << persistent.steady_stream_destroy_count
         << ",\"steady_event\":" << persistent.steady_event_destroy_count
         << ",\"steady_solver\":" << persistent.steady_solver_destroy_count
         << ",\"steady_blas\":" << persistent.steady_blas_destroy_count
         << ",\"close_stream\":" << persistent.close_stream_destroy_count
         << ",\"close_event\":" << persistent.close_event_destroy_count
         << ",\"close_solver\":" << persistent.close_solver_destroy_count
         << ",\"close_blas\":" << persistent.close_blas_destroy_count
         << "},\"transfers\":{\"static_upload_calls\":"
         << persistent.static_upload_calls
         << ",\"static_upload_bytes\":" << persistent.static_upload_bytes
         << ",\"dynamic_upload_calls\":"
         << persistent.dynamic_state_upload_calls
         << ",\"dynamic_upload_bytes\":"
         << persistent.dynamic_state_upload_bytes
         << ",\"host_controller_d2h_calls\":"
         << persistent.host_controller_d2h_calls
         << ",\"host_controller_d2h_bytes\":"
         << persistent.host_controller_d2h_bytes
         << ",\"host_diagnostics_d2h_calls\":"
         << persistent.host_diagnostics_d2h_calls
         << ",\"host_diagnostics_d2h_bytes\":"
         << persistent.host_diagnostics_d2h_bytes
         << ",\"a_to_b_h2d_calls\":" << persistent.a_to_b_h2d_calls
         << ",\"a_to_b_h2d_bytes\":" << persistent.a_to_b_h2d_bytes
         << ",\"a_to_b_d2h_calls\":" << persistent.a_to_b_d2h_calls
         << ",\"a_to_b_d2h_bytes\":" << persistent.a_to_b_d2h_bytes
         << ",\"a_to_cost_h2d_calls\":" << persistent.a_to_cost_h2d_calls
         << ",\"a_to_cost_h2d_bytes\":" << persistent.a_to_cost_h2d_bytes
         << ",\"a_to_cost_d2h_calls\":" << persistent.a_to_cost_d2h_calls
         << ",\"a_to_cost_d2h_bytes\":" << persistent.a_to_cost_d2h_bytes
         << ",\"b_to_c_h2d_calls\":" << persistent.b_to_c_h2d_calls
         << ",\"b_to_c_h2d_bytes\":" << persistent.b_to_c_h2d_bytes
         << ",\"b_to_c_d2h_calls\":" << persistent.b_to_c_d2h_calls
         << ",\"b_to_c_d2h_bytes\":" << persistent.b_to_c_d2h_bytes
         << "},\"frozen_scaling_upload_calls\":"
         << persistent.frozen_scaling_upload_calls
         << ",\"frozen_scaling_upload_bytes\":"
         << persistent.frozen_scaling_upload_bytes
         << ",\"pending_commit_device_copy_bytes\":"
         << persistent.pending_commit_device_copy_bytes
         << ",\"large_b_copy_bytes\":" << persistent.large_b_copy_bytes
         << ",\"cusolver_workspace_capacity_bytes\":"
         << persistent.cusolver_workspace_capacity_bytes
         << ",\"workspace_capacity_bytes\":"
         << persistent.workspace_capacity_bytes
         << ",\"peak_resident_bytes\":" << persistent.peak_resident_bytes
         << ",\"slots\":{\"publish\":" << persistent.slot_publish_count
         << ",\"swap\":" << persistent.slot_swap_count
         << ",\"invalidate\":" << persistent.slot_invalidate_count
         << ",\"stale_reject\":" << persistent.stale_handle_reject_count
         << "},\"device_state\":{\"static_problem_layout_builds\":"
         << persistent.static_problem_layout_builds
         << ",\"initial_entity_state_pack_calls\":"
         << persistent.initial_entity_state_pack_calls
         << ",\"initial_entity_state_pack_bytes\":"
         << persistent.initial_entity_state_pack_bytes
         << ",\"initial_entity_state_upload_calls\":"
         << persistent.initial_entity_state_upload_calls
         << ",\"initial_entity_state_upload_bytes\":"
         << persistent.initial_entity_state_upload_bytes
         << ",\"post_ready_dynamic_state_h2d_calls\":"
         << persistent.post_ready_dynamic_state_h2d_calls
         << ",\"post_ready_dynamic_state_h2d_bytes\":"
         << persistent.post_ready_dynamic_state_h2d_bytes
         << ",\"per_observation_dynamic_pack_calls\":"
         << persistent.per_observation_dynamic_pack_calls
         << ",\"per_observation_dynamic_pack_bytes\":"
         << persistent.per_observation_dynamic_pack_bytes
         << ",\"state_update_kernel_calls\":"
         << persistent.state_update_kernel_calls
         << ",\"trial_entity_state_d2h_calls\":"
         << persistent.trial_entity_state_d2h_calls
         << ",\"trial_entity_state_d2h_bytes\":"
         << persistent.trial_entity_state_d2h_bytes
         << ",\"final_entity_state_d2h_calls\":"
         << persistent.final_entity_state_d2h_calls
         << ",\"final_entity_state_d2h_bytes\":"
         << persistent.final_entity_state_d2h_bytes
         << ",\"slot_publish\":"
         << persistent.state_slot_publish_count
         << ",\"slot_swap\":" << persistent.state_slot_swap_count
         << ",\"slot_discard\":" << persistent.state_slot_discard_count
         << ",\"slot_invalidate\":"
         << persistent.state_slot_invalidate_count
         << ",\"reject_slot_swap\":"
         << persistent.reject_state_slot_swap_count
         << ",\"lineage_checks\":" << persistent.state_lineage_checks
         << ",\"lineage_violations\":"
         << persistent.state_lineage_violations
         << ",\"state_b_pair_checks\":"
         << persistent.state_b_pair_checks
         << ",\"state_b_pair_violations\":"
         << persistent.state_b_pair_violations
         << ",\"accepted_commit_device_copy_bytes\":"
         << persistent.accepted_commit_device_copy_bytes
         << ",\"cross_solve_state_handle_hits\":"
         << persistent.cross_solve_state_handle_hits
         << "},\"device_control\":{\"scalar_packet_size_bytes\":"
         << persistent.scalar_packet_size_bytes
         << ",\"scalar_packet_d2h_calls\":"
         << persistent.scalar_packet_d2h_calls
         << ",\"scalar_packet_d2h_bytes\":"
         << persistent.scalar_packet_d2h_bytes
         << ",\"steady_full_b_d2h_calls\":"
         << persistent.steady_full_b_d2h_calls
         << ",\"steady_full_b_d2h_bytes\":"
         << persistent.steady_full_b_d2h_bytes
         << ",\"steady_schur_d2h_calls\":"
         << persistent.steady_schur_d2h_calls
         << ",\"steady_schur_d2h_bytes\":"
         << persistent.steady_schur_d2h_bytes
         << ",\"steady_delta_d2h_calls\":"
         << persistent.steady_delta_d2h_calls
         << ",\"steady_delta_d2h_bytes\":"
         << persistent.steady_delta_d2h_bytes
         << ",\"steady_trial_state_d2h_calls\":"
         << persistent.steady_trial_state_d2h_calls
         << ",\"steady_trial_state_d2h_bytes\":"
         << persistent.steady_trial_state_d2h_bytes
         << ",\"final_state_materialization_operations\":"
         << persistent.final_state_materialization_operations
         << ",\"final_state_d2h_calls\":"
         << persistent.final_state_d2h_calls
         << ",\"final_state_d2h_bytes\":"
         << persistent.final_state_d2h_bytes
         << ",\"bootstrap_scaling_mirror_calls\":"
         << persistent.bootstrap_scaling_mirror_calls
         << ",\"bootstrap_scaling_mirror_bytes\":"
         << persistent.bootstrap_scaling_mirror_bytes
         << ",\"audit_mirror_b_calls\":"
         << persistent.audit_mirror_b_calls
         << ",\"audit_mirror_b_bytes\":"
         << persistent.audit_mirror_b_bytes
         << ",\"audit_mirror_state_calls\":"
         << persistent.audit_mirror_state_calls
         << ",\"audit_mirror_state_bytes\":"
         << persistent.audit_mirror_state_bytes
         << ",\"audit_mirror_schur_calls\":"
         << persistent.audit_mirror_schur_calls
         << ",\"audit_mirror_schur_bytes\":"
         << persistent.audit_mirror_schur_bytes
         << ",\"audit_mirror_delta_calls\":"
         << persistent.audit_mirror_delta_calls
         << ",\"audit_mirror_delta_bytes\":"
         << persistent.audit_mirror_delta_bytes
         << ",\"diagnostic_kernel_calls\":"
         << persistent.diagnostic_kernel_calls
         << ",\"diagnostic_reduction_failures\":"
         << persistent.diagnostic_reduction_failures
         << ",\"diagnostic_kernel_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_kernel_milliseconds)
         << ",\"diagnostic_point_status_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_point_status_milliseconds)
         << ",\"diagnostic_lm_diagonal_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_lm_diagonal_milliseconds)
         << ",\"diagnostic_pose_partial_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_pose_partial_milliseconds)
         << ",\"diagnostic_point_partial_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_point_partial_milliseconds)
         << ",\"diagnostic_merge_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_merge_milliseconds)
         << ",\"diagnostic_state_norm_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_state_norm_milliseconds)
         << ",\"diagnostic_symmetry_milliseconds\":"
         << NumberOrNull(persistent.diagnostic_symmetry_milliseconds)
         << ",\"diagnostic_packet_copy_sync_milliseconds\":"
         << NumberOrNull(
                persistent.diagnostic_packet_copy_sync_milliseconds)
         << ",\"diagnostic_total_inclusive_milliseconds\":"
         << NumberOrNull(
                persistent.diagnostic_total_inclusive_milliseconds)
         << ",\"diagnostic_all_added_inclusive_milliseconds\":"
         << NumberOrNull(
                persistent.diagnostic_all_added_inclusive_milliseconds)
         << ",\"diagnostic_workspace_bytes\":"
         << persistent.diagnostic_workspace_bytes
         << ",\"diagnostic_last_symmetry_error\":"
         << NumberOrNull(persistent.diagnostic_last_symmetry_error)
         << ",\"diagnostic_max_symmetry_error\":"
         << NumberOrNull(persistent.diagnostic_max_symmetry_error)
         << ",\"commit_token_checks\":"
         << persistent.commit_token_checks
         << ",\"commit_token_violations\":"
         << persistent.commit_token_violations
         << ",\"b_slot_swap_count\":"
         << persistent.b_slot_swap_count
         << ",\"reject_b_slot_swap_count\":"
         << persistent.reject_b_slot_swap_count
         << ",\"hot_kernel_implementation\":\""
         << (persistent.hot_kernel_transformed
                 ? "transformed"
                 : (persistent.hot_kernel_optimized ? "optimized"
                                                    : "reference"))
         << "\",\"transform_kernel_calls\":"
         << persistent.transform_kernel_calls
         << ",\"hessian_assembly_backend_requested\":"
         << JsonString(persistent.hessian_assembly_backend_requested)
         << ",\"hessian_assembly_backend_effective\":"
         << JsonString(persistent.hessian_assembly_backend_effective)
         << ",\"hessian_gradient_assembly_calls\":"
         << persistent.hessian_gradient_assembly_calls
         << ",\"pose_block_assembly_calls\":"
         << persistent.pose_block_assembly_calls
         << ",\"point_block_assembly_calls\":"
         << persistent.point_block_assembly_calls
         << ",\"edge_block_assembly_calls\":"
         << persistent.edge_block_assembly_calls
         << ",\"jacobi_damping_finalize_calls\":"
         << persistent.jacobi_damping_finalize_calls
         << ",\"gradient_summary_calls\":"
         << persistent.gradient_summary_calls
         << ",\"observation_atomic_kernel_launches\":"
         << persistent.observation_atomic_kernel_launches
         << ",\"observation_segment_partial_launches\":"
         << persistent.observation_segment_partial_launches
         << ",\"observation_segment_merge_launches\":"
         << persistent.observation_segment_merge_launches
         << ",\"output_zero_kernel_launches\":"
         << persistent.output_zero_kernel_launches
         << ",\"jacobi_finalize_kernel_launches\":"
         << persistent.jacobi_finalize_kernel_launches
         << ",\"gradient_summary_launches\":"
         << persistent.gradient_summary_launches
         << ",\"atomic_add_estimate_pose\":"
         << persistent.atomic_add_estimate_pose
         << ",\"atomic_add_estimate_point\":"
         << persistent.atomic_add_estimate_point
         << ",\"atomic_add_estimate_edge\":"
         << persistent.atomic_add_estimate_edge
         << ",\"hessian_segment_count_pose\":"
         << persistent.hessian_segment_count_pose
         << ",\"hessian_segment_count_point\":"
         << persistent.hessian_segment_count_point
         << ",\"hessian_segment_count_edge\":"
         << persistent.hessian_segment_count_edge
         << ",\"hessian_segment_size\":"
         << persistent.hessian_segment_size
         << ",\"hessian_partial_workspace_bytes\":"
         << persistent.hessian_partial_workspace_bytes
         << ",\"hessian_candidate_metadata_h2d_bytes\":"
         << persistent.hessian_candidate_metadata_h2d_bytes
         << ",\"hessian_candidate_metadata_build_calls\":"
         << persistent.hessian_candidate_metadata_build_calls
         << ",\"hessian_assembly_coverage_violations\":"
         << persistent.hessian_assembly_coverage_violations
         << ",\"arithmetic_precision_requested\":"
         << JsonString(persistent.arithmetic_precision_requested)
         << ",\"arithmetic_precision_effective\":"
         << JsonString(persistent.arithmetic_precision_effective)
         << ",\"state_storage_precision\":"
         << JsonString(persistent.state_storage_precision)
         << ",\"residual_jacobian_precision\":"
         << JsonString(persistent.residual_jacobian_precision)
         << ",\"hessian_schur_precision\":"
         << JsonString(persistent.hessian_schur_precision)
         << ",\"factorization_routine\":"
         << JsonString(persistent.factorization_routine)
         << ",\"delta_precision\":"
         << JsonString(persistent.delta_precision)
         << ",\"quaternion_plus_precision\":"
         << JsonString(persistent.quaternion_plus_precision)
         << ",\"cost_precision\":"
         << JsonString(persistent.cost_precision)
         << ",\"controller_precision\":"
         << JsonString(persistent.controller_precision)
         << ",\"float_buffer_allocation_calls\":"
         << persistent.float_buffer_allocation_calls
         << ",\"float_buffer_allocation_bytes\":"
         << persistent.float_buffer_allocation_bytes
         << ",\"float_static_upload_calls\":"
         << persistent.float_static_upload_calls
         << ",\"float_static_upload_bytes\":"
         << persistent.float_static_upload_bytes
         << ",\"float_arena_reserved_bytes\":"
         << persistent.float_arena_reserved_bytes
         << ",\"float_workspace_bytes\":"
         << persistent.float_workspace_bytes
         << ",\"double_workspace_bytes\":"
         << persistent.double_workspace_bytes
         << ",\"spotrf_calls\":" << persistent.spotrf_calls
         << ",\"spotrs_calls\":" << persistent.spotrs_calls
         << ",\"dpotrf_calls\":" << persistent.dpotrf_calls
         << ",\"dpotrs_calls\":" << persistent.dpotrs_calls
         << ",\"fp64_cost_calls\":" << persistent.fp64_cost_calls
         << ",\"float_state_update_calls\":"
         << persistent.float_state_update_calls
         << ",\"state_quantization_calls\":"
         << persistent.state_quantization_calls
         << ",\"precision_mirror_cross_hits\":"
         << persistent.precision_mirror_cross_hits
         << ",\"fixed_external_write_attempts\":"
         << persistent.fixed_external_write_attempts
         << ",\"mixed_schur_math_effective\":"
         << JsonString(persistent.mixed_schur_math_effective)
         << ",\"mixed_float_buffer_bytes\":"
         << persistent.mixed_float_buffer_bytes
         << ",\"mixed_double_buffer_bytes\":"
         << persistent.mixed_double_buffer_bytes
         << ",\"mixed_float_to_double_schur_conversion_calls\":"
         << persistent.mixed_float_to_double_schur_calls
         << ",\"mixed_float_to_double_schur_conversion_bytes\":"
         << persistent.mixed_float_to_double_schur_bytes
         << ",\"mixed_fp64_gradient_calls\":"
         << persistent.mixed_fp64_gradient_calls
         << ",\"mixed_fp64_point_inverse_calls\":"
         << persistent.mixed_fp64_point_inverse_calls
         << ",\"mixed_fp64_rhs_calls\":"
         << persistent.mixed_fp64_rhs_calls
         << ",\"mixed_fp64_back_substitution_calls\":"
         << persistent.mixed_fp64_back_substitution_calls
         << ",\"mixed_full_array_d2h_bytes\":"
         << persistent.mixed_full_array_d2h_bytes
         << ",\"mixed_scalar_packet_calls\":"
         << persistent.mixed_scalar_packet_calls
         << ",\"mixed_scalar_packet_bytes\":"
         << persistent.mixed_scalar_packet_bytes
         << ",\"mixed_serial_full_scan_kernel_count\":"
         << persistent.mixed_serial_full_scan_kernel_count
         << ",\"hessian_precision\":"
         << JsonString(persistent.hessian_precision)
         << ",\"residual_jacobian_operand_source\":"
         << JsonString(persistent.residual_jacobian_operand_source)
         << ",\"gradient_operand_precision\":"
         << JsonString(persistent.gradient_operand_precision)
         << ",\"gradient_accumulation_precision\":"
         << JsonString(persistent.gradient_accumulation_precision)
         << ",\"point_inverse_precision\":"
         << JsonString(persistent.point_inverse_precision)
         << ",\"schur_contribution_precision\":"
         << JsonString(persistent.schur_contribution_precision)
         << ",\"dense_factorization_precision\":"
         << JsonString(persistent.dense_factorization_precision)
         << ",\"mixed_fp64_gradient_from_fp64_records_calls\":"
         << persistent.mixed_fp64_gradient_from_fp64_records_calls
         << ",\"mixed_fp64_gradient_from_fp32_records_calls\":"
         << persistent.mixed_fp64_gradient_from_fp32_records_calls
         << ",\"mixed_native_fp32_point_inverse_cast_calls\":"
         << persistent.mixed_native_fp32_point_inverse_cast_calls
         << ",\"mixed_native_fp32_point_inverse_cast_bytes\":"
         << persistent.mixed_native_fp32_point_inverse_cast_bytes
         << ",\"mixed_native_fp32_transform_calls\":"
         << persistent.mixed_native_fp32_transform_calls
         << ",\"mixed_native_fp32_transform_bytes\":"
         << persistent.mixed_native_fp32_transform_bytes
         << ",\"mixed_native_fp32_schur_calls\":"
         << persistent.mixed_native_fp32_schur_calls
         << ",\"mixed_native_fp32_schur_partial_calls\":"
         << persistent.mixed_native_fp32_schur_partial_calls
         << ",\"mixed_native_fp32_schur_merge_calls\":"
         << persistent.mixed_native_fp32_schur_merge_calls
         << ",\"mixed_native_fp32_dense_conversion_calls\":"
         << persistent.mixed_native_fp32_dense_conversion_calls
         << ",\"mixed_native_fp32_dense_conversion_source_bytes\":"
         << persistent.mixed_native_fp32_dense_conversion_source_bytes
         << ",\"mixed_native_fp32_dense_conversion_destination_bytes\":"
         << persistent.mixed_native_fp32_dense_conversion_destination_bytes
         << ",\"mixed_layer_a_cast_calls\":"
         << persistent.mixed_layer_a_cast_calls
         << ",\"mixed_layer_a_cast_bytes\":"
         << persistent.mixed_layer_a_cast_bytes
         << ",\"mixed_fp64_pose_damping_calls\":"
         << persistent.mixed_fp64_pose_damping_calls
         << ",\"mixed_fp64_schur_contribution_calls\":"
         << persistent.mixed_fp64_schur_contribution_calls
         << ",\"mixed_double_edge_materialization_calls\":"
         << persistent.mixed_double_edge_materialization_calls
         << ",\"mixed_double_edge_materialization_bytes\":"
         << persistent.mixed_double_edge_materialization_bytes
         << ",\"schur_solve_backward_error_last\":"
         << NumberOrNull(persistent.schur_solve_backward_error_last)
         << ",\"schur_solve_backward_error_max\":"
         << NumberOrNull(persistent.schur_solve_backward_error_max)
         << ",\"mixed_residual_jacobian_milliseconds\":"
         << NumberOrNull(persistent.mixed_residual_jacobian_milliseconds)
         << ",\"mixed_hessian_milliseconds\":"
         << NumberOrNull(persistent.mixed_hessian_milliseconds)
         << ",\"mixed_gradient_milliseconds\":"
         << NumberOrNull(persistent.mixed_gradient_milliseconds)
         << ",\"mixed_point_inverse_milliseconds\":"
         << NumberOrNull(persistent.mixed_point_inverse_milliseconds)
         << ",\"mixed_schur_milliseconds\":"
         << NumberOrNull(persistent.mixed_schur_milliseconds)
         << ",\"mixed_rhs_milliseconds\":"
         << NumberOrNull(persistent.mixed_rhs_milliseconds)
         << ",\"mixed_back_substitution_milliseconds\":"
         << NumberOrNull(persistent.mixed_back_substitution_milliseconds)
         << ",\"mixed_diagnostics_milliseconds\":"
         << NumberOrNull(persistent.mixed_diagnostics_milliseconds)
         << ",\"transform_workspace_bytes\":"
         << persistent.transform_workspace_bytes
         << ",\"transform_kernel_milliseconds\":"
         << NumberOrNull(persistent.transform_kernel_milliseconds)
         << ",\"transformed_pair_kernel_milliseconds\":"
         << NumberOrNull(persistent.transformed_pair_kernel_milliseconds)
         << ",\"layer_c_pair_chunk_launch_calls\":"
         << persistent.layer_c_pair_chunk_launch_calls
         << ",\"layer_c_max_chunks_per_step\":"
         << persistent.layer_c_max_chunks_per_step
         << ",\"layer_c_max_chunk_bytes\":"
         << persistent.layer_c_max_chunk_bytes
         << ",\"schur_contribution_backend_requested\":"
         << JsonString(persistent.schur_contribution_backend_requested)
         << ",\"schur_contribution_backend_effective\":"
         << JsonString(persistent.schur_contribution_backend_effective)
         << ",\"schur_contribution_calls\":"
         << persistent.schur_contribution_calls
         << ",\"schur_rhs_calls\":" << persistent.schur_rhs_calls
         << ",\"factorization_calls\":" << persistent.factorization_calls
         << ",\"direct_pair_count\":" << persistent.direct_pair_count
         << ",\"segmented_pair_count\":"
         << persistent.segmented_pair_count
         << ",\"segment_count\":" << persistent.segment_count
         << ",\"max_segments_per_pair\":"
         << persistent.max_segments_per_pair
         << ",\"schur_segment_size\":" << persistent.schur_segment_size
         << ",\"schur_partial_workspace_bytes\":"
         << persistent.schur_partial_workspace_bytes
         << ",\"schur_segment_plan_builds\":"
         << persistent.schur_segment_plan_builds
         << ",\"schur_segment_metadata_h2d_bytes\":"
         << persistent.schur_segment_metadata_h2d_bytes
         << ",\"schur_workspace_init_kernel_launches\":"
         << persistent.schur_workspace_init_kernel_launches
         << ",\"schur_partial_kernel_launches\":"
         << persistent.schur_partial_kernel_launches
         << ",\"schur_merge_kernel_launches\":"
         << persistent.schur_merge_kernel_launches
         << ",\"schur_pair_contribution_chunk_launches\":"
         << persistent.schur_pair_contribution_chunk_launches
         << ",\"schur_contribution_coverage_violations\":"
         << persistent.schur_contribution_coverage_violations
         << ",\"schur_contribution_inclusive_milliseconds\":"
         << NumberOrNull(
                persistent.schur_contribution_inclusive_milliseconds)
         << ",\"solve_context_initialization_wall_milliseconds\":"
         << NumberOrNull(
                persistent.solve_context_initialization_wall_milliseconds)
         << ",\"problem_topology_preparation_wall_milliseconds\":"
         << NumberOrNull(
                persistent.problem_topology_preparation_wall_milliseconds)
         << ",\"schur_segment_plan_host_milliseconds\":"
         << NumberOrNull(persistent.schur_segment_plan_host_milliseconds)
         << ",\"layer_b_pose_milliseconds\":"
         << NumberOrNull(persistent.layer_b_pose_milliseconds)
         << ",\"layer_b_point_milliseconds\":"
         << NumberOrNull(persistent.layer_b_point_milliseconds)
         << ",\"layer_b_edge_milliseconds\":"
         << NumberOrNull(persistent.layer_b_edge_milliseconds)
         << ",\"layer_b_gradient_milliseconds\":"
         << NumberOrNull(persistent.layer_b_gradient_milliseconds)
         << ",\"layer_b_total_inclusive_milliseconds\":"
         << NumberOrNull(persistent.layer_b_total_inclusive_milliseconds)
         << ",\"schur_zero_milliseconds\":"
         << NumberOrNull(persistent.schur_zero_milliseconds)
         << ",\"schur_pose_init_milliseconds\":"
         << NumberOrNull(persistent.schur_pose_init_milliseconds)
         << ",\"schur_pair_milliseconds\":"
         << NumberOrNull(persistent.schur_pair_milliseconds)
         << ",\"schur_rhs_milliseconds\":"
         << NumberOrNull(persistent.schur_rhs_milliseconds)
         << ",\"schur_preservation_copy_milliseconds\":"
         << NumberOrNull(persistent.schur_preservation_copy_milliseconds)
         << ",\"schur_factorization_milliseconds\":"
         << NumberOrNull(persistent.schur_factorization_milliseconds)
         << ",\"schur_back_substitution_milliseconds\":"
         << NumberOrNull(persistent.schur_back_substitution_milliseconds)
         << ",\"schur_total_inclusive_milliseconds\":"
         << NumberOrNull(persistent.schur_total_inclusive_milliseconds)
         << ",\"runtime_pool_acquire_calls\":"
         << persistent.runtime_pool_acquire_calls
         << ",\"runtime_pool_cold_creates\":"
         << persistent.runtime_pool_cold_creates
         << ",\"runtime_pool_hot_leases\":"
         << persistent.runtime_pool_hot_leases
         << ",\"runtime_pool_returns\":"
         << persistent.runtime_pool_returns
         << ",\"runtime_pool_retained_entries\":"
         << persistent.runtime_pool_retained_entries
         << ",\"runtime_pool_discards\":"
         << persistent.runtime_pool_discards
         << ",\"runtime_pool_destroys\":"
         << persistent.runtime_pool_destroys
         << ",\"runtime_pool_poison_count\":"
         << persistent.runtime_pool_poison_count
         << ",\"runtime_pool_lease_generation\":"
         << persistent.runtime_pool_lease_generation
         << ",\"arena_capacity_bytes\":"
         << persistent.arena_capacity_bytes
         << ",\"arena_required_bytes\":"
         << persistent.arena_required_bytes
         << ",\"arena_retained_bytes\":"
         << persistent.arena_retained_bytes
         << ",\"arena_slice_count\":"
         << persistent.arena_slice_count
         << ",\"arena_slice_bytes\":"
         << persistent.arena_slice_bytes
         << ",\"arena_grow_calls\":"
         << persistent.arena_grow_calls
         << ",\"arena_grow_bytes\":"
         << persistent.arena_grow_bytes
         << ",\"bootstrap_scaling_full_d2h_calls\":"
         << persistent.bootstrap_scaling_full_d2h_calls
         << ",\"bootstrap_scaling_full_d2h_bytes\":"
         << persistent.bootstrap_scaling_full_d2h_bytes
         << ",\"scaling_d2d_calls\":"
         << persistent.scaling_d2d_calls
         << ",\"scaling_d2d_bytes\":"
         << persistent.scaling_d2d_bytes
         << ",\"scaling_slot_publishes\":"
         << persistent.scaling_slot_publishes
         << ",\"production_identity_fingerprint_calls\":"
         << persistent.production_identity_fingerprint_calls
         << ",\"production_identity_fingerprint_bytes\":"
         << persistent.production_identity_fingerprint_bytes
         << ",\"unified_problem_builder_calls\":"
         << persistent.unified_problem_builder_calls
         << ",\"unified_problem_builder_traversals\":"
         << persistent.unified_problem_builder_traversals
         << ",\"independent_cost_order_rebuilds\":"
         << persistent.independent_cost_order_rebuilds
         << ",\"public_visual_record_bytes\":"
         << persistent.public_visual_record_bytes
         << ",\"compact_visual_record_bytes\":"
         << persistent.compact_visual_record_bytes
         << ",\"compact_layer_a_slot_bytes\":"
         << persistent.compact_layer_a_slot_bytes
         << ",\"full_layer_a_slot_bytes\":"
         << persistent.full_layer_a_slot_bytes
         << ",\"compact_layer_a_calls\":"
         << persistent.compact_layer_a_calls
         << ",\"scalar_current_packet_calls\":"
         << persistent.scalar_current_packet_calls
         << ",\"scalar_factor_status_packet_calls\":"
         << persistent.scalar_factor_status_packet_calls
         << ",\"scalar_trial_packet_calls\":"
         << persistent.scalar_trial_packet_calls
         << ",\"redundant_synchronization_calls\":"
         << persistent.redundant_synchronization_calls
         << "}},\"v2_audit\":{\"audit_capacity_preflight_pass\":"
         << (cuda.runtime.audit_capacity_preflight_pass ? "true" : "false")
         << ",\"fault_record_capacity\":"
         << cuda.runtime.fault_record_capacity
         << ",\"secondary_diagnostic_capacity\":"
         << cuda.runtime.secondary_diagnostic_capacity
         << ",\"resource_registry_capacity\":"
         << cuda.runtime.resource_registry_capacity
         << ",\"max_open_timing_interval_capacity\":"
         << cuda.runtime.max_open_timing_interval_capacity
         << ",\"audit_overflow_count\":"
         << cuda.runtime.audit_overflow_count
         << ",\"first_audit_overflow_site\":"
         << (cuda.runtime.has_first_audit_overflow_site
                 ? std::to_string(static_cast<int>(
                       cuda.runtime.first_audit_overflow_site))
                 : "null")
         << ",\"generation_checks\":{\"solve_nonzero\":{"
         << "\"checks\":" << cuda.runtime.solve_generation_nonzero_checks
         << ",\"violations\":"
         << cuda.runtime.solve_generation_nonzero_violations
         << ",\"consistent\":"
         << (cuda.runtime.solve_generation_nonzero ? "true" : "false")
         << "},\"solve\":{\"checks\":"
         << cuda.runtime.solve_generation_consistency_checks
         << ",\"violations\":"
         << cuda.runtime.solve_generation_consistency_violations
         << ",\"consistent\":"
         << (cuda.runtime.solve_generation_consistent ? "true" : "false")
         << "},\"linearization\":{\"checks\":"
         << cuda.runtime.linearization_identity_checks
         << ",\"violations\":"
         << cuda.runtime.linearization_identity_violations
         << ",\"consistent\":"
         << (cuda.runtime.linearization_identity_consistent ? "true"
                                                             : "false")
         << "},\"topology_context\":{\"checks\":"
         << cuda.runtime.topology_context_generation_checks
         << ",\"violations\":"
         << cuda.runtime.topology_context_generation_violations
         << ",\"consistent\":"
         << (cuda.runtime.topology_context_generation_consistent ? "true"
                                                                  : "false")
         << "}},\"resource\":{\"advance_events\":"
         << cuda.runtime.resource_generation_advance_events
         << ",\"advance_violations\":"
         << cuda.runtime.resource_generation_advance_violations
         << ",\"advanced_by_this_solve\":"
         << (cuda.runtime.resource_generation_advanced_by_this_solve
                 ? "true"
                 : "false")
         << ",\"quarantine_count\":"
         << cuda.runtime.resource_quarantine_count
         << ",\"cleanup_complete\":"
         << (cuda.runtime.resource_cleanup_complete ? "true" : "false")
         << ",\"teardown_record_count\":"
         << cuda.runtime.teardown_record_count
         << ",\"teardown_records\":[";
  for (size_t i = 0; i < cuda.runtime.teardown_records.size(); ++i) {
    const CudaTeardownRecordV2& record = cuda.runtime.teardown_records[i];
    output << (i == 0 ? "" : ",")
           << "{\"teardown_type\":" << static_cast<int>(record.teardown_type)
           << ",\"device\":" << record.device
           << ",\"creation_sequence\":" << record.creation_sequence
           << ",\"attempted\":" << (record.attempted ? "true" : "false")
           << ",\"success\":" << (record.success ? "true" : "false")
           << ",\"status_domain\":" << static_cast<int>(record.status_domain)
           << ",\"raw_status_code\":" << record.raw_status_code
           << ",\"quarantined\":"
           << (record.quarantined ? "true" : "false") << '}';
  }
  output << "],\"first_cleanup_failure\":";
  if (cuda.runtime.has_first_cleanup_failure) {
    WriteStableDiagnostic(output, cuda.runtime.first_cleanup_failure);
  } else {
    output << "null";
  }
  output << "},\"timing\":{\"status\":"
         << static_cast<int>(cuda.runtime.timing_structure_v2.timing_status)
         << ",\"event_attempts\":"
         << cuda.runtime.timing_structure_v2.event_attempts
         << ",\"event_completions\":"
         << cuda.runtime.timing_structure_v2.event_completions
         << ",\"event_failures\":"
         << cuda.runtime.timing_structure_v2.event_failures
         << ",\"intervals_started\":"
         << cuda.runtime.timing_structure_v2.intervals_started
         << ",\"intervals_completed\":"
         << cuda.runtime.timing_structure_v2.intervals_completed
         << ",\"intervals_abandoned\":"
         << cuda.runtime.timing_structure_v2.intervals_abandoned
         << ",\"open_intervals_at_finalize\":"
         << cuda.runtime.timing_structure_v2.open_intervals_at_finalize
         << ",\"first_incomplete_site\":"
         << (cuda.runtime.timing_structure_v2.has_first_incomplete_site
                 ? std::to_string(static_cast<int>(
                       cuda.runtime.timing_structure_v2.first_incomplete_site))
                 : "null")
         << "},\"diagnostics\":{\"has_primary\":"
         << (cuda.runtime.diagnostic_structure_v2.has_primary ? "true"
                                                               : "false")
         << ",\"primary\":";
  if (cuda.runtime.diagnostic_structure_v2.has_primary) {
    WriteStableDiagnostic(output,
                          cuda.runtime.diagnostic_structure_v2.primary);
  } else {
    output << "null";
  }
  output << ",\"secondary_count\":"
         << cuda.runtime.diagnostic_structure_v2.secondary_count
         << ",\"secondary\":[";
  const uint64_t secondary_count = std::min<uint64_t>(
      cuda.runtime.diagnostic_structure_v2.secondary_count,
      cuda.runtime.diagnostic_structure_v2.secondary.size());
  for (uint64_t i = 0; i < secondary_count; ++i) {
    output << (i == 0 ? "" : ",");
    WriteStableDiagnostic(
        output, cuda.runtime.diagnostic_structure_v2.secondary[
                    static_cast<size_t>(i)]);
  }
  output << "]"
         << ",\"overflow_sentinel_set\":"
         << (cuda.runtime.diagnostic_structure_v2.overflow_sentinel_set
                 ? "true"
                 : "false")
         << ",\"diagnostic_overflow_count\":"
         << cuda.runtime.diagnostic_structure_v2.diagnostic_overflow_count
         << ",\"overflow_sentinel\":";
  if (cuda.runtime.diagnostic_structure_v2.overflow_sentinel_set) {
    WriteStableDiagnostic(
        output, cuda.runtime.diagnostic_structure_v2.overflow_sentinel);
  } else {
    output << "null";
  }
  output << "},\"fault_consumptions\":[";
  const uint64_t fault_count = std::min<uint64_t>(
      cuda.runtime.fault_consumption_count,
      cuda.runtime.fault_consumptions.size());
  for (uint64_t i = 0; i < fault_count; ++i) {
    const CudaFaultConsumptionV2& fault =
        cuda.runtime.fault_consumptions[static_cast<size_t>(i)];
    output << (i == 0 ? "" : ",")
           << "{\"logical_site\":" << static_cast<int>(fault.logical_site)
           << ",\"trigger_phase\":" << static_cast<int>(fault.trigger_phase)
           << ",\"operation\":" << fault.operation
           << ",\"configured_fault_kind\":"
           << static_cast<int>(fault.configured_fault_kind)
           << ",\"epoch\":" << fault.epoch
           << ",\"occurrence\":" << fault.occurrence
           << ",\"triggered\":" << (fault.triggered ? "true" : "false")
           << '}';
  }
  output << "]},\"forced_reject_probe\":{"
         << "\"forced_reject_probe_requested\":"
         << (forced_reject_probe ? "true" : "false")
         << ",\"probe_valid\":"
         << (cuda.runtime.probe_valid ? "true" : "false")
         << ",\"actual_trials\":" << cuda.runtime.actual_trials
         << ",\"accepted_trials\":" << cuda.runtime.accepted_trials
         << ",\"rejected_trials\":" << cuda.runtime.rejected_trials
         << "},\n"
         << "  \"debug_state_capture\":{\"enabled\":"
         << (capture_states ? "true" : "false")
         << ",\"root\":"
         << (capture_states ? JsonString(state_root) : "null")
         << "}\n}\n";
  // The probe is an audit artifact, not a fidelity gate. Preserve the real
  // pass/fail and trace in JSON while allowing the run harness to collect a
  // probe that happened to reject or accept naturally.
  const bool precision_experiment =
      arithmetic_precision_requested != "compatibility_default";
  const bool precision_experiment_pass =
      cuda_ran && cuda.success && state.topology_pass && state.finite_pass &&
      state.invariant_pass && std::isfinite(cuda.initial_cost) &&
      std::isfinite(cuda.final_cost);
  const int exit_code = instrumentation_ab
      ? (cuda_ran ? 0 : 1)
      : (forced_reject_probe
             ? 0
             : (precision_experiment ? (precision_experiment_pass ? 0 : 1)
                                     : (pass ? 0 : 1)));
  if (cuda_options.execution_profile != CudaExecutionProfile::kBaseline) {
    std::string shutdown_error;
    if (!ShutdownCudaRuntimePool(&shutdown_error)) {
      std::cerr << shutdown_error << '\n';
      return 1;
    }
  }
  return exit_code;
}
