#include "gpu_ba/snapshot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace colmap {
namespace gpu_ba {
namespace {

struct Sample {
  uint64_t id = 0;
  double value = 0.0;
};

struct Statistics {
  size_t count = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double maximum = 0.0;
  double rms = 0.0;
  uint64_t worst_id = 0;
};

std::string JsonString(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (const char character : value) {
    if (character == '"') stream << "\\\"";
    else if (character == '\\') stream << "\\\\";
    else if (character == '\n') stream << "\\n";
    else stream << character;
  }
  stream << '"';
  return stream.str();
}

double Percentile(const std::vector<Sample>& sorted, const double fraction) {
  if (sorted.empty()) return 0.0;
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - lower;
  return sorted[lower].value * (1.0 - weight) +
         sorted[upper].value * weight;
}

Statistics Summarize(std::vector<Sample> values) {
  Statistics result;
  result.count = values.size();
  if (values.empty()) return result;
  long double squared = 0.0;
  for (const Sample& value : values) {
    squared += static_cast<long double>(value.value) * value.value;
    if (value.value > result.maximum) {
      result.maximum = value.value;
      result.worst_id = value.id;
    }
  }
  std::sort(values.begin(), values.end(),
            [](const Sample& lhs, const Sample& rhs) {
              return lhs.value < rhs.value;
            });
  result.p50 = Percentile(values, 0.50);
  result.p95 = Percentile(values, 0.95);
  result.p99 = Percentile(values, 0.99);
  result.rms = std::sqrt(static_cast<double>(squared / values.size()));
  return result;
}

void WriteStatistics(std::ostream& output, const Statistics& value) {
  output << "{\"count\":" << value.count
         << ",\"p50\":" << value.p50
         << ",\"p95\":" << value.p95
         << ",\"p99\":" << value.p99
         << ",\"max\":" << value.maximum
         << ",\"rms\":" << value.rms
         << ",\"worst_id\":" << value.worst_id << '}';
}

std::array<double, 4> Normalize(const std::array<double, 4>& value,
                                double* norm_error) {
  const double norm = std::sqrt(std::inner_product(
      value.begin(), value.end(), value.begin(), 0.0));
  *norm_error = std::abs(norm - 1.0);
  if (!(norm > 0.0) || !std::isfinite(norm))
    return {{NAN, NAN, NAN, NAN}};
  return {{value[0] / norm, value[1] / norm, value[2] / norm,
           value[3] / norm}};
}

std::array<double, 3> CameraCenter(const std::array<double, 4>& q,
                                  const std::array<double, 3>& t) {
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];
  const double r[9] = {
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
      2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
      1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
      1.0 - 2.0 * (x * x + y * y)};
  return {{-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
           -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
           -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])}};
}

double Distance3(const std::array<double, 3>& lhs,
                 const std::array<double, 3>& rhs) {
  const double x = lhs[0] - rhs[0];
  const double y = lhs[1] - rhs[1];
  const double z = lhs[2] - rhs[2];
  return std::sqrt(x * x + y * y + z * z);
}

double RotationDegrees(const std::array<double, 4>& lhs,
                       const std::array<double, 4>& rhs) {
  if (lhs == rhs) return 0.0;
  bool exact_negation = true;
  for (size_t i = 0; i < 4; ++i)
    exact_negation = exact_negation && lhs[i] == -rhs[i];
  if (exact_negation) return 0.0;
  double lhs_error = 0.0;
  double rhs_error = 0.0;
  const auto a = Normalize(lhs, &lhs_error);
  const auto b = Normalize(rhs, &rhs_error);
  double dot = 0.0;
  for (size_t i = 0; i < 4; ++i) dot += a[i] * b[i];
  dot = std::min(1.0, std::max(0.0, std::abs(dot)));
  return 2.0 * std::acos(dot) * 180.0 / 3.14159265358979323846;
}

template <typename T, typename Id>
std::map<uint64_t, const T*> IndexBy(const std::vector<T>& values, Id id) {
  std::map<uint64_t, const T*> result;
  for (const T& value : values) result.emplace(id(value), &value);
  return result;
}

bool SameTopology(const Snapshot& lhs, const Snapshot& rhs) {
  if (lhs.cameras.size() != rhs.cameras.size() ||
      lhs.images.size() != rhs.images.size() ||
      lhs.points.size() != rhs.points.size() ||
      lhs.observations.size() != rhs.observations.size() ||
      lhs.tracks.size() != rhs.tracks.size() ||
      lhs.lidar.size() != rhs.lidar.size() ||
      lhs.parameter_blocks_source_order.size() !=
          rhs.parameter_blocks_source_order.size() ||
      lhs.parameter_blocks_canonical_order !=
          rhs.parameter_blocks_canonical_order ||
      lhs.source_insertion_order.size() != rhs.source_insertion_order.size() ||
      lhs.canonical_order.size() != rhs.canonical_order.size()) return false;
  for (size_t i = 0; i < lhs.cameras.size(); ++i) {
    const auto& a = lhs.cameras[i]; const auto& b = rhs.cameras[i];
    if (a.camera_id != b.camera_id || a.model_id != b.model_id ||
        a.width != b.width || a.height != b.height ||
        a.constant != b.constant || a.params.size() != b.params.size())
      return false;
  }
  for (size_t i = 0; i < lhs.images.size(); ++i) {
    const auto& a = lhs.images[i]; const auto& b = rhs.images[i];
    if (a.image_id != b.image_id || a.camera_id != b.camera_id ||
        a.selected != b.selected || a.pose_constant != b.pose_constant ||
        a.has_pose_parameter_blocks != b.has_pose_parameter_blocks ||
        a.constant_tvec_mask != b.constant_tvec_mask) return false;
  }
  for (size_t i = 0; i < lhs.points.size(); ++i) {
    const auto& a = lhs.points[i]; const auto& b = rhs.points[i];
    if (a.point3D_id != b.point3D_id || a.constant != b.constant ||
        a.config_role != b.config_role ||
        a.has_search_range != b.has_search_range ||
        a.search_range != b.search_range) return false;
  }
  for (size_t i = 0; i < lhs.observations.size(); ++i) {
    const auto& a = lhs.observations[i]; const auto& b = rhs.observations[i];
    if (a.source_index != b.source_index || a.image_id != b.image_id ||
        a.point2D_idx != b.point2D_idx || a.point3D_id != b.point3D_id ||
        a.pose_constant != b.pose_constant || a.xy != b.xy) return false;
  }
  for (size_t i = 0; i < lhs.tracks.size(); ++i) {
    const auto& a = lhs.tracks[i]; const auto& b = rhs.tracks[i];
    if (a.point3D_id != b.point3D_id || a.image_id != b.image_id ||
        a.point2D_idx != b.point2D_idx) return false;
  }
  for (size_t i = 0; i < lhs.lidar.size(); ++i) {
    const auto& a = lhs.lidar[i]; const auto& b = rhs.lidar[i];
    if (a.source_index != b.source_index || a.point3D_id != b.point3D_id ||
        a.lidar_type != b.lidar_type ||
        a.has_search_range != b.has_search_range ||
        a.search_range != b.search_range || a.weight != b.weight ||
        a.lidar_xyz != b.lidar_xyz || a.plane != b.plane) return false;
  }
  for (size_t i = 0; i < lhs.parameter_blocks_source_order.size(); ++i) {
    const auto& a = lhs.parameter_blocks_source_order[i];
    const auto& b = rhs.parameter_blocks_source_order[i];
    if (a.source_index != b.source_index || a.kind != b.kind ||
        a.entity_id != b.entity_id || a.ambient_size != b.ambient_size ||
        a.tangent_size != b.tangent_size || a.constant != b.constant)
      return false;
  }
  const auto same_order = [](const std::vector<OrderEntrySnapshot>& a,
                             const std::vector<OrderEntrySnapshot>& b) {
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i].source_index != b[i].source_index ||
          a[i].residual_kind != b[i].residual_kind ||
          a[i].image_id != b[i].image_id ||
          a[i].point2D_idx != b[i].point2D_idx ||
          a[i].point3D_id != b[i].point3D_id) return false;
    return true;
  };
  return same_order(lhs.source_insertion_order, rhs.source_insertion_order) &&
         same_order(lhs.canonical_order, rhs.canonical_order);
}

int Main(const int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " REFERENCE_SNAPSHOT CANDIDATE_SNAPSHOT OUTPUT_JSON\n";
    return 2;
  }
  Snapshot reference;
  Snapshot candidate;
  SnapshotReadResult read;
  std::string error;
  if (!ReadSnapshot(argv[1], &reference, &read, &error) ||
      !ReadSnapshot(argv[2], &candidate, &read, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto reference_images = IndexBy(reference.images,
      [](const ImageSnapshot& value) { return value.image_id; });
  const auto candidate_images = IndexBy(candidate.images,
      [](const ImageSnapshot& value) { return value.image_id; });
  const auto reference_points = IndexBy(reference.points,
      [](const PointSnapshot& value) { return value.point3D_id; });
  const auto candidate_points = IndexBy(candidate.points,
      [](const PointSnapshot& value) { return value.point3D_id; });
  const auto reference_cameras = IndexBy(reference.cameras,
      [](const CameraSnapshot& value) { return value.camera_id; });
  const auto candidate_cameras = IndexBy(candidate.cameras,
      [](const CameraSnapshot& value) { return value.camera_id; });

  std::vector<Sample> rotation_all, tvec_all, center_all;
  std::vector<Sample> rotation_variable, tvec_variable, center_variable;
  std::vector<Sample> rotation_fixed, tvec_fixed, center_fixed;
  std::vector<Sample> rotation_partial, tvec_partial, center_partial;
  bool fixed_images_exact = true;
  bool partial_fixed_components_exact = true;
  double quaternion_norm_error_reference = 0.0;
  double quaternion_norm_error_candidate = 0.0;
  uint32_t worst_center_image = 0;
  std::array<double, 3> worst_reference_center{{0.0, 0.0, 0.0}};
  std::array<double, 3> worst_candidate_center{{0.0, 0.0, 0.0}};
  std::array<double, 4> worst_reference_q{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> worst_candidate_q{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> worst_reference_t{{0.0, 0.0, 0.0}};
  std::array<double, 3> worst_candidate_t{{0.0, 0.0, 0.0}};
  double worst_center_distance = -1.0;
  for (const auto& entry : reference_images) {
    const auto found = candidate_images.find(entry.first);
    if (found == candidate_images.end()) continue;
    const ImageSnapshot& a = *entry.second;
    const ImageSnapshot& b = *found->second;
    double a_norm_error = 0.0;
    double b_norm_error = 0.0;
    const auto aq = Normalize(a.qvec, &a_norm_error);
    const auto bq = Normalize(b.qvec, &b_norm_error);
    quaternion_norm_error_reference =
        std::max(quaternion_norm_error_reference, a_norm_error);
    quaternion_norm_error_candidate =
        std::max(quaternion_norm_error_candidate, b_norm_error);
    const auto ac = CameraCenter(aq, a.tvec);
    const auto bc = CameraCenter(bq, b.tvec);
    const Sample rotation{entry.first, RotationDegrees(a.qvec, b.qvec)};
    const Sample tvec{entry.first, Distance3(a.tvec, b.tvec)};
    const Sample center{entry.first, Distance3(ac, bc)};
    rotation_all.push_back(rotation); tvec_all.push_back(tvec);
    center_all.push_back(center);
    if (a.pose_constant) {
      rotation_fixed.push_back(rotation); tvec_fixed.push_back(tvec);
      center_fixed.push_back(center);
      fixed_images_exact = fixed_images_exact && a.qvec == b.qvec &&
                           a.tvec == b.tvec;
    } else {
      rotation_variable.push_back(rotation); tvec_variable.push_back(tvec);
      center_variable.push_back(center);
      if (a.constant_tvec_mask != 0) {
        rotation_partial.push_back(rotation); tvec_partial.push_back(tvec);
        center_partial.push_back(center);
        for (size_t component = 0; component < 3; ++component)
          if ((a.constant_tvec_mask & (1u << component)) != 0)
            partial_fixed_components_exact =
                partial_fixed_components_exact &&
                a.tvec[component] == b.tvec[component];
      }
    }
    if (center.value > worst_center_distance) {
      worst_center_distance = center.value;
      worst_center_image = static_cast<uint32_t>(entry.first);
      worst_reference_center = ac; worst_candidate_center = bc;
      worst_reference_q = a.qvec; worst_candidate_q = b.qvec;
      worst_reference_t = a.tvec; worst_candidate_t = b.tvec;
    }
  }

  std::vector<Sample> point_all, point_variable, point_fixed;
  bool fixed_points_exact = true;
  uint64_t worst_point_id = 0;
  std::array<double, 3> worst_reference_point{{0.0, 0.0, 0.0}};
  std::array<double, 3> worst_candidate_point{{0.0, 0.0, 0.0}};
  double worst_point_distance = -1.0;
  for (const auto& entry : reference_points) {
    const auto found = candidate_points.find(entry.first);
    if (found == candidate_points.end()) continue;
    const PointSnapshot& a = *entry.second;
    const PointSnapshot& b = *found->second;
    const Sample value{entry.first, Distance3(a.xyz, b.xyz)};
    point_all.push_back(value);
    if (a.constant) {
      point_fixed.push_back(value);
      fixed_points_exact = fixed_points_exact && a.xyz == b.xyz;
    } else {
      point_variable.push_back(value);
    }
    if (value.value > worst_point_distance) {
      worst_point_distance = value.value;
      worst_point_id = entry.first;
      worst_reference_point = a.xyz;
      worst_candidate_point = b.xyz;
    }
  }

  std::vector<Sample> camera_parameters;
  bool fixed_cameras_exact = true;
  for (const auto& entry : reference_cameras) {
    const auto found = candidate_cameras.find(entry.first);
    if (found == candidate_cameras.end()) continue;
    const CameraSnapshot& a = *entry.second;
    const CameraSnapshot& b = *found->second;
    const size_t count = std::min(a.params.size(), b.params.size());
    for (size_t i = 0; i < count; ++i) {
      camera_parameters.push_back(
          {entry.first, std::abs(a.params[i] - b.params[i])});
      if (a.constant) fixed_cameras_exact =
          fixed_cameras_exact && a.params[i] == b.params[i];
    }
  }

  std::ofstream output(argv[3], std::ios::trunc);
  if (!output) return 1;
  output << std::setprecision(17)
         << "{\n  \"schema\":\"phase10p1a_precision_state_compare_v1\","
         << "\n  \"reference\":" << JsonString(argv[1])
         << ",\n  \"candidate\":" << JsonString(argv[2])
         << ",\n  \"topology_exact\":"
         << (SameTopology(reference, candidate) ? "true" : "false")
         << ",\n  \"entity_counts\":{\"reference_images\":"
         << reference_images.size() << ",\"candidate_images\":"
         << candidate_images.size() << ",\"common_images\":"
         << rotation_all.size() << ",\"reference_points\":"
         << reference_points.size() << ",\"candidate_points\":"
         << candidate_points.size() << ",\"common_points\":"
         << point_all.size() << ",\"reference_cameras\":"
         << reference_cameras.size() << ",\"candidate_cameras\":"
         << candidate_cameras.size() << "},\n  \"fixed_exact\":{"
         << "\"images\":" << (fixed_images_exact ? "true" : "false")
         << ",\"partial_translation_components\":"
         << (partial_fixed_components_exact ? "true" : "false")
         << ",\"points\":" << (fixed_points_exact ? "true" : "false")
         << ",\"cameras\":" << (fixed_cameras_exact ? "true" : "false")
         << "},\n  \"quaternion_norm_max_error\":{\"reference\":"
         << quaternion_norm_error_reference << ",\"candidate\":"
         << quaternion_norm_error_candidate << "},\n  \"pose\":{";
  const auto write_pose_group = [&](const char* name,
                                    const std::vector<Sample>& rotation,
                                    const std::vector<Sample>& tvec,
                                    const std::vector<Sample>& center,
                                    const bool first) {
    output << (first ? "\n    " : ",\n    ") << JsonString(name)
           << ":{\"rotation_degrees\":";
    WriteStatistics(output, Summarize(rotation));
    output << ",\"raw_tvec_meters\":";
    WriteStatistics(output, Summarize(tvec));
    output << ",\"camera_center_meters\":";
    WriteStatistics(output, Summarize(center));
    output << '}';
  };
  write_pose_group("all", rotation_all, tvec_all, center_all, true);
  write_pose_group("variable", rotation_variable, tvec_variable,
                   center_variable, false);
  write_pose_group("fixed", rotation_fixed, tvec_fixed, center_fixed, false);
  write_pose_group("partial_translation_mask", rotation_partial,
                   tvec_partial, center_partial, false);
  output << "\n  },\n  \"points\":{\"all\":";
  WriteStatistics(output, Summarize(point_all));
  output << ",\"variable\":";
  WriteStatistics(output, Summarize(point_variable));
  output << ",\"fixed\":";
  WriteStatistics(output, Summarize(point_fixed));
  output << "},\n  \"camera_parameter_abs\":";
  WriteStatistics(output, Summarize(camera_parameters));
  output << ",\n  \"worst_camera_center\":{\"image_id\":"
         << worst_center_image << ",\"distance_m\":"
         << std::max(0.0, worst_center_distance);
  const auto write_array = [&](const char* name, const auto& value) {
    output << "," << JsonString(name) << ":[";
    for (size_t i = 0; i < value.size(); ++i)
      output << (i == 0 ? "" : ",") << value[i];
    output << ']';
  };
  write_array("reference_qvec", worst_reference_q);
  write_array("candidate_qvec", worst_candidate_q);
  write_array("reference_tvec", worst_reference_t);
  write_array("candidate_tvec", worst_candidate_t);
  write_array("reference_center", worst_reference_center);
  write_array("candidate_center", worst_candidate_center);
  std::array<double, 3> center_delta{{
      worst_candidate_center[0] - worst_reference_center[0],
      worst_candidate_center[1] - worst_reference_center[1],
      worst_candidate_center[2] - worst_reference_center[2]}};
  write_array("delta", center_delta);
  output << "},\n  \"worst_point\":{\"point3D_id\":"
         << worst_point_id << ",\"distance_m\":"
         << std::max(0.0, worst_point_distance);
  write_array("reference_xyz", worst_reference_point);
  write_array("candidate_xyz", worst_candidate_point);
  std::array<double, 3> point_delta{{
      worst_candidate_point[0] - worst_reference_point[0],
      worst_candidate_point[1] - worst_reference_point[1],
      worst_candidate_point[2] - worst_reference_point[2]}};
  write_array("delta", point_delta);
  output << "}\n}\n";
  return output ? 0 : 1;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  return colmap::gpu_ba::Main(argc, argv);
}
