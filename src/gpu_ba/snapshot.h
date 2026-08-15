#ifndef COLMAP_SRC_GPU_BA_SNAPSHOT_H_
#define COLMAP_SRC_GPU_BA_SNAPSHOT_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace colmap {
namespace gpu_ba {

constexpr uint32_t kSnapshotSchemaVersion = 1;
constexpr uint32_t kSnapshotLittleEndianMarker = 0x01020304u;

enum class BaKind : uint8_t { kLocal = 0, kGlobal = 1, kWhole = 2 };
enum class ResidualKind : uint8_t { kVisual = 0, kLidar = 1 };
enum class ParameterKind : uint8_t {
  kQuaternion = 0,
  kTranslation = 1,
  kPoint3D = 2,
  kCamera = 3,
};

struct SnapshotMetadata {
  std::string snapshot_id;
  BaKind ba_kind = BaKind::kLocal;
  uint64_t registered_image_count = 0;
  uint64_t ba_call_index = 0;
  uint64_t refinement_index = 0;
  uint32_t trigger_image_id = 0;
  std::string optimize_phrase;
  std::string backend = "ceres_cpu";
  std::string loss_function;
  std::string lidar_residual_mode = "legacy_exact";
  std::string lidar_correspondence_version = "colmap-pcd-cpu-v1";
  std::string schur_mode = "deterministic";
  bool refine_focal_length = false;
  bool refine_principal_point = false;
  bool refine_extra_params = false;
  bool refine_extrinsics = true;
  double proj_lidar_weight = 0.0;
  double icp_lidar_weight = 0.0;
  double icp_ground_lidar_weight = 0.0;
  double function_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  double parameter_tolerance = 0.0;
  int32_t max_num_iterations = 0;
  int32_t max_linear_solver_iterations = 0;
  int32_t max_consecutive_invalid_steps = 0;
};

struct CameraSnapshot {
  uint32_t camera_id = 0;
  int32_t model_id = -1;
  uint64_t width = 0;
  uint64_t height = 0;
  bool constant = true;
  std::vector<double> params;
};

struct ImageSnapshot {
  uint32_t image_id = 0;
  uint32_t camera_id = 0;
  bool selected = false;
  bool pose_constant = true;
  bool has_pose_parameter_blocks = false;
  uint8_t constant_tvec_mask = 0;
  std::array<double, 4> qvec{{1.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> tvec{{0.0, 0.0, 0.0}};
};

struct PointSnapshot {
  uint64_t point3D_id = 0;
  bool constant = false;
  uint8_t config_role = 0;
  bool has_search_range = false;
  double search_range = 0.0;
  std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
};

struct ObservationSnapshot {
  uint64_t source_index = 0;
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
  uint64_t point3D_id = 0;
  bool pose_constant = false;
  std::array<double, 2> xy{{0.0, 0.0}};
};

struct TrackElementSnapshot {
  uint64_t point3D_id = 0;
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
};

struct LidarSnapshot {
  uint64_t source_index = 0;
  uint64_t point3D_id = 0;
  uint8_t lidar_type = 0;
  bool has_search_range = false;
  double search_range = 0.0;
  double weight = 0.0;
  std::array<double, 3> lidar_xyz{{0.0, 0.0, 0.0}};
  std::array<double, 4> plane{{0.0, 0.0, 0.0, 0.0}};
};

struct ParameterBlockSnapshot {
  uint64_t source_index = 0;
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_id = 0;
  uint32_t ambient_size = 0;
  uint32_t tangent_size = 0;
  bool constant = false;
};

struct OrderEntrySnapshot {
  uint64_t source_index = 0;
  ResidualKind residual_kind = ResidualKind::kVisual;
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
  uint64_t point3D_id = 0;
};

struct Snapshot {
  SnapshotMetadata metadata;
  std::vector<CameraSnapshot> cameras;
  std::vector<ImageSnapshot> images;
  std::vector<PointSnapshot> points;
  std::vector<ObservationSnapshot> observations;
  std::vector<TrackElementSnapshot> tracks;
  std::vector<LidarSnapshot> lidar;
  std::vector<ParameterBlockSnapshot> parameter_blocks_source_order;
  std::vector<uint64_t> parameter_blocks_canonical_order;
  std::vector<OrderEntrySnapshot> source_insertion_order;
  std::vector<OrderEntrySnapshot> canonical_order;
};

struct SnapshotIntegrity {
  std::string manifest_core_sha256;
  std::string manifest_sha256;
  std::string payload_sha256;
  std::string lidar_correspondence_sha256;
  std::string source_order_sha256;
  std::string canonical_order_sha256;
  uint64_t payload_bytes = 0;
};

struct SnapshotWriteResult {
  std::string prefix_path;
  std::string manifest_path;
  std::string payload_path;
  SnapshotIntegrity integrity;
};

struct SnapshotReadResult {
  std::string prefix_path;
  std::string manifest_path;
  std::string payload_path;
  SnapshotIntegrity integrity;
};

std::string BaKindName(BaKind kind);
bool ParseBaKind(const std::string& value, BaKind* kind);

uint32_t Crc32(const std::vector<uint8_t>& data);
std::string Sha256Hex(const std::vector<uint8_t>& data);
std::string Sha256Hex(const std::string& data);

bool ValidateSnapshot(const Snapshot& snapshot, std::string* error);

bool WriteSnapshot(const Snapshot& snapshot,
                   const std::string& output_dir,
                   SnapshotWriteResult* result,
                   std::string* error);

bool ReadSnapshot(const std::string& snapshot_path,
                  Snapshot* snapshot,
                  SnapshotReadResult* result,
                  std::string* error);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_SNAPSHOT_H_
