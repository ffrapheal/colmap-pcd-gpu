#ifndef COLMAP_SRC_EXE_ONLINE_I3DGS_REPLAY_H_
#define COLMAP_SRC_EXE_ONLINE_I3DGS_REPLAY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

struct OnlineI3dgsReplayCamera {
  std::string model;
  size_t width = 0;
  size_t height = 0;
  std::vector<double> params;
};

struct OnlineI3dgsReplayFrameInput {
  size_t frame_index = 0;
  uint32_t image_id = 0;
  std::string image_name;
  std::string image_path;
  std::string camera_path;
  std::string odometry_path;
  std::string scan_path;
  std::string image_sha256;
  std::string camera_sha256;
  std::string odometry_sha256;
  std::string scan_sha256;
  uint64_t scan_size_bytes = 0;
  std::array<double, 4> qvec = {{1.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> tvec = {{0.0, 0.0, 0.0}};
};

struct OnlineI3dgsReplayControllerConfig {
  OnlineI3dgsReplayCamera camera;
  size_t expected_frame_count = 0;
  std::string output_path;
  size_t ba_window_size = 20;
};

struct OnlineI3dgsReplayControllerFrameResult {
  bool success = false;
  bool ready_to_flush = false;
  uint64_t global_ba_call_count = 0;
  std::string termination;
  std::string audit_json;
  std::vector<std::string> ba_pass_json;
  std::string graph_snapshot_json;
};

struct OnlineI3dgsReplayControllerFinishResult {
  bool success = false;
  uint64_t global_ba_call_count = 0;
  std::string termination;
  std::string summary_json;
  std::vector<std::string> ba_pass_json;
  std::string graph_snapshot_json;
};

class OnlineI3dgsReplayController {
 public:
  virtual ~OnlineI3dgsReplayController() = default;

  virtual OnlineI3dgsReplayControllerFrameResult ProcessFrame(
      const OnlineI3dgsReplayFrameInput& frame) = 0;
  virtual OnlineI3dgsReplayControllerFinishResult Finish(
      bool full_sequence) = 0;
  virtual bool WriteModel(const std::string& staging_path,
                          std::string* error) const = 0;
};

using OnlineI3dgsReplayControllerFactory = std::function<
    std::unique_ptr<OnlineI3dgsReplayController>(
        const OnlineI3dgsReplayControllerConfig&, std::string*)>;

struct OnlineI3dgsReplayOptions {
  std::string sealed_artifact;
  std::string output_path;
  size_t max_frames = 0;
  size_t required_frame_count = 246;
  bool verify_runtime_environment = true;
  std::string executable_path;
  size_t ba_window_size = 20;
};

enum class OnlineI3dgsReplayStatus {
  COMPLETED,
  SMOKE_COMPLETED,
  INCOMPLETE,
  REJECTED,
};

struct OnlineI3dgsReplayResult {
  OnlineI3dgsReplayStatus status = OnlineI3dgsReplayStatus::REJECTED;
  size_t processed_frame_count = 0;
  size_t last_consistent_frame_index = 0;
  uint64_t global_ba_call_count = 0;
  std::string error;

  bool IsSuccess() const {
    return status == OnlineI3dgsReplayStatus::COMPLETED ||
           status == OnlineI3dgsReplayStatus::SMOKE_COMPLETED;
  }
};

OnlineI3dgsReplayResult RunOnlineI3dgsReplay(
    const OnlineI3dgsReplayOptions& options,
    const OnlineI3dgsReplayControllerFactory& controller_factory);

OnlineI3dgsReplayControllerFactory CreateOnlineI3dgsControllerFactory();

void RequestOnlineI3dgsReplayStopForTesting();
void ResetOnlineI3dgsReplayStopForTesting();

}  // namespace colmap

#endif  // COLMAP_SRC_EXE_ONLINE_I3DGS_REPLAY_H_
