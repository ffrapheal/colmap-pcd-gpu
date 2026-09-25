#define TEST_NAME "exe/online_i3dgs_replay_test"
#include "util/testing.h"

#include "exe/online_i3dgs_replay.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <jsoncpp/json/json.h>
#include <openssl/sha.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <boost/filesystem.hpp>

using namespace colmap;

namespace {

std::string CompactJson(const Json::Value& value) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  builder["commentStyle"] = "None";
  std::string output = Json::writeString(builder, value);
  if (!output.empty() && output.back() == '\n') output.pop_back();
  return output;
}

std::string Sha256(const std::string& value) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  BOOST_REQUIRE(::SHA256(
                    reinterpret_cast<const unsigned char*>(value.data()),
                    value.size(), digest.data()) != nullptr);
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const unsigned char byte : digest) {
    stream << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return stream.str();
}

std::string ReadFile(const boost::filesystem::path& path) {
  std::ifstream stream(path.string(), std::ios::binary);
  BOOST_REQUIRE_MESSAGE(stream.is_open(), path.string());
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

void WriteFile(const boost::filesystem::path& path,
               const std::string& contents) {
  std::ofstream stream(path.string(), std::ios::binary | std::ios::trunc);
  BOOST_REQUIRE_MESSAGE(stream.is_open(), path.string());
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  BOOST_REQUIRE(stream.good());
}

void WriteJson(const boost::filesystem::path& path, const Json::Value& value) {
  WriteFile(path, CompactJson(value) + "\n");
}

int64_t Nanoseconds(const timespec& value) {
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

Json::Value FileRecord(const boost::filesystem::path& path) {
  struct stat file_stat;
  BOOST_REQUIRE_EQUAL(::stat(path.string().c_str(), &file_stat), 0);
  char* raw_real_path = ::realpath(path.string().c_str(), nullptr);
  BOOST_REQUIRE(raw_real_path != nullptr);
  const std::string real_path(raw_real_path);
  std::free(raw_real_path);
  std::ostringstream mode;
  mode << "0o" << std::oct << (file_stat.st_mode & 07777);
  Json::Value record(Json::objectValue);
  record["path"] = boost::filesystem::absolute(path).string();
  record["real_path"] = real_path;
  record["size_bytes"] = Json::UInt64(file_stat.st_size);
  record["mtime_ns"] = Json::UInt64(Nanoseconds(file_stat.st_mtim));
  record["mtime_utc"] = "test";
  record["ctime_ns"] = Json::UInt64(Nanoseconds(file_stat.st_ctim));
  record["ctime_utc"] = "test";
  record["sha256"] = Sha256(ReadFile(path));
  record["device"] = Json::UInt64(file_stat.st_dev);
  record["inode"] = Json::UInt64(file_stat.st_ino);
  record["mode"] = "test";
  record["mode_octal"] = mode.str();
  record["path_is_symlink"] = false;
  return record;
}

Json::Value IdentityMatrix(const size_t rows, const size_t columns) {
  Json::Value matrix(Json::arrayValue);
  for (size_t row = 0; row < rows; ++row) {
    Json::Value values(Json::arrayValue);
    for (size_t column = 0; column < columns; ++column) {
      values.append(row == column ? 1.0 : 0.0);
    }
    matrix.append(std::move(values));
  }
  return matrix;
}

Json::Value Pose(const size_t frame_index) {
  Json::Value pose(Json::objectValue);
  pose["timestamp"] = std::to_string(frame_index);
  pose["source_values"] = Json::Value(Json::arrayValue);
  for (size_t index = 0; index < 8; ++index) {
    pose["source_values"].append(index == 0 ? static_cast<double>(frame_index)
                                            : 0.0);
  }
  Json::Value fastlio(Json::objectValue);
  fastlio["convention"] = "T_wc";
  fastlio["translation_m"] = Json::Value(Json::arrayValue);
  fastlio["translation_m"].append(0.0);
  fastlio["translation_m"].append(0.0);
  fastlio["translation_m"].append(0.0);
  fastlio["source_quaternion_wxyz"] = Json::Value(Json::arrayValue);
  fastlio["normalized_quaternion_wxyz"] = Json::Value(Json::arrayValue);
  for (const double value : {1.0, 0.0, 0.0, 0.0}) {
    fastlio["source_quaternion_wxyz"].append(value);
    fastlio["normalized_quaternion_wxyz"].append(value);
  }
  fastlio["source_quaternion_norm"] = 1.0;
  fastlio["T_wc"] = IdentityMatrix(4, 4);
  pose["fastlio"] = fastlio;

  Json::Value prior(Json::objectValue);
  prior["convention"] = "T_cw";
  prior["camera_center_world_m"] = Json::Value(Json::arrayValue);
  prior["translation_cw_m"] = Json::Value(Json::arrayValue);
  for (size_t index = 0; index < 3; ++index) {
    prior["camera_center_world_m"].append(0.0);
    prior["translation_cw_m"].append(0.0);
  }
  prior["rotation_cw"] = IdentityMatrix(3, 3);
  prior["T_cw"] = IdentityMatrix(4, 4);
  prior["pose_prior_ply_values"] = Json::Value(Json::arrayValue);
  for (size_t index = 0; index < 6; ++index) {
    prior["pose_prior_ply_values"].append(0.0);
  }
  pose["colmap_prior"] = prior;
  return pose;
}

Json::Value CamRecord() {
  Json::Value cam(Json::objectValue);
  cam["format"] = "MVE_CAM";
  cam["world_to_camera_3x4"] = Json::Value(Json::arrayValue);
  for (size_t row = 0; row < 3; ++row) {
    Json::Value values(Json::arrayValue);
    for (size_t column = 0; column < 4; ++column) {
      values.append(row == column ? 1.0 : 0.0);
    }
    cam["world_to_camera_3x4"].append(std::move(values));
  }
  cam["calibration_values"] = Json::Value(Json::arrayValue);
  for (size_t index = 0; index < 6; ++index) {
    cam["calibration_values"].append(index == 0 ? 1.0 : 0.0);
  }
  return cam;
}

Json::Value CameraRecord(const Json::Value& source_file) {
  Json::Value camera(Json::objectValue);
  camera["model"] = "OPENCV";
  camera["width"] = Json::UInt64(16);
  camera["height"] = Json::UInt64(16);
  camera["params"] = Json::Value(Json::arrayValue);
  for (const double value : {10.0, 10.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0}) {
    camera["params"].append(value);
  }
  camera["param_order"] = Json::Value(Json::arrayValue);
  for (const char* value : {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"}) {
    camera["param_order"].append(value);
  }
  camera["source_distortion_model"] = "plumb_bob";
  camera["source_distortion"] = Json::Value(Json::arrayValue);
  for (size_t index = 0; index < 5; ++index) {
    camera["source_distortion"].append(0.0);
  }
  camera["source_file"] = source_file;
  return camera;
}

class ReplayFixture {
 public:
  explicit ReplayFixture(const size_t frame_count) : frame_count_(frame_count) {
    static size_t sequence = 0;
    root_ = boost::filesystem::temp_directory_path() /
            ("online-i3dgs-replay-test-" + std::to_string(::getpid()) + "-" +
             std::to_string(++sequence));
    session_ = root_ / "session";
    phase1_ = root_ / "phase1";
    boost::filesystem::create_directories(session_);
    boost::filesystem::create_directories(phase1_ / "logs");
    boost::filesystem::create_directories(root_ / "i3dgs");
    CreateInputs();
    CreateManifest();
    CreateConfig();
    CreateEvents();
    CreateSummary();
  }

  ~ReplayFixture() { boost::filesystem::remove_all(root_); }

  const boost::filesystem::path& Phase1() const { return phase1_; }
  boost::filesystem::path Output(const std::string& name) const {
    return root_ / name;
  }
  const std::vector<std::array<boost::filesystem::path, 4>>& FramePaths() const {
    return frame_paths_;
  }

  OnlineI3dgsReplayOptions Options(const boost::filesystem::path& output,
                                   const size_t max_frames = 0) const {
    OnlineI3dgsReplayOptions options;
    options.sealed_artifact = phase1_.string();
    options.output_path = output.string();
    options.max_frames = max_frames;
    options.required_frame_count = frame_count_;
    options.verify_runtime_environment = false;
    return options;
  }

 private:
  void CreateInputs() {
    const std::array<std::string, 4> prefixes =
        {{"imgs_", "imgs_", "odoms_", "scans_"}};
    const std::array<std::string, 4> suffixes =
        {{".jpg", ".CAM", ".txt", ".pcd"}};
    for (size_t frame_index = 1; frame_index <= frame_count_; ++frame_index) {
      std::array<boost::filesystem::path, 4> paths;
      for (size_t kind = 0; kind < paths.size(); ++kind) {
        paths[kind] = session_ /
                      (prefixes[kind] + std::to_string(frame_index) +
                       suffixes[kind]);
        WriteFile(paths[kind], "frame-" + std::to_string(frame_index) + "-" +
                                       std::to_string(kind) + "\n");
      }
      frame_paths_.push_back(paths);
    }
    camera_source_ = root_ / "camera.json";
    binary_ = root_ / "colmap";
    implementation_ = root_ / "pose-conversion.py";
    pose_prior_ = phase1_ / "pose-prior.ply";
    WriteFile(camera_source_, "{}\n");
    WriteFile(binary_, "test binary\n");
    BOOST_REQUIRE_EQUAL(::chmod(binary_.string().c_str(), 0700), 0);
    WriteFile(implementation_, "# test\n");
    WriteFile(pose_prior_, "ply\n");
  }

  void CreateManifest() {
    Json::Value manifest(Json::objectValue);
    manifest["schema"] = "online_i3dgs_phase1_input_manifest_v1";
    manifest["phase"] = 1;
    manifest["run_id"] = "test-phase1";
    manifest["generated_at"] = "test";
    manifest["artifact_dir"] = boost::filesystem::canonical(phase1_).string();
    manifest["hash_algorithm"] = "SHA-256";
    manifest["session"]["path"] = boost::filesystem::absolute(session_).string();
    manifest["session"]["real_path"] =
        boost::filesystem::canonical(session_).string();
    manifest["session"]["read_only_policy"] = true;
    manifest["session"]["inventory_file_count"] =
        Json::UInt64(frame_count_ * 4);
    manifest["session"]["inventory_relative_paths_sha256"] =
        std::string(64, '0');
    manifest["frame_count"] = Json::UInt64(frame_count_);
    manifest["frame_range"] = Json::Value(Json::arrayValue);
    manifest["frame_range"].append(Json::UInt64(1));
    manifest["frame_range"].append(Json::UInt64(frame_count_));
    manifest["frame_order"] = "NUMERIC_SUFFIX_ASCENDING";
    manifest["frames"] = Json::Value(Json::arrayValue);
    for (size_t offset = 0; offset < frame_count_; ++offset) {
      Json::Value frame(Json::objectValue);
      frame["frame_index"] = Json::UInt64(offset + 1);
      frame["image_name"] = "imgs_" + std::to_string(offset + 1) + ".jpg";
      frame["image"]["width"] = Json::UInt64(16);
      frame["image"]["height"] = Json::UInt64(16);
      frame["files"]["jpg"] = FileRecord(frame_paths_[offset][0]);
      frame["files"]["cam"] = FileRecord(frame_paths_[offset][1]);
      frame["files"]["odom"] = FileRecord(frame_paths_[offset][2]);
      frame["files"]["scan"] = FileRecord(frame_paths_[offset][3]);
      frame["cam"] = CamRecord();
      frame["pose"] = Pose(offset + 1);
      manifest["frames"].append(std::move(frame));
    }
    manifest["camera"] = CameraRecord(FileRecord(camera_source_));
    manifest["session_assets"] = Json::Value(Json::arrayValue);
    manifest["required_session_assets"] = Json::Value(Json::arrayValue);
    const Json::Value binary = FileRecord(binary_);
    manifest["binaries"]["frontend"] = binary;
    manifest["binaries"]["mapper"] = binary;
    manifest["binaries"]["texrecon"] = binary;
    manifest["repository"]["commit"] = std::string(40, '1');
    manifest["repository"]["tree"] = std::string(40, '2');
    manifest["repository"]["dirty"] = false;
    manifest["repository"]["status"] = Json::Value(Json::arrayValue);
    manifest["repository"]["status_sha256"] = Sha256("");
    manifest["repository"]["key_source_files"] = Json::Value(Json::arrayValue);
    manifest["repository"]["key_source_files"].append(
        FileRecord(implementation_));
    manifest["repository"]["key_source_snapshot_set_sha256"] =
        std::string(64, '3');
    manifest["repository"]["untracked_key_source_files"] =
        Json::Value(Json::arrayValue);
    manifest["i3dgs"]["path"] =
        boost::filesystem::canonical(root_ / "i3dgs").string();
    manifest["i3dgs"]["commit"] =
        "cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42";
    manifest["i3dgs"]["expected_commit"] =
        "cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42";
    manifest["i3dgs"]["dirty"] = false;
    manifest["i3dgs"]["status"] = Json::Value(Json::arrayValue);
    manifest["i3dgs"]["status_sha256"] = Sha256("");
    manifest["pose_conversion"]["fastlio_pose_convention"] = "T_wc";
    manifest["pose_conversion"]["colmap_prior_convention"] = "T_cw";
    manifest["pose_conversion"]["implementation_file"] =
        FileRecord(implementation_);
    manifest["stability_verification"] = Json::Value(Json::objectValue);
    manifest_ = std::move(manifest);
    WriteJson(phase1_ / "input-manifest.json", manifest_);
  }

  void CreateConfig() {
    Json::Value config(Json::objectValue);
    config["schema"] = "online_i3dgs_phase1_resolved_config_v1";
    config["phase"] = 1;
    config["run_mode"] = "PREPARE_ONLY_DRY_RUN";
    config["session_dir"] = boost::filesystem::canonical(session_).string();
    config["artifact_dir"] = boost::filesystem::canonical(phase1_).string();
    config["expected_frame_count"] = Json::UInt64(frame_count_);
    config["resolved_frame_count"] = Json::UInt64(frame_count_);
    config["input_manifest_sha256"] =
        Sha256(ReadFile(phase1_ / "input-manifest.json"));
    config["input_policy"]["artifact_must_not_exist"] = true;
    config["input_policy"]["overwrite_allowed"] = false;
    config["input_policy"]["input_mutation_detection"] =
        "FULL_REHASH_BEFORE_AND_AFTER_REPLAY";
    config["replay"]["event_order"] = "ARRIVED_BY_NUMERIC_FRAME_INDEX";
    config["replay"]["visibility"] = "CURRENT_AND_HISTORY_ONLY";
    config["replay"]["release_future_paths"] = false;
    config["replay"]["release_future_features"] = false;
    config["camera"] = manifest_["camera"];
    config["camera"].removeMember("source_file");
    config["frontend_baseline_not_executed"]["binary"] =
        manifest_["binaries"]["frontend"];
    Json::Value& mapper = config["mapper_baseline_not_executed"];
    mapper["binary"] = manifest_["binaries"]["mapper"];
    mapper["Mapper.online_mode"] = 1;
    mapper["Mapper.ba_global_enabled"] = 0;
    mapper["Mapper.ba_refine_focal_length"] = 0;
    mapper["Mapper.ba_refine_principal_point"] = 0;
    mapper["Mapper.ba_refine_extra_params"] = 0;
    config["texrecon_not_executed"]["binary"] =
        manifest_["binaries"]["texrecon"];
    config["pose_conversion"] = manifest_["pose_conversion"];
    config["i3dgs_reference"] = manifest_["i3dgs"];
    config_ = std::move(config);
    WriteJson(phase1_ / "resolved-config.json", config_);
  }

  void CreateEvents() {
    std::string previous_hash;
    std::string output;
    for (size_t offset = 0; offset < frame_count_; ++offset) {
      Json::Value event(Json::objectValue);
      event["schema"] = "online_i3dgs_phase1_frame_event_v1";
      event["event_sequence"] = Json::UInt64(offset + 1);
      event["event_type"] = "ARRIVED";
      event["frame_index"] = Json::UInt64(offset + 1);
      event["image_name"] = "imgs_" + std::to_string(offset + 1) + ".jpg";
      event["state_before"] = "SEALED";
      event["state_after"] = "ARRIVED";
      event["phase1_disposition"] = "PREPARED_ONLY_NO_MAPPER";
      event["attempt_no"] = 0;
      event["registration_attempted"] = false;
      const std::array<const char*, 4> kinds = {{"jpg", "cam", "odom", "scan"}};
      for (size_t kind = 0; kind < kinds.size(); ++kind) {
        const Json::Value record = FileRecord(frame_paths_[offset][kind]);
        event["released_inputs"][kinds[kind]]["path"] = record["path"];
        event["released_inputs"][kinds[kind]]["sha256"] = record["sha256"];
      }
      event["released_pose"] = Pose(offset + 1);
      Json::Value& causality = event["causality"];
      causality["visibility_rule"] = "CURRENT_AND_HISTORY_ONLY";
      causality["visible_frame_indices"] = Json::Value(Json::arrayValue);
      causality["released_pose_frame_indices"] = Json::Value(Json::arrayValue);
      for (size_t index = 1; index <= offset + 1; ++index) {
        causality["visible_frame_indices"].append(Json::UInt64(index));
        causality["released_pose_frame_indices"].append(Json::UInt64(index));
      }
      causality["max_visible_frame_index"] = Json::UInt64(offset + 1);
      causality["visible_feature_frame_indices"] = Json::Value(Json::arrayValue);
      causality["visible_match_pairs"] = Json::Value(Json::arrayValue);
      causality["future_paths_exposed"] = false;
      causality["future_features_exposed"] = false;
      if (offset == 0) {
        event["previous_event_sha256"] = Json::Value(Json::nullValue);
      } else {
        event["previous_event_sha256"] = previous_hash;
      }
      previous_hash = Sha256(CompactJson(event));
      event["event_sha256"] = previous_hash;
      output += CompactJson(event) + "\n";
    }
    event_chain_tail_ = previous_hash;
    WriteFile(phase1_ / "logs/frames.jsonl", output);
  }

  void CreateSummary() {
    Json::Value summary(Json::objectValue);
    summary["schema"] = "online_i3dgs_phase1_run_summary_v1";
    summary["status"] = "COMPLETED";
    summary["frame_count"] = Json::UInt64(frame_count_);
    summary["causal_event_count"] = Json::UInt64(frame_count_);
    summary["future_path_exposure_count"] = Json::UInt64(0);
    summary["event_chain_tail_sha256"] = event_chain_tail_;
    const auto add_output = [&](const char* name,
                                const boost::filesystem::path& path) {
      summary["outputs"][name]["path"] = path.string();
      summary["outputs"][name]["sha256"] = Sha256(ReadFile(path));
    };
    add_output("input_manifest", phase1_ / "input-manifest.json");
    add_output("resolved_config", phase1_ / "resolved-config.json");
    add_output("frames", phase1_ / "logs/frames.jsonl");
    add_output("pose_prior", pose_prior_);
    WriteJson(phase1_ / "run-summary.json", summary);
  }

  size_t frame_count_;
  boost::filesystem::path root_;
  boost::filesystem::path session_;
  boost::filesystem::path phase1_;
  boost::filesystem::path camera_source_;
  boost::filesystem::path binary_;
  boost::filesystem::path implementation_;
  boost::filesystem::path pose_prior_;
  std::vector<std::array<boost::filesystem::path, 4>> frame_paths_;
  Json::Value manifest_;
  Json::Value config_;
  std::string event_chain_tail_;
};

struct FakeControl {
  std::vector<std::array<std::string, 4>> expected_paths;
  std::vector<size_t> observed_frames;
  size_t fail_at_frame = 0;
  size_t global_ba_at_frame = 0;
  size_t interrupt_after_frame = 0;
  std::string mutate_on_finish_path;
  bool future_isolation_ok = true;
  bool factory_called = false;
  bool model_written = false;
  size_t observed_ba_window_size = 0;
  std::function<void(size_t)> before_frame;
  std::map<size_t, std::vector<std::string>> frame_ba_records;
  std::vector<std::string> finish_ba_records;
};

class FakeReplayController final : public OnlineI3dgsReplayController {
 public:
  explicit FakeReplayController(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  OnlineI3dgsReplayControllerFrameResult ProcessFrame(
      const OnlineI3dgsReplayFrameInput& frame) override {
    if (control_->before_frame) control_->before_frame(frame.frame_index);
    const size_t offset = frame.frame_index - 1;
    if (offset >= control_->expected_paths.size()) {
      control_->future_isolation_ok = false;
    } else {
      const std::array<std::string, 4>& expected = control_->expected_paths[offset];
      control_->future_isolation_ok =
          control_->future_isolation_ok && frame.image_path == expected[0] &&
          frame.camera_path == expected[1] &&
          frame.odometry_path == expected[2] && frame.scan_path == expected[3];
    }
    control_->observed_frames.push_back(frame.frame_index);
    OnlineI3dgsReplayControllerFrameResult result;
    result.success = frame.frame_index != control_->fail_at_frame;
    result.ready_to_flush =
        frame.frame_index == control_->expected_paths.size();
    result.global_ba_call_count =
        frame.frame_index == control_->global_ba_at_frame ? 1 : 0;
    result.termination = result.success ? "POSE_ONLY" : "INJECTED_FRAME_FAILURE";
    Json::Value audit(Json::objectValue);
    audit["frame_index"] = Json::UInt64(frame.frame_index);
    audit["success"] = result.success;
    result.audit_json = CompactJson(audit);
    Json::Value graph(Json::objectValue);
    graph["version"] = Json::UInt64(frame.frame_index);
    result.graph_snapshot_json = CompactJson(graph);
    result.ba_pass_json = control_->frame_ba_records[frame.frame_index];
    if (frame.frame_index == control_->interrupt_after_frame) {
      RequestOnlineI3dgsReplayStopForTesting();
    }
    return result;
  }

  OnlineI3dgsReplayControllerFinishResult Finish(bool) override {
    if (!control_->mutate_on_finish_path.empty()) {
      std::ofstream stream(control_->mutate_on_finish_path, std::ios::app);
      stream << "changed\n";
    }
    OnlineI3dgsReplayControllerFinishResult result;
    result.success = true;
    result.termination = "COMPLETE";
    Json::Value summary(Json::objectValue);
    summary["known_pose_registered_count"] =
        Json::UInt64(control_->observed_frames.size());
    summary["global_ba_call_count"] = Json::UInt64(0);
    result.summary_json = CompactJson(summary);
    Json::Value graph(Json::objectValue);
    graph["final"] = true;
    result.graph_snapshot_json = CompactJson(graph);
    result.ba_pass_json = control_->finish_ba_records;
    return result;
  }

  bool WriteModel(const std::string& staging_path,
                  std::string* error) const override {
    std::ofstream stream(staging_path + "/images.bin", std::ios::binary);
    if (!stream.is_open()) {
      if (error != nullptr) *error = "cannot create fake model";
      return false;
    }
    stream << "model\n";
    control_->model_written = true;
    return true;
  }

 private:
  std::shared_ptr<FakeControl> control_;
};

std::shared_ptr<FakeControl> MakeControl(const ReplayFixture& fixture) {
  std::shared_ptr<FakeControl> control(new FakeControl());
  for (const auto& paths : fixture.FramePaths()) {
    control->expected_paths.push_back(
        {{boost::filesystem::absolute(paths[0]).string(),
          boost::filesystem::absolute(paths[1]).string(),
          boost::filesystem::absolute(paths[2]).string(),
          boost::filesystem::absolute(paths[3]).string()}});
  }
  return control;
}

OnlineI3dgsReplayControllerFactory Factory(
    const std::shared_ptr<FakeControl>& control) {
  return [control](const OnlineI3dgsReplayControllerConfig& config,
                   std::string*) -> std::unique_ptr<OnlineI3dgsReplayController> {
    control->factory_called = true;
    control->observed_ba_window_size = config.ba_window_size;
    return std::unique_ptr<OnlineI3dgsReplayController>(
        new FakeReplayController(control));
  };
}

Json::Value ReadJson(const boost::filesystem::path& path) {
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  Json::Value output;
  std::string errors;
  std::istringstream stream(ReadFile(path));
  BOOST_REQUIRE(Json::parseFromStream(builder, stream, &output, &errors));
  return output;
}

std::vector<Json::Value> ReadJsonLines(const boost::filesystem::path& path) {
  std::vector<Json::Value> records;
  std::istringstream lines(ReadFile(path));
  std::string line;
  while (std::getline(lines, line)) {
    Json::CharReaderBuilder builder;
    Json::Value record;
    std::string errors;
    std::istringstream input(line);
    BOOST_REQUIRE(Json::parseFromStream(builder, input, &record, &errors));
    records.push_back(std::move(record));
  }
  return records;
}

}  // namespace

BOOST_AUTO_TEST_CASE(RejectsTamperedEventHashChainBeforeCreatingOutput) {
  ReplayFixture fixture(3);
  const boost::filesystem::path frames = fixture.Phase1() / "logs/frames.jsonl";
  std::string contents = ReadFile(frames);
  const std::string marker = "\"event_sha256\":\"";
  const size_t position = contents.find(marker);
  BOOST_REQUIRE(position != std::string::npos);
  contents[position + marker.size()] =
      contents[position + marker.size()] == '0' ? '1' : '0';
  WriteFile(frames, contents);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  const boost::filesystem::path output = fixture.Output("tampered-output");
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::REJECTED);
  BOOST_CHECK(result.error.find("chain") != std::string::npos);
  BOOST_CHECK(!control->factory_called);
  BOOST_CHECK(!boost::filesystem::exists(output));
}

BOOST_AUTO_TEST_CASE(ReleasesOnlyCurrentFramePathsToController) {
  ReplayFixture fixture(3);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  const boost::filesystem::path output = fixture.Output("smoke-output");
  OnlineI3dgsReplayOptions options = fixture.Options(output, 2);
  options.ba_window_size = 10;
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(options, Factory(control));
  BOOST_REQUIRE(result.status == OnlineI3dgsReplayStatus::SMOKE_COMPLETED);
  BOOST_CHECK(control->future_isolation_ok);
  BOOST_REQUIRE_EQUAL(control->observed_frames.size(), 2);
  BOOST_CHECK_EQUAL(control->observed_frames[0], 1);
  BOOST_CHECK_EQUAL(control->observed_frames[1], 2);
  BOOST_CHECK(boost::filesystem::exists(output / "SMOKE_COMPLETED"));
  BOOST_CHECK(!boost::filesystem::exists(output / "COMPLETED"));
  BOOST_CHECK(!boost::filesystem::exists(output / "INCOMPLETE"));
  const Json::Value resolved_config = ReadJson(output / "resolved-config.json");
  BOOST_CHECK_EQUAL(resolved_config["ba_window_size"].asUInt64(), 10);
  BOOST_CHECK_EQUAL(control->observed_ba_window_size, 10);
  const Json::Value backend = resolved_config["backend"];
  BOOST_CHECK_EQUAL(backend["audit_profile"].asString(), "production");
  BOOST_CHECK_EQUAL(backend["arithmetic_precision"].asString(), "fp32_mixed");
  BOOST_CHECK_EQUAL(backend["loss_function_type"].asString(), "SOFT_L1");
  BOOST_CHECK_EQUAL(backend["loss_function_scale"].asDouble(), 1.0);
  BOOST_CHECK_EQUAL(backend["max_num_iterations"].asInt(), 25);
  BOOST_CHECK_EQUAL(backend["gradient_tolerance"].asDouble(), 10.0);
  BOOST_CHECK_EQUAL(backend["proj_lidar_constraint_weight"].asDouble(), 10.0);
  BOOST_CHECK_EQUAL(backend["icp_lidar_constraint_weight"].asDouble(), 10.0);
  BOOST_CHECK_EQUAL(backend["icp_ground_lidar_constraint_weight"].asDouble(), 10.0);
  const Json::Value association = resolved_config["association"];
  BOOST_REQUIRE(association.isObject());
  BOOST_CHECK(association["local_lidar_kdtree_only"].asBool());
  BOOST_CHECK_EQUAL(association["min_proj_num"].asInt(), 1);
  BOOST_CHECK_EQUAL(association["kdtree_max_search_range"].asDouble(), 0.15);
  BOOST_CHECK_EQUAL(association["kdtree_min_search_range"].asDouble(), 0.05);
  BOOST_CHECK_EQUAL(association["search_range_drop_speed"].asDouble(), 0.01);
  BOOST_CHECK_EQUAL(association["ba_match_features_threshold"].asInt(), 200);
}

BOOST_AUTO_TEST_CASE(RejectsInvalidWindowBeforeCreatingOutput) {
  ReplayFixture fixture(3);
  for (const size_t window_size : {size_t{0}, size_t{1}, size_t{21}}) {
    const std::shared_ptr<FakeControl> control = MakeControl(fixture);
    const boost::filesystem::path output = fixture.Output("invalid-window");
    OnlineI3dgsReplayOptions options = fixture.Options(output);
    options.ba_window_size = window_size;
    const OnlineI3dgsReplayResult result =
        RunOnlineI3dgsReplay(options, Factory(control));
    BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::REJECTED);
    BOOST_CHECK(!control->factory_called);
    BOOST_CHECK(!boost::filesystem::exists(output));
  }
}

BOOST_AUTO_TEST_CASE(AppendsFrameAndBaLogsWithContinuousSequenceThroughFlush) {
  ReplayFixture fixture(3);
  const auto control = MakeControl(fixture);
  const auto output = fixture.Output("append-output");
  control->frame_ba_records[1] = {"{\"kind\":\"first\"}",
                                  "{\"kind\":\"second\"}"};
  control->frame_ba_records[3] = {"{\"kind\":\"third\"}"};
  control->finish_ba_records = {"{\"kind\":\"flush\"}"};
  std::array<ino_t, 2> log_inodes{};
  std::array<std::string, 2> prefixes;
  const std::array<const char*, 2> log_names =
      {{"logs/frames.jsonl", "logs/ba-passes.jsonl"}};
  control->before_frame = [&](size_t frame_index) {
    for (size_t index = 0; index < log_names.size(); ++index) {
      const auto path = output / log_names[index];
      struct stat log_stat;
      BOOST_REQUIRE_EQUAL(::stat(path.string().c_str(), &log_stat), 0);
      const std::string contents = ReadFile(path);
      if (frame_index == 1) {
        log_inodes[index] = log_stat.st_ino;
      } else {
        BOOST_CHECK_EQUAL(log_stat.st_ino, log_inodes[index]);
        BOOST_CHECK_EQUAL(contents.substr(0, prefixes[index].size()),
                          prefixes[index]);
      }
      prefixes[index] = contents;
    }
  };
  const auto result = RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_REQUIRE(result.status == OnlineI3dgsReplayStatus::COMPLETED);
  const auto frames = ReadJsonLines(output / log_names[0]);
  const auto passes = ReadJsonLines(output / log_names[1]);
  BOOST_REQUIRE_EQUAL(frames.size(), 3);
  BOOST_REQUIRE_EQUAL(passes.size(), 4);
  for (size_t index = 0; index < frames.size(); ++index) {
    BOOST_CHECK_EQUAL(frames[index]["event_sequence"].asUInt64(), index + 1);
  }
  for (size_t index = 0; index < passes.size(); ++index) {
    BOOST_CHECK_EQUAL(passes[index]["replay_pass_sequence"].asUInt64(), index + 1);
    BOOST_CHECK_EQUAL(passes[index]["replay_frame_index"].asUInt64(),
                      index < 2 ? 1 : 3);
  }
  BOOST_CHECK_EQUAL(passes.back()["kind"].asString(), "flush");
}

BOOST_AUTO_TEST_CASE(RejectsRedirectedAppendLogWithoutChangingTarget) {
  ReplayFixture fixture(3);
  const auto control = MakeControl(fixture);
  const auto output = fixture.Output("redirect-output");
  const auto target = fixture.Output("untouched-log");
  WriteFile(target, "keep\n");
  control->before_frame = [&](size_t frame_index) {
    if (frame_index != 2) return;
    const auto log = output / "logs/frames.jsonl";
    BOOST_REQUIRE_EQUAL(::unlink(log.string().c_str()), 0);
    BOOST_REQUIRE_EQUAL(::symlink(target.string().c_str(), log.string().c_str()), 0);
  };
  const auto result = RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(result.last_consistent_frame_index, 1);
  BOOST_CHECK_EQUAL(ReadFile(target), "keep\n");
  BOOST_CHECK(!control->model_written);
}

BOOST_AUTO_TEST_CASE(MalformedBaBatchPreservesPreviousLogPrefix) {
  ReplayFixture fixture(3);
  const auto control = MakeControl(fixture);
  const auto output = fixture.Output("malformed-ba-output");
  control->frame_ba_records[1] = {"{\"kind\":\"valid\"}"};
  control->frame_ba_records[2] = {"{}", "not-json"};
  const auto result = RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(result.last_consistent_frame_index, 1);
  const auto frames = ReadJsonLines(output / "logs/frames.jsonl");
  const auto passes = ReadJsonLines(output / "logs/ba-passes.jsonl");
  BOOST_REQUIRE_EQUAL(frames.size(), 1);
  BOOST_REQUIRE_EQUAL(passes.size(), 1);
  BOOST_CHECK_EQUAL(passes[0]["replay_pass_sequence"].asUInt64(), 1);
}

BOOST_AUTO_TEST_CASE(RefusesExistingOutputWithoutChangingIt) {
  ReplayFixture fixture(2);
  const boost::filesystem::path output = fixture.Output("existing-output");
  boost::filesystem::create_directory(output);
  WriteFile(output / "sentinel", "keep\n");
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::REJECTED);
  BOOST_CHECK(!control->factory_called);
  BOOST_CHECK_EQUAL(ReadFile(output / "sentinel"), "keep\n");
}

BOOST_AUTO_TEST_CASE(InterruptStopsBeforeOpeningTheNextFrame) {
  ReplayFixture fixture(3);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  control->interrupt_after_frame = 1;
  const boost::filesystem::path output = fixture.Output("interrupt-output");
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(result.processed_frame_count, 1);
  BOOST_REQUIRE_EQUAL(control->observed_frames.size(), 1);
  BOOST_CHECK_EQUAL(control->observed_frames.front(), 1);
  BOOST_CHECK(boost::filesystem::exists(output / "INCOMPLETE"));
  BOOST_CHECK(!boost::filesystem::exists(output / "COMPLETED"));
}

BOOST_AUTO_TEST_CASE(FrameFailurePersistsIncompleteAndLastConsistentFrame) {
  ReplayFixture fixture(3);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  control->fail_at_frame = 2;
  const boost::filesystem::path output = fixture.Output("failure-output");
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(result.last_consistent_frame_index, 1);
  const Json::Value summary = ReadJson(output / "run-summary.json");
  BOOST_CHECK_EQUAL(summary["status"].asString(), "INCOMPLETE");
  BOOST_CHECK_EQUAL(summary["last_consistent_frame_index"].asUInt64(), 1);
  BOOST_CHECK(boost::filesystem::exists(output / "INCOMPLETE"));
}

BOOST_AUTO_TEST_CASE(Full246ReplayPublishesCompletionSummaryAndModel) {
  ReplayFixture fixture(246);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  const boost::filesystem::path output = fixture.Output("full-output");
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_REQUIRE(result.status == OnlineI3dgsReplayStatus::COMPLETED);
  BOOST_CHECK_EQUAL(result.processed_frame_count, 246);
  BOOST_CHECK(control->model_written);
  const Json::Value summary = ReadJson(output / "run-summary.json");
  BOOST_CHECK_EQUAL(summary["status"].asString(), "COMPLETED");
  BOOST_CHECK(summary["complete"].asBool());
  BOOST_CHECK_EQUAL(summary["expected_frame_count"].asUInt64(), 246);
  BOOST_CHECK_EQUAL(summary["processed_frame_count"].asUInt64(), 246);
  BOOST_CHECK_EQUAL(summary["global_ba_call_count"].asUInt64(), 0);
  BOOST_CHECK_EQUAL(summary["input_stability"].asString(), "VERIFIED");
  BOOST_CHECK(boost::filesystem::exists(output / "models/final/images.bin"));
  BOOST_CHECK(boost::filesystem::exists(output / "COMPLETED"));
  BOOST_CHECK(!boost::filesystem::exists(output / "INCOMPLETE"));
}

BOOST_AUTO_TEST_CASE(RejectsNonzeroGlobalBaCountAndStopsImmediately) {
  ReplayFixture fixture(3);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  control->global_ba_at_frame = 2;
  const boost::filesystem::path output = fixture.Output("global-ba-output");
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(result.global_ba_call_count, 1);
  BOOST_REQUIRE_EQUAL(control->observed_frames.size(), 2);
  BOOST_CHECK(result.error.find("global_ba_call_count") != std::string::npos);
  BOOST_CHECK(boost::filesystem::exists(output / "INCOMPLETE"));
}

BOOST_AUTO_TEST_CASE(RejectsInputChangedAfterControllerFinish) {
  ReplayFixture fixture(2);
  const std::shared_ptr<FakeControl> control = MakeControl(fixture);
  control->mutate_on_finish_path = fixture.FramePaths()[1][0].string();
  const boost::filesystem::path output = fixture.Output("mutation-output");
  const OnlineI3dgsReplayResult result =
      RunOnlineI3dgsReplay(fixture.Options(output), Factory(control));
  BOOST_CHECK(result.status == OnlineI3dgsReplayStatus::INCOMPLETE);
  BOOST_CHECK(result.error.find("changed") != std::string::npos);
  BOOST_CHECK(!control->model_written);
  BOOST_CHECK(boost::filesystem::exists(output / "INCOMPLETE"));
  BOOST_CHECK(!boost::filesystem::exists(output / "COMPLETED"));
}
