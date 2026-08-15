#define TEST_NAME "gpu_ba/snapshot"
#include "util/testing.h"

#include <fstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <unistd.h>

#include "gpu_ba/snapshot.h"

namespace colmap {
namespace gpu_ba {
namespace {

Snapshot MakeSnapshot() {
  Snapshot snapshot;
  snapshot.metadata.snapshot_id =
      "local-reg2-call1-refine0-trigger35-phraselocal";
  snapshot.metadata.ba_kind = BaKind::kLocal;
  snapshot.metadata.registered_image_count = 2;
  snapshot.metadata.ba_call_index = 1;
  snapshot.metadata.trigger_image_id = 35;
  snapshot.metadata.optimize_phrase = "local";
  snapshot.metadata.loss_function = "trivial";
  snapshot.metadata.proj_lidar_weight = 1.0;
  snapshot.metadata.icp_lidar_weight = 10.0;
  snapshot.metadata.icp_ground_lidar_weight = 20.0;
  snapshot.metadata.gradient_tolerance = 1.0;
  snapshot.metadata.max_num_iterations = 25;

  CameraSnapshot camera;
  camera.camera_id = 1;
  camera.model_id = 4;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0};
  snapshot.cameras.push_back(camera);

  ImageSnapshot image30;
  image30.image_id = 30;
  image30.camera_id = 1;
  image30.selected = true;
  image30.pose_constant = true;
  image30.qvec = {{1.0, 0.0, 0.0, 0.0}};
  snapshot.images.push_back(image30);
  ImageSnapshot image35 = image30;
  image35.image_id = 35;
  image35.pose_constant = false;
  image35.has_pose_parameter_blocks = true;
  image35.tvec = {{0.1, 0.2, 0.3}};
  snapshot.images.push_back(image35);

  PointSnapshot point7;
  point7.point3D_id = 7;
  point7.xyz = {{1.0, 2.0, 3.0}};
  point7.config_role = 1;
  point7.has_search_range = true;
  point7.search_range = 0.15;
  snapshot.points.push_back(point7);

  ObservationSnapshot observation35;
  observation35.source_index = 0;
  observation35.image_id = 35;
  observation35.point2D_idx = 11;
  observation35.point3D_id = 7;
  observation35.xy = {{12.5, 23.5}};
  snapshot.observations.push_back(observation35);
  ObservationSnapshot observation30 = observation35;
  observation30.source_index = 1;
  observation30.image_id = 30;
  observation30.point2D_idx = 9;
  observation30.pose_constant = true;
  snapshot.observations.push_back(observation30);

  snapshot.tracks.push_back({7, 30, 9});
  snapshot.tracks.push_back({7, 35, 11});

  LidarSnapshot lidar;
  lidar.source_index = 2;
  lidar.point3D_id = 7;
  lidar.lidar_type = 1;
  lidar.has_search_range = true;
  lidar.search_range = 0.15;
  lidar.weight = 10.0;
  lidar.lidar_xyz = {{1.0, 2.0, 2.9}};
  lidar.plane = {{0.0, 0.0, 1.0, -2.9}};
  snapshot.lidar.push_back(lidar);

  snapshot.parameter_blocks_source_order.push_back(
      {0, ParameterKind::kQuaternion, 35, 4, 3, false});
  snapshot.parameter_blocks_source_order.push_back(
      {1, ParameterKind::kTranslation, 35, 3, 3, false});
  snapshot.parameter_blocks_source_order.push_back(
      {2, ParameterKind::kPoint3D, 7, 3, 3, false});
  snapshot.parameter_blocks_source_order.push_back(
      {3, ParameterKind::kCamera, 1, 8, 0, true});
  snapshot.parameter_blocks_canonical_order = {0, 1, 2, 3};

  snapshot.source_insertion_order.push_back(
      {0, ResidualKind::kVisual, 35, 11, 7});
  snapshot.source_insertion_order.push_back(
      {1, ResidualKind::kVisual, 30, 9, 7});
  snapshot.source_insertion_order.push_back(
      {2, ResidualKind::kLidar, UINT32_MAX, UINT32_MAX, 7});
  snapshot.canonical_order = {snapshot.source_insertion_order[1],
                              snapshot.source_insertion_order[0],
                              snapshot.source_insertion_order[2]};
  return snapshot;
}

std::string TempRoot() {
  return "/tmp/colmap_gpu_ba_snapshot_test_" + std::to_string(getpid());
}

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  BOOST_REQUIRE(file.is_open());
  const std::streamoff size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0) file.read(reinterpret_cast<char*>(bytes.data()), size);
  BOOST_REQUIRE(file.good() || file.eof());
  return bytes;
}

void WriteBytes(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  BOOST_REQUIRE(file.is_open());
  if (!bytes.empty()) {
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  BOOST_REQUIRE(file.good());
}

struct Fixture {
  Fixture() { boost::filesystem::remove_all(TempRoot()); }
  ~Fixture() { boost::filesystem::remove_all(TempRoot()); }
};

}  // namespace

BOOST_FIXTURE_TEST_CASE(RoundTripAndDeterminism, Fixture) {
  const Snapshot original = MakeSnapshot();
  SnapshotWriteResult first;
  SnapshotWriteResult second;
  std::string error;
  BOOST_REQUIRE(WriteSnapshot(original, TempRoot() + "/first", &first, &error));
  BOOST_REQUIRE(WriteSnapshot(original, TempRoot() + "/second", &second, &error));
  BOOST_CHECK_EQUAL(first.integrity.payload_sha256,
                    second.integrity.payload_sha256);
  BOOST_CHECK_EQUAL(first.integrity.manifest_sha256,
                    second.integrity.manifest_sha256);

  Snapshot loaded;
  SnapshotReadResult read_result;
  BOOST_REQUIRE(ReadSnapshot(first.prefix_path, &loaded, &read_result, &error));
  BOOST_CHECK_EQUAL(loaded.metadata.snapshot_id,
                    original.metadata.snapshot_id);
  BOOST_CHECK_EQUAL(loaded.images.size(), original.images.size());
  BOOST_CHECK_EQUAL(loaded.points.size(), original.points.size());
  BOOST_CHECK_EQUAL(loaded.observations.size(), original.observations.size());
  BOOST_CHECK_EQUAL(loaded.lidar.size(), original.lidar.size());
  BOOST_CHECK_EQUAL(read_result.integrity.payload_sha256,
                    first.integrity.payload_sha256);

  SnapshotWriteResult duplicate;
  BOOST_CHECK(!WriteSnapshot(original, TempRoot() + "/first", &duplicate, &error));
  BOOST_CHECK(error.find("overwrite") != std::string::npos);
}

BOOST_FIXTURE_TEST_CASE(CorruptionAndTruncation, Fixture) {
  SnapshotWriteResult result;
  std::string error;
  BOOST_REQUIRE(WriteSnapshot(MakeSnapshot(), TempRoot() + "/valid", &result,
                              &error));
  const std::vector<uint8_t> valid_payload = ReadBytes(result.payload_path);
  const std::vector<uint8_t> valid_manifest = ReadBytes(result.manifest_path);

  auto MakeInvalid = [&](const std::string& name,
                         std::vector<uint8_t> payload,
                         std::vector<uint8_t> manifest) {
    const std::string prefix = TempRoot() + "/" + name + "/snapshot";
    boost::filesystem::create_directories(TempRoot() + "/" + name);
    WriteBytes(prefix + ".payload.bin", payload);
    WriteBytes(prefix + ".manifest.json", manifest);
    return prefix;
  };

  std::vector<uint8_t> corrupted = valid_payload;
  corrupted.back() ^= 0x80;
  Snapshot loaded;
  SnapshotReadResult read_result;
  std::string prefix =
      MakeInvalid("crc", corrupted, valid_manifest);
  BOOST_CHECK(!ReadSnapshot(prefix, &loaded, &read_result, &error));
  BOOST_CHECK(error.find("CRC32") != std::string::npos);

  std::vector<uint8_t> truncated(valid_payload.begin(),
                                 valid_payload.begin() + valid_payload.size() / 2);
  prefix = MakeInvalid("truncated", truncated, valid_manifest);
  BOOST_CHECK(!ReadSnapshot(prefix, &loaded, &read_result, &error));

  std::vector<uint8_t> bad_endian = valid_payload;
  bad_endian[12] ^= 0xff;
  prefix = MakeInvalid("endian", bad_endian, valid_manifest);
  BOOST_CHECK(!ReadSnapshot(prefix, &loaded, &read_result, &error));
  BOOST_CHECK(error.find("little-endian") != std::string::npos);

  std::vector<uint8_t> bad_schema = valid_payload;
  bad_schema[8] = 2;
  prefix = MakeInvalid("schema", bad_schema, valid_manifest);
  BOOST_CHECK(!ReadSnapshot(prefix, &loaded, &read_result, &error));
  BOOST_CHECK(error.find("schema") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(SemanticCrossSectionValidation) {
  Snapshot snapshot = MakeSnapshot();
  std::string error;
  BOOST_REQUIRE(ValidateSnapshot(snapshot, &error));

  snapshot.observations[0].point2D_idx += 1;
  BOOST_CHECK(!ValidateSnapshot(snapshot, &error));
  BOOST_CHECK(error.find("visual residual") != std::string::npos);
}

}  // namespace gpu_ba
}  // namespace colmap
