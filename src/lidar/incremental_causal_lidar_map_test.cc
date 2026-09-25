#define TEST_NAME "lidar/incremental_causal_lidar_map"
#include "util/testing.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "lidar/incremental_causal_lidar_map.h"

namespace colmap {
namespace lidar {
namespace {

struct TestPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float intensity = 1.0f;
  float normal_x = 0.0f;
  float normal_y = 0.0f;
  float normal_z = 0.0f;
  float curvature = 0.0f;
};

class TempPcd {
 public:
  TempPcd() {
    static std::atomic<uint64_t> next_id{0};
    path_ = "/tmp/incremental_causal_lidar_map_test_" +
            std::to_string(static_cast<uint64_t>(getpid())) + "_" +
            std::to_string(next_id.fetch_add(1)) + ".pcd";
  }

  ~TempPcd() { std::remove(path_.c_str()); }

  TempPcd(const TempPcd&) = delete;
  TempPcd& operator=(const TempPcd&) = delete;

  const std::string& path() const { return path_; }

  void Write(const std::vector<TestPoint>& points,
             const bool binary = true,
             const std::string& viewpoint = "0 0 0 1 0 0 0") const {
    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("cannot create temporary PCD");
    file << "# .PCD v0.7\n"
         << "VERSION 0.7\n"
         << "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
         << "SIZE 4 4 4 4 4 4 4 4\n"
         << "TYPE F F F F F F F F\n"
         << "COUNT 1 1 1 1 1 1 1 1\n"
         << "WIDTH " << points.size() << "\n"
         << "HEIGHT 1\n"
         << "VIEWPOINT " << viewpoint << "\n"
         << "POINTS " << points.size() << "\n"
         << "DATA " << (binary ? "binary" : "ascii") << "\n";
    if (binary) {
      for (const TestPoint& point : points) {
        const std::array<float, 8> values = {{
            point.x,       point.y,        point.z,       point.intensity,
            point.normal_x, point.normal_y, point.normal_z, point.curvature,
        }};
        for (const float value : values) WriteLittleEndianFloat(value, &file);
      }
    } else {
      file << std::setprecision(std::numeric_limits<float>::max_digits10);
      for (const TestPoint& point : points) {
        file << point.x << ' ' << point.y << ' ' << point.z << ' '
             << point.intensity << ' ' << point.normal_x << ' '
             << point.normal_y << ' ' << point.normal_z << ' '
             << point.curvature << '\n';
      }
    }
    if (!file) throw std::runtime_error("cannot write temporary PCD");
  }

  void WriteRaw(const std::string& contents) const {
    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("cannot create temporary PCD");
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!file) throw std::runtime_error("cannot write temporary PCD");
  }

 private:
  static void WriteLittleEndianFloat(const float value, std::ofstream* file) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::array<char, 4> bytes = {{
        static_cast<char>(bits & 0xffu),
        static_cast<char>((bits >> 8) & 0xffu),
        static_cast<char>((bits >> 16) & 0xffu),
        static_cast<char>((bits >> 24) & 0xffu),
    }};
    file->write(bytes.data(), bytes.size());
  }

  std::string path_;
};

void AppendLittleEndianFloat(const float value, std::string* bytes) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bytes->push_back(static_cast<char>(bits & 0xffu));
  bytes->push_back(static_cast<char>((bits >> 8) & 0xffu));
  bytes->push_back(static_cast<char>((bits >> 16) & 0xffu));
  bytes->push_back(static_cast<char>((bits >> 24) & 0xffu));
}

void AppendLittleEndianUint16(const uint16_t value, std::string* bytes) {
  bytes->push_back(static_cast<char>(value & 0xffu));
  bytes->push_back(static_cast<char>((value >> 8) & 0xffu));
}

std::string MakeSinglePointBinaryPcd(const TestPoint& point) {
  std::string bytes =
      "VERSION 0.7\n"
      "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
      "SIZE 4 4 4 4 4 4 4 4\n"
      "TYPE F F F F F F F F\n"
      "COUNT 1 1 1 1 1 1 1 1\n"
      "WIDTH 1\n"
      "HEIGHT 1\n"
      "POINTS 1\n"
      "DATA binary\n";
  const std::array<float, 8> values = {{
      point.x,       point.y,        point.z,       point.intensity,
      point.normal_x, point.normal_y, point.normal_z, point.curvature,
  }};
  for (const float value : values) AppendLittleEndianFloat(value, &bytes);
  return bytes;
}

bool ContainsPoint(const std::vector<float>& xyz,
                   const std::array<float, 3>& point) {
  for (size_t i = 0; i < xyz.size() / 3; ++i) {
    if (xyz[3 * i] == point[0] && xyz[3 * i + 1] == point[1] &&
        xyz[3 * i + 2] == point[2]) {
      return true;
    }
  }
  return false;
}

TestPoint PointInColmapWorld(const float x,
                            const float y,
                            const float z,
                            const std::array<float, 3>& colmap_normal =
                                {{0.0f, 0.0f, 0.0f}}) {
  TestPoint point;
  point.x = z;
  point.y = -x;
  point.z = -y;
  point.normal_x = colmap_normal[2];
  point.normal_y = -colmap_normal[0];
  point.normal_z = -colmap_normal[1];
  return point;
}

VoxelKey KeyForColmapPoint(const float x, const float y, const float z) {
  return VoxelKey{static_cast<int64_t>(
                      std::floor(static_cast<double>(x) * 100.0)),
                  static_cast<int64_t>(
                      std::floor(static_cast<double>(y) * 100.0)),
                  static_cast<int64_t>(
                      std::floor(static_cast<double>(z) * 100.0))};
}

ScanSource MakeSource(const uint64_t index, const std::string& path) {
  ScanSource source;
  source.scan_index = index;
  source.pcd_path = path;
  source.input_frame = LidarCoordinateFrame::FASTLIO_WORLD;
  return source;
}

bool TestStatFile(const std::string& path,
                  uint64_t* size,
                  std::string* error) {
  struct stat metadata;
  if (stat(path.c_str(), &metadata) != 0 || metadata.st_size < 0) {
    *error = "test stat failed";
    return false;
  }
  *size = static_cast<uint64_t>(metadata.st_size);
  return true;
}

bool TestReadFile(const std::string& path,
                  std::vector<uint8_t>* bytes,
                  std::string* error) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    *error = "test read open failed";
    return false;
  }
  const std::streamoff size = file.tellg();
  if (size < 0) {
    *error = "test read size failed";
    return false;
  }
  file.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(size));
  if (size > 0 &&
      !file.read(reinterpret_cast<char*>(bytes->data()),
                 static_cast<std::streamsize>(size))) {
    *error = "test read failed";
    return false;
  }
  return true;
}

struct EstimatorCall {
  std::vector<float> support_xyz;
  std::vector<float> query_xyz;
  float outer_radius = 0.0f;
  float inner_radius = 0.0f;
};

struct EstimatorCapture {
  std::vector<EstimatorCall> calls;
};

void EmitNeighborhoodNormals(const std::vector<float>& support_xyz,
                             const std::vector<float>& query_xyz,
                             const float radius,
                             const bool outer,
                             std::vector<float>* output) {
  output->clear();
  output->reserve(query_xyz.size() / 3 * 4);
  const float squared_radius = radius * radius;
  for (size_t query_index = 0; query_index < query_xyz.size() / 3;
       ++query_index) {
    uint64_t neighbor_count = 0;
    double coordinate_sum = 0.0;
    for (size_t support_index = 0; support_index < support_xyz.size() / 3;
         ++support_index) {
      const float dx = support_xyz[3 * support_index] -
                       query_xyz[3 * query_index];
      const float dy = support_xyz[3 * support_index + 1] -
                       query_xyz[3 * query_index + 1];
      const float dz = support_xyz[3 * support_index + 2] -
                       query_xyz[3 * query_index + 2];
      const float squared_distance = dx * dx + dy * dy + dz * dz;
      if (squared_distance <= squared_radius) {
        ++neighbor_count;
        coordinate_sum += support_xyz[3 * support_index] +
                          support_xyz[3 * support_index + 1] +
                          support_xyz[3 * support_index + 2];
      }
    }
    if (outer) {
      output->push_back(1.0f);
      output->push_back(static_cast<float>(neighbor_count) * 0.01f);
      output->push_back(static_cast<float>(coordinate_sum) * 0.0001f);
    } else {
      output->push_back(static_cast<float>(coordinate_sum) * 0.0001f);
      output->push_back(1.0f);
      output->push_back(static_cast<float>(neighbor_count) * 0.01f);
    }
    output->push_back(static_cast<float>(neighbor_count));
  }
}

bool NeighborhoodEstimator(const std::vector<float>& support_xyz,
                           const std::vector<float>& query_xyz,
                           const float outer_radius,
                           const float inner_radius,
                           std::vector<float>* outer_normals,
                           std::vector<float>* inner_normals,
                           CudaNormalEstimationTiming* timing,
                           std::string* error,
                           EstimatorCapture* capture = nullptr) {
  (void)error;
  if (capture != nullptr) {
    capture->calls.push_back(
        EstimatorCall{support_xyz, query_xyz, outer_radius, inner_radius});
  }
  EmitNeighborhoodNormals(support_xyz, query_xyz, outer_radius, true,
                          outer_normals);
  EmitNeighborhoodNormals(support_xyz, query_xyz, inner_radius, false,
                          inner_normals);
  if (timing != nullptr) {
    timing->upload_ms = 0.1;
    timing->index_ms = 0.2;
    timing->normals_ms = 0.3;
    timing->download_ms = 0.4;
  }
  return true;
}

IncrementalCausalLidarMapDependencies FakeDependencies(
    const std::shared_ptr<EstimatorCapture>& capture = nullptr) {
  IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [capture](const std::vector<float>& support_xyz,
                const std::vector<float>& query_xyz,
                const float outer_radius,
                const float inner_radius,
                std::vector<float>* outer_normals,
                std::vector<float>* inner_normals,
                CudaNormalEstimationTiming* timing,
                std::string* error) {
        return NeighborhoodEstimator(support_xyz, query_xyz, outer_radius,
                                     inner_radius, outer_normals, inner_normals,
                                     timing, error, capture.get());
      };
  return dependencies;
}

LidarNormalValue NormalizedExpected(const std::vector<float>& values,
                                    const size_t query_index) {
  LidarNormalValue expected;
  const float nx = values[4 * query_index];
  const float ny = values[4 * query_index + 1];
  const float nz = values[4 * query_index + 2];
  if (std::isnan(nx) && std::isnan(ny) && std::isnan(nz) &&
      std::isnan(values[4 * query_index + 3])) {
    return expected;
  }
  const double norm = std::sqrt(static_cast<double>(nx) * nx +
                                static_cast<double>(ny) * ny +
                                static_cast<double>(nz) * nz);
  expected.normal = {{static_cast<float>(nx / norm),
                      static_cast<float>(ny / norm),
                      static_cast<float>(nz / norm)}};
  for (float& component : expected.normal) {
    if (component == 0.0f) component = 0.0f;
  }
  expected.curvature = values[4 * query_index + 3] == 0.0f
                           ? 0.0f
                           : values[4 * query_index + 3];
  expected.valid = true;
  return expected;
}

void CheckNormalMatches(const LidarNormalValue& actual,
                        const LidarNormalValue& expected) {
  BOOST_REQUIRE_EQUAL(actual.valid, expected.valid);
  if (!actual.valid) {
    BOOST_CHECK_EQUAL(actual.normal[0], 0.0f);
    BOOST_CHECK_EQUAL(actual.normal[1], 0.0f);
    BOOST_CHECK_EQUAL(actual.normal[2], 0.0f);
    BOOST_CHECK_EQUAL(actual.curvature, 0.0f);
    return;
  }
  BOOST_CHECK_EQUAL(actual.normal[0], expected.normal[0]);
  BOOST_CHECK_EQUAL(actual.normal[1], expected.normal[1]);
  BOOST_CHECK_EQUAL(actual.normal[2], expected.normal[2]);
  BOOST_CHECK_EQUAL(actual.curvature, expected.curvature);
}

std::vector<float> SnapshotXyz(
    const std::shared_ptr<const LidarMapSnapshot>& snapshot,
    const std::vector<VoxelKey>& keys) {
  std::vector<float> xyz;
  xyz.reserve(keys.size() * 3);
  for (const VoxelKey& key : keys) {
    LidarVoxelRecord record;
    if (!snapshot->FindVoxel(key, &record)) {
      throw std::runtime_error("snapshot key disappeared");
    }
    xyz.insert(xyz.end(), record.centroid.begin(), record.centroid.end());
  }
  return xyz;
}

void CheckFloatVectorsBitwiseEqual(const std::vector<float>& actual,
                                   const std::vector<float>& expected) {
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  if (!actual.empty()) {
    BOOST_CHECK_EQUAL(
        std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(float)),
        0);
  }
}

struct BruteForceNearestPlane {
  bool found = false;
  LidarVoxelRecord record;
  double squared_distance = 0.0;
};

BruteForceNearestPlane FindNearestPlaneBruteForce(
    const std::shared_ptr<const LidarMapSnapshot>& snapshot,
    const std::array<double, 3>& query,
    const LidarNormalScale scale,
    const double max_distance) {
  BruteForceNearestPlane result;
  const double max_squared_distance = max_distance * max_distance;
  for (const VoxelKey& key : snapshot->VoxelKeys()) {
    LidarVoxelRecord record;
    if (!snapshot->FindVoxel(key, &record)) {
      throw std::runtime_error("snapshot key disappeared during brute force");
    }
    const LidarNormalValue& normal =
        scale == LidarNormalScale::INNER_0_05_M ? record.inner_normal
                                                : record.outer_normal;
    if (!normal.valid) continue;
    const double dx = static_cast<double>(record.centroid[0]) - query[0];
    const double dy = static_cast<double>(record.centroid[1]) - query[1];
    const double dz = static_cast<double>(record.centroid[2]) - query[2];
    const double squared_distance = dx * dx + dy * dy + dz * dz;
    if (squared_distance > max_squared_distance) continue;
    if (!result.found || squared_distance < result.squared_distance ||
        (squared_distance == result.squared_distance &&
         key < result.record.key)) {
      result.found = true;
      result.record = record;
      result.squared_distance = squared_distance;
    }
  }
  return result;
}

void CheckNearestMatchesBruteForce(
    const std::shared_ptr<const LidarMapSnapshot>& snapshot,
    const std::array<double, 3>& query,
    const LidarNormalScale scale,
    const double max_distance) {
  const BruteForceNearestPlane expected =
      FindNearestPlaneBruteForce(snapshot, query, scale, max_distance);
  const NearestPlaneResult actual =
      snapshot->FindNearestPlane(query, scale, max_distance);
  BOOST_REQUIRE_MESSAGE(actual.ok, actual.error);
  BOOST_REQUIRE_EQUAL(actual.found, expected.found);
  if (!expected.found) return;
  const LidarNormalValue& expected_normal =
      scale == LidarNormalScale::INNER_0_05_M
          ? expected.record.inner_normal
          : expected.record.outer_normal;
  BOOST_CHECK(actual.plane.key == expected.record.key);
  BOOST_CHECK(actual.plane.frame == LidarCoordinateFrame::COLMAP_WORLD);
  BOOST_CHECK(actual.plane.scale == scale);
  BOOST_CHECK(actual.plane.point == expected.record.centroid);
  BOOST_CHECK(actual.plane.normal == expected_normal.normal);
  BOOST_CHECK_EQUAL(actual.plane.curvature, expected_normal.curvature);
  BOOST_CHECK_EQUAL(actual.plane.normal_revision, expected_normal.revision);
  BOOST_CHECK_EQUAL(actual.plane.voxel_count, expected.record.count);
  BOOST_CHECK_EQUAL(actual.squared_distance, expected.squared_distance);
}

enum class FailureMode {
  NONE,
  STAT_FALSE,
  STAT_THROW,
  READ_FALSE,
  READ_THROW,
  HASH_FALSE,
  HASH_THROW,
  NORMAL_FALSE,
  NORMAL_THROW,
  NORMAL_WRONG_LENGTH,
  NORMAL_PARTIAL_NAN,
  NORMAL_PARTIAL_INF,
  NORMAL_NAN_TIMING,
  NORMAL_NEGATIVE_TIMING,
  NORMAL_ZERO_VECTOR,
};

struct FailureControl {
  FailureMode mode = FailureMode::NONE;
  uint64_t stat_calls = 0;
  uint64_t read_calls = 0;
  uint64_t hash_calls = 0;
  uint64_t normal_calls = 0;
};

IncrementalCausalLidarMapDependencies ControlledDependencies(
    const std::shared_ptr<FailureControl>& control) {
  IncrementalCausalLidarMapDependencies dependencies;
  dependencies.stat_file =
      [control](const std::string& path, uint64_t* size, std::string* error) {
        ++control->stat_calls;
        if (control->mode == FailureMode::STAT_THROW) {
          throw std::runtime_error("injected stat throw");
        }
        if (control->mode == FailureMode::STAT_FALSE) {
          *error = "injected stat false";
          return false;
        }
        return TestStatFile(path, size, error);
      };
  dependencies.read_file =
      [control](const std::string& path,
                std::vector<uint8_t>* bytes,
                std::string* error) {
        ++control->read_calls;
        if (control->mode == FailureMode::READ_THROW) {
          throw std::runtime_error("injected read throw");
        }
        if (control->mode == FailureMode::READ_FALSE) {
          *error = "injected read false";
          return false;
        }
        return TestReadFile(path, bytes, error);
      };
  dependencies.sha256 =
      [control](const std::vector<uint8_t>& bytes,
                std::string* digest,
                std::string* error) {
        (void)bytes;
        ++control->hash_calls;
        if (control->mode == FailureMode::HASH_THROW) {
          throw std::runtime_error("injected hash throw");
        }
        if (control->mode == FailureMode::HASH_FALSE) {
          *error = "injected hash false";
          return false;
        }
        *digest = std::string(64, 'a');
        return true;
      };
  dependencies.estimate_normals =
      [control](const std::vector<float>& support_xyz,
                const std::vector<float>& query_xyz,
                const float outer_radius,
                const float inner_radius,
                std::vector<float>* outer_normals,
                std::vector<float>* inner_normals,
                CudaNormalEstimationTiming* timing,
                std::string* error) {
        ++control->normal_calls;
        if (control->mode == FailureMode::NORMAL_THROW) {
          throw std::runtime_error("injected normal throw");
        }
        if (control->mode == FailureMode::NORMAL_FALSE) {
          *error = "injected normal false";
          return false;
        }
        NeighborhoodEstimator(support_xyz, query_xyz, outer_radius,
                              inner_radius, outer_normals, inner_normals,
                              timing, error);
        if (control->mode == FailureMode::NORMAL_WRONG_LENGTH) {
          outer_normals->pop_back();
        } else if (control->mode == FailureMode::NORMAL_PARTIAL_NAN) {
          (*outer_normals)[0] = std::numeric_limits<float>::quiet_NaN();
        } else if (control->mode == FailureMode::NORMAL_PARTIAL_INF) {
          (*inner_normals)[1] = std::numeric_limits<float>::infinity();
        } else if (control->mode == FailureMode::NORMAL_NAN_TIMING) {
          timing->index_ms = std::numeric_limits<double>::quiet_NaN();
        } else if (control->mode == FailureMode::NORMAL_NEGATIVE_TIMING) {
          timing->download_ms = -0.1;
        } else if (control->mode == FailureMode::NORMAL_ZERO_VECTOR) {
          (*outer_normals)[0] = 0.0f;
          (*outer_normals)[1] = 0.0f;
          (*outer_normals)[2] = 0.0f;
        }
        return true;
      };
  return dependencies;
}

BOOST_AUTO_TEST_CASE(CausalSequenceRejectsBeforeIoAndKeepsOldSnapshot) {
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write({PointInColmapWorld(0.0f, 0.0f, 1.0f)});
  scan2.Write({PointInColmapWorld(0.02f, 0.0f, 1.0f)});
  const std::shared_ptr<FailureControl> control(new FailureControl());
  IncrementalCausalLidarMap map(ControlledDependencies(control));

  const std::shared_ptr<const LidarMapSnapshot> initial = map.GetSnapshot();
  BOOST_REQUIRE_EQUAL(initial->Version(), 0);
  BOOST_REQUIRE_EQUAL(initial->MaxScanIndex(), 0);
  BOOST_REQUIRE_EQUAL(initial->VoxelCount(), 0);

  AppendScanAudit audit1;
  std::string error;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(1, scan1.path()), &audit1,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot1 = map.GetSnapshot();
  const std::string snapshot1_hash = snapshot1->SnapshotSha256();
  const NearestPlaneResult query1 = snapshot1->FindNearestPlane(
      {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.01);
  BOOST_REQUIRE(query1.ok);
  BOOST_REQUIRE(query1.found);
  const uint64_t stat_calls = control->stat_calls;
  const uint64_t read_calls = control->read_calls;
  const uint64_t hash_calls = control->hash_calls;

  BOOST_CHECK(!map.AppendScan(
      MakeSource(3, "/tmp/nonexistent_future_incremental_scan.pcd"), nullptr,
      &error));
  BOOST_CHECK(!error.empty());
  BOOST_CHECK_EQUAL(control->stat_calls, stat_calls);
  BOOST_CHECK_EQUAL(control->read_calls, read_calls);
  BOOST_CHECK_EQUAL(control->hash_calls, hash_calls);
  BOOST_CHECK(map.GetSnapshot() == snapshot1);
  BOOST_CHECK_EQUAL(map.GetSnapshot()->SnapshotSha256(), snapshot1_hash);

  BOOST_CHECK(!map.AppendScan(MakeSource(1, scan1.path()), nullptr, &error));
  BOOST_CHECK_EQUAL(control->stat_calls, stat_calls);
  BOOST_CHECK_EQUAL(control->read_calls, read_calls);
  BOOST_CHECK_EQUAL(control->hash_calls, hash_calls);
  BOOST_CHECK(map.GetSnapshot() == snapshot1);

  AppendScanAudit audit2;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(2, scan2.path()), &audit2,
                                       &error),
                        error);
  BOOST_CHECK_EQUAL(map.GetSnapshot()->Version(), 2);
  BOOST_CHECK_EQUAL(map.GetSnapshot()->MaxScanIndex(), 2);
  BOOST_CHECK_EQUAL(snapshot1->Version(), 1);
  BOOST_CHECK_EQUAL(snapshot1->SnapshotSha256(), snapshot1_hash);
  const NearestPlaneResult query1_after = snapshot1->FindNearestPlane(
      {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.01);
  BOOST_REQUIRE(query1_after.ok);
  BOOST_REQUIRE(query1_after.found);
  BOOST_CHECK(query1_after.plane.key == query1.plane.key);
  BOOST_CHECK_EQUAL(query1_after.plane.normal_revision,
                    query1.plane.normal_revision);
}

BOOST_AUTO_TEST_CASE(FullMapNormalUpdateSkipsCpuNeighborhoodSearch) {
  TempPcd scan;
  scan.Write({PointInColmapWorld(0.041f, 0.0f, 1.0f),
              PointInColmapWorld(0.001f, 0.0f, 1.0f),
              PointInColmapWorld(0.021f, 0.0f, 1.0f)});
  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMap map(FakeDependencies(capture));
  AppendScanAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(1, scan.path()), &audit,
                                      &error), error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  BOOST_CHECK_EQUAL(audit.map_voxel_count, 3);
  BOOST_CHECK_EQUAL(audit.map_block_count, 1);
  BOOST_CHECK_EQUAL(audit.mutable_block_count, 1);
  BOOST_CHECK_EQUAL(audit.affected_query_count, 3);
  BOOST_CHECK_EQUAL(audit.support_voxel_count, 3);
  BOOST_CHECK(audit.all_voxels_queried);
  for (const NeighborhoodSearchAudit* search :
       {&audit.affected_search, &audit.support_search}) {
    BOOST_CHECK_EQUAL(search->query_count, 0);
    BOOST_CHECK_EQUAL(search->block_probe_count, 0);
    BOOST_CHECK_EQUAL(search->visited_block_count, 0);
    BOOST_CHECK_EQUAL(search->visited_voxel_count, 0);
    BOOST_CHECK_EQUAL(search->hit_count, 0);
    BOOST_CHECK_EQUAL(search->inserted_key_count, 0);
  }
  BOOST_REQUIRE_EQUAL(capture->calls.size(), 1);
  const std::vector<float> expected_xyz =
      SnapshotXyz(snapshot, snapshot->VoxelKeys());
  BOOST_CHECK(capture->calls.front().support_xyz == expected_xyz);
  BOOST_CHECK(capture->calls.front().query_xyz == expected_xyz);
  const std::vector<std::string> stages = {
      "input_read", "input_hash", "snapshot_copy", "parse_and_voxelize",
      "update_centroids", "affected_search", "support_search",
      "pack_normal_inputs", "normal_estimation", "decode_normals",
      "writeback_normals", "block_derived", "spatial_index",
      "nearest_index", "snapshot_finalize"};
  BOOST_REQUIRE_EQUAL(audit.stage_milliseconds.size(), stages.size());
  double accounted_milliseconds = 0.0;
  for (const std::string& stage : stages) {
    BOOST_REQUIRE_EQUAL(audit.stage_milliseconds.count(stage), 1);
    const double elapsed = audit.stage_milliseconds.at(stage);
    BOOST_CHECK(std::isfinite(elapsed));
    BOOST_CHECK_GE(elapsed, 0.0);
    accounted_milliseconds += elapsed;
  }
  BOOST_CHECK(std::isfinite(audit.total_milliseconds));
  BOOST_CHECK_GE(audit.total_milliseconds, accounted_milliseconds);
  BOOST_CHECK_EQUAL(audit.geometry_sha256_after,
                    snapshot->GeometrySha256());
  BOOST_CHECK_EQUAL(audit.snapshot_sha256_after,
                    snapshot->SnapshotSha256());
}

BOOST_AUTO_TEST_CASE(PcdCoordinateAndSourceNormalAuditIsExact) {
  TempPcd binary;
  TestPoint transformed;
  transformed.x = 1.0f;
  transformed.y = 2.0f;
  transformed.z = 3.0f;
  transformed.normal_x = 3.0f;
  transformed.normal_y = 4.0f;
  transformed.normal_z = 0.0f;
  TestPoint zero_normal = PointInColmapWorld(4.0f, 5.0f, 6.0f);
  TestPoint dropped = zero_normal;
  dropped.curvature = std::numeric_limits<float>::infinity();
  binary.Write({transformed, zero_normal, dropped}, true,
               "100 200 300 0.5 0.5 0.5 0.5");

  IncrementalCausalLidarMap map(FakeDependencies());
  AppendScanAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, binary.path()), &audit, &error), error);
  BOOST_CHECK(audit.source_frame == LidarCoordinateFrame::FASTLIO_WORLD);
  BOOST_CHECK(audit.output_frame == LidarCoordinateFrame::COLMAP_WORLD);
  BOOST_CHECK(audit.derived_normal_frame ==
              LidarCoordinateFrame::COLMAP_WORLD);
  BOOST_CHECK_EQUAL(audit.point_transform_count_min, 1);
  BOOST_CHECK_EQUAL(audit.point_transform_count_max, 1);
  BOOST_CHECK_EQUAL(audit.source_normal_transform_count_min, 1);
  BOOST_CHECK_EQUAL(audit.source_normal_transform_count_max, 1);
  BOOST_CHECK_EQUAL(audit.derived_normal_transform_count, 0);
  BOOST_CHECK_EQUAL(audit.input_record_count, 3);
  BOOST_CHECK_EQUAL(audit.accepted_record_count, 2);
  BOOST_CHECK_EQUAL(audit.dropped_non_finite_record_count, 1);
  BOOST_CHECK_EQUAL(audit.zero_source_normal_count, 1);
  BOOST_CHECK_EQUAL(audit.nonzero_source_normal_count, 1);
  BOOST_REQUIRE(audit.has_first_accepted_record);
  BOOST_CHECK_EQUAL(audit.first_transformed_point[0], -2.0f);
  BOOST_CHECK_EQUAL(audit.first_transformed_point[1], -3.0f);
  BOOST_CHECK_EQUAL(audit.first_transformed_point[2], 1.0f);
  BOOST_CHECK_SMALL(audit.first_transformed_source_normal[0] + 0.8f, 1e-6f);
  BOOST_CHECK_EQUAL(audit.first_transformed_source_normal[1], 0.0f);
  BOOST_CHECK_SMALL(audit.first_transformed_source_normal[2] - 0.6f, 1e-6f);
  BOOST_CHECK_EQUAL(audit.opened_scan_index, 1);
  BOOST_CHECK_EQUAL(audit.file_sha256.size(), 64);

  LidarVoxelRecord record;
  BOOST_REQUIRE(map.GetSnapshot()->FindVoxel(
      KeyForColmapPoint(-2.0f, -3.0f, 1.0f), &record));
  BOOST_CHECK_EQUAL(record.centroid[0], -2.0f);
  BOOST_CHECK_EQUAL(record.centroid[1], -3.0f);
  BOOST_CHECK_EQUAL(record.centroid[2], 1.0f);

  TempPcd ascii;
  ascii.Write({PointInColmapWorld(7.0f, 8.0f, 9.0f)}, false,
              "-999 -999 -999 0 1 0 0");
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(2, ascii.path()), &audit, &error), error);
  BOOST_REQUIRE(map.GetSnapshot()->FindVoxel(
      KeyForColmapPoint(7.0f, 8.0f, 9.0f), &record));
  BOOST_CHECK_EQUAL(record.centroid[0], 7.0f);
  BOOST_CHECK_EQUAL(record.centroid[1], 8.0f);
  BOOST_CHECK_EQUAL(record.centroid[2], 9.0f);
}

BOOST_AUTO_TEST_CASE(PcdReorderedFieldsExtraFieldAndDefaultCountAreParsed) {
  TempPcd scan;
  std::string bytes =
      "VERSION 0.7\n"
      "FIELDS ring normal_z y curvature x intensity normal_x z normal_y\n"
      "SIZE 2 4 4 4 4 4 4 4 4\n"
      "TYPE U F F F F F F F F\n"
      "WIDTH 1\n"
      "HEIGHT 1\n"
      "POINTS 1\n"
      "DATA binary\n";
  AppendLittleEndianUint16(42, &bytes);
  AppendLittleEndianFloat(0.0f, &bytes);
  AppendLittleEndianFloat(2.0f, &bytes);
  AppendLittleEndianFloat(0.25f, &bytes);
  AppendLittleEndianFloat(1.0f, &bytes);
  AppendLittleEndianFloat(9.0f, &bytes);
  AppendLittleEndianFloat(3.0f, &bytes);
  AppendLittleEndianFloat(3.0f, &bytes);
  AppendLittleEndianFloat(4.0f, &bytes);
  scan.WriteRaw(bytes);

  IncrementalCausalLidarMap map(FakeDependencies());
  AppendScanAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan.path()), &audit, &error), error);
  BOOST_REQUIRE(audit.has_first_accepted_record);
  BOOST_CHECK_EQUAL(audit.input_record_count, 1);
  BOOST_CHECK_EQUAL(audit.accepted_record_count, 1);
  BOOST_CHECK_EQUAL(audit.first_transformed_point[0], -2.0f);
  BOOST_CHECK_EQUAL(audit.first_transformed_point[1], -3.0f);
  BOOST_CHECK_EQUAL(audit.first_transformed_point[2], 1.0f);
  BOOST_CHECK_SMALL(audit.first_transformed_source_normal[0] + 0.8f, 1e-6f);
  BOOST_CHECK_EQUAL(audit.first_transformed_source_normal[1], 0.0f);
  BOOST_CHECK_SMALL(audit.first_transformed_source_normal[2] - 0.6f, 1e-6f);

  LidarVoxelRecord record;
  BOOST_REQUIRE(map.GetSnapshot()->FindVoxel(
      KeyForColmapPoint(-2.0f, -3.0f, 1.0f), &record));
  BOOST_CHECK_EQUAL(record.centroid[0], -2.0f);
  BOOST_CHECK_EQUAL(record.centroid[1], -3.0f);
  BOOST_CHECK_EQUAL(record.centroid[2], 1.0f);
}

BOOST_AUTO_TEST_CASE(InputFrameTransformCountsAndAllInvalidAreRejected) {
  TempPcd valid;
  valid.Write({PointInColmapWorld(0.0f, 0.0f, 0.0f)});
  const std::shared_ptr<FailureControl> control(new FailureControl());
  IncrementalCausalLidarMap map(ControlledDependencies(control));
  const std::shared_ptr<const LidarMapSnapshot> initial = map.GetSnapshot();
  std::string error;

  ScanSource source = MakeSource(1, valid.path());
  source.input_frame = LidarCoordinateFrame::COLMAP_WORLD;
  BOOST_CHECK(!map.AppendScan(source, nullptr, &error));
  BOOST_CHECK_EQUAL(control->stat_calls, 0);
  source.input_frame = LidarCoordinateFrame::FASTLIO_WORLD;
  source.point_transform_count = 1;
  BOOST_CHECK(!map.AppendScan(source, nullptr, &error));
  BOOST_CHECK_EQUAL(control->stat_calls, 0);
  source.point_transform_count = 0;
  source.normal_transform_count = 1;
  BOOST_CHECK(!map.AppendScan(source, nullptr, &error));
  BOOST_CHECK_EQUAL(control->stat_calls, 0);
  BOOST_CHECK(map.GetSnapshot() == initial);

  TempPcd invalid;
  TestPoint point = PointInColmapWorld(0.0f, 0.0f, 0.0f);
  point.normal_y = std::numeric_limits<float>::quiet_NaN();
  invalid.Write({point});
  source = MakeSource(1, invalid.path());
  source.expected_sha256 = std::string(64, 'a');
  BOOST_CHECK(!map.AppendScan(source, nullptr, &error));
  BOOST_CHECK(error.find("no finite") != std::string::npos);
  BOOST_CHECK(map.GetSnapshot() == initial);
  BOOST_CHECK_EQUAL(initial->Version(), 0);
}

BOOST_AUTO_TEST_CASE(VoxelBoundariesAccumulatorsAndCountOnlyAreDeterministic) {
  const float positive_zero = 0.0f;
  const float negative_zero = -0.0f;
  const float positive_boundary = std::nextafter(
      0.01f, std::numeric_limits<float>::infinity());
  const float negative_tiny = std::nextafter(
      0.0f, -std::numeric_limits<float>::infinity());
  const float negative_boundary = std::nextafter(
      -0.01f, -std::numeric_limits<float>::infinity());
  TempPcd scan1;
  scan1.Write({PointInColmapWorld(positive_zero, 0.0f, 0.0f),
               PointInColmapWorld(negative_zero, 0.0f, 0.0f),
               PointInColmapWorld(positive_boundary, 0.0f, 0.0f),
               PointInColmapWorld(negative_tiny, 0.0f, 0.0f),
               PointInColmapWorld(negative_boundary, 0.0f, 0.0f)});
  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMap map(FakeDependencies(capture));
  AppendScanAudit audit1;
  std::string error;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(1, scan1.path()), &audit1,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot1 = map.GetSnapshot();
  BOOST_CHECK_EQUAL(snapshot1->VoxelCount(), 4);
  BOOST_CHECK_EQUAL(KeyForColmapPoint(positive_boundary, 0.0f, 0.0f).x, 1);
  BOOST_CHECK_EQUAL(KeyForColmapPoint(negative_tiny, 0.0f, 0.0f).x, -1);
  BOOST_CHECK_EQUAL(KeyForColmapPoint(negative_boundary, 0.0f, 0.0f).x, -2);

  LidarVoxelRecord zero_record;
  BOOST_REQUIRE(snapshot1->FindVoxel(VoxelKey{0, 0, 0}, &zero_record));
  BOOST_CHECK_EQUAL(zero_record.count, 2);
  BOOST_CHECK_EQUAL(zero_record.xyz_sum[0], 0.0);
  BOOST_CHECK(!std::signbit(zero_record.xyz_sum[0]));
  BOOST_CHECK(!std::signbit(zero_record.centroid[0]));
  const uint64_t revision_before = zero_record.outer_normal.revision;
  const std::string geometry_before = snapshot1->GeometrySha256();
  const std::string snapshot_before = snapshot1->SnapshotSha256();
  const size_t estimator_calls_before = capture->calls.size();
  const NearestPlaneResult nearest_before = snapshot1->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(nearest_before.ok);
  BOOST_REQUIRE(nearest_before.found);
  BOOST_CHECK(nearest_before.plane.key == zero_record.key);
  BOOST_CHECK_EQUAL(nearest_before.plane.voxel_count, 2);
  BOOST_CHECK_EQUAL(nearest_before.plane.normal_revision, revision_before);

  TempPcd scan2;
  scan2.Write({PointInColmapWorld(positive_zero, 0.0f, 0.0f)});
  AppendScanAudit audit2;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(2, scan2.path()), &audit2,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot2 = map.GetSnapshot();
  LidarVoxelRecord updated_zero_record;
  BOOST_REQUIRE(snapshot2->FindVoxel(VoxelKey{0, 0, 0},
                                     &updated_zero_record));
  BOOST_CHECK_EQUAL(updated_zero_record.count, 3);
  BOOST_CHECK_EQUAL(updated_zero_record.first_scan_index, 1);
  BOOST_CHECK_EQUAL(updated_zero_record.last_scan_index, 2);
  BOOST_CHECK_EQUAL(updated_zero_record.outer_normal.revision, revision_before);
  BOOST_CHECK_EQUAL(audit2.touched_voxel_count, 1);
  BOOST_CHECK_EQUAL(audit2.new_voxel_count, 0);
  BOOST_CHECK_EQUAL(audit2.geometry_changed_voxel_count, 0);
  BOOST_CHECK_EQUAL(audit2.count_only_voxel_count, 1);
  BOOST_CHECK_EQUAL(audit2.affected_query_count, 0);
  BOOST_CHECK_EQUAL(audit2.support_voxel_count, 0);
  BOOST_CHECK_EQUAL(capture->calls.size(), estimator_calls_before);
  for (const NeighborhoodSearchAudit* search :
       {&audit2.affected_search, &audit2.support_search}) {
    BOOST_CHECK_EQUAL(search->query_count, 0);
    BOOST_CHECK_EQUAL(search->block_probe_count, 0);
    BOOST_CHECK_EQUAL(search->visited_block_count, 0);
    BOOST_CHECK_EQUAL(search->visited_voxel_count, 0);
    BOOST_CHECK_EQUAL(search->hit_count, 0);
    BOOST_CHECK_EQUAL(search->inserted_key_count, 0);
  }
  BOOST_CHECK_EQUAL(snapshot2->GeometrySha256(), geometry_before);
  BOOST_CHECK_NE(snapshot2->SnapshotSha256(), snapshot_before);
  BOOST_REQUIRE_EQUAL(audit2.stage_milliseconds.count("nearest_index"), 1);
  BOOST_CHECK(std::isfinite(audit2.stage_milliseconds.at("nearest_index")));
  BOOST_CHECK_GE(audit2.stage_milliseconds.at("nearest_index"), 0.0);

  const NearestPlaneResult nearest_after = snapshot2->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(nearest_after.ok);
  BOOST_REQUIRE(nearest_after.found);
  BOOST_CHECK(nearest_after.plane.key == zero_record.key);
  BOOST_CHECK_EQUAL(nearest_after.plane.voxel_count, 3);
  BOOST_CHECK_EQUAL(nearest_after.plane.normal_revision, revision_before);
  const NearestPlaneResult retained_nearest = snapshot1->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(retained_nearest.ok);
  BOOST_REQUIRE(retained_nearest.found);
  BOOST_CHECK_EQUAL(retained_nearest.plane.voxel_count, 2);
  BOOST_CHECK_EQUAL(retained_nearest.plane.normal_revision, revision_before);

  BOOST_CHECK_EQUAL(VoxelKeyToBlockKey(VoxelKey{-1, 0, 0}).x, -1);
  BOOST_CHECK_EQUAL(VoxelKeyToBlockKey(VoxelKey{-32, 0, 0}).x, -1);
  BOOST_CHECK_EQUAL(VoxelKeyToBlockKey(VoxelKey{-33, 0, 0}).x, -2);
  BOOST_CHECK_EQUAL(VoxelKeyToBlockKey(VoxelKey{31, 0, 0}).x, 0);
  BOOST_CHECK_EQUAL(VoxelKeyToBlockKey(VoxelKey{32, 0, 0}).x, 1);
}

BOOST_AUTO_TEST_CASE(InjectedFailuresRollbackAndSameScanRetries) {
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write({PointInColmapWorld(0.0f, 0.0f, 1.0f),
               PointInColmapWorld(0.01f, 0.0f, 1.0f)});
  scan2.Write({PointInColmapWorld(0.03f, 0.0f, 1.0f)});
  const std::vector<FailureMode> failure_modes = {
      FailureMode::STAT_FALSE,
      FailureMode::STAT_THROW,
      FailureMode::READ_FALSE,
      FailureMode::READ_THROW,
      FailureMode::HASH_FALSE,
      FailureMode::HASH_THROW,
      FailureMode::NORMAL_FALSE,
      FailureMode::NORMAL_THROW,
      FailureMode::NORMAL_WRONG_LENGTH,
      FailureMode::NORMAL_PARTIAL_NAN,
      FailureMode::NORMAL_PARTIAL_INF,
      FailureMode::NORMAL_NAN_TIMING,
      FailureMode::NORMAL_NEGATIVE_TIMING,
      FailureMode::NORMAL_ZERO_VECTOR,
  };

  for (const FailureMode failure_mode : failure_modes) {
    const std::shared_ptr<FailureControl> control(new FailureControl());
    IncrementalCausalLidarMap map(ControlledDependencies(control));
    std::string error;
    BOOST_REQUIRE_MESSAGE(
        map.AppendScan(MakeSource(1, scan1.path()), nullptr, &error), error);
    const std::shared_ptr<const LidarMapSnapshot> before = map.GetSnapshot();
    const std::string geometry_before = before->GeometrySha256();
    const std::string snapshot_before = before->SnapshotSha256();
    const NearestPlaneResult query_before = before->FindNearestPlane(
        {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.02);
    BOOST_REQUIRE(query_before.ok);
    BOOST_REQUIRE(query_before.found);
    control->mode = failure_mode;
    AppendScanAudit audit;
    BOOST_CHECK(!map.AppendScan(MakeSource(2, scan2.path()), &audit, &error));
    BOOST_CHECK(!error.empty());
    BOOST_CHECK(map.GetSnapshot() == before);
    BOOST_CHECK_EQUAL(map.GetSnapshot()->Version(), 1);
    BOOST_CHECK_EQUAL(map.GetSnapshot()->GeometrySha256(), geometry_before);
    BOOST_CHECK_EQUAL(map.GetSnapshot()->SnapshotSha256(), snapshot_before);
    BOOST_CHECK_EQUAL(audit.version_after, 0);
    const NearestPlaneResult query_after_failure =
        map.GetSnapshot()->FindNearestPlane(
            {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.02);
    BOOST_REQUIRE(query_after_failure.ok);
    BOOST_REQUIRE(query_after_failure.found);
    BOOST_CHECK(query_after_failure.plane.key == query_before.plane.key);
    BOOST_CHECK_EQUAL(query_after_failure.plane.normal_revision,
                      query_before.plane.normal_revision);

    control->mode = FailureMode::NONE;
    BOOST_REQUIRE_MESSAGE(
        map.AppendScan(MakeSource(2, scan2.path()), &audit, &error), error);
    BOOST_CHECK_EQUAL(map.GetSnapshot()->Version(), 2);
    BOOST_CHECK_EQUAL(map.GetSnapshot()->MaxScanIndex(), 2);
  }
}

BOOST_AUTO_TEST_CASE(SchemaAndManifestFailuresRollbackAndRetry) {
  TempPcd scan;
  scan.WriteRaw(
      "VERSION 0.7\n"
      "FIELDS x y z intensity normal_x normal_y normal_z\n"
      "SIZE 4 4 4 4 4 4 4\n"
      "TYPE F F F F F F F\n"
      "COUNT 1 1 1 1 1 1 1\n"
      "WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n"
      "0 0 0 1 0 0 0\n");
  IncrementalCausalLidarMap map(FakeDependencies());
  const std::shared_ptr<const LidarMapSnapshot> initial = map.GetSnapshot();
  std::string error;
  BOOST_CHECK(!map.AppendScan(MakeSource(1, scan.path()), nullptr, &error));
  BOOST_CHECK(error.find("curvature") != std::string::npos);
  BOOST_CHECK(map.GetSnapshot() == initial);

  scan.Write({PointInColmapWorld(0.0f, 0.0f, 1.0f)});
  ScanSource source = MakeSource(1, scan.path());
  source.expected_size_bytes = 1;
  BOOST_CHECK(!map.AppendScan(source, nullptr, &error));
  BOOST_CHECK(map.GetSnapshot() == initial);
  source.expected_size_bytes = 0;
  source.expected_sha256 = std::string(64, 'f');
  BOOST_CHECK(!map.AppendScan(source, nullptr, &error));
  BOOST_CHECK(map.GetSnapshot() == initial);
  source.expected_sha256.clear();
  BOOST_REQUIRE_MESSAGE(map.AppendScan(source, nullptr, &error), error);
  BOOST_CHECK_EQUAL(map.GetSnapshot()->Version(), 1);
}

BOOST_AUTO_TEST_CASE(PcdOverflowAndBinaryPayloadLengthFailuresRollback) {
  TempPcd scan;
  IncrementalCausalLidarMap map(FakeDependencies());
  const std::shared_ptr<const LidarMapSnapshot> initial = map.GetSnapshot();
  std::string error;

  scan.WriteRaw(
      "VERSION 0.7\n"
      "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
      "SIZE 4 4 4 4 4 4 4 4\n"
      "TYPE F F F F F F F F\n"
      "COUNT 1 1 1 1 1 1 1 1\n"
      "WIDTH 18446744073709551615\n"
      "HEIGHT 2\n"
      "DATA ascii\n");
  BOOST_CHECK(!map.AppendScan(MakeSource(1, scan.path()), nullptr, &error));
  BOOST_CHECK(error.find("overflows") != std::string::npos);
  BOOST_CHECK(map.GetSnapshot() == initial);

  const TestPoint point = PointInColmapWorld(0.1f, 0.2f, 0.3f);
  const std::string complete = MakeSinglePointBinaryPcd(point);
  scan.WriteRaw(complete.substr(0, complete.size() - 1));
  BOOST_CHECK(!map.AppendScan(MakeSource(1, scan.path()), nullptr, &error));
  BOOST_CHECK(error.find("payload size") != std::string::npos);
  BOOST_CHECK(map.GetSnapshot() == initial);

  scan.WriteRaw(complete + std::string(1, '\0'));
  BOOST_CHECK(!map.AppendScan(MakeSource(1, scan.path()), nullptr, &error));
  BOOST_CHECK(error.find("payload size") != std::string::npos);
  BOOST_CHECK(map.GetSnapshot() == initial);

  scan.WriteRaw(complete);
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan.path()), nullptr, &error), error);
  BOOST_CHECK_EQUAL(map.GetSnapshot()->Version(), 1);
}

BOOST_AUTO_TEST_CASE(
    CentroidChangeUsesConservativeDirtyGridAndPreservesOldSnapshot) {
  const float old_center = 0.001f;
  const float incoming = 0.009f;
  const float new_center = static_cast<float>(
      (static_cast<double>(old_center) + incoming) / 2.0);
  const float radius =
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters;
  const float old_only = old_center - radius;
  const float new_only = new_center + radius;
  const float recursive_only = new_only + radius;
  BOOST_CHECK_LE(std::abs(static_cast<double>(old_only) - old_center),
                 static_cast<double>(radius));
  BOOST_CHECK_GT(std::abs(static_cast<double>(old_only) - new_center),
                 static_cast<double>(radius));
  BOOST_CHECK_LE(std::abs(static_cast<double>(new_only) - new_center),
                 static_cast<double>(radius));
  BOOST_CHECK_GT(std::abs(static_cast<double>(new_only) - old_center),
                 static_cast<double>(radius));
  BOOST_CHECK_GT(std::abs(static_cast<double>(recursive_only) - new_center),
                 static_cast<double>(radius));

  std::vector<TestPoint> first_points = {
      PointInColmapWorld(old_center, 0.0f, 1.0f),
      PointInColmapWorld(old_only, 0.0f, 1.0f),
      PointInColmapWorld(new_only, 0.0f, 1.0f),
      PointInColmapWorld(recursive_only, 0.0f, 1.0f),
  };
  for (int i = 0; i < 12; ++i) {
    first_points.push_back(
        PointInColmapWorld(5.0f + 0.02f * i, 0.0f, 1.0f));
  }
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write(first_points);
  scan2.Write({PointInColmapWorld(incoming, 0.0f, 1.0f)});
  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMap map(FakeDependencies(capture));
  std::string error;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(1, scan1.path()), nullptr,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot1 = map.GetSnapshot();
  const NearestPlaneResult old_nearest = snapshot1->FindNearestPlane(
      {{static_cast<double>(old_center), 0.0, 1.0}},
      LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(old_nearest.ok);
  BOOST_REQUIRE(old_nearest.found);
  BOOST_CHECK_EQUAL(old_nearest.plane.voxel_count, 1);
  BOOST_CHECK_EQUAL(old_nearest.plane.normal_revision, 1);
  LidarVoxelRecord far_before;
  BOOST_REQUIRE(snapshot1->FindVoxel(
      KeyForColmapPoint(5.0f, 0.0f, 1.0f), &far_before));
  AppendScanAudit audit;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(2, scan2.path()), &audit,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  BOOST_CHECK_EQUAL(audit.geometry_changed_voxel_count, 1);
  BOOST_CHECK_EQUAL(audit.new_voxel_count, 0);
  BOOST_CHECK_EQUAL(audit.affected_query_count, 4);
  BOOST_CHECK_EQUAL(audit.support_voxel_count, snapshot->VoxelCount());
  BOOST_CHECK(!audit.all_voxels_queried);

  LidarVoxelRecord old_record;
  LidarVoxelRecord old_only_record;
  LidarVoxelRecord new_only_record;
  LidarVoxelRecord recursive_record;
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(old_center, 0.0f, 1.0f), &old_record));
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(old_only, 0.0f, 1.0f), &old_only_record));
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(new_only, 0.0f, 1.0f), &new_only_record));
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(recursive_only, 0.0f, 1.0f), &recursive_record));
  BOOST_CHECK_EQUAL(old_record.outer_normal.revision, 2);
  BOOST_CHECK_EQUAL(old_record.count, 2);
  BOOST_CHECK_EQUAL(old_record.centroid[0], new_center);
  BOOST_CHECK_EQUAL(old_only_record.outer_normal.revision, 2);
  BOOST_CHECK_EQUAL(new_only_record.outer_normal.revision, 2);
  BOOST_CHECK_EQUAL(recursive_record.outer_normal.revision, 2);
  const NearestPlaneResult new_nearest = snapshot->FindNearestPlane(
      {{static_cast<double>(new_center), 0.0, 1.0}},
      LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(new_nearest.ok);
  BOOST_REQUIRE(new_nearest.found);
  BOOST_CHECK(new_nearest.plane.key == old_record.key);
  BOOST_CHECK_EQUAL(new_nearest.plane.point[0], new_center);
  BOOST_CHECK_EQUAL(new_nearest.plane.voxel_count, 2);
  BOOST_CHECK_EQUAL(new_nearest.plane.normal_revision, 2);
  const NearestPlaneResult retained_old_nearest =
      snapshot1->FindNearestPlane(
          {{static_cast<double>(old_center), 0.0, 1.0}},
          LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(retained_old_nearest.ok);
  BOOST_REQUIRE(retained_old_nearest.found);
  BOOST_CHECK_EQUAL(retained_old_nearest.plane.point[0], old_center);
  BOOST_CHECK_EQUAL(retained_old_nearest.plane.voxel_count, 1);
  BOOST_CHECK_EQUAL(retained_old_nearest.plane.normal_revision, 1);
  LidarVoxelRecord far_after;
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(5.0f, 0.0f, 1.0f), &far_after));
  BOOST_CHECK_EQUAL(far_after.count, far_before.count);
  BOOST_CHECK_EQUAL(far_after.centroid[0], far_before.centroid[0]);
  BOOST_CHECK_EQUAL(far_after.centroid[1], far_before.centroid[1]);
  BOOST_CHECK_EQUAL(far_after.centroid[2], far_before.centroid[2]);
  CheckNormalMatches(far_after.outer_normal, far_before.outer_normal);
  CheckNormalMatches(far_after.inner_normal, far_before.inner_normal);
  BOOST_CHECK_EQUAL(far_after.outer_normal.revision, 1);
  BOOST_CHECK_EQUAL(far_after.inner_normal.revision, 1);
  LidarVoxelRecord far_retained;
  BOOST_REQUIRE(snapshot1->FindVoxel(
      KeyForColmapPoint(5.0f, 0.0f, 1.0f), &far_retained));
  BOOST_CHECK_EQUAL(far_retained.outer_normal.revision, 1);
  BOOST_CHECK_EQUAL(far_retained.inner_normal.revision, 1);
  CheckNormalMatches(far_retained.outer_normal, far_before.outer_normal);
  CheckNormalMatches(far_retained.inner_normal, far_before.inner_normal);

  const std::vector<VoxelKey> all_keys = snapshot->VoxelKeys();
  std::vector<VoxelKey> affected_keys;
  for (const VoxelKey& key : all_keys) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    if (record.outer_normal.revision == 2) affected_keys.push_back(key);
  }
  const std::vector<float> all_xyz = SnapshotXyz(snapshot, all_keys);
  const std::vector<float> affected_xyz = SnapshotXyz(snapshot, affected_keys);
  BOOST_REQUIRE_EQUAL(capture->calls.size(), 2);
  BOOST_CHECK(capture->calls.back().support_xyz == all_xyz);
  BOOST_CHECK(capture->calls.back().query_xyz == affected_xyz);
  for (const NeighborhoodSearchAudit* search :
       {&audit.affected_search, &audit.support_search}) {
    BOOST_CHECK_EQUAL(search->query_count, 0);
    BOOST_CHECK_EQUAL(search->block_probe_count, 0);
    BOOST_CHECK_EQUAL(search->visited_block_count, 0);
    BOOST_CHECK_EQUAL(search->visited_voxel_count, 0);
    BOOST_CHECK_EQUAL(search->hit_count, 0);
    BOOST_CHECK_EQUAL(search->inserted_key_count, 0);
  }
  std::vector<float> outer_reference;
  std::vector<float> inner_reference;
  CudaNormalEstimationTiming timing;
  BOOST_REQUIRE(NeighborhoodEstimator(
      all_xyz, affected_xyz,
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters,
      IncrementalCausalLidarMap::kInnerNormalRadiusMeters, &outer_reference,
      &inner_reference, &timing, &error));
  for (size_t i = 0; i < affected_keys.size(); ++i) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(affected_keys[i], &record));
    CheckNormalMatches(record.outer_normal,
                       NormalizedExpected(outer_reference, i));
    CheckNormalMatches(record.inner_normal,
                       NormalizedExpected(inner_reference, i));
  }
}

BOOST_AUTO_TEST_CASE(ChangedAndNewCellsMarkBothHaloBoundaries) {
  const float changed_old = 0.161f;
  const float changed_incoming = 0.169f;
  const float changed_new = static_cast<float>(
      (static_cast<double>(changed_old) + changed_incoming) / 2.0);
  const float new_voxel = -0.319f;
  const float negative_boundary_witness = -0.47f;
  const float positive_boundary_witness = 0.33f;
  const float negative_remote = -0.63f;
  const float positive_remote = 0.49f;
  const float radius =
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters;
  BOOST_CHECK_GT(std::abs(static_cast<double>(negative_boundary_witness) -
                          static_cast<double>(new_voxel)),
                 static_cast<double>(radius));
  BOOST_CHECK_GT(std::abs(static_cast<double>(positive_boundary_witness) -
                          static_cast<double>(changed_new)),
                 static_cast<double>(radius));

  TempPcd scan1;
  TempPcd scan2;
  scan1.Write({PointInColmapWorld(negative_remote, 0.0f, 1.0f),
               PointInColmapWorld(negative_boundary_witness, 0.0f, 1.0f),
               PointInColmapWorld(-0.15f, 0.0f, 1.0f),
               PointInColmapWorld(0.01f, 0.0f, 1.0f),
               PointInColmapWorld(changed_old, 0.0f, 1.0f),
               PointInColmapWorld(positive_boundary_witness, 0.0f, 1.0f),
               PointInColmapWorld(positive_remote, 0.0f, 1.0f)});
  scan2.Write({PointInColmapWorld(changed_incoming, 0.0f, 1.0f),
               PointInColmapWorld(new_voxel, 0.0f, 1.0f)});

  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMap map(FakeDependencies(capture));
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan1.path()), nullptr, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> old_snapshot = map.GetSnapshot();
  const std::string old_snapshot_hash = old_snapshot->SnapshotSha256();
  LidarVoxelRecord negative_remote_before;
  LidarVoxelRecord positive_remote_before;
  BOOST_REQUIRE(old_snapshot->FindVoxel(
      KeyForColmapPoint(negative_remote, 0.0f, 1.0f),
      &negative_remote_before));
  BOOST_REQUIRE(old_snapshot->FindVoxel(
      KeyForColmapPoint(positive_remote, 0.0f, 1.0f),
      &positive_remote_before));

  AppendScanAudit audit;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(2, scan2.path()), &audit, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  BOOST_CHECK_EQUAL(audit.geometry_changed_voxel_count, 1);
  BOOST_CHECK_EQUAL(audit.new_voxel_count, 1);
  BOOST_CHECK_EQUAL(audit.affected_query_count, 6);
  BOOST_CHECK_EQUAL(audit.support_voxel_count, snapshot->VoxelCount());
  BOOST_CHECK(!audit.all_voxels_queried);

  std::vector<VoxelKey> expected_affected_keys = {
      KeyForColmapPoint(negative_boundary_witness, 0.0f, 1.0f),
      KeyForColmapPoint(new_voxel, 0.0f, 1.0f),
      KeyForColmapPoint(-0.15f, 0.0f, 1.0f),
      KeyForColmapPoint(0.01f, 0.0f, 1.0f),
      KeyForColmapPoint(changed_old, 0.0f, 1.0f),
      KeyForColmapPoint(positive_boundary_witness, 0.0f, 1.0f),
  };
  std::sort(expected_affected_keys.begin(), expected_affected_keys.end());
  for (const VoxelKey& key : expected_affected_keys) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    BOOST_CHECK_EQUAL(record.outer_normal.revision, 2);
    BOOST_CHECK_EQUAL(record.inner_normal.revision, 2);
  }

  for (const std::pair<float, LidarVoxelRecord>& remote :
       {std::make_pair(negative_remote, negative_remote_before),
        std::make_pair(positive_remote, positive_remote_before)}) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(
        KeyForColmapPoint(remote.first, 0.0f, 1.0f), &record));
    BOOST_CHECK_EQUAL(record.outer_normal.revision, 1);
    BOOST_CHECK_EQUAL(record.inner_normal.revision, 1);
    CheckNormalMatches(record.outer_normal, remote.second.outer_normal);
    CheckNormalMatches(record.inner_normal, remote.second.inner_normal);
  }

  BOOST_REQUIRE_EQUAL(capture->calls.size(), 2);
  const std::vector<float> support_xyz =
      SnapshotXyz(snapshot, snapshot->VoxelKeys());
  const std::vector<float> query_xyz =
      SnapshotXyz(snapshot, expected_affected_keys);
  BOOST_CHECK(capture->calls.back().support_xyz == support_xyz);
  BOOST_CHECK(capture->calls.back().query_xyz == query_xyz);
  BOOST_CHECK_EQUAL(old_snapshot->SnapshotSha256(), old_snapshot_hash);
  LidarVoxelRecord old_changed_record;
  BOOST_REQUIRE(old_snapshot->FindVoxel(
      KeyForColmapPoint(changed_old, 0.0f, 1.0f), &old_changed_record));
  BOOST_CHECK_EQUAL(old_changed_record.centroid[0], changed_old);
  BOOST_CHECK_EQUAL(old_changed_record.outer_normal.revision, 1);
}

BOOST_AUTO_TEST_CASE(
    PackedNormalInputsPreserveGlobalKeyOrderAndHeldSnapshot) {
  const float changed_old = 0.001f;
  const float changed_incoming = 0.009f;
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write({PointInColmapWorld(0.011f, -0.66f, -1.31f),
               PointInColmapWorld(-0.011f, 0.66f, 1.31f),
               PointInColmapWorld(changed_old, 0.66f, 1.31f),
               PointInColmapWorld(0.021f, 0.0f, 0.0f)});
  scan2.Write({PointInColmapWorld(changed_incoming, 0.66f, 1.31f)});

  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMap map(FakeDependencies(capture));
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan1.path()), nullptr, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> held_snapshot =
      map.GetSnapshot();
  const std::string held_geometry_hash = held_snapshot->GeometrySha256();
  const std::string held_snapshot_hash = held_snapshot->SnapshotSha256();
  const std::vector<VoxelKey> held_keys = held_snapshot->VoxelKeys();
  const std::vector<float> held_xyz = SnapshotXyz(held_snapshot, held_keys);

  AppendScanAudit audit;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(2, scan2.path()), &audit, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  const std::vector<VoxelKey> sorted_keys = snapshot->VoxelKeys();
  BOOST_REQUIRE_EQUAL(sorted_keys.size(), 4);
  for (size_t key_index = 1; key_index < sorted_keys.size(); ++key_index) {
    BOOST_CHECK(sorted_keys[key_index - 1] < sorted_keys[key_index]);
  }
  BOOST_CHECK(VoxelKeyToBlockKey(sorted_keys[2]) <
              VoxelKeyToBlockKey(sorted_keys[1]));

  std::vector<VoxelKey> affected_keys;
  for (const VoxelKey& key : sorted_keys) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    if (record.outer_normal.revision == 2) affected_keys.push_back(key);
  }
  BOOST_REQUIRE_EQUAL(affected_keys.size(), 2);
  const std::vector<float> expected_support =
      SnapshotXyz(snapshot, sorted_keys);
  const std::vector<float> expected_query =
      SnapshotXyz(snapshot, affected_keys);
  BOOST_REQUIRE_EQUAL(capture->calls.size(), 2);
  CheckFloatVectorsBitwiseEqual(capture->calls.back().support_xyz,
                                expected_support);
  CheckFloatVectorsBitwiseEqual(capture->calls.back().query_xyz,
                                expected_query);
  BOOST_CHECK_EQUAL(audit.support_voxel_count, sorted_keys.size());
  BOOST_CHECK_EQUAL(audit.affected_query_count, affected_keys.size());

  std::vector<float> outer_reference;
  std::vector<float> inner_reference;
  CudaNormalEstimationTiming timing;
  BOOST_REQUIRE(NeighborhoodEstimator(
      expected_support, expected_query,
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters,
      IncrementalCausalLidarMap::kInnerNormalRadiusMeters, &outer_reference,
      &inner_reference, &timing, &error));
  for (size_t query_index = 0; query_index < affected_keys.size();
       ++query_index) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(affected_keys[query_index], &record));
    CheckNormalMatches(record.outer_normal,
                       NormalizedExpected(outer_reference, query_index));
    CheckNormalMatches(record.inner_normal,
                       NormalizedExpected(inner_reference, query_index));
    BOOST_CHECK_EQUAL(record.outer_normal.revision, 2);
    BOOST_CHECK_EQUAL(record.inner_normal.revision, 2);
  }

  BOOST_CHECK_EQUAL(held_snapshot->GeometrySha256(), held_geometry_hash);
  BOOST_CHECK_EQUAL(held_snapshot->SnapshotSha256(), held_snapshot_hash);
  const std::vector<VoxelKey> held_keys_after = held_snapshot->VoxelKeys();
  BOOST_REQUIRE_EQUAL(held_keys_after.size(), held_keys.size());
  for (size_t key_index = 0; key_index < held_keys.size(); ++key_index) {
    BOOST_CHECK(held_keys_after[key_index] == held_keys[key_index]);
  }
  CheckFloatVectorsBitwiseEqual(
      SnapshotXyz(held_snapshot, held_keys_after), held_xyz);

  TempPcd scan3;
  scan3.Write({PointInColmapWorld(-2.0f, 0.0f, 1.0f),
               PointInColmapWorld(0.015f, 0.8f, 0.0f),
               PointInColmapWorld(2.0f, 0.0f, 1.0f),
               PointInColmapWorld(0.029f, 0.0f, 0.0f),
               PointInColmapWorld(0.003f, 0.66f, 1.31f)});
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(3, scan3.path()), &audit, &error), error);
  const auto merged_snapshot = map.GetSnapshot();
  const auto merged_keys = merged_snapshot->VoxelKeys();
  BOOST_CHECK_EQUAL(audit.new_voxel_count, 3);
  BOOST_CHECK_EQUAL(audit.geometry_changed_voxel_count, 2);
  BOOST_REQUIRE_EQUAL(merged_keys.size(), 7);
  BOOST_REQUIRE_EQUAL(capture->calls.size(), 3);
  CheckFloatVectorsBitwiseEqual(capture->calls.back().support_xyz,
                                SnapshotXyz(merged_snapshot, merged_keys));
  CheckFloatVectorsBitwiseEqual(SnapshotXyz(snapshot, sorted_keys),
                                expected_support);
  BOOST_CHECK_EQUAL(held_snapshot->SnapshotSha256(), held_snapshot_hash);
}

BOOST_AUTO_TEST_CASE(NormalRadiusBoundariesAreClosedAtBothScales) {
  const float inner =
      IncrementalCausalLidarMap::kInnerNormalRadiusMeters;
  const float outer =
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters;
  TempPcd scan;
  scan.Write({PointInColmapWorld(0.0f, 0.0f, 1.0f),
              PointInColmapWorld(inner, 0.0f, 1.0f),
              PointInColmapWorld(outer, 0.0f, 1.0f)});
  IncrementalCausalLidarMap map(FakeDependencies());
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan.path()), nullptr, &error), error);
  LidarVoxelRecord origin;
  BOOST_REQUIRE(map.GetSnapshot()->FindVoxel(
      KeyForColmapPoint(0.0f, 0.0f, 1.0f), &origin));
  BOOST_CHECK_EQUAL(origin.outer_normal.curvature, 3.0f);
  BOOST_CHECK_EQUAL(origin.inner_normal.curvature, 2.0f);
}

BOOST_AUTO_TEST_CASE(FloatRoundedRadiusBoundaryMatchesFullCudaReference) {
  const float positive_boundary = 0.07500000298f;
  const float negative_boundary = -0.07500001043f;
  const float radius =
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters;
  const float cuda_delta = positive_boundary - negative_boundary;
  BOOST_REQUIRE_LE(cuda_delta * cuda_delta, radius * radius);
  BOOST_REQUIRE_GT(static_cast<double>(positive_boundary) -
                       static_cast<double>(negative_boundary),
                   static_cast<double>(radius));

  std::vector<TestPoint> first_points = {
      PointInColmapWorld(negative_boundary, 0.0f, 1.0f),
      PointInColmapWorld(negative_boundary, 0.02f, 1.0f),
      PointInColmapWorld(negative_boundary, 0.0f, 1.02f),
      PointInColmapWorld(negative_boundary + 0.01f, 0.01f, 0.99f),
  };
  for (int i = 0; i < 12; ++i) {
    first_points.push_back(
        PointInColmapWorld(4.0f + 0.4f * i, 2.0f, 1.0f));
  }
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write(first_points);
  scan2.Write({PointInColmapWorld(positive_boundary, 0.0f, 1.0f)});

  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [capture](const std::vector<float>& support_xyz,
                const std::vector<float>& query_xyz,
                const float outer_radius,
                const float inner_radius,
                std::vector<float>* outer_normals,
                std::vector<float>* inner_normals,
                CudaNormalEstimationTiming* timing,
                std::string* error) {
        capture->calls.push_back(
            EstimatorCall{support_xyz, query_xyz, outer_radius, inner_radius});
        return EstimateDualRadiusNormalsForQueriesCuda(
            support_xyz, query_xyz, outer_radius, inner_radius, outer_normals,
            inner_normals, timing, error);
      };
  IncrementalCausalLidarMap map(dependencies);
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan1.path()), nullptr, &error), error);
  AppendScanAudit audit;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(2, scan2.path()), &audit, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  BOOST_REQUIRE_EQUAL(capture->calls.size(), 2);
  const EstimatorCall& local_call = capture->calls.back();
  const std::array<float, 3> negative_point{{negative_boundary, 0.0f, 1.0f}};
  const std::array<float, 3> positive_point{{positive_boundary, 0.0f, 1.0f}};
  BOOST_CHECK(ContainsPoint(local_call.query_xyz, negative_point));
  BOOST_CHECK(ContainsPoint(local_call.support_xyz, negative_point));
  BOOST_CHECK(ContainsPoint(local_call.support_xyz, positive_point));
  BOOST_CHECK_LT(audit.affected_query_count, snapshot->VoxelCount());
  BOOST_CHECK_EQUAL(audit.support_voxel_count, snapshot->VoxelCount());
  BOOST_CHECK(!audit.all_voxels_queried);

  LidarVoxelRecord boundary_record;
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(negative_boundary, 0.0f, 1.0f), &boundary_record));
  BOOST_CHECK_EQUAL(boundary_record.outer_normal.revision, 2);

  const std::vector<VoxelKey> all_keys = snapshot->VoxelKeys();
  std::vector<VoxelKey> affected_keys;
  for (const VoxelKey& key : all_keys) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    if (record.outer_normal.revision == 2) affected_keys.push_back(key);
  }
  const std::vector<float> all_xyz = SnapshotXyz(snapshot, all_keys);
  const std::vector<float> affected_xyz = SnapshotXyz(snapshot, affected_keys);
  std::vector<float> outer_reference;
  std::vector<float> inner_reference;
  CudaNormalEstimationTiming timing;
  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            all_xyz, affected_xyz,
                            IncrementalCausalLidarMap::kOuterNormalRadiusMeters,
                            IncrementalCausalLidarMap::kInnerNormalRadiusMeters,
                            &outer_reference, &inner_reference, &timing, &error),
                        error);
  for (size_t i = 0; i < affected_keys.size(); ++i) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(affected_keys[i], &record));
    CheckNormalMatches(record.outer_normal,
                       NormalizedExpected(outer_reference, i));
    CheckNormalMatches(record.inner_normal,
                       NormalizedExpected(inner_reference, i));
  }
}

BOOST_AUTO_TEST_CASE(
    LocalCudaUpdateMatchesFullSupportQueryReferenceAndRetainsDistantNormals) {
  std::vector<TestPoint> first_points;
  for (int y = -5; y <= 5; ++y) {
    for (int x = -5; x <= 5; ++x) {
      first_points.push_back(PointInColmapWorld(
          0.01f * x, 0.01f * y, 1.0f));
      first_points.push_back(PointInColmapWorld(
          2.0f + 0.01f * x, 0.01f * y, 1.0f));
    }
  }
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write(first_points);
  scan2.Write({PointInColmapWorld(0.12f, 0.0f, 1.0f)});
  const std::shared_ptr<EstimatorCapture> capture(new EstimatorCapture());
  IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [capture](const std::vector<float>& support_xyz,
                const std::vector<float>& query_xyz,
                const float outer_radius,
                const float inner_radius,
                std::vector<float>* outer_normals,
                std::vector<float>* inner_normals,
                CudaNormalEstimationTiming* timing,
                std::string* error) {
        capture->calls.push_back(
            EstimatorCall{support_xyz, query_xyz, outer_radius, inner_radius});
        return EstimateDualRadiusNormalsForQueriesCuda(
            support_xyz, query_xyz, outer_radius, inner_radius, outer_normals,
            inner_normals, timing, error);
      };
  IncrementalCausalLidarMap map(dependencies);
  std::string error;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(1, scan1.path()), nullptr,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> old_snapshot = map.GetSnapshot();
  const std::string old_snapshot_hash = old_snapshot->SnapshotSha256();
  AppendScanAudit audit;
  BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(2, scan2.path()), &audit,
                                       &error),
                        error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  BOOST_REQUIRE_LT(audit.affected_query_count, snapshot->VoxelCount());
  BOOST_REQUIRE_EQUAL(audit.support_voxel_count, snapshot->VoxelCount());
  BOOST_CHECK(!audit.all_voxels_queried);

  const std::vector<VoxelKey> all_keys = snapshot->VoxelKeys();
  std::vector<VoxelKey> affected_keys;
  for (const VoxelKey& key : all_keys) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    if (record.outer_normal.revision == 2) affected_keys.push_back(key);
  }
  BOOST_REQUIRE_EQUAL(affected_keys.size(), audit.affected_query_count);
  const std::vector<float> all_xyz = SnapshotXyz(snapshot, all_keys);
  const std::vector<float> affected_xyz = SnapshotXyz(snapshot, affected_keys);
  BOOST_REQUIRE_EQUAL(capture->calls.size(), 2);
  BOOST_CHECK(capture->calls.back().support_xyz == all_xyz);
  BOOST_CHECK(capture->calls.back().query_xyz == affected_xyz);
  std::vector<float> outer_reference;
  std::vector<float> inner_reference;
  CudaNormalEstimationTiming timing;
  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            all_xyz, affected_xyz,
                            IncrementalCausalLidarMap::kOuterNormalRadiusMeters,
                            IncrementalCausalLidarMap::kInnerNormalRadiusMeters,
                            &outer_reference, &inner_reference, &timing, &error),
                        error);
  for (size_t i = 0; i < affected_keys.size(); ++i) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(affected_keys[i], &record));
    CheckNormalMatches(record.outer_normal,
                       NormalizedExpected(outer_reference, i));
    CheckNormalMatches(record.inner_normal,
                       NormalizedExpected(inner_reference, i));
  }

  size_t retained_count = 0;
  for (const VoxelKey& key : all_keys) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    if (record.outer_normal.revision == 2) continue;
    ++retained_count;
    LidarVoxelRecord old_record;
    BOOST_REQUIRE(old_snapshot->FindVoxel(key, &old_record));
    BOOST_CHECK_EQUAL(record.outer_normal.revision, 1);
    BOOST_CHECK_EQUAL(record.inner_normal.revision, 1);
    CheckNormalMatches(record.outer_normal, old_record.outer_normal);
    CheckNormalMatches(record.inner_normal, old_record.inner_normal);
  }
  BOOST_CHECK_GT(retained_count, 0);
  BOOST_CHECK_EQUAL(old_snapshot->SnapshotSha256(), old_snapshot_hash);
  for (const NeighborhoodSearchAudit* search :
       {&audit.affected_search, &audit.support_search}) {
    BOOST_CHECK_EQUAL(search->query_count, 0);
    BOOST_CHECK_EQUAL(search->block_probe_count, 0);
    BOOST_CHECK_EQUAL(search->visited_block_count, 0);
    BOOST_CHECK_EQUAL(search->visited_voxel_count, 0);
    BOOST_CHECK_EQUAL(search->hit_count, 0);
    BOOST_CHECK_EQUAL(search->inserted_key_count, 0);
  }

  LidarVoxelRecord center;
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(0.0f, 0.0f, 1.0f), &center));
  BOOST_REQUIRE(center.outer_normal.valid);
  BOOST_REQUIRE(center.inner_normal.valid);
  BOOST_CHECK_GT(std::abs(center.outer_normal.normal[2]), 0.99f);
  BOOST_CHECK_GT(std::abs(center.inner_normal.normal[2]), 0.99f);
  BOOST_CHECK_LT(center.outer_normal.curvature, 1e-4f);
  BOOST_CHECK_LT(center.inner_normal.curvature, 1e-4f);
}

BOOST_AUTO_TEST_CASE(NearestPlaneHonorsValidityDistanceTieAndScale) {
  TempPcd scan;
  std::vector<TestPoint> sparse_points = {
      PointInColmapWorld(0.0f, 0.0f, 0.0f),
      PointInColmapWorld(0.02f, 0.0f, 0.0f),
      PointInColmapWorld(0.5f, 0.0f, 0.0f),
      PointInColmapWorld(0.52f, 0.0f, 0.0f),
      PointInColmapWorld(-0.125f, 20.0f, 0.0f),
      PointInColmapWorld(0.125f, 20.0f, 0.0f),
      PointInColmapWorld(0.875f, 40.0f, 0.0f),
      PointInColmapWorld(1.125f, 40.0f, 0.0f),
  };
  for (int i = -64; i <= 64; ++i) {
    sparse_points.push_back(PointInColmapWorld(
        static_cast<float>(i) * 0.5f,
        20.5f + static_cast<float>((i + 64) % 5) * 0.25f,
        static_cast<float>((i + 64) % 7) * 0.125f));
  }
  scan.Write(sparse_points);
  IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [](const std::vector<float>& support_xyz,
         const std::vector<float>& query_xyz,
         const float outer_radius,
         const float inner_radius,
         std::vector<float>* outer,
         std::vector<float>* inner,
         CudaNormalEstimationTiming* timing,
         std::string* error) {
        (void)support_xyz;
        (void)outer_radius;
        (void)inner_radius;
        (void)timing;
        (void)error;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        outer->clear();
        inner->clear();
        for (size_t i = 0; i < query_xyz.size() / 3; ++i) {
          const float x = query_xyz[3 * i];
          const float y = query_xyz[3 * i + 1];
          const float z = query_xyz[3 * i + 2];
          if (x == 0.0f && y == 0.0f && z == 0.0f) {
            outer->insert(outer->end(), {nan, nan, nan, nan});
          } else {
            outer->insert(outer->end(), {1.0f, 0.0f, 0.0f, 0.1f});
          }
          if (x == 0.5f && y == 0.0f && z == 0.0f) {
            inner->insert(inner->end(), {nan, nan, nan, nan});
          } else {
            inner->insert(inner->end(), {0.0f, 1.0f, 0.0f, 0.2f});
          }
        }
        return true;
      };
  IncrementalCausalLidarMap map(dependencies);
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan.path()), nullptr, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> snapshot = map.GetSnapshot();
  BOOST_REQUIRE_GT(snapshot->BlockCount(), 20);

  for (int i = 0; i < 48; ++i) {
    const std::array<double, 3> query = {{
        -24.0 + static_cast<double>(i) * 1.03125,
        20.625 + static_cast<double>((i * 7) % 9) * 0.0625,
        static_cast<double>((i * 11) % 7) * 0.125,
    }};
    const std::array<double, 4> radii = {{0.0, 0.125, 0.75, 2.0}};
    const double radius = radii[static_cast<size_t>(i) % radii.size()];
    CheckNearestMatchesBruteForce(
        snapshot, query, LidarNormalScale::OUTER_0_15_M, radius);
    CheckNearestMatchesBruteForce(
        snapshot, query, LidarNormalScale::INNER_0_05_M, radius);
  }

  LidarVoxelRecord valid_record;
  BOOST_REQUIRE(snapshot->FindVoxel(
      KeyForColmapPoint(0.02f, 0.0f, 0.0f), &valid_record));
  const double exact_distance =
      std::abs(static_cast<double>(valid_record.centroid[0]));

  NearestPlaneResult result = snapshot->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 0.005);
  BOOST_REQUIRE(result.ok);
  BOOST_CHECK(!result.found);
  BOOST_CHECK_LT(result.visited_block_count, snapshot->BlockCount());
  CheckNearestMatchesBruteForce(snapshot, {{0.0, 0.0, 0.0}},
                                LidarNormalScale::OUTER_0_15_M, 0.005);
  result = snapshot->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M,
      exact_distance);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE(result.found);
  BOOST_CHECK(result.plane.key == valid_record.key);
  BOOST_CHECK_EQUAL(result.squared_distance, exact_distance * exact_distance);
  BOOST_CHECK_LT(result.visited_block_count, snapshot->BlockCount());
  CheckNearestMatchesBruteForce(snapshot, {{0.0, 0.0, 0.0}},
                                LidarNormalScale::OUTER_0_15_M,
                                exact_distance);
  result = snapshot->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::INNER_0_05_M, 0.0);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE(result.found);
  BOOST_CHECK_EQUAL(result.plane.normal[0], 0.0f);
  BOOST_CHECK_EQUAL(result.plane.normal[1], 1.0f);
  BOOST_CHECK_LT(result.visited_block_count, snapshot->BlockCount());
  CheckNearestMatchesBruteForce(snapshot, {{0.0, 0.0, 0.0}},
                                LidarNormalScale::INNER_0_05_M, 0.0);

  result = snapshot->FindNearestPlane(
      {{0.5, 0.0, 0.0}}, LidarNormalScale::INNER_0_05_M, 0.03);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE(result.found);
  BOOST_CHECK(result.plane.key == KeyForColmapPoint(0.52f, 0.0f, 0.0f));
  CheckNearestMatchesBruteForce(snapshot, {{0.5, 0.0, 0.0}},
                                LidarNormalScale::INNER_0_05_M, 0.03);
  result = snapshot->FindNearestPlane(
      {{0.5, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 0.0);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE(result.found);
  BOOST_CHECK(result.plane.key == KeyForColmapPoint(0.5f, 0.0f, 0.0f));

  LidarAabb validity_bounds;
  validity_bounds.min = {{-0.01, -0.01, -0.01}};
  validity_bounds.max = {{0.03, 0.01, 0.01}};
  const PlaneCollectionResult outer_planes = snapshot->CollectPlanesInAabb(
      validity_bounds, LidarNormalScale::OUTER_0_15_M);
  const PlaneCollectionResult inner_planes = snapshot->CollectPlanesInAabb(
      validity_bounds, LidarNormalScale::INNER_0_05_M);
  BOOST_REQUIRE(outer_planes.ok);
  BOOST_REQUIRE(inner_planes.ok);
  BOOST_CHECK_EQUAL(outer_planes.planes.size(), 1);
  BOOST_CHECK_EQUAL(inner_planes.planes.size(), 2);
  BOOST_CHECK_LT(outer_planes.visited_block_count, snapshot->BlockCount());
  BOOST_CHECK_LT(inner_planes.visited_block_count, snapshot->BlockCount());

  LidarAabb all_bounds;
  all_bounds.min = {{-1e100, -1e100, -1e100}};
  all_bounds.max = {{1e100, 1e100, 1e100}};
  const PlaneCollectionResult all_planes = snapshot->CollectPlanesInAabb(
      all_bounds, LidarNormalScale::INNER_0_05_M);
  BOOST_REQUIRE(all_planes.ok);
  BOOST_CHECK_EQUAL(all_planes.visited_block_count, snapshot->BlockCount());

  result = snapshot->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarCoordinateFrame::FASTLIO_WORLD,
      LidarNormalScale::OUTER_0_15_M, 1.0);
  BOOST_CHECK(!result.ok);
  result = snapshot->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, static_cast<LidarNormalScale>(99), 1.0);
  BOOST_CHECK(!result.ok);
  result = snapshot->FindNearestPlane(
      {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 1000.0);
  BOOST_CHECK(!result.ok);
  BOOST_CHECK(result.error.find("block-key interval") != std::string::npos);

  const double tie_radius = 0.125;
  result = snapshot->FindNearestPlane(
      {{0.0, 20.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, tie_radius);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE(result.found);
  BOOST_CHECK(result.plane.key ==
              KeyForColmapPoint(-0.125f, 20.0f, 0.0f));
  BOOST_CHECK_EQUAL(result.squared_distance, tie_radius * tie_radius);
  CheckNearestMatchesBruteForce(snapshot, {{0.0, 20.0, 0.0}},
                                LidarNormalScale::OUTER_0_15_M, tie_radius);

  const double precision_query_x = 1.0 + std::ldexp(1.0, -25);
  BOOST_REQUIRE_EQUAL(static_cast<float>(precision_query_x), 1.0f);
  result = snapshot->FindNearestPlane(
      {{precision_query_x, 40.0, 0.0}},
      LidarNormalScale::OUTER_0_15_M, 0.13);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE(result.found);
  BOOST_CHECK(result.plane.key == KeyForColmapPoint(1.125f, 40.0f, 0.0f));
  CheckNearestMatchesBruteForce(
      snapshot, {{precision_query_x, 40.0, 0.0}},
      LidarNormalScale::OUTER_0_15_M, 0.13);
}

BOOST_AUTO_TEST_CASE(AabbPlanesAreKeyOrderedAndInputsAreValidated) {
  TempPcd scan;
  scan.Write({PointInColmapWorld(0.01f, 0.33f, 0.0f),
              PointInColmapWorld(0.02f, 0.00f, 0.0f),
              PointInColmapWorld(-0.40f, -0.40f, 0.0f)});
  IncrementalCausalLidarMap map(FakeDependencies());
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scan.path()), nullptr, &error), error);
  LidarAabb bounds;
  bounds.min = {{-1.0, -1.0, -1.0}};
  bounds.max = {{1.0, 1.0, 1.0}};
  PlaneCollectionResult result = map.GetSnapshot()->CollectPlanesInAabb(
      bounds, LidarNormalScale::OUTER_0_15_M);
  BOOST_REQUIRE(result.ok);
  BOOST_REQUIRE_EQUAL(result.planes.size(), 3);
  BOOST_CHECK(std::is_sorted(
      result.planes.begin(), result.planes.end(),
      [](const PlaneSample& lhs, const PlaneSample& rhs) {
        return lhs.key < rhs.key;
      }));

  bounds.frame = LidarCoordinateFrame::FASTLIO_WORLD;
  result = map.GetSnapshot()->CollectPlanesInAabb(
      bounds, LidarNormalScale::OUTER_0_15_M);
  BOOST_CHECK(!result.ok);
  bounds.frame = LidarCoordinateFrame::COLMAP_WORLD;
  bounds.min[0] = 2.0;
  bounds.max[0] = 1.0;
  result = map.GetSnapshot()->CollectPlanesInAabb(
      bounds, LidarNormalScale::OUTER_0_15_M);
  BOOST_CHECK(!result.ok);
  bounds.min[0] = -1.0;
  bounds.max[0] = 1.0;
  result = map.GetSnapshot()->CollectPlanesInAabb(
      bounds, static_cast<LidarNormalScale>(99));
  BOOST_CHECK(!result.ok);
}

BOOST_AUTO_TEST_CASE(HashGoldenAndSnapshotOutlivesMap) {
  TempPcd scan1;
  TempPcd scan2;
  scan1.Write({
      PointInColmapWorld(0.0f, 0.0f, 1.0f),
      PointInColmapWorld(0.02f, 0.01f, 1.0f),
  });
  scan2.Write({PointInColmapWorld(0.3f, 0.0f, 1.0f)});
  std::string error;
  std::shared_ptr<const LidarMapSnapshot> snapshot;
  {
    IncrementalCausalLidarMap map(FakeDependencies());
    BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(1, scan1.path()), nullptr,
                                         &error),
                          error);
    BOOST_REQUIRE_MESSAGE(map.AppendScan(MakeSource(2, scan2.path()), nullptr,
                                         &error),
                          error);
    snapshot = map.GetSnapshot();
  }

  BOOST_CHECK_EQUAL(snapshot->GeometrySha256(),
                    "9c87a2309fb584fd1c79ba79edbbea2c9a98da8ab2790562cfcd4fbe523cb8c9");
  BOOST_CHECK_EQUAL(snapshot->SnapshotSha256(),
                    "2be80c9b44adeddd13596b65cce3406b805c0dc904ea0942af24b67eae4ff4c7");
  for (const VoxelKey& key : snapshot->VoxelKeys()) {
    LidarVoxelRecord record;
    BOOST_REQUIRE(snapshot->FindVoxel(key, &record));
    BOOST_CHECK_EQUAL(record.outer_normal.revision, 2);
    BOOST_CHECK_EQUAL(record.inner_normal.revision, 2);
  }
  IncrementalCausalLidarMap second(FakeDependencies());
  BOOST_REQUIRE_MESSAGE(second.AppendScan(MakeSource(1, scan1.path()), nullptr,
                                          &error),
                        error);
  BOOST_REQUIRE_MESSAGE(second.AppendScan(MakeSource(2, scan2.path()), nullptr,
                                          &error),
                        error);
  BOOST_CHECK_EQUAL(snapshot->GeometrySha256(),
                    second.GetSnapshot()->GeometrySha256());
  BOOST_CHECK_EQUAL(snapshot->SnapshotSha256(),
                    second.GetSnapshot()->SnapshotSha256());
  const std::string hash_before_query = snapshot->SnapshotSha256();
  const NearestPlaneResult nearest = snapshot->FindNearestPlane(
      {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.1);
  BOOST_REQUIRE(nearest.ok);
  const LidarAabb bounds{{{-1.0, -1.0, 0.0}}, {{1.0, 1.0, 2.0}},
                         LidarCoordinateFrame::COLMAP_WORLD};
  BOOST_REQUIRE(snapshot->CollectPlanesInAabb(
                            bounds, LidarNormalScale::INNER_0_05_M)
                    .ok);
  BOOST_CHECK_EQUAL(snapshot->SnapshotSha256(), hash_before_query);

  LidarVoxelRecord copy;
  const VoxelKey first_key = snapshot->VoxelKeys().front();
  BOOST_REQUIRE(snapshot->FindVoxel(first_key, &copy));
  const std::array<float, 3> owned_centroid = copy.centroid;
  copy.centroid = {{999.0f, 999.0f, 999.0f}};
  copy.outer_normal.normal = {{0.0f, 0.0f, 0.0f}};
  LidarVoxelRecord reread;
  BOOST_REQUIRE(snapshot->FindVoxel(first_key, &reread));
  BOOST_CHECK(reread.centroid == owned_centroid);
  BOOST_CHECK_EQUAL(snapshot->SnapshotSha256(), hash_before_query);
}

BOOST_AUTO_TEST_CASE(SnapshotReadersRemainStableDuringSingleWriterAppends) {
  std::vector<std::unique_ptr<TempPcd>> scans;
  for (int i = 0; i < 17; ++i) {
    scans.emplace_back(new TempPcd());
  }
  scans[0]->Write({PointInColmapWorld(0.0f, 0.0f, 1.0f),
                   PointInColmapWorld(0.02f, 0.01f, 1.0f),
                   PointInColmapWorld(-0.02f, 0.01f, 1.01f)});
  for (size_t i = 1; i < scans.size(); ++i) {
    scans[i]->Write({PointInColmapWorld(
        2.0f + static_cast<float>(i) * 0.4f, 1.0f, 1.0f)});
  }

  IncrementalCausalLidarMap map(FakeDependencies());
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(MakeSource(1, scans[0]->path()), nullptr, &error), error);
  const std::shared_ptr<const LidarMapSnapshot> old_snapshot = map.GetSnapshot();
  const std::string old_geometry_hash = old_snapshot->GeometrySha256();
  const std::string old_snapshot_hash = old_snapshot->SnapshotSha256();
  const size_t old_voxel_count = old_snapshot->VoxelCount();
  const VoxelKey old_key = KeyForColmapPoint(0.0f, 0.0f, 1.0f);
  LidarVoxelRecord old_record;
  BOOST_REQUIRE(old_snapshot->FindVoxel(old_key, &old_record));
  const NearestPlaneResult old_nearest = old_snapshot->FindNearestPlane(
      {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.1);
  BOOST_REQUIRE(old_nearest.ok);
  BOOST_REQUIRE(old_nearest.found);

  constexpr size_t kReaderCount = 4;
  std::atomic<int> ready_readers{0};
  std::atomic<int> active_readers{0};
  std::atomic<bool> start{false};
  std::atomic<bool> reader_failed{false};
  std::atomic<size_t> writer_progress{1};
  std::array<std::atomic<size_t>, kReaderCount> reader_progress;
  for (std::atomic<size_t>& progress : reader_progress) progress.store(0);
  std::vector<std::thread> readers;
  for (size_t reader_index = 0; reader_index < kReaderCount; ++reader_index) {
    readers.emplace_back([&, reader_index]() {
      const std::shared_ptr<const LidarMapSnapshot> held = old_snapshot;
      ready_readers.fetch_add(1);
      while (!start.load()) std::this_thread::yield();
      active_readers.fetch_add(1);
      while (true) {
        LidarVoxelRecord record;
        const NearestPlaneResult nearest = held->FindNearestPlane(
            {{0.0, 0.0, 1.0}}, LidarNormalScale::OUTER_0_15_M, 0.1);
        if (held->Version() != 1 || held->VoxelCount() != old_voxel_count ||
            held->GeometrySha256() != old_geometry_hash ||
            held->SnapshotSha256() != old_snapshot_hash ||
            !held->FindVoxel(old_key, &record) ||
            record.centroid != old_record.centroid || !nearest.ok ||
            !nearest.found || nearest.plane.key != old_nearest.plane.key ||
            nearest.plane.normal_revision !=
                old_nearest.plane.normal_revision) {
          reader_failed.store(true);
        }
        const size_t progress = writer_progress.load();
        reader_progress[reader_index].store(progress);
        if (progress == scans.size()) break;
        std::this_thread::yield();
      }
    });
  }

  bool writer_succeeded = true;
  std::string writer_error;
  std::thread writer([&]() {
    while (!start.load()) std::this_thread::yield();
    while (active_readers.load() != static_cast<int>(kReaderCount)) {
      std::this_thread::yield();
    }
    for (size_t i = 1; i < scans.size(); ++i) {
      if (!map.AppendScan(MakeSource(i + 1, scans[i]->path()), nullptr,
                          &writer_error)) {
        writer_succeeded = false;
        break;
      }
      const size_t published_version = i + 1;
      writer_progress.store(published_version);
      for (size_t reader_index = 0; reader_index < kReaderCount;
           ++reader_index) {
        while (reader_progress[reader_index].load() < published_version) {
          std::this_thread::yield();
        }
      }
    }
    if (!writer_succeeded) writer_progress.store(scans.size());
  });
  while (ready_readers.load() != static_cast<int>(kReaderCount)) {
    std::this_thread::yield();
  }
  start.store(true);
  for (std::thread& reader : readers) reader.join();
  writer.join();

  BOOST_REQUIRE_MESSAGE(writer_succeeded, writer_error);
  BOOST_CHECK(!reader_failed.load());
  BOOST_CHECK_EQUAL(old_snapshot->Version(), 1);
  BOOST_CHECK_EQUAL(old_snapshot->GeometrySha256(), old_geometry_hash);
  BOOST_CHECK_EQUAL(old_snapshot->SnapshotSha256(), old_snapshot_hash);
  BOOST_CHECK_EQUAL(map.GetSnapshot()->Version(), scans.size());
  BOOST_CHECK_GT(map.GetSnapshot()->VoxelCount(), old_voxel_count);
}

BOOST_AUTO_TEST_CASE(SignedZeroAndInvalidNormalPayloadsHashCanonically) {
  TempPcd positive_zero_scan;
  TempPcd negative_zero_scan;
  positive_zero_scan.Write({PointInColmapWorld(0.0f, 0.0f, 0.0f)});
  negative_zero_scan.Write({PointInColmapWorld(-0.0f, 0.0f, 0.0f)});
  const auto invalid_estimator =
      [](const std::vector<float>& support_xyz,
         const std::vector<float>& query_xyz,
         const float outer_radius,
         const float inner_radius,
         std::vector<float>* outer,
         std::vector<float>* inner,
         CudaNormalEstimationTiming* timing,
         std::string* error) {
        (void)support_xyz;
        (void)outer_radius;
        (void)inner_radius;
        (void)timing;
        (void)error;
        const float nan1 = std::nanf("1");
        const float nan2 = std::nanf("2");
        outer->assign(query_xyz.size() / 3 * 4, nan1);
        inner->assign(query_xyz.size() / 3 * 4, nan2);
        return true;
      };
  IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals = invalid_estimator;
  IncrementalCausalLidarMap positive_map(dependencies);
  IncrementalCausalLidarMap negative_map(dependencies);
  std::string error;
  BOOST_REQUIRE_MESSAGE(positive_map.AppendScan(
                            MakeSource(1, positive_zero_scan.path()), nullptr,
                            &error),
                        error);
  BOOST_REQUIRE_MESSAGE(negative_map.AppendScan(
                            MakeSource(1, negative_zero_scan.path()), nullptr,
                            &error),
                        error);
  BOOST_CHECK_EQUAL(positive_map.GetSnapshot()->GeometrySha256(),
                    negative_map.GetSnapshot()->GeometrySha256());
  BOOST_CHECK_EQUAL(positive_map.GetSnapshot()->SnapshotSha256(),
                    negative_map.GetSnapshot()->SnapshotSha256());
  const NearestPlaneResult nearest =
      positive_map.GetSnapshot()->FindNearestPlane(
          {{0.0, 0.0, 0.0}}, LidarNormalScale::OUTER_0_15_M, 1.0);
  BOOST_REQUIRE(nearest.ok);
  BOOST_CHECK(!nearest.found);
}

}  // namespace
}  // namespace lidar
}  // namespace colmap
