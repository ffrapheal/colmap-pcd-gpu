#define TEST_NAME "lidar/cuda_normal_estimation"
#include "util/testing.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "lidar/cuda_normal_estimation.h"

namespace colmap {
namespace lidar {
namespace {

constexpr float kOuterRadius = 0.15f;
constexpr float kInnerRadius = 0.05f;
constexpr int kCoordBias = 1 << 20;

int CellForCoordinate(float coordinate, float inverse_cell_size) {
  return static_cast<int>(std::floor(
      static_cast<double>(coordinate * inverse_cell_size)));
}

float SmallestCoordinateForCell(int target_cell, float inverse_cell_size) {
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  const float positive_infinity = std::numeric_limits<float>::infinity();
  float coordinate = static_cast<float>(target_cell) / inverse_cell_size;

  while (CellForCoordinate(coordinate, inverse_cell_size) < target_cell) {
    coordinate = std::nextafter(coordinate, positive_infinity);
  }
  while (CellForCoordinate(coordinate, inverse_cell_size) > target_cell) {
    coordinate = std::nextafter(coordinate, negative_infinity);
  }
  while (CellForCoordinate(std::nextafter(coordinate, negative_infinity),
                           inverse_cell_size) == target_cell) {
    coordinate = std::nextafter(coordinate, negative_infinity);
  }
  return coordinate;
}

std::vector<float> MakePlaneSupport() {
  std::vector<float> xyz;
  for (int y = -10; y <= 10; ++y) {
    for (int x = -10; x <= 10; ++x) {
      xyz.push_back(0.01f * x);
      xyz.push_back(0.01f * y);
      xyz.push_back(1.0f);
    }
  }
  return xyz;
}

std::vector<float> MakeOrthogonalPlaneSupport() {
  std::vector<float> xyz;
  for (int y = -10; y <= 10; ++y) {
    for (int x = -10; x <= 10; ++x) {
      xyz.push_back(0.01f * x);
      xyz.push_back(0.01f * y);
      xyz.push_back(1.0f);
    }
  }
  for (int z = -10; z <= 10; ++z) {
    for (int y = -10; y <= 10; ++y) {
      xyz.push_back(2.0f);
      xyz.push_back(0.01f * y);
      xyz.push_back(1.0f + 0.01f * z);
    }
  }
  return xyz;
}

void CheckPlaneNormals(const std::vector<float>& normals,
                       size_t query_count) {
  BOOST_REQUIRE_EQUAL(normals.size(), query_count * 4);
  for (size_t i = 0; i < query_count; ++i) {
    const float nx = normals[4 * i];
    const float ny = normals[4 * i + 1];
    const float nz = normals[4 * i + 2];
    const float curvature = normals[4 * i + 3];
    BOOST_REQUIRE(std::isfinite(nx));
    BOOST_REQUIRE(std::isfinite(ny));
    BOOST_REQUIRE(std::isfinite(nz));
    BOOST_REQUIRE(std::isfinite(curvature));
    BOOST_CHECK_SMALL(nx, 1e-4f);
    BOOST_CHECK_SMALL(ny, 1e-4f);
    BOOST_CHECK_CLOSE(std::abs(nz), 1.0f, 1e-3f);
    BOOST_CHECK_SMALL(curvature, 1e-4f);
  }
}

void CheckNegativeAxisNormal(const std::vector<float>& normals,
                             size_t query_index,
                             size_t expected_axis) {
  BOOST_REQUIRE_LT(query_index * 4 + 3, normals.size());
  for (size_t axis = 0; axis < 3; ++axis) {
    const float component = normals[query_index * 4 + axis];
    BOOST_REQUIRE(std::isfinite(component));
    if (axis == expected_axis) {
      BOOST_CHECK_CLOSE(-component, 1.0f, 1e-3f);
    } else {
      BOOST_CHECK_SMALL(component, 1e-4f);
    }
  }
  const float curvature = normals[query_index * 4 + 3];
  BOOST_REQUIRE(std::isfinite(curvature));
  BOOST_CHECK_SMALL(curvature, 1e-4f);
}

void SetStaleResults(std::vector<float>* outer_normals,
                     std::vector<float>* inner_normals,
                     CudaNormalEstimationTiming* timing) {
  *outer_normals = {1.0f, 2.0f, 3.0f, 4.0f};
  *inner_normals = {5.0f, 6.0f, 7.0f, 8.0f};
  timing->upload_ms = 1.0;
  timing->index_ms = 2.0;
  timing->normals_ms = 3.0;
  timing->download_ms = 4.0;
}

void CheckResultsCleared(const std::vector<float>& outer_normals,
                         const std::vector<float>& inner_normals,
                         const CudaNormalEstimationTiming& timing,
                         const std::string& case_name) {
  BOOST_CHECK_MESSAGE(outer_normals.empty(),
                      case_name + ": outer output was not cleared");
  BOOST_CHECK_MESSAGE(inner_normals.empty(),
                      case_name + ": inner output was not cleared");
  BOOST_CHECK_MESSAGE(timing.upload_ms == 0.0,
                      case_name + ": upload timing was not cleared");
  BOOST_CHECK_MESSAGE(timing.index_ms == 0.0,
                      case_name + ": index timing was not cleared");
  BOOST_CHECK_MESSAGE(timing.normals_ms == 0.0,
                      case_name + ": normals timing was not cleared");
  BOOST_CHECK_MESSAGE(timing.download_ms == 0.0,
                      case_name + ": download timing was not cleared");
}

void CheckInvalidInput(const std::string& case_name,
                       const std::vector<float>& support_xyz,
                       const std::vector<float>& query_xyz,
                       float outer_radius,
                       float inner_radius,
                       const std::string& expected_error) {
  std::vector<float> outer_normals;
  std::vector<float> inner_normals;
  CudaNormalEstimationTiming timing;
  std::string error = "stale error";
  SetStaleResults(&outer_normals, &inner_normals, &timing);

  const bool success = EstimateDualRadiusNormalsForQueriesCuda(
      support_xyz, query_xyz, outer_radius, inner_radius, &outer_normals,
      &inner_normals, &timing, &error);
  BOOST_CHECK_MESSAGE(!success, case_name + ": unexpectedly succeeded");
  BOOST_CHECK_MESSAGE(!error.empty(), case_name + ": error was empty");
  BOOST_CHECK_MESSAGE(
      error.find(expected_error) != std::string::npos,
      case_name + ": unexpected error: " + error);
  CheckResultsCleared(outer_normals, inner_normals, timing, case_name);
}

}  // namespace

BOOST_AUTO_TEST_CASE(PlaneSupportEstimatesQueryNormals) {
  const std::vector<float> support_xyz = MakePlaneSupport();
  const std::vector<float> query_xyz = {
      -0.035f, -0.025f, 1.0f,
       0.005f,  0.015f, 1.0f,
       0.047f, -0.033f, 1.0f,
  };
  std::vector<float> outer_normals;
  std::vector<float> inner_normals;
  CudaNormalEstimationTiming timing;
  std::string error;

  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            support_xyz, query_xyz, kOuterRadius,
                            kInnerRadius, &outer_normals, &inner_normals,
                            &timing, &error),
                        error);
  CheckPlaneNormals(outer_normals, query_xyz.size() / 3);
  CheckPlaneNormals(inner_normals, query_xyz.size() / 3);
}

BOOST_AUTO_TEST_CASE(QueryWithoutSupportProducesNaNs) {
  const std::vector<float> support_xyz = MakePlaneSupport();
  const std::vector<float> query_xyz = {3.0f, -4.0f, 2.0f};
  std::vector<float> outer_normals;
  std::vector<float> inner_normals;
  std::string error;

  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            support_xyz, query_xyz, kOuterRadius,
                            kInnerRadius, &outer_normals, &inner_normals,
                            nullptr, &error),
                        error);
  BOOST_REQUIRE_EQUAL(outer_normals.size(), 4);
  BOOST_REQUIRE_EQUAL(inner_normals.size(), 4);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK(std::isnan(outer_normals[i]));
    BOOST_CHECK(std::isnan(inner_normals[i]));
  }
}

BOOST_AUTO_TEST_CASE(CompatibilityWrapperMatchesQueryApi) {
  const std::vector<float> xyz = MakePlaneSupport();
  std::vector<float> query_outer;
  std::vector<float> query_inner;
  std::vector<float> wrapper_outer;
  std::vector<float> wrapper_inner;
  std::string error;

  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            xyz, xyz, kOuterRadius, kInnerRadius, &query_outer,
                            &query_inner, nullptr, &error),
                        error);
  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsCuda(
                            xyz, kOuterRadius, kInnerRadius, &wrapper_outer,
                            &wrapper_inner, nullptr, &error),
                        error);
  BOOST_REQUIRE_EQUAL(query_outer.size(), wrapper_outer.size());
  BOOST_REQUIRE_EQUAL(query_inner.size(), wrapper_inner.size());
  for (size_t i = 0; i < query_outer.size(); ++i) {
    BOOST_CHECK((std::isnan(query_outer[i]) &&
                 std::isnan(wrapper_outer[i])) ||
                query_outer[i] == wrapper_outer[i]);
    BOOST_CHECK((std::isnan(query_inner[i]) &&
                 std::isnan(wrapper_inner[i])) ||
                query_inner[i] == wrapper_inner[i]);
  }
}

BOOST_AUTO_TEST_CASE(QueryOutputPreservesReverseInputOrder) {
  const std::vector<float> support_xyz = MakeOrthogonalPlaneSupport();
  const std::vector<float> query_xyz = {
      2.0f, 0.0f, 1.0f,
      0.0f, 0.0f, 1.0f,
  };
  std::vector<float> outer_normals;
  std::vector<float> inner_normals;
  std::string error;

  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            support_xyz, query_xyz, kOuterRadius,
                            kInnerRadius, &outer_normals, &inner_normals,
                            nullptr, &error),
                        error);
  BOOST_REQUIRE_EQUAL(outer_normals.size(), query_xyz.size() / 3 * 4);
  BOOST_REQUIRE_EQUAL(inner_normals.size(), query_xyz.size() / 3 * 4);
  CheckNegativeAxisNormal(outer_normals, 0, 0);
  CheckNegativeAxisNormal(outer_normals, 1, 2);
  CheckNegativeAxisNormal(inner_normals, 0, 0);
  CheckNegativeAxisNormal(inner_normals, 1, 2);
}

BOOST_AUTO_TEST_CASE(PackedCellKeyAdjacentBoundaryCellsAreAccepted) {
  const float inverse_cell_size = 1.0f / kOuterRadius;
  const float positive_coordinate =
      SmallestCoordinateForCell(kCoordBias - 1, inverse_cell_size);
  const float negative_coordinate =
      SmallestCoordinateForCell(-kCoordBias + 1, inverse_cell_size);
  BOOST_REQUIRE_EQUAL(
      CellForCoordinate(positive_coordinate, inverse_cell_size),
      kCoordBias - 1);
  BOOST_REQUIRE_EQUAL(
      CellForCoordinate(negative_coordinate, inverse_cell_size),
      -kCoordBias + 1);

  const std::vector<float> xyz = {
      positive_coordinate, 0.0f, 0.0f,
      negative_coordinate, 0.0f, 0.0f,
  };
  std::vector<float> outer_normals;
  std::vector<float> inner_normals;
  std::string error;

  BOOST_REQUIRE_MESSAGE(EstimateDualRadiusNormalsForQueriesCuda(
                            xyz, xyz, kOuterRadius, kInnerRadius,
                            &outer_normals, &inner_normals, nullptr, &error),
                        error);
  BOOST_REQUIRE_EQUAL(outer_normals.size(), 8);
  BOOST_REQUIRE_EQUAL(inner_normals.size(), 8);
}

BOOST_AUTO_TEST_CASE(InvalidInputClearsResultsAndTiming) {
  const std::vector<float> valid_xyz = {0.0f, 0.0f, 1.0f};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const float inverse_cell_size = 1.0f / kOuterRadius;
  const float positive_boundary_coordinate =
      SmallestCoordinateForCell(kCoordBias, inverse_cell_size);
  const float negative_boundary_coordinate =
      SmallestCoordinateForCell(-kCoordBias, inverse_cell_size);

  CheckInvalidInput("empty support", {}, valid_xyz, kOuterRadius,
                    kInnerRadius, "support_xyz must contain three floats");
  CheckInvalidInput("empty query", valid_xyz, {}, kOuterRadius, kInnerRadius,
                    "query_xyz must contain three floats");
  CheckInvalidInput("malformed support", {0.0f, 0.0f}, valid_xyz,
                    kOuterRadius, kInnerRadius,
                    "support_xyz must contain three floats");
  CheckInvalidInput("malformed query", valid_xyz, {0.0f, 0.0f},
                    kOuterRadius, kInnerRadius,
                    "query_xyz must contain three floats");
  CheckInvalidInput("support NaN", {0.0f, 0.0f, nan}, valid_xyz,
                    kOuterRadius, kInnerRadius,
                    "support_xyz contains non-finite XYZ");
  CheckInvalidInput("query NaN", valid_xyz, {0.0f, nan, 1.0f},
                    kOuterRadius, kInnerRadius,
                    "query_xyz contains non-finite XYZ");
  CheckInvalidInput("query infinity", valid_xyz,
                    {0.0f, 0.0f, infinity}, kOuterRadius, kInnerRadius,
                    "query_xyz contains non-finite XYZ");
  BOOST_REQUIRE_EQUAL(
      CellForCoordinate(positive_boundary_coordinate, inverse_cell_size),
      kCoordBias);
  BOOST_REQUIRE_EQUAL(
      CellForCoordinate(negative_boundary_coordinate, inverse_cell_size),
      -kCoordBias);
  CheckInvalidInput("positive packed-cell boundary", valid_xyz,
                    {positive_boundary_coordinate, 0.0f, 1.0f},
                    kOuterRadius, kInnerRadius,
                    "query_xyz coordinate exceeds packed cell-key range");
  CheckInvalidInput("negative packed-cell boundary", valid_xyz,
                    {negative_boundary_coordinate, 0.0f, 1.0f},
                    kOuterRadius, kInnerRadius,
                    "query_xyz coordinate exceeds packed cell-key range");

  const std::string invalid_radii_error =
      "radii must be finite and satisfy 0 < inner_radius <= outer_radius";
  CheckInvalidInput("zero outer radius", valid_xyz, valid_xyz, 0.0f,
                    kInnerRadius, invalid_radii_error);
  CheckInvalidInput("negative outer radius", valid_xyz, valid_xyz, -1.0f,
                    kInnerRadius, invalid_radii_error);
  CheckInvalidInput("NaN outer radius", valid_xyz, valid_xyz, nan,
                    kInnerRadius, invalid_radii_error);
  CheckInvalidInput("infinite outer radius", valid_xyz, valid_xyz, infinity,
                    kInnerRadius, invalid_radii_error);
  CheckInvalidInput("zero inner radius", valid_xyz, valid_xyz, kOuterRadius,
                    0.0f, invalid_radii_error);
  CheckInvalidInput("negative inner radius", valid_xyz, valid_xyz,
                    kOuterRadius, -kInnerRadius, invalid_radii_error);
  CheckInvalidInput("NaN inner radius", valid_xyz, valid_xyz, kOuterRadius,
                    nan, invalid_radii_error);
  CheckInvalidInput("infinite inner radius", valid_xyz, valid_xyz,
                    kOuterRadius, infinity, invalid_radii_error);
  CheckInvalidInput("inner radius exceeds outer", valid_xyz, valid_xyz,
                    kInnerRadius, kOuterRadius, invalid_radii_error);

  const std::string radius_range_error =
      "radii exceed supported CUDA float arithmetic range";
  CheckInvalidInput("outer radius square overflow", valid_xyz, valid_xyz,
                    std::numeric_limits<float>::max(), kInnerRadius,
                    radius_range_error);
  CheckInvalidInput("inner radius square underflow", valid_xyz, valid_xyz,
                    kOuterRadius, std::numeric_limits<float>::min(),
                    radius_range_error);
}

}  // namespace lidar
}  // namespace colmap
