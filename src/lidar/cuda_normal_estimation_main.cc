#include "lidar/cuda_normal_estimation.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::string outer_output;
  std::string inner_output;
  float outer_radius = 0.15f;
  float inner_radius = 0.05f;
};

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program
            << " --input cloud.pcd --outer_output outer.pcd"
               " --inner_output inner.pcd [--outer_radius 0.15]"
               " [--inner_radius 0.05]\n";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (i + 1 >= argc) return false;
    const std::string value = argv[++i];
    if (argument == "--input") {
      options->input = value;
    } else if (argument == "--outer_output") {
      options->outer_output = value;
    } else if (argument == "--inner_output") {
      options->inner_output = value;
    } else if (argument == "--outer_radius") {
      options->outer_radius = std::strtof(value.c_str(), nullptr);
    } else if (argument == "--inner_radius") {
      options->inner_radius = std::strtof(value.c_str(), nullptr);
    } else {
      return false;
    }
  }
  return !options->input.empty() && !options->outer_output.empty() &&
         !options->inner_output.empty();
}

double MillisecondsSince(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  using Point = pcl::PointXYZINormal;
  pcl::PointCloud<Point> cloud;
  auto stage_start = std::chrono::steady_clock::now();
  if (pcl::io::loadPCDFile<Point>(options.input, cloud) < 0) {
    std::cerr << "Failed to read " << options.input << "\n";
    return EXIT_FAILURE;
  }
  const double read_ms = MillisecondsSince(stage_start);
  std::vector<float> xyz;
  xyz.reserve(cloud.size() * 3);
  for (const Point& point : cloud.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      std::cerr << "Input contains non-finite XYZ; filter it before CUDA normal"
                   " estimation\n";
      return EXIT_FAILURE;
    }
    xyz.push_back(point.x);
    xyz.push_back(point.y);
    xyz.push_back(point.z);
  }

  std::vector<float> outer_normals;
  std::vector<float> inner_normals;
  colmap::lidar::CudaNormalEstimationTiming timing;
  std::string error;
  if (!colmap::lidar::EstimateDualRadiusNormalsCuda(
          xyz, options.outer_radius, options.inner_radius, &outer_normals,
          &inner_normals, &timing, &error)) {
    std::cerr << "CUDA normal estimation failed: " << error << "\n";
    return EXIT_FAILURE;
  }

  pcl::PointCloud<Point> outer_cloud = cloud;
  pcl::PointCloud<Point> inner_cloud = cloud;
  size_t outer_valid = 0;
  size_t inner_valid = 0;
  for (size_t i = 0; i < cloud.size(); ++i) {
    Point& outer = outer_cloud.points[i];
    Point& inner = inner_cloud.points[i];
    outer.normal_x = outer_normals[4 * i];
    outer.normal_y = outer_normals[4 * i + 1];
    outer.normal_z = outer_normals[4 * i + 2];
    outer.curvature = outer_normals[4 * i + 3];
    inner.normal_x = inner_normals[4 * i];
    inner.normal_y = inner_normals[4 * i + 1];
    inner.normal_z = inner_normals[4 * i + 2];
    inner.curvature = inner_normals[4 * i + 3];
    outer_valid += std::isfinite(outer.normal_x);
    inner_valid += std::isfinite(inner.normal_x);
  }
  outer_cloud.is_dense = outer_valid == cloud.size();
  inner_cloud.is_dense = inner_valid == cloud.size();

  stage_start = std::chrono::steady_clock::now();
  if (pcl::io::savePCDFileBinary(options.outer_output, outer_cloud) < 0 ||
      pcl::io::savePCDFileBinary(options.inner_output, inner_cloud) < 0) {
    std::cerr << "Failed to write CUDA normal outputs\n";
    return EXIT_FAILURE;
  }
  const double write_ms = MillisecondsSince(stage_start);

  std::cout << std::fixed << std::setprecision(3)
            << "points=" << cloud.size() << " outer_radius="
            << options.outer_radius << " inner_radius=" << options.inner_radius
            << " outer_valid=" << outer_valid << " inner_valid=" << inner_valid
            << " read_ms=" << read_ms << " upload_ms=" << timing.upload_ms
            << " index_ms=" << timing.index_ms
            << " normals_ms=" << timing.normals_ms
            << " download_ms=" << timing.download_ms
            << " write_ms=" << write_ms << "\n";
  return EXIT_SUCCESS;
}
