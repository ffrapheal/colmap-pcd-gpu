// Copyright (c) 2026, NVIDIA Corporation.
// All rights reserved.

#ifndef COLMAP_SRC_SFM_NONBA_PROFILER_H_
#define COLMAP_SRC_SFM_NONBA_PROFILER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace colmap {

// Stable identifiers for the Mapper stages captured by the non-BA profiler.
// Formal partition accounting uses exclusive wall time only. Inclusive wall
// time is retained as a diagnostic for each call boundary.
enum class NonBaStageId : uint8_t {
  kMapperRoot = 0,
  kDatabaseCorrespondenceLoad,
  kLidarLoad,
  kLidarPlyLoadTransformAndIndex,
  kInitialPair,
  kInitialLidarProjection,
  kInitialAbsolutePose,
  kInitialPoseRefine,
  kInitialCommit,
  kFindNextImages,
  kRegisterNextImage,
  kRegisterCollect2D3D,
  kRegisterAbsolutePose,
  kRegisterPoseRefine,
  kRegisterCommit,
  kTriangulateImage,
  kLocalRefinement,
  kLocalFindBundle,
  kLocalVariablePointCollection,
  kLocalLidarProjectionPreparation,
  kLocalLidarProjectionAndMatching,
  kLocalKdQueries,
  kLocalBaConfig,
  kLocalBaSolve,
  kLocalMergeTracks,
  kLocalCompleteTracks,
  kLocalCompleteImage,
  kLocalFilterInImages,
  kLocalFilterModified,
  kLocalLidarOutlier,
  kGlobalRefinement,
  kGlobalPreComplete,
  kGlobalPreMerge,
  kGlobalRetriangulate,
  kGlobalAdjustment,
  kGlobalPoints3DAndConfigCopy,
  kGlobalNegativeDepthScan,
  kGlobalImageConfigSelection,
  kGlobalVariablePointCollection,
  kGlobalLidarKdPreparation,
  kGlobalKdQueries,
  kGlobalBaConfig,
  kGlobalBaSolve,
  kGlobalPostComplete,
  kGlobalPostMerge,
  kGlobalFilterAll,
  kGlobalFilterImages,
  kExtractColors,
  kWriteModel,
  kCudaRuntimeShutdown,
  kCount,
};

enum class NonBaPartition : uint8_t {
  kRootDiagnostic = 0,
  kNonBa,
  kBaExcluded,
};

struct NonBaStageInfo {
  NonBaStageInfo(const char* name,
                 const NonBaPartition partition,
                 const char* item_unit,
                 const bool yield_defined = false)
      : name(name),
        partition(partition),
        input_item_unit(item_unit),
        output_item_unit(item_unit),
        yield_defined(yield_defined) {}

  NonBaStageInfo(const char* name,
                 const NonBaPartition partition,
                 const char* input_item_unit,
                 const char* output_item_unit,
                 const bool yield_defined = false)
      : name(name),
        partition(partition),
        input_item_unit(input_item_unit),
        output_item_unit(output_item_unit),
        yield_defined(yield_defined) {}

  const char* name;
  NonBaPartition partition;
  const char* input_item_unit;
  const char* output_item_unit;
  // True only when a zero output is a meaningful algorithmic zero-yield
  // result. Setup, selection/configuration, copying, output, and shutdown
  // stages still report output items but are excluded from zero-yield totals.
  bool yield_defined;
};

inline const std::array<NonBaStageInfo,
                        static_cast<size_t>(NonBaStageId::kCount)>&
NonBaStageInfos() {
  static const std::array<NonBaStageInfo,
                          static_cast<size_t>(NonBaStageId::kCount)>
      kInfos = {{
          {"mapper_root", NonBaPartition::kNonBa, "models"},
          {"database_correspondence_load", NonBaPartition::kNonBa, "images"},
          {"lidar_load", NonBaPartition::kNonBa, "maps"},
          {"lidar_ply_load_transform_submap_kdtree_index",
           NonBaPartition::kNonBa, "maps"},
          {"initial_pair", NonBaPartition::kNonBa, "images", true},
          {"initial_lidar_projection", NonBaPartition::kNonBa,
           "feature_matches", "projected_matches", true},
          {"initial_absolute_pose_ransac", NonBaPartition::kNonBa,
           "correspondences", "inliers", true},
          {"initial_pose_refine", NonBaPartition::kNonBa, "inliers", true},
          {"initial_commit", NonBaPartition::kNonBa, "inliers",
           "observations", true},
          {"find_next_images", NonBaPartition::kNonBa, "candidate_images",
           true},
          {"register_next_image", NonBaPartition::kNonBa, "observations",
           "registered_images", true},
          {"register_collect_2d3d", NonBaPartition::kNonBa,
           "points2D", "correspondences", true},
          {"register_absolute_pose_ransac", NonBaPartition::kNonBa,
           "correspondences", "inliers", true},
          {"register_pose_refine", NonBaPartition::kNonBa, "inliers", true},
          {"register_commit", NonBaPartition::kNonBa, "inliers",
           "observations", true},
          {"triangulate_image", NonBaPartition::kNonBa, "observations",
           true},
          {"local_refinement", NonBaPartition::kNonBa,
           "modified_points3D", "adjusted_observations", true},
          {"local_find_bundle", NonBaPartition::kNonBa, "points3D",
           "images", true},
          {"local_variable_point_collection", NonBaPartition::kNonBa,
           "points3D"},
          {"local_lidar_projection_preparation", NonBaPartition::kNonBa,
           "points3D", "candidate_points"},
          {"local_lidar_projection_matching", NonBaPartition::kNonBa,
           "candidate_points", "accepted_lidar_constraints", true},
          {"local_kd_queries", NonBaPartition::kNonBa, "candidate_points",
           "accepted_lidar_constraints", true},
          {"local_ba_config", NonBaPartition::kNonBa, "points3D"},
          {"local_ba_solve", NonBaPartition::kBaExcluded, "points3D",
           "adjusted_observations", true},
          {"local_merge_tracks", NonBaPartition::kNonBa, "points3D",
           "merged_observations", true},
          {"local_complete_tracks", NonBaPartition::kNonBa, "points3D",
           "completed_observations", true},
          {"local_complete_image", NonBaPartition::kNonBa, "observations",
           "completed_observations", true},
          {"local_filter_in_images", NonBaPartition::kNonBa, "images",
           "filtered_observations", true},
          {"local_filter_modified", NonBaPartition::kNonBa, "points3D",
           "filtered_observations", true},
          {"local_lidar_outlier", NonBaPartition::kNonBa, "points3D",
           "filtered_observations", true},
          {"global_refinement", NonBaPartition::kNonBa,
           "observations", "changed_observations", true},
          {"global_pre_complete", NonBaPartition::kNonBa, "points3D",
           "completed_observations", true},
          {"global_pre_merge", NonBaPartition::kNonBa, "points3D",
           "merged_observations", true},
          {"global_retriangulate", NonBaPartition::kNonBa, "observations",
           true},
          {"global_adjustment", NonBaPartition::kNonBa, "registered_images",
           "successful_solves", true},
          {"global_points3d_copy_access", NonBaPartition::kNonBa, "points3D"},
          {"global_pre_negative_depth_scan", NonBaPartition::kNonBa,
           "observations", "filtered_observations", true},
          {"global_image_config_selection", NonBaPartition::kNonBa, "images",
           "configured_images"},
          {"global_variable_point_collection", NonBaPartition::kNonBa,
           "points3D"},
          {"global_lidar_kd_preparation", NonBaPartition::kNonBa,
           "points3D"},
          {"global_kd_queries", NonBaPartition::kNonBa, "points3D",
           "accepted_lidar_constraints", true},
          {"global_ba_config", NonBaPartition::kNonBa, "points3D",
           "config_items"},
          {"global_ba_solve", NonBaPartition::kBaExcluded,
           "adjusted_observations", true},
          {"global_post_complete", NonBaPartition::kNonBa, "points3D",
           "completed_observations", true},
          {"global_post_merge", NonBaPartition::kNonBa, "points3D",
           "merged_observations", true},
          {"global_post_filter_all", NonBaPartition::kNonBa, "observations",
           "filtered_observations", true},
          {"global_post_filter_images", NonBaPartition::kNonBa, "images",
           "filtered_images", true},
          {"extract_colors", NonBaPartition::kNonBa, "images"},
          {"write_model", NonBaPartition::kNonBa, "points3D"},
          {"cuda_runtime_shutdown", NonBaPartition::kNonBa, "runtimes"},
      }};
  return kInfos;
}

struct NonBaStageToken {
  size_t depth = 0;
};

// The sink interface lets instrumented code hold a nullable pointer. The
// controller only constructs an enabled profiler on the on path. A null pointer
// performs only the minimum branch and does not construct a profiler, allocate
// record storage, lock, read a clock, or write a file.
class NonBaStageSink {
 public:
  virtual ~NonBaStageSink() = default;
  virtual NonBaStageToken Begin(NonBaStageId stage_id,
                                uint64_t input_items) = 0;
  virtual void End(const NonBaStageToken& token, uint64_t output_items) = 0;
};

class NonBaStageScope {
 public:
  NonBaStageScope(NonBaStageSink* sink,
                  const NonBaStageId stage_id,
                  const uint64_t input_items = 0)
      : sink_(sink) {
    if (sink_ != nullptr) {
      token_ = sink_->Begin(stage_id, input_items);
    }
  }

  template <typename InputItemsFunction,
            typename std::enable_if<!std::is_integral<
                typename std::decay<InputItemsFunction>::type>::value,
                                    int>::type = 0>
  NonBaStageScope(NonBaStageSink* sink,
                  const NonBaStageId stage_id,
                  InputItemsFunction input_items_function)
      : sink_(sink) {
    if (sink_ != nullptr) {
      token_ = sink_->Begin(
          stage_id, static_cast<uint64_t>(input_items_function()));
    }
  }

  ~NonBaStageScope() {
    if (sink_ != nullptr) {
      sink_->End(token_, output_items_);
    }
  }

  NonBaStageScope(const NonBaStageScope&) = delete;
  NonBaStageScope& operator=(const NonBaStageScope&) = delete;

  void SetOutputItems(const uint64_t output_items) {
    if (sink_ != nullptr) {
      output_items_ = output_items;
    }
  }

 private:
  NonBaStageSink* sink_;
  NonBaStageToken token_;
  uint64_t output_items_ = 0;
};

struct NonBaStageSnapshot {
  uint64_t calls = 0;
  int64_t exclusive_ns = 0;
  int64_t inclusive_ns = 0;
  uint64_t input_items = 0;
  uint64_t input_items_max = 0;
  uint64_t output_items = 0;
  uint64_t output_items_max = 0;
  uint64_t zero_yield_calls = 0;
  int64_t zero_yield_exclusive_ns = 0;
};

struct NonBaStageDistributionSnapshot {
  int64_t exclusive_p50_ns = 0;
  int64_t exclusive_p95_ns = 0;
  int64_t exclusive_max_ns = 0;
  int64_t inclusive_p50_ns = 0;
  int64_t inclusive_p95_ns = 0;
  int64_t inclusive_max_ns = 0;
};

struct NonBaPartitionSnapshot {
  int64_t mapper_root_inclusive_ns = 0;
  int64_t nonba_exclusive_ns = 0;
  int64_t ba_excluded_exclusive_ns = 0;
  int64_t unclassified_exclusive_ns = 0;
  int64_t closure_error_ns = 0;
};

class NonBaStageProfiler final : public NonBaStageSink {
 public:
  using NowFunction = int64_t (*)();

  explicit NonBaStageProfiler(NowFunction now = &SteadyNowNs) : now_(now) {
    assert(now_ != nullptr);
  }

  NonBaStageToken Begin(const NonBaStageId stage_id,
                        const uint64_t input_items) override {
    const int64_t start_ns = now_();
    const NonBaStageToken token{stack_.size()};
    stack_.push_back({stage_id, start_ns, 0, input_items});
    return token;
  }

  void End(const NonBaStageToken& token, const uint64_t output_items) override {
    assert(!stack_.empty());
    assert(token.depth + 1 == stack_.size());

    const int64_t stop_ns = now_();
    const Frame frame = stack_.back();
    stack_.pop_back();

    const int64_t inclusive_ns = std::max<int64_t>(0, stop_ns - frame.start_ns);
    const int64_t exclusive_ns =
        std::max<int64_t>(0, inclusive_ns - frame.child_inclusive_ns);
    if (!stack_.empty()) {
      stack_.back().child_inclusive_ns += inclusive_ns;
    }

    Aggregate& aggregate = aggregates_[Index(frame.stage_id)];
    ++aggregate.calls;
    aggregate.exclusive_ns += exclusive_ns;
    aggregate.inclusive_ns += inclusive_ns;
    aggregate.input_items += frame.input_items;
    aggregate.input_items_max =
        std::max(aggregate.input_items_max, frame.input_items);
    aggregate.output_items += output_items;
    aggregate.output_items_max =
        std::max(aggregate.output_items_max, output_items);
    aggregate.exclusive_samples_ns.push_back(exclusive_ns);
    aggregate.inclusive_samples_ns.push_back(inclusive_ns);
    if (NonBaStageInfos()[Index(frame.stage_id)].yield_defined &&
        output_items == 0) {
      ++aggregate.zero_yield_calls;
      aggregate.zero_yield_exclusive_ns += exclusive_ns;
    }
  }

  NonBaStageSnapshot Snapshot(const NonBaStageId stage_id) const {
    const Aggregate& aggregate = aggregates_[Index(stage_id)];
    NonBaStageSnapshot snapshot;
    snapshot.calls = aggregate.calls;
    snapshot.exclusive_ns = aggregate.exclusive_ns;
    snapshot.inclusive_ns = aggregate.inclusive_ns;
    snapshot.input_items = aggregate.input_items;
    snapshot.input_items_max = aggregate.input_items_max;
    snapshot.output_items = aggregate.output_items;
    snapshot.output_items_max = aggregate.output_items_max;
    snapshot.zero_yield_calls = aggregate.zero_yield_calls;
    snapshot.zero_yield_exclusive_ns = aggregate.zero_yield_exclusive_ns;
    return snapshot;
  }

  NonBaStageDistributionSnapshot DistributionSnapshot(
      const NonBaStageId stage_id) const {
    const Aggregate& aggregate = aggregates_[Index(stage_id)];
    NonBaStageDistributionSnapshot snapshot;
    snapshot.exclusive_p50_ns =
        PercentileNs(aggregate.exclusive_samples_ns, 0.50);
    snapshot.exclusive_p95_ns =
        PercentileNs(aggregate.exclusive_samples_ns, 0.95);
    snapshot.exclusive_max_ns = MaxNs(aggregate.exclusive_samples_ns);
    snapshot.inclusive_p50_ns =
        PercentileNs(aggregate.inclusive_samples_ns, 0.50);
    snapshot.inclusive_p95_ns =
        PercentileNs(aggregate.inclusive_samples_ns, 0.95);
    snapshot.inclusive_max_ns = MaxNs(aggregate.inclusive_samples_ns);
    return snapshot;
  }

  NonBaPartitionSnapshot PartitionSnapshot() const {
    NonBaPartitionSnapshot snapshot;
    const Aggregate& root = aggregates_[Index(NonBaStageId::kMapperRoot)];
    snapshot.mapper_root_inclusive_ns = root.inclusive_ns;
    snapshot.nonba_exclusive_ns = root.exclusive_ns;
    snapshot.unclassified_exclusive_ns = root.exclusive_ns;
    for (size_t i = 0; i < aggregates_.size(); ++i) {
      if (i == Index(NonBaStageId::kMapperRoot)) {
        continue;
      }
      const NonBaPartition partition = NonBaStageInfos()[i].partition;
      if (partition == NonBaPartition::kNonBa) {
        snapshot.nonba_exclusive_ns += aggregates_[i].exclusive_ns;
      } else if (partition == NonBaPartition::kBaExcluded) {
        snapshot.ba_excluded_exclusive_ns += aggregates_[i].exclusive_ns;
      }
    }
    snapshot.closure_error_ns =
        snapshot.mapper_root_inclusive_ns - snapshot.nonba_exclusive_ns -
        snapshot.ba_excluded_exclusive_ns;
    return snapshot;
  }

  bool WriteJson(const std::string& path) const {
    if (!stack_.empty() || path.empty()) {
      return false;
    }

    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream.is_open()) {
      return false;
    }
    stream << std::fixed << std::setprecision(9);

    const NonBaPartitionSnapshot partition = PartitionSnapshot();

    stream << "{\n"
           << "  \"schema\": \"colmap_nonba_stage_profile_v1\",\n"
           << "  \"clock\": \"std::chrono::steady_clock\",\n"
           << "  \"wall_unit\": \"milliseconds\",\n"
           << "  \"percentile_method\": \"nearest_rank\",\n"
           << "  \"authoritative_total_wall_source\": "
              "\"/usr/bin/time -v around the complete Mapper process\",\n"
           << "  \"formal_partition_uses\": \"exclusive_wall_ms\",\n"
           << "  \"inclusive_wall_is_diagnostic\": true,\n"
           << "  \"parallel_stage_semantics\": "
              "\"caller_boundary_wall_not_worker_time_sum\",\n"
           << "  \"partition\": {\n"
           << "    \"mapper_root_inclusive_wall_ms\": "
           << NsToMs(partition.mapper_root_inclusive_ns) << ",\n"
           << "    \"nonba_exclusive_wall_ms\": "
           << NsToMs(partition.nonba_exclusive_ns) << ",\n"
           << "    \"ba_solve_excluded_exclusive_wall_ms\": "
           << NsToMs(partition.ba_excluded_exclusive_ns) << ",\n"
           << "    \"uninstrumented_remainder_exclusive_wall_ms\": "
           << NsToMs(partition.unclassified_exclusive_ns) << ",\n"
           << "    \"closure_error_ms\": "
           << NsToMs(partition.closure_error_ns)
           << "\n  },\n"
           << "  \"stages\": [\n";

    for (size_t i = 0; i < aggregates_.size(); ++i) {
      const Aggregate& aggregate = aggregates_[i];
      const NonBaStageInfo& info = NonBaStageInfos()[i];
      stream << "    {\"stage_id\": " << i << ", \"name\": \""
             << info.name << "\", \"parent\": \""
             << ParentName(static_cast<NonBaStageId>(i))
             << "\", \"phase\": \""
             << PhaseName(static_cast<NonBaStageId>(i))
             << "\", \"exclusive_bucket\": \""
             << ExclusiveBucketName(static_cast<NonBaStageId>(i))
             << "\", \"boundary_contents\": \""
             << BoundaryContents(static_cast<NonBaStageId>(i))
             << "\", \"partition\": \""
             << PartitionName(info.partition)
             << "\", \"ba_excluded\": "
             << (info.partition == NonBaPartition::kBaExcluded ? "true"
                                                               : "false")
             << ", \"yield_defined\": "
             << (info.yield_defined ? "true" : "false")
             << ", \"input_item_unit\": \"" << info.input_item_unit
             << "\", \"output_item_unit\": \"" << info.output_item_unit
             << "\", \"calls\": " << aggregate.calls
             << ", \"exclusive_wall_ms\": "
             << NsToMs(aggregate.exclusive_ns)
             << ", \"inclusive_wall_diagnostic_ms\": "
             << NsToMs(aggregate.inclusive_ns)
             << ", \"exclusive_call_ms\": {\"p50\": "
             << NsToMs(PercentileNs(aggregate.exclusive_samples_ns, 0.50))
             << ", \"p95\": "
             << NsToMs(PercentileNs(aggregate.exclusive_samples_ns, 0.95))
             << ", \"max\": "
             << NsToMs(MaxNs(aggregate.exclusive_samples_ns))
             << "}, \"inclusive_call_diagnostic_ms\": {\"p50\": "
             << NsToMs(PercentileNs(aggregate.inclusive_samples_ns, 0.50))
             << ", \"p95\": "
             << NsToMs(PercentileNs(aggregate.inclusive_samples_ns, 0.95))
             << ", \"max\": "
             << NsToMs(MaxNs(aggregate.inclusive_samples_ns))
             << "}, \"input_items\": {\"sum\": " << aggregate.input_items
             << ", \"max\": " << aggregate.input_items_max
             << "}, \"output_items\": {\"sum\": "
             << aggregate.output_items << ", \"max\": "
             << aggregate.output_items_max << "}"
             << ", \"zero_yield_calls\": " << aggregate.zero_yield_calls
             << ", \"zero_yield_exclusive_wall_ms\": "
             << NsToMs(aggregate.zero_yield_exclusive_ns) << "}"
             << (i + 1 == aggregates_.size() ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
    stream.close();
    return stream.good();
  }

 private:
  struct Frame {
    NonBaStageId stage_id;
    int64_t start_ns;
    int64_t child_inclusive_ns;
    uint64_t input_items;
  };

  struct Aggregate {
    uint64_t calls = 0;
    int64_t exclusive_ns = 0;
    int64_t inclusive_ns = 0;
    uint64_t input_items = 0;
    uint64_t input_items_max = 0;
    uint64_t output_items = 0;
    uint64_t output_items_max = 0;
    uint64_t zero_yield_calls = 0;
    int64_t zero_yield_exclusive_ns = 0;
    std::vector<int64_t> exclusive_samples_ns;
    std::vector<int64_t> inclusive_samples_ns;
  };

  static int64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  static size_t Index(const NonBaStageId stage_id) {
    return static_cast<size_t>(stage_id);
  }

  static double NsToMs(const int64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1000000.0;
  }

  static int64_t MaxNs(const std::vector<int64_t>& samples) {
    if (samples.empty()) {
      return 0;
    }
    return *std::max_element(samples.begin(), samples.end());
  }

  static int64_t PercentileNs(const std::vector<int64_t>& samples,
                              const double quantile) {
    if (samples.empty()) {
      return 0;
    }
    std::vector<int64_t> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    const size_t rank = static_cast<size_t>(
        std::ceil(quantile * static_cast<double>(sorted_samples.size())));
    const size_t index = std::max<size_t>(1, rank) - 1;
    return sorted_samples[std::min(index, sorted_samples.size() - 1)];
  }

  static const char* ParentName(const NonBaStageId stage_id) {
    switch (stage_id) {
      case NonBaStageId::kMapperRoot:
        return "none";
      case NonBaStageId::kLidarPlyLoadTransformAndIndex:
        return "lidar_load";
      case NonBaStageId::kInitialLidarProjection:
      case NonBaStageId::kInitialAbsolutePose:
      case NonBaStageId::kInitialPoseRefine:
      case NonBaStageId::kInitialCommit:
        return "initial_pair";
      case NonBaStageId::kRegisterCollect2D3D:
      case NonBaStageId::kRegisterAbsolutePose:
      case NonBaStageId::kRegisterPoseRefine:
      case NonBaStageId::kRegisterCommit:
        return "register_next_image";
      case NonBaStageId::kLocalFindBundle:
      case NonBaStageId::kLocalBaConfig:
      case NonBaStageId::kLocalFilterInImages:
      case NonBaStageId::kLocalFilterModified:
      case NonBaStageId::kLocalLidarOutlier:
        return "local_refinement";
      case NonBaStageId::kLocalLidarProjectionPreparation:
      case NonBaStageId::kLocalLidarProjectionAndMatching:
      case NonBaStageId::kLocalKdQueries:
      case NonBaStageId::kLocalBaSolve:
      case NonBaStageId::kLocalMergeTracks:
      case NonBaStageId::kLocalCompleteTracks:
      case NonBaStageId::kLocalCompleteImage:
        return "local_ba_config";
      case NonBaStageId::kLocalVariablePointCollection:
        return "local_lidar_projection_preparation";
      case NonBaStageId::kGlobalPoints3DAndConfigCopy:
      case NonBaStageId::kGlobalNegativeDepthScan:
      case NonBaStageId::kGlobalImageConfigSelection:
      case NonBaStageId::kGlobalVariablePointCollection:
      case NonBaStageId::kGlobalLidarKdPreparation:
      case NonBaStageId::kGlobalBaConfig:
        return "global_adjustment";
      case NonBaStageId::kGlobalKdQueries:
        return "global_lidar_kd_preparation";
      case NonBaStageId::kGlobalBaSolve:
        return "global_ba_config";
      case NonBaStageId::kGlobalPreComplete:
      case NonBaStageId::kGlobalPreMerge:
      case NonBaStageId::kGlobalRetriangulate:
      case NonBaStageId::kGlobalPostComplete:
      case NonBaStageId::kGlobalPostMerge:
      case NonBaStageId::kGlobalFilterAll:
      case NonBaStageId::kGlobalFilterImages:
        return "global_refinement";
      case NonBaStageId::kGlobalAdjustment:
        return "global_refinement_or_mapper_root";
      default:
        return "mapper_root";
    }
  }

  static const char* PhaseName(const NonBaStageId stage_id) {
    const size_t index = Index(stage_id);
    if (stage_id == NonBaStageId::kMapperRoot) return "root";
    if (index <= Index(NonBaStageId::kLidarPlyLoadTransformAndIndex)) {
      return "input";
    }
    if (index <= Index(NonBaStageId::kInitialCommit)) return "initial";
    if (index <= Index(NonBaStageId::kTriangulateImage)) return "register";
    if (index <= Index(NonBaStageId::kLocalLidarOutlier)) return "local";
    if (index <= Index(NonBaStageId::kGlobalFilterImages)) return "global";
    return "output_shutdown";
  }

  static const char* ExclusiveBucketName(const NonBaStageId stage_id) {
    switch (stage_id) {
      case NonBaStageId::kMapperRoot:
        return "unclassified";
      case NonBaStageId::kLidarLoad:
        return "point_cloud_load_other";
      case NonBaStageId::kInitialPair:
        return "initial_pair_other";
      case NonBaStageId::kRegisterNextImage:
        return "register_other";
      case NonBaStageId::kLocalRefinement:
        return "local_other";
      case NonBaStageId::kGlobalRefinement:
        return "global_other";
      case NonBaStageId::kGlobalAdjustment:
        return "global_adjustment_other";
      default:
        return NonBaStageInfos()[Index(stage_id)].name;
    }
  }

  static const char* BoundaryContents(const NonBaStageId stage_id) {
    switch (stage_id) {
      case NonBaStageId::kMapperRoot:
        return "controller thread root; exclusive wall is unclassified";
      case NonBaStageId::kDatabaseCorrespondenceLoad:
        return "database open plus DatabaseCache correspondence and image load";
      case NonBaStageId::kLidarLoad:
        return "point cloud owner wrapper including object setup";
      case NonBaStageId::kLidarPlyLoadTransformAndIndex:
        return "single owner call containing PLY read coordinate transform "
               "submap build and KD index build";
      case NonBaStageId::kInitialPair:
        return "entire initial pair registration including nested stages";
      case NonBaStageId::kInitialLidarProjection:
        return "single LiDAR SetNewImage owner call containing prepare project and feature match";
      case NonBaStageId::kRegisterNextImage:
        return "entire next image registration including nested stages";
      case NonBaStageId::kLocalRefinement:
        return "entire iterative local refinement call including nested stages";
      case NonBaStageId::kLocalBaConfig:
        return "inclusive-content boundary for local BA configuration and its "
               "nested projection KD solve merge and completion stages; "
               "exclusive wall is unclassified local-config work";
      case NonBaStageId::kLocalLidarProjectionPreparation:
        return "inclusive-content boundary for the combined variable-point "
               "collection and LiDAR candidate split; child exclusive owns "
               "the combined loop";
      case NonBaStageId::kLocalVariablePointCollection:
        return "combined variable-point collection and LiDAR projection/KD "
               "candidate classification loop";
      case NonBaStageId::kLocalLidarProjectionAndMatching:
        return "BundleAdjustmentConfig Project2Image and "
               "MatchVariablePoint2LidarPoint owner boundaries";
      case NonBaStageId::kLocalKdQueries:
        return "BundleAdjustmentConfig MatchClosestLidarPoint owner boundaries";
      case NonBaStageId::kLocalBaSolve:
      case NonBaStageId::kGlobalBaSolve:
        return "BundleAdjuster Solve caller wall boundary; parallel worker times are not summed";
      case NonBaStageId::kGlobalRefinement:
        return "entire iterative global refinement call including nested stages";
      case NonBaStageId::kGlobalAdjustment:
        return "entire global adjustment wrapper including nested config KD and BA stages";
      case NonBaStageId::kGlobalPoints3DAndConfigCopy:
        return "Reconstruction Points3D value copy and access at owner boundary";
      case NonBaStageId::kGlobalLidarKdPreparation:
        return "inclusive-content boundary for the original combined LiDAR "
               "range-preparation KD-query and constraint-creation loop; "
               "child exclusive owns the combined loop";
      case NonBaStageId::kGlobalKdQueries:
        return "original combined per-variable-point range preparation "
               "nearest LiDAR query and accepted-constraint creation loop";
      case NonBaStageId::kGlobalBaConfig:
        return "BundleAdjuster construction binding and destruction boundary "
               "with BA Solve nested and excluded";
      case NonBaStageId::kGlobalFilterAll:
        return "single FilterAllPoints3D owner boundary containing negative-"
               "depth reprojection-error and triangulation-angle filtering";
      case NonBaStageId::kWriteModel:
        return "fresh Mapper callback Reconstruction Write plus project options write";
      default:
        return "direct caller boundary";
    }
  }

  static const char* PartitionName(const NonBaPartition partition) {
    switch (partition) {
      case NonBaPartition::kRootDiagnostic:
        return "root_diagnostic";
      case NonBaPartition::kNonBa:
        return "nonba";
      case NonBaPartition::kBaExcluded:
        return "ba_excluded";
    }
    return "unknown";
  }

  NowFunction now_;
  std::vector<Frame> stack_;
  std::array<Aggregate, static_cast<size_t>(NonBaStageId::kCount)>
      aggregates_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_NONBA_PROFILER_H_
