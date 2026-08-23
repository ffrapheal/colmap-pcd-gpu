#include "sfm/nonba_profiler.h"
#include "lidar/pcd_projection.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

int64_t g_now_ns = 0;
int g_now_calls = 0;

int64_t FakeNowNs() {
  ++g_now_calls;
  return g_now_ns;
}

void EarlyReturn(colmap::NonBaStageSink* profiler) {
  g_now_ns = 310;
  colmap::NonBaStageScope stage(
      profiler, colmap::NonBaStageId::kFindNextImages, 4);
  g_now_ns = 320;
  return;
}

void RecordDuration(colmap::NonBaStageSink* profiler,
                    const int64_t start_ns,
                    const int64_t stop_ns) {
  g_now_ns = start_ns;
  colmap::NonBaStageScope stage(
      profiler, colmap::NonBaStageId::kDatabaseCorrespondenceLoad, 2);
  stage.SetOutputItems(1);
  g_now_ns = stop_ns;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 3);
  const std::string off_path = argv[1];
  const std::string on_path = argv[2];
  std::remove(off_path.c_str());
  std::remove(on_path.c_str());

  // Off: the nullable sink executes no clock reads and creates no output.
  g_now_calls = 0;
  int disabled_input_calls = 0;
  {
    colmap::NonBaStageScope disabled_scope(
        nullptr, colmap::NonBaStageId::kMapperRoot,
        [&disabled_input_calls]() {
          ++disabled_input_calls;
          return 1;
        });
    disabled_scope.SetOutputItems(1);
  }
  assert(g_now_calls == 0);
  assert(disabled_input_calls == 0);
  assert(!std::ifstream(off_path).good());

  // A zero output is not a zero-yield result for stages whose output is only
  // setup/configuration bookkeeping.
  colmap::NonBaStageProfiler no_yield_profiler(&FakeNowNs);
  g_now_ns = 0;
  {
    colmap::NonBaStageScope configuration(
        &no_yield_profiler, colmap::NonBaStageId::kGlobalBaConfig, 3);
    g_now_ns = 10;
  }
  const colmap::NonBaStageSnapshot configuration =
      no_yield_profiler.Snapshot(colmap::NonBaStageId::kGlobalBaConfig);
  assert(configuration.calls == 1);
  assert(configuration.output_items == 0);
  assert(configuration.zero_yield_calls == 0);
  assert(configuration.zero_yield_exclusive_ns == 0);

  colmap::NonBaStageProfiler profiler(&FakeNowNs);

  colmap::lidar::PcdProjectionOptions projection_options;
  colmap::lidar::PcdProj projector(projection_options);
  assert(projector.NonBaProfiler() == nullptr);
  const int hook_now_calls = g_now_calls;
  {
    colmap::lidar::PcdProj::NonBaProfilerBinding null_binding(
        &projector, nullptr);
    assert(projector.NonBaProfiler() == nullptr);
  }
  assert(projector.NonBaProfiler() == nullptr);
  {
    colmap::lidar::PcdProj::NonBaProfilerBinding outer_binding(
        &projector, &profiler);
    assert(projector.NonBaProfiler() == &profiler);
    try {
      colmap::lidar::PcdProj::NonBaProfilerBinding nested_binding(
          &projector, &no_yield_profiler);
      assert(projector.NonBaProfiler() == &no_yield_profiler);
      throw std::runtime_error("synthetic binding unwind");
    } catch (const std::runtime_error&) {
      assert(projector.NonBaProfiler() == &profiler);
    }
  }
  assert(projector.NonBaProfiler() == nullptr);
  assert(g_now_calls == hook_now_calls);
  assert(!std::ifstream(off_path).good());

  g_now_ns = 0;
  {
    colmap::NonBaStageScope root(
        &profiler, colmap::NonBaStageId::kMapperRoot, 1);

    g_now_ns = 10;
    {
      colmap::NonBaStageScope parent(
          &profiler, colmap::NonBaStageId::kLocalRefinement, 5);

      g_now_ns = 20;
      {
        colmap::NonBaStageScope ba(
            &profiler, colmap::NonBaStageId::kLocalBaSolve, 7);
        ba.SetOutputItems(7);
        g_now_ns = 50;
      }

      g_now_ns = 55;
      {
        colmap::NonBaStageScope zero_yield(
            &profiler, colmap::NonBaStageId::kLocalMergeTracks, 11);
        g_now_ns = 65;
      }

      parent.SetOutputItems(1);
      g_now_ns = 90;
    }

    RecordDuration(&profiler, 100, 110);
    RecordDuration(&profiler, 120, 140);
    RecordDuration(&profiler, 150, 180);
    RecordDuration(&profiler, 190, 230);
    RecordDuration(&profiler, 240, 340);

    EarlyReturn(&profiler);

    g_now_ns = 340;
    {
      colmap::NonBaStageScope projection_matching(
          &profiler,
          colmap::NonBaStageId::kLocalLidarProjectionAndMatching, 6);

      g_now_ns = 350;
      {
        colmap::NonBaStageScope project2image_loop(
            &profiler, colmap::NonBaStageId::kLocalProject2ImageLoop, 6);

        g_now_ns = 360;
        {
          colmap::NonBaStageScope set_new_image(
              &profiler, colmap::NonBaStageId::kLocalPcdSetNewImage, 1);

          g_now_ns = 370;
          {
            colmap::NonBaStageScope feature_collection(
                &profiler,
                colmap::NonBaStageId::kLocalPcdFeatureCollectionIndex, 20);
            feature_collection.SetOutputItems(12);
            g_now_ns = 380;
          }

          g_now_ns = 385;
          {
            colmap::NonBaStageScope search_submap(
                &profiler, colmap::NonBaStageId::kLocalPcdSearchSubmap, 1);
            search_submap.SetOutputItems(4);
            g_now_ns = 395;
          }

          g_now_ns = 400;
          {
            colmap::NonBaStageScope image_map_projection(
                &profiler, colmap::NonBaStageId::kLocalPcdImageMapProj, 123);
            image_map_projection.SetOutputItems(9);
            g_now_ns = 415;
          }

          g_now_ns = 420;
          {
            colmap::NonBaStageScope association_extraction(
                &profiler,
                colmap::NonBaStageId::kLocalPcdAssociationExtraction, 20);
            association_extraction.SetOutputItems(7);
            g_now_ns = 430;
          }

          set_new_image.SetOutputItems(7);
          g_now_ns = 440;
        }

        project2image_loop.SetOutputItems(1);
        g_now_ns = 450;
      }

      g_now_ns = 460;
      {
        colmap::NonBaStageScope match_variable_point_loop(
            &profiler,
            colmap::NonBaStageId::kLocalMatchVariablePointToLidarLoop, 6);
        match_variable_point_loop.SetOutputItems(5);
        g_now_ns = 480;
      }

      projection_matching.SetOutputItems(5);
      g_now_ns = 500;
    }

    root.SetOutputItems(1);
    g_now_ns = 550;
  }

  const colmap::NonBaStageSnapshot parent =
      profiler.Snapshot(colmap::NonBaStageId::kLocalRefinement);
  assert(parent.calls == 1);
  assert(parent.inclusive_ns == 80);
  assert(parent.exclusive_ns == 40);
  assert(parent.input_items == 5);
  assert(parent.input_items_max == 5);

  const colmap::NonBaStageSnapshot ba =
      profiler.Snapshot(colmap::NonBaStageId::kLocalBaSolve);
  assert(ba.calls == 1);
  assert(ba.exclusive_ns == 30);

  const colmap::NonBaStageSnapshot zero_yield =
      profiler.Snapshot(colmap::NonBaStageId::kLocalMergeTracks);
  assert(zero_yield.zero_yield_calls == 1);
  assert(zero_yield.zero_yield_exclusive_ns == 10);

  const colmap::NonBaStageSnapshot early_return =
      profiler.Snapshot(colmap::NonBaStageId::kFindNextImages);
  assert(early_return.calls == 1);
  assert(early_return.exclusive_ns == 10);

  const colmap::NonBaStageSnapshot distribution_stage =
      profiler.Snapshot(colmap::NonBaStageId::kDatabaseCorrespondenceLoad);
  assert(distribution_stage.calls == 5);
  assert(distribution_stage.input_items == 10);
  assert(distribution_stage.input_items_max == 2);
  assert(distribution_stage.output_items == 5);
  assert(distribution_stage.output_items_max == 1);
  const colmap::NonBaStageDistributionSnapshot distribution =
      profiler.DistributionSnapshot(
          colmap::NonBaStageId::kDatabaseCorrespondenceLoad);
  assert(distribution.exclusive_p50_ns == 30);
  assert(distribution.exclusive_p95_ns == 100);
  assert(distribution.exclusive_max_ns == 100);

  const colmap::NonBaStageSnapshot projection_matching = profiler.Snapshot(
      colmap::NonBaStageId::kLocalLidarProjectionAndMatching);
  const colmap::NonBaStageSnapshot project2image_loop = profiler.Snapshot(
      colmap::NonBaStageId::kLocalProject2ImageLoop);
  const colmap::NonBaStageSnapshot set_new_image = profiler.Snapshot(
      colmap::NonBaStageId::kLocalPcdSetNewImage);
  const colmap::NonBaStageSnapshot feature_collection = profiler.Snapshot(
      colmap::NonBaStageId::kLocalPcdFeatureCollectionIndex);
  const colmap::NonBaStageSnapshot search_submap = profiler.Snapshot(
      colmap::NonBaStageId::kLocalPcdSearchSubmap);
  const colmap::NonBaStageSnapshot image_map_projection = profiler.Snapshot(
      colmap::NonBaStageId::kLocalPcdImageMapProj);
  const colmap::NonBaStageSnapshot association_extraction = profiler.Snapshot(
      colmap::NonBaStageId::kLocalPcdAssociationExtraction);
  const colmap::NonBaStageSnapshot match_variable_point_loop =
      profiler.Snapshot(
          colmap::NonBaStageId::kLocalMatchVariablePointToLidarLoop);

  assert(projection_matching.inclusive_ns == 160);
  assert(projection_matching.exclusive_ns == 40);
  assert(projection_matching.calls == 1);
  assert(projection_matching.input_items == 6);
  assert(projection_matching.output_items == 5);
  assert(project2image_loop.calls == 1);
  assert(project2image_loop.input_items == 6);
  assert(project2image_loop.output_items == 1);
  assert(project2image_loop.inclusive_ns == 100);
  assert(project2image_loop.exclusive_ns == 20);
  assert(set_new_image.calls == 1);
  assert(set_new_image.input_items == 1);
  assert(set_new_image.output_items == 7);
  assert(set_new_image.inclusive_ns == 80);
  assert(set_new_image.exclusive_ns == 35);
  assert(feature_collection.calls == 1);
  assert(feature_collection.input_items == 20);
  assert(feature_collection.output_items == 12);
  assert(feature_collection.inclusive_ns == 10);
  assert(search_submap.calls == 1);
  assert(search_submap.input_items == 1);
  assert(search_submap.output_items == 4);
  assert(search_submap.inclusive_ns == 10);
  assert(image_map_projection.calls == 1);
  assert(image_map_projection.input_items == 123);
  assert(image_map_projection.output_items == 9);
  assert(image_map_projection.inclusive_ns == 15);
  assert(association_extraction.calls == 1);
  assert(association_extraction.input_items == 20);
  assert(association_extraction.output_items == 7);
  assert(association_extraction.inclusive_ns == 10);
  assert(match_variable_point_loop.calls == 1);
  assert(match_variable_point_loop.input_items == 6);
  assert(match_variable_point_loop.output_items == 5);
  assert(match_variable_point_loop.inclusive_ns == 20);

  assert(project2image_loop.output_items == set_new_image.calls);
  assert(feature_collection.calls == set_new_image.calls);
  assert(search_submap.calls == set_new_image.calls);
  assert(image_map_projection.calls == set_new_image.calls);
  assert(association_extraction.calls == set_new_image.calls);
  assert(set_new_image.inclusive_ns ==
         set_new_image.exclusive_ns + feature_collection.inclusive_ns +
             search_submap.inclusive_ns + image_map_projection.inclusive_ns +
             association_extraction.inclusive_ns);
  assert(project2image_loop.inclusive_ns ==
         project2image_loop.exclusive_ns + set_new_image.inclusive_ns);
  assert(projection_matching.inclusive_ns ==
         projection_matching.exclusive_ns + project2image_loop.inclusive_ns +
             match_variable_point_loop.inclusive_ns);

  const auto& stage_infos = colmap::NonBaStageInfos();
  const auto& project2image_info = stage_infos[static_cast<size_t>(
      colmap::NonBaStageId::kLocalProject2ImageLoop)];
  const auto& image_map_projection_info = stage_infos[static_cast<size_t>(
      colmap::NonBaStageId::kLocalPcdImageMapProj)];
  const auto assert_units = [&stage_infos](
                                const colmap::NonBaStageId stage_id,
                                const std::string& input_unit,
                                const std::string& output_unit) {
    const auto& info = stage_infos[static_cast<size_t>(stage_id)];
    assert(std::string(info.input_item_unit) == input_unit);
    assert(std::string(info.output_item_unit) == output_unit);
  };
  assert(std::string(project2image_info.input_item_unit) ==
         "project2image_calls");
  assert(std::string(project2image_info.output_item_unit) ==
         "new_projected_images");
  assert(std::string(image_map_projection_info.input_item_unit) ==
         "selected_lidar_points");
  assert(std::string(image_map_projection_info.output_item_unit) ==
         "projected_feature_pixels");
  assert_units(colmap::NonBaStageId::kLocalLidarProjectionAndMatching,
               "candidate_points", "accepted_lidar_constraints");
  assert_units(colmap::NonBaStageId::kLocalPcdSetNewImage,
               "images", "new_unique_point3D_lidar_associations");
  assert_units(colmap::NonBaStageId::kLocalPcdFeatureCollectionIndex,
               "points2D", "valid_unique_feature_pixels");
  assert_units(colmap::NonBaStageId::kLocalPcdSearchSubmap,
               "images", "selected_lidar_nodes");
  assert_units(colmap::NonBaStageId::kLocalPcdAssociationExtraction,
               "points2D", "new_unique_point3D_lidar_associations");
  assert_units(colmap::NonBaStageId::kLocalMatchVariablePointToLidarLoop,
               "candidate_points", "accepted_lidar_constraints");

  const colmap::NonBaStageSnapshot root =
      profiler.Snapshot(colmap::NonBaStageId::kMapperRoot);
  assert(root.inclusive_ns == 550);
  assert(root.exclusive_ns == 100);

  const colmap::NonBaPartitionSnapshot partition =
      profiler.PartitionSnapshot();
  assert(partition.mapper_root_inclusive_ns == 550);
  assert(partition.nonba_exclusive_ns == 520);
  assert(partition.ba_excluded_exclusive_ns == 30);
  assert(partition.unclassified_exclusive_ns == 100);
  assert(partition.closure_error_ns == 0);

  assert(profiler.WriteJson(on_path));
  assert(std::ifstream(on_path).good());
  return 0;
}
