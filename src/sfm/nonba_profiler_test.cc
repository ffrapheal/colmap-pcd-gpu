#include "sfm/nonba_profiler.h"

#include <cassert>
#include <cstdio>
#include <fstream>
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
  {
    colmap::NonBaStageScope disabled_scope(
        nullptr, colmap::NonBaStageId::kMapperRoot, 1);
    disabled_scope.SetOutputItems(1);
  }
  assert(g_now_calls == 0);
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

    root.SetOutputItems(1);
    g_now_ns = 400;
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

  const colmap::NonBaStageSnapshot root =
      profiler.Snapshot(colmap::NonBaStageId::kMapperRoot);
  assert(root.inclusive_ns == 400);
  assert(root.exclusive_ns == 110);

  const colmap::NonBaPartitionSnapshot partition =
      profiler.PartitionSnapshot();
  assert(partition.mapper_root_inclusive_ns == 400);
  assert(partition.nonba_exclusive_ns == 370);
  assert(partition.ba_excluded_exclusive_ns == 30);
  assert(partition.unclassified_exclusive_ns == 110);
  assert(partition.closure_error_ns == 0);

  assert(profiler.WriteJson(on_path));
  assert(std::ifstream(on_path).good());
  return 0;
}
