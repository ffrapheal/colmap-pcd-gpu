#include "sfm/incremental_mapper.h"

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

using Options = colmap::IncrementalMapper::Options;
using Route = Options::LocalLidarRoute;

std::pair<size_t, size_t> CountRoutes(
    const Options& options, const std::vector<size_t>& track_lengths) {
  size_t projection = 0;
  size_t kdtree = 0;
  for (const size_t track_length : track_lengths) {
    if (options.LocalLidarRouteForTrackLength(track_length) ==
        Route::PROJECTION) {
      ++projection;
    } else {
      ++kdtree;
    }
  }
  return {projection, kdtree};
}

}  // namespace

int main() {
  Options options;
  assert(options.min_proj_num == 1);
  assert(!options.local_lidar_kdtree_only);

  options.min_proj_num = 1;
  assert(options.LocalLidarRouteForTrackLength(3) == Route::PROJECTION);
  assert(options.LocalLidarRouteForTrackLength(4) == Route::KDTREE);
  assert(options.LocalLidarRouteForTrackLength(5) == Route::KDTREE);

  options.local_lidar_kdtree_only = true;
  for (const size_t track_length : {size_t{0}, size_t{3}, size_t{4},
                                    size_t{5}}) {
    assert(options.LocalLidarRouteForTrackLength(track_length) ==
           Route::KDTREE);
  }

  const std::vector<size_t> fixed_track_lengths = {0, 1, 2, 3, 4, 5, 8};
  options.local_lidar_kdtree_only = false;
  assert(CountRoutes(options, fixed_track_lengths) ==
         std::make_pair(size_t{4}, size_t{3}));
  options.local_lidar_kdtree_only = true;
  assert(CountRoutes(options, fixed_track_lengths) ==
         std::make_pair(size_t{0}, size_t{7}));
  return 0;
}
