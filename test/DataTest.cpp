
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <limits.h>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <valarray>
#include <vector>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "Data.h"

TEST(DataTest, Sparse) {
  mlight::SparseFeatureIns<int> ins;
  std::vector<mlight::SparseFeatureIns<int>::Fid> expect;
  for (int i = 0; i < 10; ++i) {
    ins.addFeature(i);
    expect.push_back(i);
  }
  EXPECT_EQ(expect, ins.allFeatureIds());
}

TEST(DataTest, Dense) {
  mlight::DenseFeatureIns<int> ins(8, -1);
  ins.setFeature(0, 1);
  ins.setFeature(3, 3);
  ins.setFeature(7, 0);

  for (const auto& pr : std::vector<std::pair<int, int>>{
    {0, 1},
    {1, -1},
    {3, 3},
    {5, -1},
    {7, 0},
  }) {
    EXPECT_EQ(pr.second, ins.getFeature(pr.first));
    EXPECT_EQ(pr.second, ins.getFeatureDouble(pr.first));
    const auto &baseIns = ins;
    EXPECT_EQ(pr.second,
              baseIns.getFeatureDouble(
                  static_cast<mlight::DataInstanceBase::Fid>(pr.first)));
  }
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}

