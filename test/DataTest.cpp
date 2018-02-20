
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

TEST(DataTest, Basic) {
  mini_ml::SparseFeatureIns<int> ins;
  std::vector<mini_ml::SparseFeatureIns<int>::Fid> expect;
  for (int i = 0; i < 10; ++i) {
    ins.addFeature(i);
    expect.push_back(i);
  }
  EXPECT_EQ(expect, ins.allFeatureIds());
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}

