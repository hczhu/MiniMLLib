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

#include <folly/Range.h>
#include <folly/Format.h>
#include <folly/gen/Base.h>
#include <folly/gen/String.h>

#include "prototypes/LinearReg.h"

using namespace mini_ml;

void quasarSpectraLinearReg() {
  std::ifstream ifs("data/quasar_train.csv");
  auto readLine = [&] (std::ifstream& ifs){
    std::string line;
    ifs >> line;
    using namespace folly::gen;
    return split(line, ',') | eachTo<double>() | as<std::vector>();
  };
  std::vector<std::vector<double>> X;
  for (auto x : readLine(ifs)) {
    X.push_back(std::vector<double>(1, x));
  }
  const auto Y = readLine(ifs);
  CHECK_EQ(Y.size(), X.size());
  ifs.close();

  {
    const auto theta = fitLSM(X, Y);
    CHECK_EQ(2, theta.size());
    LOG(INFO) << theta[0] << " " << theta[1] << std::endl;
    std::ofstream ofs("data/quasar_train_visual.csv");
    using namespace folly::gen;
    auto printValues = [&](const std::vector<double>& v,
                          const std::string& name) {
      ofs << name << " " << (from(v) | unsplit(',')) << std::endl;
    };
    ofs << (from(X) | rconcat | unsplit(',')) << std::endl;
    printValues(Y, "Observed");
    auto lY = Y;
    for (int i = 0; i < X.size(); ++i) {
      lY[i] = theta[0] * X[i][0] + theta[1];
    }
    printValues(lY, "Linear-reg");
    auto W = Y;
    for (double sm : {5, 1, 10, 100, 1000}) {
      for (int i = 0; i < X.size(); ++i) {
        auto x = X[i][0];
        for (int j = 0; j < W.size(); ++j) {
          W[j] = exp(-(x - X[j][0]) * (x - X[j][0]) / 2 / sm / sm);
        }
        auto theta = fitLSM(X, Y, 0, W);
        lY[i] = theta[0] * x + theta[1];
      }
      printValues(lY,
                  folly::sformat("Locally-weighted-linear-reg-sigma={}", sm));
    }
  }
}

// http://cs229.stanford.edu/ps/ps1/ps1.pdf
int main(int argc, char* argv[]) {
  quasarSpectraLinearReg();
  return 0;
}

