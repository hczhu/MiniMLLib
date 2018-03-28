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
  using namespace folly::gen;
  std::ifstream trainIfs("data/quasar_train.csv");
  std::ifstream testIfs("data/quasar_test.csv");
  auto readLine = [&] (std::ifstream& ifs){
    std::string line;
    ifs >> line;
    ifs.ignore(2);
    return split(line, ',') | eachTo<double>() | as<std::vector>();
  };
  auto smoothY = [](const std::vector<double>& X, const std::vector<double>& Y,
                    double sm,
                    std::vector<std::vector<double>>* thetas = nullptr) {
    auto W = Y;
    auto sY = Y;
    for (int i = 0; i < X.size(); ++i) {
      auto x = X[i];
      for (int j = 0; j < W.size(); ++j) {
        W[j] = exp(-(x - X[j]) * (x - X[j]) / 2 / sm / sm);
      }
      auto theta = fitLSM(from(X) | map([](double x) {
                            return std::vector<double>(1, x);
                          }) | as<std::vector>(),
                          Y, 0, W);
      sY[i] = theta[0] * x + theta[1];
      if (thetas) {
        thetas->push_back(std::move(theta));
      }
    }
    return sY;
  };

  auto X = readLine(trainIfs);
  CHECK(X == readLine(testIfs));

  std::vector<std::vector<double>> trainY;
  std::vector<std::vector<double>> testY;
  while (trainIfs.peek() != EOF) {
    trainY.push_back(readLine(trainIfs));
    CHECK_EQ(trainY.back().size(), X.size());
  }
  while (testIfs.peek() != EOF) {
    testY.push_back(readLine(testIfs));
    CHECK_EQ(testY.back().size(), X.size());
  }
  const auto XX = from(X) |
                  map([](double x) { return std::vector<double>(1, x); }) |
                  as<std::vector>();

  // data visualization
  {
    auto lY = trainY[0];
    const auto theta = fitLSM(XX, lY);
    CHECK_EQ(2, theta.size());
    std::ofstream ofs("data/quasar_train_visual.csv");
    auto printValues = [&](const std::vector<double>& v,
                          const std::string& name) {
      ofs << name << " " << (from(v) | unsplit(',')) << std::endl;
    };
    ofs << (from(X) | unsplit(',')) << std::endl;
    printValues(lY, "Observed");
    for (int i = 0; i < X.size(); ++i) {
      lY[i] = theta[0] * X[i] + theta[1];
    }
    printValues(lY, "Linear-reg");
    for (double sm : {5, 1, 10, 100, 1000}) {
      lY = smoothY(X, trainY[0], sm);
      printValues(lY,
                  folly::sformat("Locally-weighted-linear-reg-sigma={}", sm));
    }
  }
  // smooth all
  for (auto& y : trainY) {
    y = smoothY(X, y, 5);
  }
  for (auto& y : testY) {
    y = smoothY(X, y, 5);
  }
  LOG(INFO) << "Smoothed all.";

  int cutoff = 150;
  auto distance = [&, cutoff](const auto& Y1, const auto& Y2, bool right = true) {
    double res = 0;
    int begin = 0;
    int end = cutoff;
    if (right) {
      begin = cutoff;
      end = Y1.size();
    }
    for (int i = begin; i < end; ++i) {
      res += (Y1[i] - Y2[i]) * (Y1[i] - Y2[i]);
    }
    return res;
  };
  const int K = 3;
  auto estimate = [&, cutoff] (const auto& Ys, const auto& Y) {
    std::vector<std::pair<double, int>> disIdx;
    for (int i = 0; i < Ys.size(); ++i) {
      disIdx.emplace_back(distance(Y, Ys[i]), i);
    }
    std::sort(disIdx.begin(), disIdx.end());
    std::vector<double> eY(0, Y.size());
    const auto h = disIdx.back().first;
    double wSum = 0;
    for (int i = 0; i < K; ++i) {
      double w = std::max(0.0, 1 - disIdx[i].first / h);
      wSum = w;
      auto idx = disIdx[i].second;
      for (int j = 0; j < cutoff; ++j) {
        eY[j] += w * Ys[idx][j];
      }
    }
    for (auto& y : eY) {
      y /= wSum;
    }
    return eY;
  };
  double errorSum = 0;
  for (const auto& y : trainY) {
    auto ey = estimate(trainY, y);
     errorSum += distance(ey, y, false);
  }
  LOG(INFO) << "Training error: " << errorSum / trainY.size();
}

// http://cs229.stanford.edu/ps/ps1/ps1.pdf
int main(int argc, char* argv[]) {
  quasarSpectraLinearReg();
  return 0;
}

