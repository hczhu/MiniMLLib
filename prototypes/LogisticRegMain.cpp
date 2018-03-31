
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
#include <fstream>
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

#include "prototypes/LogisticReg.h"

using namespace mini_ml;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::vector<double>> X;
  {
    double x1, x2;
    std::ifstream f("data/logistic_x.txt");
    while (f >> x1 >> x2) {
      X.push_back({x1, x2});
    }
  }
  std::vector<int> Y;
  {
    std::ifstream f("data/logistic_y.txt");
    double y;
    while (f >> y) {
      Y.push_back(y);
    }
  }
  Options options;
  options.useNewton = true;
  options.miniBatchSize = Y.size();
  options.minThetaDiffNorm = 1e-6;
  options.chooseBestLoglossTheta = true;
  options.chooseBestErrorRateTheta = false;
  options.randomInit = false;
  auto theta = fitLR(X, Y, options);
  {
    std::ofstream f("data/logistic_xy.txt");
    for (auto th : theta) {
      f << th << " ";
    }
    f << std::endl;
    for (int i = 0; i < Y.size(); ++i) {
      f << X[i][0] << " " << X[i][1] << " " << Y[i] << std::endl;
    }
  }
  for (auto name : {"data/data_a.txt", "data/data_b.txt"}) {
    LOG(INFO) << "========= training data in " << name << " ================";
    std::ifstream f(name);
    double y, x1, x2;
    std::vector<int> Y;
    std::vector<std::vector<double>> X;
    while (f >> y >> x1 >> x2) {
      Y.push_back(y);
      X.push_back({x1, x2});
    }
    Options options;
    options.stopIfZeroError = false;
    options.numEpoch = 1000000000;
    options.randomInit = false;
    options.miniBatchSize = 100;
    options.minThetaDiffNorm = 1e-15;
    options.learningRate = 10;
    options.momentumMultiplier = 0;
    options.lrDecay = 1;
    options.L2 = 0;
    auto theta = fitLR(X, Y, options);
    {
      std::ofstream f(std::string(name) + ".lr");
      for (auto th : theta) {
        f << th << " ";
      }
      f << std::endl;
      for (int i = 0; i < Y.size(); ++i) {
        f << X[i][0] << " " << X[i][1] << " " << Y[i] << std::endl;
      }
    }
  }
  return 0;
}
