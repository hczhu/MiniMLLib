#pragma once

#include <vector>

namespace mini_ml {

struct Options {
  double L2 = 0;
  int maxItr = 100000;
  double learningRate = 5;
  // Decay every round.
  double lrDecay = 0.999;
  double exitThetaDelta = 1e-5;
  bool randomInit = true;
  bool useNewton = false;
};

// Minimize log-loss.
std::vector<double> fitLR(const std::vector<std::vector<double>>& X,
                          const std::vector<int>& Y,
                          Options options = {},
                          const std::vector<double>& W = {});

}  // namespace mini_ml
