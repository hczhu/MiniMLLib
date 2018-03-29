#pragma once

#include <vector>

namespace mini_ml {

struct Options {
  double L2 = 0;
  int numEpoch = 1000;
  double learningRate = 1;
  // Decay every round.
  double lrDecay = 0.999;
  bool randomInit = false;
  bool useNewton = false;
  int miniBatchSize = 32;
  double momentumMultiplier = 0.9;
  double minThetaDiffNorm = 1e-2;
  bool chooseBestLoglossTheta = true;
  bool chooseBestErrorRateTheta = true;
};

// Minimize log-loss.
std::vector<double> fitLR(const std::vector<std::vector<double>>& X,
                          const std::vector<int>& Y,
                          Options options = {},
                          const std::vector<double>& W = {});

}  // namespace mini_ml
