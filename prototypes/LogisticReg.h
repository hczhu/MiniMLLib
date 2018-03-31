#pragma once

#include <vector>

namespace mini_ml {

struct Options {
  double L2 = 0;
  int numEpoch = 10000000;
  double learningRate = 1;
  // Decay every round.
  double lrDecay = 0.999;
  bool randomInit = false;
  bool useNewton = false;
  int miniBatchSize = 100;
  double momentumMultiplier = 0.5;
  double minThetaDiffNorm = 1e-5;
  bool chooseBestLoglossTheta = true;
  bool chooseBestErrorRateTheta = true;
  bool stopIfZeroError = false;
};

enum class ResCode : int {
  CONVERGED = 0,
  NOT_CONVERGED = 1,
  EARLY_TERM = 2,
};

// Minimize log-loss.
std::vector<double> fitLR(const std::vector<std::vector<double>>& X,
                          const std::vector<int>& Y,
                          Options options = {},
                          const std::vector<double>& W = {});

}  // namespace mini_ml
