#include "prototypes/LogisticReg.h"

#include <glog/logging.h>
#include <armadillo>

namespace mini_ml {

std::vector<double> fitLR(const std::vector<std::vector<double>>& X,
                          const std::vector<int>& Y,
                          Options options,
                          const std::vector<double>& W) {
  for (auto y : Y) {
    CHECK(y == -1 || y == 1) << "y can't  must be -1 or 1: " << y;
  }
  auto sigmod = [](double z) {
    return 1.0 /(1 + exp(-z));
  };
  const int n = X.size();
  const int m = X[0].size();
  arma::vec theta(m + 1, arma::fill::randu);
  if (!options.randomInit) {
    theta.fill(0);
  }
  arma::Mat<double> X1(n, m + 1);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      X1(i, j) = X[i][j];
    }
    X1(i, m) = 1;
  }
  arma::Col<int> vY(Y);
  auto predProb = [&] {
    arma::vec z = (X1 * theta) % vY;
    z.for_each([&](double& val) {
      val = sigmod(val);
    });
    return z;
  };
  auto logloss = [&] {
    return -arma::sum(arma::log(predProb()));
  };
  LOG(INFO) << "Initial log-loss = " << logloss();
  for (int itr = 0; itr < options.maxItr; ++itr) {
    auto probs = predProb();
    arma::vec dtheta = (((1.0 - probs) % vY).t() * X1).t();
    if (options.useNewton) {

    } else {
      dtheta *= options.learningRate;
      options.learningRate *= options.lrDecay;
    }
    auto maxDeltaRatio = arma::max(arma::abs(dtheta)) /
                         std::max(1.0, arma::max(arma::abs(theta)));
    theta += dtheta;
    LOG_EVERY_N(INFO, 10) << "Iteration #" << itr
                          << " learning rate = " << options.learningRate
                          << " theta update ratio max = " << maxDeltaRatio
                          << " log-loss = " << -arma::sum(arma::log(probs));
    if (maxDeltaRatio < options.exitThetaDeltaRatio) {
      LOG(INFO) << "The update " << maxDeltaRatio << " is too small."
                << " Finishing training.";
      break;
    }
  }
  return arma::conv_to<std::vector<double>>::from(theta);
}

}  // namespace mini_ml
