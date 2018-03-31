#include "prototypes/LogisticReg.h"

#include <armadillo>
#include <folly/gen/Base.h>
#include <folly/gen/String.h>
#include <glog/logging.h>

DEFINE_int32(log_every_n, 10, "Log logloss every N epoch.");
DEFINE_int32(num_epoch, 0, "Number of epochs");

namespace mini_ml {

std::vector<double> fitLR(const std::vector<std::vector<double>>& X,
                          const std::vector<int>& Y,
                          Options options,
                          const std::vector<double>& W) {
  for (auto y : Y) {
    CHECK(y == -1 || y == 1) << "y can't  must be -1 or 1: " << y;
  }
  auto sigmod = [](double z) {
    constexpr int kCutoff = 100;
    constexpr double eps = 1e-50;
    if (abs(z) > kCutoff) {
      return z > 0 ? (1.0 - eps) : eps;
    }
    return 1.0 /(1 + exp(-z));
  };
  const int n = options.miniBatchSize;
  const int m = X[0].size();
  arma::vec theta(m + 1, arma::fill::randn);
  if (!options.randomInit) {
    theta.fill(0);
  }
  arma::Mat<double> X1(n, m + 1, arma::fill::ones);
  arma::Col<int> vY(options.miniBatchSize);
  auto predProb = [&] {
    // Return P{ y = vY[i] | X1[i] }
    // The distance between the origin and the projected point of 'X1[i]'
    // on the vector of 'theta'.
    arma::vec margin = (X1 * theta) % vY;
    margin.for_each([&](double& val) {
      val = sigmod(val);
    });
    return margin;
  };
  auto loglossAndError = [&]() -> std::pair<double, int> {
    double res = 0;
    const arma::vec theta1 = theta(arma::span(0, m - 1));
    int error = 0;
    for (int i = 0; i < X.size(); ++i) {
      auto margin = Y[i] * (arma::dot(arma::vec(X[i]), theta1) + theta(m));
      if (margin < 0) {
        ++error;
      }
      res += log(sigmod(margin));
    }
    return {-res / X.size() +
                0.5 * options.L2 * arma::norm(theta1) * arma::norm(theta1),
            error};
  };
  auto checkGrad = [&] (const arma::vec& dtheta) {
    constexpr double eps = 1e-6;
    auto ll = loglossAndError().first;
    for (int idx = 0; idx < m + 1; ++idx) {
      theta(idx) += eps;
      auto di = (loglossAndError().first - ll) / eps;
      CHECK_NEAR(di, dtheta(idx), 1e-10);
    }
    return "";
  };
  int numBatches =
      (X.size() + options.miniBatchSize - 1) / options.miniBatchSize;
  decltype(theta) momentum = theta * 0;
  double best = 1e20;
  decltype(theta) bestTheta = theta;
  int bestEpoch = 0;
  if (options.useNewton) {
    CHECK_EQ(1, numBatches);
  }
  if (FLAGS_num_epoch > 0) {
    options.numEpoch = FLAGS_num_epoch;
  }
  for (int epoch = 0; epoch < options.numEpoch; ++epoch) {
    auto prevTheta = theta;
    for (int batch = 0; batch < numBatches; ++batch) {
      // Prepare data for this batch
      if (epoch == 0 || numBatches > 1) {
        for (int i = 0; i < n; ++i) {
          int instanceId = (n * batch + i) % Y.size();
          vY(i) = Y[instanceId];
          for (int j = 0; j < m; ++j) {
            X1(i, j) = X[instanceId][j];
          }
        }
      }
      auto probs = predProb();
      // Minimize log-loss:
      //  -Sigma(log P{ vY[i] | X1[i] }) / n
      //   + L2 / 2 * norm(theta) * norm(theta)
      arma::vec dtheta =
          ((((probs - 1.0) % vY).t() * X1).t() / n) + (options.L2 * theta);
      using namespace folly::gen;
      VLOG(1) << (from(arma::conv_to<std::vector<double>>::from(dtheta)) |
                  unsplit(','))
              << checkGrad(dtheta);
      if (options.useNewton) {
        arma::mat H(m + 1, m + 1, arma::fill::zeros);
        for (int i = 0; i < n; ++i) {
          H += (X1.row(i).t() * X1.row(i)) * (probs(i) * (1 - probs(i))) / n;
        }
        for (int i = 0; i < m + 1; ++i) {
          H(i, i) += options.L2;
        }
        dtheta = arma::inv(H) * dtheta;
      } else {
        momentum = momentum * options.momentumMultiplier +
                   dtheta * (1 - options.momentumMultiplier);
        dtheta = momentum * options.learningRate;
      }
      theta -= dtheta;
    }
    double ll;
    int er;
    std::tie(ll, er) = loglossAndError();
    if (options.chooseBestErrorRateTheta) {
      if (er <= best) {
        best = er;
        bestTheta = theta;
        bestEpoch = epoch;
      }
    } else if (ll <= best) {
      best = ll;
      bestTheta = theta;
      bestEpoch = epoch;
    }
    LOG_EVERY_N(INFO, FLAGS_log_every_n)
        << "Epoch #" << epoch << " learning rate = " << options.learningRate
        << " theta diff norm = " << arma::norm(prevTheta - theta)
        << " theta norm = " << arma::norm(theta)
        << " logloss (with L2=" << options.L2 << "): " << ll
        << " error rate = " << er;
    if (arma::norm(prevTheta - theta) < options.minThetaDiffNorm) {
      LOG(INFO) << "Exiting earlier due to that the theta update is too small "
                   "in the last epoch.";
      break;
    }
    if (er == 0 && options.stopIfZeroError) {
      LOG(INFO) << "Training data got classified perfectly. Exiting...";
      break;
    }
    prevTheta = theta;
    options.learningRate *= options.lrDecay;
  }
  LOG(INFO) << "Best "
            << (options.chooseBestErrorRateTheta ? "#error = " : "logloss = ")
            << best << " at epoch #" << bestEpoch;
  return arma::conv_to<std::vector<double>>::from(
      (options.chooseBestLoglossTheta || options.chooseBestErrorRateTheta)
          ? bestTheta
          : theta);
}

}  // namespace mini_ml
