#include "prototypes/LinearReg.h"

#include <armadillo>
#include <glog/logging.h>

namespace mini_ml {

std::vector<double> fitLSM(const std::vector<std::vector<double>> &X,
                           const std::vector<double> &Y, double L2) {
  const int n = X.size();
  const int m = X[0].size();
  CHECK_EQ(n, Y.size());
  arma::Mat<double> X1(n, m + 1);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      X1(i, j) = X[i][j];
    }
    X1(i, m) = 1;
  }
  decltype(X1) Xt = X1.t();
  decltype(X1) X2 = Xt * X1;
  CHECK_EQ(X2.n_rows, m + 1);
  CHECK_EQ(X2.n_cols, m + 1);
  for (int i = 0; i < m; ++i) {
    X2(i, i) += L2;
  }
  arma::vec Yv(Y);
  arma::vec theta = arma::inv(X2) * (Xt * Yv);
  // LOG(INFO) << "Theta := " << theta;
  auto norm = arma::norm(X1 * theta - Yv);
  auto thetaNorm = arma::norm(theta(arma::span(0, m - 1)));
  LOG(INFO) << "Squared error with L2 (" << L2
            << "):= " << norm * norm << " theta square = "
            << thetaNorm * thetaNorm;
  return arma::conv_to<std::vector<double>>::from(theta);
}

}  // namespace mini_ml
