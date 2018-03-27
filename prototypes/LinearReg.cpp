#include "prototypes/LinearReg.h"

#include <armadillo>
#include <glog/logging.h>

namespace mini_ml {

std::vector<double> fitLSM(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& Y, double L2,
                           const std::vector<double>& W) {
  const int n = X.size();
  const int m = X[0].size();
  CHECK_EQ(n, Y.size());
  // X' * diag(W) * X
  // 'm + 1' for the intercept.
  arma::Mat<double> XtWX(m + 1, m + 1, arma::fill::zeros);
  // TODO: Use OpenMP
  for (int i = 0; i < n; ++i) {
    const double w = W.empty() ? 1 : W[i];
    const auto& x = X[i];
    for (int j = 0; j < m; ++j) {
      for (int k = j; k < m; ++k) {
        XtWX(j, k) += w * x[j] * x[k];
      }
      XtWX(j, m) += w * x[j];
    }
    XtWX(m, m) += w;
  }
  for (int j = 0; j <= m; ++j) {
    for (int k = 0; k < j; ++k) {
      XtWX(j, k) = XtWX(k, j);
    }
  }
  for (int i = 0; i < m; ++i) {
    XtWX(i, i) += L2;
  }
  arma::vec Yv(Y);
  if (!W.empty()) {
    for (int i = 0; i < n; ++i) {
      Yv(i) *= W[i];
    }
  }
  arma::mat Xt(m + 1, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      Xt(j, i) = X[i][j];
    }
    Xt(m, i) = 1;
  }
  arma::vec theta = arma::inv(XtWX) * (Xt * Yv);
  // LOG(INFO) << "Theta = " << theta;
  arma::vec error = (theta.t() * Xt - arma::vec(Y).t()).t();
  double sqError = 0;
  for (int i = 0; i < n; ++i) {
    sqError += (W.empty() ? 1 : W[i]) * error(i) * error(i);
  }
  auto thetaNorm = arma::norm(theta(arma::span(0, m - 1)));
  LOG(INFO) << "Squared error with L2 (" << L2 << "):= " << sqError
            << " theta square = " << thetaNorm * thetaNorm;
  return arma::conv_to<std::vector<double>>::from(theta);
}

}  // namespace mini_ml
