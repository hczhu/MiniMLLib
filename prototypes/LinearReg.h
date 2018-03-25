#pragma once

#include <vector>

#include <armadillo>
#include <glog/logging.h>

namespace mini_ml {

// Minimize |(X * theta + intercept - Y)|^2 + 0.5 * l2 * |theta|^2
std::vector<double> fitLSM(const std::vector<std::vector<double>> &X,
                           const std::vector<double> &Y, double l2 = 0);

}  // namespace mini_ml
