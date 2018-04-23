#pragma once

#include <vector>

namespace mlight {

// Minimize | W * (X * theta + intercept - Y)|^2 + 0.5 * l2 * |theta|^2
std::vector<double> fitLSM(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& Y, double L2 = 0,
                           const std::vector<double>& W = {});

}  // namespace mlight
