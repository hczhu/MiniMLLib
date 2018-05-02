#pragma once

#include <exception>
#include <vector>

namespace mlight {

class NoImpExp : public std::exception {};

class FeatureBatch {
 public:
  using Mat = std::vector<std::vector<double>>;
  explicit FeatureBatch(int n);
  bool hasLabels() const {
    return false;
  }

  int numInstances() const {
    return n_;
  }
  
  virtual std::vector<int> getLabels() const {
    throw NoImpExp();
  }

  virtual std::vector<double> getLabelValues() const {
    throw NoImpExp();
  }

  // Every column is an instance.
  virtual Mat getDenseFeatures() const {
    throw NoImpExp();
  }

  // Return np.matmul(mat, w).
  // 'mat' is the the feature matrix represented by this class.
  virtual std::vector<double> dot(const std::vector<double>& w) const {
    throw NoImpExp();
  }

  // Return np.matmul(mat.T, w).
  // 'mat' is the the feature matrix represented by this class.
  // Used for prediction.
  virtual std::vector<double> tdot(const std::vector<double>& w) const {
    throw NoImpExp();
  }

 private:
  int n_ = 0;
};

}  // namespace mlight
