#pragma once

#include <exception>
#include <vector>

namespace mlight {

class NoImpExp : public std::exception {};

// Read-only feature batch
class FeatureBatch {
 public:
  using Mat = std::vector<std::vector<double>>;
  explicit FeatureBatch(int n);
  virtual ~FeatureBatch();
  bool hasLabels() const {
    return false;
  }

  int numInstances() const {
    return n_;
  }
  
  virtual std::vector<int> getLabels() const {
    throw NoImpExp();
  }

  virtual std::vector<float> getLabelValues() const {
    throw NoImpExp();
  }

  // Return np.matmul(mat, w).
  // 'mat' is the the feature matrix represented by this class.
  // Each column is an instance.
  virtual std::vector<double> dot(const std::vector<double>& w) const {
    throw NoImpExp();
  }

  // Return np.matmul(mat.T, w).
  // 'mat' is the the feature matrix represented by this class.
  // Used for prediction.
  virtual std::vector<double> tdot(const std::vector<double>& w) const {
    throw NoImpExp();
  }

  // return np.matmul(W, mat)
  virtual Mat matmul(const Mat& W) const {
    throw NoImpExp();
  }

 private:
  int n_ = 0;
};

}  // namespace mlight
