#pragma once

#include <set>
#include <vector>

#include <folly/Optional.h>
#include <glog/logging.h>

namespace mini_ml {

// Read-only instance
class DataInstanceBase {
 public:
  using Fid = uint64_t;
  DataInstanceBase() = default;
  virtual ~DataInstanceBase() = default;

  virtual folly::Optional<double> getFeatureDouble(Fid fid) const {
    LOG(FATAL) << "Not implemented!";
  }
  virtual std::vector<std::pair<Fid, double>> allFeaturesDouble() const {
    LOG(FATAL) << "Not implemented!";
  }
  virtual std::vector<Fid> allFeatureIds() const {
    LOG(FATAL) << "Not implemented!";
  }
};

template<typename FID>
class SparseFeatureIns : public DataInstanceBase {
 public:
  using DataInstanceBase::Fid;
  SparseFeatureIns() = default;
  bool addFeature(FID fid) {
    return features_.insert(fid).second;
  }
  std::vector<Fid> allFeatureIds() const override {
    return std::vector<Fid>{features_.begin(), features_.end()};
  }

 private:
  std::set<FID> features_;
};


}  // namespace mini_ml
