#pragma once

#include <set>
#include <vector>

#include <folly/Optional.h>
#include <glog/logging.h>

namespace mini_ml {

// Read-only instance
class DataInstanceBase {
 public:
  // TODO: add value and pair iterators.
  using Fid = uint64_t;
  DataInstanceBase() = default;
  virtual ~DataInstanceBase() = default;

  virtual folly::Optional<double> getFeatureDouble(Fid fid) const {
    LOG(FATAL) << "Not implemented!";
  }
  virtual double getFeatureDouble(int idx) const {
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

template<typename Value>
class DenseFeatureIns : public DataInstanceBase {
 public:
  using ValueType = Value;
  using DataInstanceBase::Fid;

  DenseFeatureIns(size_t dim, Value defaultValue = static_cast<Value>(0))
    : values_(dim, defaultValue) {}

  double getFeatureDouble(int idx) const override {
    return values_[idx];
  }
  Value getFeature(int idx) const {
    return values_[idx];
  }
  size_t dimention() const {
    return values_.size();
  }
  bool setFeature(size_t idx, Value v) {
    CHECK_LT(idx, values_.size());
    values_[idx] = v;
    return true;
  }

 private:
  std::vector<ValueType> values_;
};




}  // namespace mini_ml
