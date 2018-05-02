#pragma once

#include "Data.h"

namespace mlight {

class DataLoader {
 public:
  class NoMoreDataExp : public std::exception {};

  struct Config {
    int nFeatures = 0;
    int nInstances = 0;
    bool hasLabels = false;
    int batchSize = 256;
    size_t maxMemMbPerBatch = 32;
    std::string filename = "";
  };

  virtual const FeatureBatch& currentBatch() const = 0;
  virtual bool hasNextBatch() const = 0;
  virtual const FeatureBatch& nextBatch() = 0;

  // Start over from the first data batch.
  virtual void rewind() = 0;

 protected:
  explicit DataLoader(const Config& config);

 private: 
  Config config_;
};

}  // namespace mlight
