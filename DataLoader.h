#pragma once

#include <memory>

#include "Data.h"
#include "Registry.h"

namespace mlight {

class DataLoader {
 public:
  class NoMoreDataExp : public std::exception {};

  struct Config {
    std::string name = "";
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

  virtual ~DataLoader();

 protected:
  explicit DataLoader(Config config);

 private: 
  Config config_;
};

std::unique_ptr<DataLoader> getDataLoader(DataLoader::Config config);

}  // namespace mlight

DECLARE_REGISTRY(
    DataLoaderReg, std::string,
    std::function<std::unique_ptr<DataLoader>(DataLoader::Config)>);

#define REGISTER_DATA_LOADER(name, className) \
  REGISTER(DataLoaderReg, name, [](DataLoader::Config config) { \
    return std::make_unique<className>(std::move(config)); \
  })

#define DATA_LOADER_REGISTRY REGISTRY(DataLoaderReg)
