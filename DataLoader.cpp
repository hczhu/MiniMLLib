#include "DataLoader.h"
#include "Registry.h"

namespace mlight {

DataLoader::~DataLoader() = default;

namespace {

template <typename V>
class CsvFileLoader : public DataLoader {

};


}

std::unique_ptr<DataLoader> getDataLoader(DataLoader::Config config) {
  auto itr = DATA_LOADER_REGISTRY.find(config.name);
  if (itr == DATA_LOADER_REGISTRY.end()) {
    return nullptr;
  }
  return itr->second(std::move(config));
}


}  // namespace mlight
