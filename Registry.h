#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include <glog/logging.h>

namespace mlight {

template <typename K, typename V>
struct Registry : std::unordered_map<K, V> {
  void registerValue(const K& key, const V& value, const std::string &name) {
    LOG_IF(FATAL, this->count(key)) << name << " is already registered.";
    this->emplace(std::move(key), std::move(value));
  }
  static Registry& get() {
    static Registry registry;
    return registry;
  }
};

struct InitFunc {
  InitFunc(std::function<void()> func) {
    func();
  }
};

} // namespace mlight

#define DECLARE_REGISTRY(name, keyType, valueType) \
  namespace mlight { namespace registries { \
    struct name##Registry##T : public mlight::Registry<keyType, valueType> {}; \
  }}

#define REGISTRY(name) \
    (::mlight::registries::name##Registry##T::get())

#define CONCAT(a, b) a##b
#define CONCAT1(a, b) CONCAT(a, b)
#define APPEND_LINE_NO(name) CONCAT1(name, __LINE__)

#define REGISTER(name, key, value) \
  namespace { \
    static mlight::InitFunc APPEND_LINE_NO(name)([&] { \
      REGISTRY(name).registerValue(key, value, #name ":" #key); \
    }); \
  }
