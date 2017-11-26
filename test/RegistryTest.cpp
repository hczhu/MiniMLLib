#include <functional>

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "Registry.h"

template<typename T>
class _DisplayType;

template<typename T>
void _displayType(T&& t);

#define PEEK(x) LOG(INFO) << #x << ": [" << (x) << "]"

DECLARE_REGISTRY(MathFunc, std::string, std::function<int(int)>)
REGISTER(MathFunc, "square", [](int a) -> int {
  return a * a;
});
REGISTER(MathFunc, "inc", [](int a) -> int {
  return a + 1;
});
REGISTER(MathFunc, "dec", [](int a) -> int {
  return a - 1;
});

TEST(RegistryTest, Simple) {
  EXPECT_EQ(100, REGISTRY(MathFunc)["square"](10));
  EXPECT_EQ(11, REGISTRY(MathFunc)["inc"](10));
  EXPECT_EQ(9, REGISTRY(MathFunc)["dec"](10));
}

int main(int argc, char* argv[]) {
  folly::init(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}

