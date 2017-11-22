#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <armadillo>

TEST(MatrixTest, MatrixInverse) {
  const int n = 4;
  arma::Mat<double> A(4, 4);
  for (int i = 1; i <= n; ++i ) {
    int ele = 1;
    for (int j = 0; j < n; ++j, ele *= i) {
      A(i - 1, j) = ele;
    }
  }
  LOG(INFO) << "A:\n" << A;
  LOG(INFO) << "det(A) = " << arma::det(A);
  EXPECT_LT(std::abs(arma::det(A) - 12.0), 1e-6);
  auto iA = arma::inv(A);
  LOG(INFO) << "A':\n" << iA;
  LOG(INFO) << "A * A':\n" << A * iA;
  EXPECT_LT(arma::norm(((A * iA) - arma::eye<arma::mat>(n, n)), 2), 1e-9);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}

