#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <armadillo>

TEST(MatrixTest, Basic) {
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
  arma::mat iA = arma::inv(A);
  LOG(INFO) << "A':\n" << iA;
  LOG(INFO) << "A * A':\n" << A * iA;
  EXPECT_LT(arma::norm(((A * iA) - arma::eye<arma::mat>(n, n)), 2), 1e-9);

  arma::mat A1 = 1 - iA;
  for (int i = 0; i < A1.n_rows; ++i) {
    for (int j = 0; j < A1.n_cols; ++j) {
      EXPECT_NEAR(A1(i, j), 1 - iA(i, j), 1e-10);
    }
  }

  arma::mat A2 = 0.5 * iA;
  CHECK_EQ(A2.n_cols, iA.n_cols);
  CHECK_EQ(A2.n_rows, iA.n_rows);
  for (int i = 0; i < A2.n_rows; ++i) {
    for (int j = 0; j < A2.n_cols; ++j) {
      EXPECT_NEAR(A2(i, j), 0.5 * iA(i, j), 1e-10);
    }
  }

  arma::vec v1 = {1.0, 2.0, 3.0};
  arma::vec v2 = 0.5 * v1;
  for (int i = 0; i < v2.size(); ++i) {
    EXPECT_NEAR(v1(i), v2(i), 1e-10);
  }
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}

