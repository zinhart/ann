#include <gtest/gtest.h>
#include "ann/loss_function.hh"
#include <random>
#include <limits>
using namespace zinhart::error_metrics;
const unsigned int STOP = 100;

TEST(loss_function_test, cross_entropy_multi_class_objective)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double kth_target, kth_output, epsilon = 1.e-30; 
  zinhart::error_metrics::loss_function loss;
  kth_target = dist(mt), kth_output = dist(mt);
  ASSERT_EQ(  kth_target * log(kth_output + epsilon), loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, LOSS_FUNCTION_TYPE::OBJECTIVE, kth_output, kth_target, epsilon));
}

TEST(loss_function_test, cross_entropy_multi_class_derivative)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double kth_target, kth_output, epsilon; 
  zinhart::error_metrics::loss_function loss;
  kth_target = dist(mt), kth_output = dist(mt);
  ASSERT_EQ(kth_output - kth_target , loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, LOSS_FUNCTION_TYPE::DERIVATIVE, kth_output, kth_target, epsilon) );
}

TEST(loss_function_test, mean_square_error_objective)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double kth_target, kth_output;
  zinhart::error_metrics::loss_function loss;
  kth_target = dist(mt), kth_output = dist(mt);
  ASSERT_EQ( (kth_output - kth_target) * (kth_output - kth_target), loss(LOSS_FUNCTION_NAME::MSE, LOSS_FUNCTION_TYPE::OBJECTIVE, kth_output, kth_target) );
}
TEST(loss_function_test, mean_square_error_derivative)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double kth_target, kth_output;
  zinhart::error_metrics::loss_function loss;
  kth_target = dist(mt), kth_output = dist(mt);
  ASSERT_EQ( double (2.0) * (kth_output - kth_target), loss(LOSS_FUNCTION_NAME::MSE, LOSS_FUNCTION_TYPE::DERIVATIVE, kth_output, kth_target));
}

