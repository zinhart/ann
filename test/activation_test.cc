#include "ann/activation.hh"
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
using namespace zinhart;

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> pos_real(0, std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> reals(std::numeric_limits<double>::min(), std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> neg_real(std::numeric_limits<double>::min(), -1 );
/*
 * CUDA WRAPPERS
 * */
/*
 * ACTIVATION OBJECTIVE
 * */
/*TEST(Layer_Test, call_activation_identity)
{
  Layer L;
  double x = reals(mt);
  ASSERT_EQ(x, call_activation(L, x, LAYER_NAME::IDENTITY, ACTIVATION::OBJECTIVE));
}
TEST(Layer_Test, call_activation_softmax)
{
  Layer L;
  double x = reals(mt);
  ASSERT_EQ(std::exp(x), call_activation(L, x, LAYER_NAME::SOFTMAX, ACTIVATION::OBJECTIVE));
}
TEST(Layer_Test, call_activation_tanh)
{
  Layer L;
  double x = reals(mt);
  ASSERT_EQ(std::tanh(-x), call_activation(L, x, LAYER_NAME::TANH, ACTIVATION::OBJECTIVE));
}
TEST(Layer_Test, call_activation_relu)
{
  Layer L;
  double x = pos_real(mt);
  ASSERT_EQ(x, call_activation(L, x, LAYER_NAME::RELU, ACTIVATION::OBJECTIVE));
  x = neg_real(mt);
  ASSERT_EQ(0, call_activation(L, x, LAYER_NAME::RELU, ACTIVATION::OBJECTIVE));
}
TEST(Layer_Test, call_activation_leaky_relu)
{
  Layer L;
  double x = pos_real(mt);
  double leakage_coefficient = 0.1;
  ASSERT_EQ(x, call_activation(L, x, leakage_coefficient,LAYER_NAME::LEAKY_RELU, ACTIVATION::OBJECTIVE));
  x = neg_real(mt);
  ASSERT_EQ((x * leakage_coefficient), call_activation(L, x, leakage_coefficient,LAYER_NAME::LEAKY_RELU, ACTIVATION::OBJECTIVE));
}
 // ACTIVATION DERIVATIVE
TEST(Layer_Test, call_activation_identity_derivative)
{
  Layer L;
  double x = reals(mt);
  ASSERT_EQ(1.0, call_activation(L, x, LAYER_NAME::IDENTITY, ACTIVATION::DERIVATIVE));
}
TEST(Layer_Test, call_activation_softmax_derivative)
{
  Layer L;
  double x = reals(mt);
  ASSERT_EQ( x * (1.0 - x ), call_activation(L, x, LAYER_NAME::SOFTMAX, ACTIVATION::DERIVATIVE));
}
TEST(Layer_Test, call_activation_tanh_derivative)
{
  Layer L;
  double x = reals(mt);
  ASSERT_EQ(1.0 - (x*x), call_activation(L, x, LAYER_NAME::TANH, ACTIVATION::DERIVATIVE));
}
TEST(Layer_Test, call_activation_relu_derivative)
{
  Layer L;
  double x = pos_real(mt);
  ASSERT_EQ(1.0, call_activation(L, x, LAYER_NAME::RELU, ACTIVATION::DERIVATIVE));
  x = neg_real(mt);
  ASSERT_EQ(0, call_activation(L, x, LAYER_NAME::RELU, ACTIVATION::DERIVATIVE));
}
TEST(Layer_Test, call_activation_leaky_relu_derivative)
{
  Layer L;
  double x = pos_real(mt);
  double leakage_coefficient = 0.1;
  ASSERT_EQ(1.0, call_activation(L, x, leakage_coefficient,LAYER_NAME::LEAKY_RELU, ACTIVATION::DERIVATIVE));
  x = neg_real(mt);
  ASSERT_EQ(leakage_coefficient, call_activation(L, x, leakage_coefficient,LAYER_NAME::LEAKY_RELU, ACTIVATION::DERIVATIVE));
}*/
