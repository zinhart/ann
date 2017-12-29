#include "ann/activation.hh"
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <memory>
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
TEST(activation_test, call_activation_identity)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max());
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  std::shared_ptr<double> activation;
  std::shared_ptr<double> activation_copy;
  double * device_activations;
  cudaError_t error_id;
  std::uint16_t activation_size = Z_plus(mt);
  activation = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
  activation_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );//will store the results of call activation
  error_id = cudaMalloc( (void **) &device_activations, activation_size* sizeof(double));
  if(error_id != cudaSuccess)
	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  for(std::int16_t i = 0; i < activation_size; ++i)
  {	
	activation.get()[i] = real(mt);
	activation_copy.get()[i] = activation.get()[i];
  }
  ASSERT_EQ(call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::OBJECTIVE, activation_copy.get(), activation_size), 0);
  for(std::int16_t i = 0; i < activation_size; ++i)
  {
	//ASSERT_EQ(activation.get()[i],activation_copy.get()[i]);
  }/**/
  error_id = cudaFree(device_activations);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}
/*TEST(Layer_Test, call_activation_softmax)
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
