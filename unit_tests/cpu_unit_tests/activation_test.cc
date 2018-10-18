#include <ann/ann.hh>
#include <ann/layer.hh>
#include <gtest/gtest.h>
#include <random>
#include <utility>
#include <limits>
#include <type_traits>
#include <algorithm>

using namespace zinhart::models::layers;
using namespace zinhart::function_space;

TEST(activation_test, identity_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::identity_layer(), objective(), layer_activations_ptr, layer_length, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::identity_layer(), *(layer_activations_ptr_test + i) + bias);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::identity_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::identity_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::identity_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::identity_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, sigmoid_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::sigmoid_layer(), objective(), layer_activations_ptr, layer_length, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i) + bias);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::sigmoid_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::sigmoid_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, softplus_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::softplus_layer(), objective(), layer_activations_ptr, layer_length, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::softplus_layer(), *(layer_activations_ptr_test + i) + bias);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::softplus_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::softplus_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::softplus_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::softplus_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, tanh_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::tanh_layer(), objective(), layer_activations_ptr, layer_length, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::tanh_layer(), *(layer_activations_ptr_test + i) + bias);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::tanh_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::tanh_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::tanh_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::tanh_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, relu_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::relu_layer(), objective(), layer_activations_ptr, layer_length, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::relu_layer(), *(layer_activations_ptr_test + i) + bias);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::relu_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::relu_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr,layer_length);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, leaky_relu_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double coefficient{real_dist(mt)};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::leaky_relu_layer(), objective(), layer_activations_ptr, layer_length, coefficient, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::leaky_relu_layer(), *(layer_activations_ptr_test + i) + bias, coefficient);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::leaky_relu_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr, layer_length, coefficient);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::leaky_relu_layer(), *(layer_activations_ptr_test + i), coefficient);

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::leaky_relu_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr, layer_length, coefficient);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::leaky_relu_layer(), *(layer_activations_ptr_test + i), coefficient);

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, exp_leaky_relu_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double coefficient{real_dist(mt)};
  double bias{real_dist(mt)};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
  }

  // perform activation
  a.activate(layer_info::exp_leaky_relu_layer(), objective(), layer_activations_ptr, layer_length, coefficient, bias);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::exp_leaky_relu_layer(), *(layer_activations_ptr_test + i) + bias, coefficient);

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative for output_layer
  a.activate(layer_info::exp_leaky_relu_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, error_ptr, layer_activations_ptr, layer_length, coefficient);

  // perform activation derivative test for output_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = error_test_ptr[i] * a.derivative(layer_info::exp_leaky_relu_layer(), *(layer_activations_ptr_test + i), coefficient);

  // validate activation derivative for output_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // perform activation derivative for hidden_layer
  a.activate(layer_info::exp_leaky_relu_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, layer_activations_ptr, layer_length, coefficient);

  // perform activation derivative test for hidden_layer
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) *= a.derivative(layer_info::exp_leaky_relu_layer(), *(layer_activations_ptr_test + i), coefficient);

  // validate activation derivative for hidden_layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
}

TEST(activation_test, softmax_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(2, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};
  double * error_ptr{nullptr};
  double * error_test_ptr{nullptr};
  double * jacobian_ptr{nullptr};
  double * jacobian_test_ptr{nullptr};
  double bias{real_dist(mt)};
  double sum{0.0};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  error_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  jacobian_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  jacobian_test_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );


  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
	error_ptr[i] = real_dist(mt);
	error_test_ptr[i] = error_ptr[i];
	jacobian_ptr[i] = 0;
	jacobian_test_ptr[i] = 0;
  }

  // perform activation
  a.activate(layer_info::softmax_layer(), objective(), layer_activations_ptr, layer_length, bias);

  // perform activation test
  double max = *std::max_element(layer_activations_ptr_test, layer_activations_ptr_test + layer_length);
  for(i = 0; i < layer_length; ++i)
  {
	*(layer_activations_ptr_test + i) = std::exp( *(layer_activations_ptr_test + i) + bias - max );
	sum += *(layer_activations_ptr_test + i);
  }

  for(i = 0; i < layer_length; ++i)
  {
	*(layer_activations_ptr_test + i) /= sum;
  }

  // validate activation
  for(i = 0; i < layer_length; ++i)
  {
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );
  }

  // perform activation derivative for output layer
  a.activate(layer_info::softmax_layer(), layer_info::output_layer(), derivative(), layer_deltas_ptr, jacobian_ptr, error_ptr, layer_activations_ptr, layer_length);

  // perform activation test for output layer
  for(i = 0; i < layer_length; ++i)
  {
  	for(j = 0; j < layer_length; ++j)
	  *(jacobian_test_ptr + j) = (j == i) ? *(layer_activations_ptr_test + i) * (double{1.0} - *(layer_activations_ptr_test + i)) : -*(layer_activations_ptr_test + j) * *(layer_activations_ptr_test + i);
   
	sum = 0;
  	for(j = 0; j < layer_length; ++j)
  	  sum += jacobian_test_ptr[j] * layer_activations_ptr_test[j];
	
	*(layer_deltas_ptr_test + i ) = *(error_test_ptr + i) * sum;
  }

  // validate activation derivative for output layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );


  // perform activation derivative for hidden layer
  a.activate(layer_info::softmax_layer(), layer_info::hidden_layer(), derivative(), layer_deltas_ptr, jacobian_ptr, layer_activations_ptr, layer_length);

  // perform activation test for hidden layer
  for(i = 0; i < layer_length; ++i)
  {
  	for(j = 0; j < layer_length; ++j)
	  *(jacobian_test_ptr + j) = (j == i) ? *(layer_activations_ptr_test + i) * (double{1.0} - *(layer_activations_ptr_test + i)) : -*(layer_activations_ptr_test + j) * *(layer_activations_ptr_test + i);
   
	sum = 0;
  	for(j = 0; j < layer_length; ++j)
  	  sum += jacobian_test_ptr[j] * layer_activations_ptr_test[j];
	
	*(layer_deltas_ptr_test + i ) *= sum;
  }

  // validate activation derivative for hidden layer
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
  mkl_free(error_ptr);
  mkl_free(error_test_ptr);
  mkl_free(jacobian_ptr);
  mkl_free(jacobian_test_ptr);
}
/*
TEST(activation_test, batch_norm_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 100);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector legnths loop counters
  const std::uint32_t alignment{64};
  std::uint32_t layer_length{neuron_dist(mt)}, i{0}, j{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};

  // an activation object
  activation<double> a;

  // allocate vectors
  layer_activations_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_activations_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );
  layer_deltas_ptr_test = (double*) mkl_malloc( layer_length * sizeof( double ), alignment );

  // initialize vectors
  for(i = 0; i < layer_length; ++i)
  {
	layer_activations_ptr[i] = real_dist(mt);
	layer_activations_ptr_test[i] = layer_activations_ptr[i];
	layer_deltas_ptr[i] = real_dist(mt);
	layer_deltas_ptr_test[i] = layer_deltas_ptr[i];
  }

  // perform activation
  a.activate(layer_info::sigmoid_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  a.activate(layer_info::sigmoid_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::sigmoid_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}*/
