#include <ann/ann.hh>
#include <ann/layer.hh>
#include <gtest/gtest.h>
#include <random>
#include <utility>

using namespace zinhart::models::layers;
using namespace zinhart::function_space;

TEST(layer_test, identity_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::identity_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::identity_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::identity_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::identity_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, sigmoid_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::sigmoid_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::sigmoid_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::sigmoid_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, softplus_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::softplus_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::softplus_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::softplus_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::softplus_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, tanh_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::tanh_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::tanh_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::tanh_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::tanh_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, relu_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::relu_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::relu_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::relu_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, leaky_relu_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::leaky_relu_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::leaky_relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::leaky_relu_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::leaky_relu_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, exp_leaky_relu_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::exp_leaky_relu_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::exp_leaky_relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::exp_leaky_relu_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::exp_leaky_relu_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}
/*
TEST(layer_test, softmax_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::sigmoid_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::sigmoid_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::sigmoid_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, batch_norm_activation)
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

  // a layer object
  layer<double> l;

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
  l.activate(layer_info::sigmoid_layer(), objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = l.objective(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  l.activate(layer_info::sigmoid_layer(), derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = l.derivative(layer_info::sigmoid_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}*/
