#include <ann/ann.hh>
#include <ann/layer.hh>
#include <gtest/gtest.h>
#include <random>
#include <utility>
#include <memory>
using namespace zinhart::models::layers;
using namespace zinhart::function_space;

TEST(layer_test, identity_layer)
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

  // an identity layer
  std::shared_ptr<layer<double>> identity = std::make_shared<identity_layer<double>>();
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
  identity->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::identity_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  identity->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::identity_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

TEST(layer_test, sigmoid_layer)
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

  // a sigmoid layer
  std::shared_ptr<layer<double>> sigmoid = std::make_shared<sigmoid_layer<double>>();
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
  sigmoid->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::sigmoid_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  sigmoid->activate(derivative(), layer_deltas_ptr, layer_length);

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
}

TEST(layer_test, softplus_layer)
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

  // a softplus layer
  std::shared_ptr<layer<double>> softplus = std::make_shared<softplus_layer<double>>();
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
  softplus->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::softplus_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  softplus->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::softplus_layer(), *(layer_deltas_ptr_test + i));

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
 
  // a tanh layer
  std::shared_ptr<layer<double>> tanh = std::make_shared<tanh_layer<double>>();
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
  tanh->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::tanh_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  tanh->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::tanh_layer(), *(layer_deltas_ptr_test + i));

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

  // a relu layer
  std::shared_ptr<layer<double>> relu = std::make_shared<relu_layer<double>>();
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
  relu->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  relu->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::relu_layer(), *(layer_deltas_ptr_test + i));

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

  // a leaky_relu layer
  std::shared_ptr<layer<double>> leaky_relu = std::make_shared<leaky_relu_layer<double>>();
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
  leaky_relu->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::leaky_relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  leaky_relu->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::leaky_relu_layer(), *(layer_deltas_ptr_test + i));

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

  // an exp_leaky_relu layer
  std::shared_ptr<layer<double>> exp_leaky_relu = std::make_shared<exp_leaky_relu_layer<double>>();
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
  exp_leaky_relu->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = a.objective(layer_info::exp_leaky_relu_layer(), *(layer_activations_ptr_test + i));

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  exp_leaky_relu->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	*(layer_deltas_ptr_test + i) = a.derivative(layer_info::exp_leaky_relu_layer(), *(layer_deltas_ptr_test + i));

  // validate activation derivative
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_deltas_ptr + i), *(layer_deltas_ptr_test + i) );

  // cleanup
  mkl_free(layer_activations_ptr);
  mkl_free(layer_activations_ptr_test);
  mkl_free(layer_deltas_ptr);
  mkl_free(layer_deltas_ptr_test);
}

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
  double sum{0};
  double * layer_activations_ptr{nullptr};
  double * layer_activations_ptr_test{nullptr};
  double * layer_deltas_ptr{nullptr};
  double * layer_deltas_ptr_test{nullptr};

  // a softmax layer
  std::shared_ptr<layer<double>> softmax = std::make_shared<softmax_layer<double>>();
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
  softmax->activate(objective(), layer_activations_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
	sum += std::exp( *(layer_activations_ptr_test + i) );

  for(i = 0; i < layer_length; ++i)
	*(layer_activations_ptr_test + i) = std::exp(*(layer_activations_ptr_test + i)) / sum; 

  // validate activation
  for(i = 0; i < layer_length; ++i)
	ASSERT_DOUBLE_EQ(*(layer_activations_ptr + i), *(layer_activations_ptr_test + i) );

  // perform activation derivative
  softmax->activate(derivative(), layer_deltas_ptr, layer_length);

  // perform activation test
  for(i = 0; i < layer_length; ++i)
  	for(j = 0; j < layer_length; ++j)
	  *(layer_deltas_ptr_test + j) = (j == i) ? *(layer_deltas_ptr_test + i) * (double{1.0} - *(layer_deltas_ptr_test + i)) : -*(layer_deltas_ptr_test + j) * *(layer_deltas_ptr_test + i);

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
