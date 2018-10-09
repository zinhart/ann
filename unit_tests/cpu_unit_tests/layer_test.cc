#include <ann/ann.hh>
#include <ann/layer.hh>
#include <concurrent_routines/concurrent_routines.hh>
#include <vector>
#include <gtest/gtest.h>
#include <random>
#include <utility>
using namespace zinhart::models::layers;
using namespace zinhart::function_space;
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




































/*

TEST(layer_test, identity_activation)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(2, 10);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  const std::uint32_t alignment = 64;
  const std::uint32_t n_threads{thread_dist(mt)};
  std::uint32_t total_activations_length{neuron_dist(mt) * n_threads}, thread_activation_length{total_activations_length / n_threads}, 
				i{0}, thread_id{0}, start{0}, stop{0};
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_deltas_ptr{nullptr};
  double * total_deltas_ptr_test{nullptr};
  std::vector<layer<double>> total_layers;
  layer_info::layer_type type;

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );

  // initialize
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = real_dist(mt);
	total_activations_ptr_test[i] = total_activations_ptr[i];
	total_deltas_ptr[i] = real_dist(mt);
	total_deltas_ptr_test[i] = total_deltas_ptr[i];
  }

  // initialize layers
  for(thread_id = 0, start = 0, stop = thread_activation_length; stop != total_activations_length; ++thread_id, start += thread_activation_length, stop += thread_activation_length)
  {	total_layers.push_back(layer<double>(start, stop, total_activations_ptr, total_deltas_ptr)); }
  
  // validate thread_offsets
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	ASSERT_EQ(total_layers[thread_id].get_start_index(), thread_id * thread_activation_length )<<"Asserts that the start index is at thread boundary"<<thread_id<<"\n";
	ASSERT_EQ(total_layers[thread_id].get_stop_index(), (thread_id + 1) * thread_activation_length )<<"Asserts that the stop index is at thread boundary "<<thread_id<<"\n";
  }

  // perform activation on a per thread basis
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	total_layers[thread_id].activate(type.identity, objective());
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  *(total_activations_ptr_test + i) = total_layers[thread_id].objective( type.identity, *(total_activations_ptr_test + i) );
	}
  }

  // validate objective activation values on a per thread basis, since this is the identity function they should not have changed
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  ASSERT_DOUBLE_EQ(*(total_activations_ptr + i),*(total_activations_ptr_test + i))<<"validate identity objective activation, failure in thread "<<thread_id<<" element "<<i <<"\n";
	}
  }

  // perform activation derivative on a per thread basis
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	total_layers[thread_id].activate(type.identity, derivative());
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  *(total_activations_ptr_test + i) = total_layers[thread_id].derivative( type.identity, *(total_activations_ptr_test + i) );
	}
  }

  // validate derivative of activation values on a per thread basis, since this is the identity function they should be 1
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  ASSERT_DOUBLE_EQ(*(total_activations_ptr + i),*(total_activations_ptr_test + i))<<"validate identity derivative activation, failure in thread "<<thread_id<<" element "<<i <<"\n";
	  ASSERT_DOUBLE_EQ(*(total_activations_ptr + i), double{1.0} )<<"validate identity derivative activation is 1, failure in thread "<<thread_id<<" element "<<i <<"\n";
	}
  }
  // cleanup
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_deltas_ptr);
  mkl_free(total_deltas_ptr_test);
}

*/
