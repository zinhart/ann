#include <ann/ann.hh>
#include <ann/layer.hh>
#include <vector>
#include <gmock/gmock.h>
#include <random>
using namespace zinhart::models::layers;
using namespace zinhart::function_space;
/*
TEST(layer_test, init_test)
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
  double * total_deltas_ptr{nullptr};
  std::vector<layer<double>> total_layers;


  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );

  // initialize
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = real_dist(mt);
	total_deltas_ptr[i] = real_dist(mt);
  }

  // initialize layers
  for(thread_id = 0, start = 0, stop = thread_activation_length;  thread_id < n_threads; ++thread_id, start += thread_activation_length, stop += thread_activation_length)
  {	total_layers.push_back(layer<double>(start, stop, total_activations_ptr, total_deltas_ptr)); }
  
  // validate thread offsets
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	ASSERT_EQ(total_layers[thread_id].get_start_index(), thread_id * thread_activation_length )<<"Asserts that the start index is at thread boundary"<<thread_id<<"\n";
	ASSERT_EQ(total_layers[thread_id].get_stop_index(), (thread_id + 1) * thread_activation_length )<<"Asserts that the stop index is at thread boundary "<<thread_id<<"\n";
  }
  
  // cleanup
  mkl_free(total_activations_ptr);
  mkl_free(total_deltas_ptr);
}
*/
TEST(layer_test, sigmoid_activation)
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
/*
  // initialize layers
  for(thread_id = 0, start = 0, stop = thread_activation_length; /thread_id < n_threads; ++thread_id, start += thread_activation_length, stop += thread_activation_length)
  {	total_layers.push_back(layer<double>(start, stop, total_activations_ptr, total_deltas_ptr)); }
  
  // validate thread_offsets
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	ASSERT_EQ(total_layers[thread_id].get_start_index(), thread_id * thread_activation_length )<<"Asserts that the start index is at thread boundary"<<thread_id<<"\n";
	ASSERT_EQ(total_layers[thread_id].get_stop_index(), (thread_id + 1) * thread_activation_length )<<"Asserts that the stop index is at thread boundary "<<thread_id<<"\n";
  }
  std::cout<<"total layers size"<<total_layers.size()<<"\n";

  // perform activation on a per thread basis
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  std::cout<<"i: "<<i<<"\n";
	  std::cout<< "BEFORE" <<*(total_activations_ptr + i) <<" "<< *(total_activations_ptr_test + i) <<" "<<"\n";
//	  *(total_activations_ptr_test + i) = total_layers[thread_id].objective( type.sigmoid, *(total_activations_ptr_test + i) );
	}
	total_layers[thread_id].activate(type.sigmoid, objective());
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  std::cout<<"i: "<<i<<"\n";
	  std::cout<< "AFTER" <<*(total_activations_ptr + i) <<" "<< *(total_activations_ptr_test + i) <<" "<<"\n";
//	  *(total_activations_ptr_test + i) = total_layers[thread_id].objective( type.sigmoid, *(total_activations_ptr_test + i) );
	}
  }

  // validate objective activation values on a per thread basis, since this is the sigmoid function they should not have changed
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  ASSERT_DOUBLE_EQ(*(total_activations_ptr + i),*(total_activations_ptr_test + i))<<"validate sigmoid objective activation, failure in thread "<<thread_id<<" element "<<i <<"\n";
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

  // validate derivative of activation values on a per thread basis
  for(thread_id = 0; thread_id < total_layers.size(); ++thread_id)
  {
	for(i = total_layers[thread_id].get_start_index(); i < total_layers[thread_id].get_stop_index(); ++i)
	{
	  ASSERT_DOUBLE_EQ(*(total_activations_ptr + i),*(total_activations_ptr_test + i))<<"validate sigmoid derivative activation, failure in thread "<<thread_id<<" element "<<i <<"\n";
	}
  }
*/
  // cleanup
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_deltas_ptr);
  mkl_free(total_deltas_ptr_test);
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
