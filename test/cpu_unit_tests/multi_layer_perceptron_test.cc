#include "ann/models/multi_layer_perceptron.hh"
#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
#include <vector>
#include <iomanip>
using namespace zinhart::models;
using namespace zinhart::activation;
TEST(multi_layer_perceptron, forward_propagate)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1,5000);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types());// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length, total_hidden_weights_length, total_bias_length, total_case_length, total_cases;
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  double * current_inputs_ptr{nullptr};
  double * activation_ptr{nullptr};
  double * total_cases_ptr{nullptr};


  // loop counters misc vars
  std::uint32_t i, ith_layer, ith_case, thread_id, activation_offset, n_layers{layer_dist(mt)};
  const std::uint32_t n_threads{thread_dist(mt)};
  // variables necessary for forward_propagation
  const std::uint32_t input_layer{0};
  std::uint32_t output_layer{0};
  std::uint32_t case_index{0};
  std::uint32_t m{0}, n{0}, k{0};
  double alpha{0.0}, beta{0.0};


  // the thread pool
  zinhart::parallel::thread_pool pool(n_threads);

  // the model
  multi_layer_perceptron<double> model;


  // set layers
  std::vector<LAYER_INFO> total_layers;
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = neuron_dist(mt);
  total_layers.push_back(a_layer);
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	a_layer.first = ACTIVATION_NAME(layer_dist(mt));
	a_layer.second = neuron_dist(mt);  
	total_layers.push_back(a_layer);
  }
  
  // set total case length 
  total_case_length = total_layers[input_layer].second * neuron_dist(mt);
  total_cases = total_case_length / total_layers[input_layer].second;
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  total_bias_length = total_layers.size() - 1;

  std::uint32_t alignment = 64;

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_weights_ptr = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_cases_ptr = (double*) mkl_malloc( total_case_length * sizeof( double ), alignment );

  // set random training data 
  for(i = 0; i < total_case_length; ++i)
	total_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
	total_activations_ptr[i] = 0.0;
  for(i = 0; i < total_hidden_weights_length; ++i)
	total_hidden_weights_ptr[i] = real_dist(mt);

  // lambda to call member function multi_layer_perceptron.forward propagate
  auto forward_propagate_lambda = [&model, &total_layers, &total_cases_ptr, &total_activations_ptr, &total_hidden_weights_ptr, &total_bias_ptr]
								  (const std::uint32_t & ith_training_case, const std::uint32_t & activations, const std::uint32_t & weights, const std::uint32_t & thread_index)
								  {
									model.forward_propagate(total_layers,
															total_cases_ptr, ith_training_case,
															total_activations_ptr, activations,
															total_hidden_weights_ptr, weights,
															total_bias_ptr,
															thread_index
										                   );
								  }; 
  // begin forward propagation
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// this inner for loop is the validation loop
	for(ith_case = 0; ith_case < total_cases; ith_case+=total_layers[input_layer].second)
	{

	}

  }

  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
}
