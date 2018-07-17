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
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types());// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_bias_length{0}, total_case_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  double * current_inputs_ptr{nullptr};
  double * total_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};

  // loop counters misc vars
  std::uint32_t i{0}, j{0}, ith_layer{0},ith_case{0}, thread_id{0}, thread_stride{0}, n_layers{layer_dist(mt)};
  const std::uint32_t n_threads{thread_dist(mt)};
  // variables necessary for forward_propagation
  const std::uint32_t input_layer{0};
  std::uint32_t current_layer{0};
  std::uint32_t previous_layer{0};
  std::uint32_t current_layer_index{0};
  std::uint32_t previous_layer_index{0};
  std::uint32_t weight_index{0};
  std::uint32_t current_threads_activation_index{0};
  std::uint32_t case_index{0};
  std::uint32_t m{0}, n{0}, k{0};
  double alpha{1.0}, beta{0.0};


  // the thread pool & futures
  zinhart::parallel::thread_pool pool(n_threads);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;

  // the model
  multi_layer_perceptron<double> model;
  zinhart::activation::activation_function af;


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
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_case_length = total_layers[input_layer].second * case_dist(mt);
  total_cases = total_case_length / total_layers[input_layer].second;
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  thread_stride = total_activations_length;// important!
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
  {
	total_activations_ptr[i] = 0.0;
	total_activations_ptr_test[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
	total_hidden_weights_ptr[i] = real_dist(mt);
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);


  // lambda to call member function multi_layer_perceptron.forward propagate
  auto fprop_model = [](std::vector<LAYER_INFO> & layers,
						double * total_cases_ptr_init, const std::uint32_t ith_training_case,
						double * total_activations_ptr_init, const std::uint32_t activations_length,
						double * total_hidden_weights_ptr_init, const std::uint32_t weights_length,
						double * total_bias_ptr_init,
						const std::uint32_t total_threads, const std::uint32_t thread_index
					   )
					   {
					     multi_layer_perceptron<double> mlp;
					     mlp.forward_propagate(layers,
											   total_cases_ptr_init, ith_training_case,
											   total_activations_ptr_init, activations_length,
											   total_hidden_weights_ptr_init, weights_length,
											   total_bias_ptr_init,
											   total_threads,
											   thread_index
										      );
					  }; 
  // BEGIN FORWARD PROP
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	const double * current_training_case{total_cases_ptr + (ith_case * total_layers[input_layer].second)};
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
	  results.push_back(pool.add_task(fprop_model, std::ref(total_layers), total_cases_ptr, ith_case, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
//	  std::string s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
// 	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
//	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * thread_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_activation_ptr, n
				 );
//	  zinhart::serial::print_matrix_row_major(current_threads_activation_ptr, 1, total_layers[current_layer].second, "parallel activation vector");
	  // add in bias, consider using neaumaer sum
	  for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		total_activations_ptr_test[i] += total_bias_ptr[previous_layer];
	  
	  // apply activation functions
	  for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_activations_ptr_test[i]);
//	  zinhart::serial::print_matrix_row_major(current_threads_activation_ptr, 1, total_layers[current_layer].second, "parallel activation vector");
	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer].second;

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
//	  std::cout<<"\n";
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr = current_threads_activation_ptr + current_layer_index;
		const double * prior_layer_ptr = current_threads_activation_ptr + previous_layer_index; 
//		s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
//	  	zinhart::serial::print_matrix_row_major(current_weight_matrix, total_layers[current_layer].second, total_layers[previous_layer].second, s);
//	    zinhart::serial::print_matrix_row_major(prior_layer_ptr, total_layers[previous_layer].second, 1, "Inputs");
		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[previous_layer].second;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_ptr, n
				   );
  
//		zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "parallel activation vector");
		// add in bias, consider using neaumaer sum
		for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		  total_activations_ptr_test[i] += total_bias_ptr[previous_layer];
		
		// apply activation functions
		for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		  af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_activations_ptr_test[i]);
//		zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "parallel activation vector");
		
		// update weight matrix index	
		weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer].second;
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
// 		std::cout<<"\n";
	   }

	  // synchronize w.r.t the current thread 
	  results[thread_id].get();

	  // validate
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
	  //zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
	  //zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");
	}
	results.clear();
  }
  // END FORWARD PROP
  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
}