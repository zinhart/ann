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
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types() - 1);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_bias_length{0}, total_case_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_hidden_input_ptr{nullptr};
  double * total_hidden_input_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  double * current_inputs_ptr{nullptr};
  double * total_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};
  double * current_threads_hidden_input_ptr{nullptr};

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
  total_hidden_input_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_input_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
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
	total_hidden_input_ptr[i] = 0.0;
	total_hidden_input_ptr_test[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
	total_hidden_weights_ptr[i] = real_dist(mt);
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);


  // lambda to call member function multi_layer_perceptron.forward propagate
  auto fprop_model = [](std::vector<LAYER_INFO> & layers,
						double * total_cases_ptr_init, const std::uint32_t ith_training_case,
						double * total_hidden_inputs_init, double * total_activations_ptr_init, const std::uint32_t activations_length,
						double * total_hidden_weights_ptr_init, const std::uint32_t weights_length,
						double * total_bias_ptr_init,
						const std::uint32_t total_threads, const std::uint32_t thread_index
					   )
					   {
					     multi_layer_perceptron<double> mlp;
					     mlp.forward_propagate(layers,
											   total_cases_ptr_init, ith_training_case,
											   total_hidden_inputs_init, total_activations_ptr_init, activations_length,
											   total_hidden_weights_ptr_init, weights_length,
											   total_bias_ptr_init,
											   total_threads,
											   thread_index
										      );
					  }; 
  // BEGIN FORWARD PROP
  for(ith_case = 0; ith_case < total_cases/*1*/; ++ith_case)
  {
	const double * current_training_case{total_cases_ptr + (ith_case * total_layers[input_layer].second)};
	for(thread_id = 0; thread_id < n_threads/*1*/; ++thread_id)
	{
	  results.push_back(pool.add_task(fprop_model, std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
/*	  std::string s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
 	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
	  */
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * thread_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_hidden_input_ptr, n
				 );
//	  zinhart::serial::print_matrix_row_major(current_threads_hidden_input_ptr, 1, total_layers[current_layer].second, "pre bias and activation vector");

	  // add in bias
	  for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
	  {
		total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		// apply activation functions
		total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
	  }
		
	 // std::cout<<zinhart::activation::get_activation_name(total_layers[current_layer].first)<<"\n"; 

	//  zinhart::serial::print_matrix_row_major(current_threads_activation_ptr, 1, total_layers[current_layer].second, "activation vector");
	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer].second;

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
	 // std::cout<<"\n";
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		double * current_layer_Wx{total_hidden_input_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 
/*		s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
	  	zinhart::serial::print_matrix_row_major(current_weight_matrix, total_layers[current_layer].second, total_layers[previous_layer].second, s);
	    zinhart::serial::print_matrix_row_major(prior_layer_ptr, total_layers[previous_layer].second, 1, "Inputs");*/
		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[previous_layer].second;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_Wx, n
				   );
  
	//	zinhart::serial::print_matrix_row_major(current_layer_Wx, 1, total_layers[current_layer].second, "pre bias and activation vector");

		// add in bias
		for(i = current_threads_activation_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		{
		  total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		  // apply activation functions
		  total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
		}
		//zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "pre activation vector");
  	//	std::cout<<"ACTIVATION: "<<zinhart::activation::get_activation_name(total_layers[current_layer].first)<<"\n"; 

	//	zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "activation vector");
		/**/
	
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
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
/*
	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr_test, 1, total_activations_length, "serial hidden inputs vector");
	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr, 1, total_activations_length, "parallel hidden inputs vector");
	  zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
	  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");*/
	}
	results.clear();
  }
  // END FORWARD PROP
  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_hidden_input_ptr);
  mkl_free(total_hidden_input_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
}

TEST(multi_layer_perceptron, get_results)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types() - 1);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_bias_length{0}, total_case_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_hidden_input_ptr{nullptr};
  double * total_hidden_input_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  double * current_inputs_ptr{nullptr};
  double * total_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};
  double * current_threads_hidden_input_ptr{nullptr};
  double * outputs_ptr{nullptr};
  double * outputs_ptr_test{nullptr};

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

  const std::uint32_t alignment{64};
  const std::uint32_t output_layer_nodes{total_layers[total_layers.size() - 1].second};
  std::uint32_t output_layer_index{0};
  for(i = 1; i < total_layers.size(); ++i)
	output_layer_index += total_layers[i].second;

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_input_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_input_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  outputs_ptr = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  outputs_ptr_test = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
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
	total_hidden_input_ptr[i] = 0.0;
	total_hidden_input_ptr_test[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
	total_hidden_weights_ptr[i] = real_dist(mt);
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);
  for(i = 0; i < output_layer_nodes; ++i)
  {
	outputs_ptr[i] = 0.0;
	outputs_ptr_test[i] = 0.0;
  }


  // lambda to call member function multi_layer_perceptron.forward propagate
  auto fprop_model = [](std::vector<LAYER_INFO> & layers,
						double * total_cases_ptr_init, const std::uint32_t ith_training_case,
						double * total_hidden_inputs_init, double * total_activations_ptr_init, const std::uint32_t activations_length,
						double * total_hidden_weights_ptr_init, const std::uint32_t weights_length,
						double * total_bias_ptr_init,
						const std::uint32_t total_threads, const std::uint32_t thread_index
					   )
					   {
					     multi_layer_perceptron<double> mlp;
					     mlp.forward_propagate(layers,
											   total_cases_ptr_init, ith_training_case,
											   total_hidden_inputs_init, total_activations_ptr_init, activations_length,
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
	  results.push_back(pool.add_task(fprop_model, std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
/*	  std::string s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
 	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
	  */
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * thread_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_hidden_input_ptr, n
				 );
//	  zinhart::serial::print_matrix_row_major(current_threads_hidden_input_ptr, 1, total_layers[current_layer].second, "pre bias and activation vector");

	  // add in bias
	  for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
	  {
		total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		// apply activation functions
		total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
		// save outputs
		if(current_layer == total_layers.size() - 1)
		  outputs_ptr_test[j] = total_activations_ptr_test[i];
	  }
		
	 // std::cout<<zinhart::activation::get_activation_name(total_layers[current_layer].first)<<"\n"; 

	//  zinhart::serial::print_matrix_row_major(current_threads_activation_ptr, 1, total_layers[current_layer].second, "activation vector");
	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer].second;

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
	  //std::cout<<"\n";
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		double * current_layer_Wx{total_hidden_input_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 
/*		s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
	  	zinhart::serial::print_matrix_row_major(current_weight_matrix, total_layers[current_layer].second, total_layers[previous_layer].second, s);
	    zinhart::serial::print_matrix_row_major(prior_layer_ptr, total_layers[previous_layer].second, 1, "Inputs");*/
		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[previous_layer].second;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_Wx, n
				   );
  
	//	zinhart::serial::print_matrix_row_major(current_layer_Wx, 1, total_layers[current_layer].second, "pre bias and activation vector");

		// add in bias
		for(i = current_threads_activation_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		{
		  total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		  // apply activation functions
		  total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
		  // save outputs
		  if(current_layer == total_layers.size() - 1)
			outputs_ptr_test[j] = total_activations_ptr_test[i];

		}
		//zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "pre activation vector");
  	//	std::cout<<"ACTIVATION: "<<zinhart::activation::get_activation_name(total_layers[current_layer].first)<<"\n"; 

	//	zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "activation vector");
		/**/
	
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
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);

	  multi_layer_perceptron<double> mlp;
	  mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr,
					  n_threads,
					  thread_id
					 );
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";
/*
	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr_test, 1, total_activations_length, "serial hidden inputs vector");
	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr, 1, total_activations_length, "parallel hidden inputs vector");
	  zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
	  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");*/
	}
	results.clear();
  }
  // END FORWARD PROP
  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_hidden_input_ptr);
  mkl_free(total_hidden_input_ptr_test);
  mkl_free(outputs_ptr);
  mkl_free(outputs_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
}
TEST(multi_layer_perceptron, backward_propagate)
{

  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types() - 1);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> loss_function_dist(0, 1);
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_gradient_length{0}, total_bias_length{0}, total_case_length{0}, total_targets_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_deltas_ptr{nullptr};
  double * total_deltas_ptr_test{nullptr};
  double * total_hidden_input_ptr{nullptr};
  double * total_hidden_input_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_gradient_ptr{nullptr};
  double * total_gradient_ptr_test{nullptr};
  double * total_bias_ptr{nullptr};
  double * total_targets_ptr{nullptr};
  double * current_inputs_ptr{nullptr};
  double * total_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};
  double * current_threads_hidden_input_ptr{nullptr};
  double * current_threads_delta_ptr{nullptr};
  double * current_threads_gradient_ptr{nullptr};
  double * outputs_ptr{nullptr};
  double * outputs_ptr_test{nullptr};

  // loop counters misc vars
  std::uint32_t i{0}, j{0}, ith_layer{0},ith_case{0}, thread_id{0}, thread_stride{0}, gradient_stride{0}, n_layers{layer_dist(mt)};
  const std::uint32_t n_threads{thread_dist(mt)};
  // variables necessary for forward_propagation & backward propagation
  const std::uint32_t input_layer{0};
  std::uint32_t current_layer{0};
  std::uint32_t previous_layer{0};
  std::uint32_t current_layer_index{0};
  std::uint32_t previous_layer_index{0};
  std::uint32_t weight_index{0};
  std::uint32_t current_threads_activation_index{0};
  std::uint32_t current_threads_gradient_index{0};
  std::uint32_t case_index{0};
  std::uint32_t m{0}, n{0}, k{0};
  double alpha{1.0}, beta{0.0}, error{0.0};


  // the thread pool & futures
  zinhart::parallel::thread_pool pool(n_threads);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;

  // the model
  multi_layer_perceptron<double> model;
  zinhart::activation::activation_function af;
  zinhart::error_metrics::loss_function loss;


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
  const std::uint32_t output_layer{total_layers.size() - 1};
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
  gradient_stride = total_hidden_weights_length;
  total_gradient_length = total_hidden_weights_length * n_threads;// important!
  total_bias_length = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer].second * total_cases;


  const std::uint32_t alignment{64};
  const std::uint32_t output_layer_nodes{total_layers[total_layers.size() - 1].second};
  std::uint32_t output_layer_index{0};
  for(i = 1; i < total_layers.size(); ++i)
	output_layer_index += total_layers[i].second;

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_input_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_input_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  outputs_ptr = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  outputs_ptr_test = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  total_hidden_weights_ptr = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_gradient_ptr = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  total_gradient_ptr_test = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_cases_ptr = (double*) mkl_malloc( total_case_length * sizeof( double ), alignment );
  total_targets_ptr = (double*) mkl_malloc(total_targets_length * sizeof(double), alignment );

  // set random training data 
  for(i = 0; i < total_case_length; ++i)
	total_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_targets_length; ++i)
	total_targets_ptr[i] = 0.0;
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = 0.0;
	total_activations_ptr_test[i] = 0.0;
	total_hidden_input_ptr[i] = 0.0;
	total_hidden_input_ptr_test[i] = 0.0;
	total_deltas_ptr[i] = 0.0;
	total_deltas_ptr_test[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
  {
	total_hidden_weights_ptr[i] = real_dist(mt);
	total_gradient_ptr[i] = 0.0;
	total_gradient_ptr_test[i] = 0.0;
  }
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);
  for(i = 0; i < output_layer_nodes; ++i)
  {
	outputs_ptr[i] = 0.0;
	outputs_ptr_test[i] = 0.0;
  }


  // lambda to call member function multi_layer_perceptron.forward propagate
  auto fprop_model = [](std::vector<LAYER_INFO> & layers,
						double * total_cases_ptr_init, const std::uint32_t ith_training_case,
						double * total_hidden_inputs_init, double * total_activations_ptr_init, const std::uint32_t activations_length,
						double * total_hidden_weights_ptr_init, const std::uint32_t weights_length,
						double * total_bias_ptr_init,
						const std::uint32_t total_threads, const std::uint32_t thread_index
					   )
					   {
					     multi_layer_perceptron<double> mlp;
					     mlp.forward_propagate(layers,
											   total_cases_ptr_init, ith_training_case,
											   total_hidden_inputs_init, total_activations_ptr_init, activations_length,
											   total_hidden_weights_ptr_init, weights_length,
											   total_bias_ptr_init,
											   total_threads,
											   thread_index
										      );
					  }; 
  // BEGIN FORWARD & BACKWARD PROP
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	const double * current_training_case{total_cases_ptr + (ith_case * total_layers[input_layer].second)};
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
	  results.push_back(pool.add_task(fprop_model, std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
/*	  std::string s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
 	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
	  */
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * thread_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_hidden_input_ptr, n
				 );
//	  zinhart::serial::print_matrix_row_major(current_threads_hidden_input_ptr, 1, total_layers[current_layer].second, "pre bias and activation vector");

	  // add in bias
	  for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
	  {
		total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		// apply activation functions
		total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
		// save outputs
		if(current_layer == total_layers.size() - 1)
		  outputs_ptr_test[j] = total_activations_ptr_test[i];
	  }
		
	 // std::cout<<zinhart::activation::get_activation_name(total_layers[current_layer].first)<<"\n"; 

	//  zinhart::serial::print_matrix_row_major(current_threads_activation_ptr, 1, total_layers[current_layer].second, "activation vector");
	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer].second;

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
	  //std::cout<<"\n";
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		double * current_layer_Wx{total_hidden_input_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 
/*		s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
	  	zinhart::serial::print_matrix_row_major(current_weight_matrix, total_layers[current_layer].second, total_layers[previous_layer].second, s);
	    zinhart::serial::print_matrix_row_major(prior_layer_ptr, total_layers[previous_layer].second, 1, "Inputs");*/
		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[previous_layer].second;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_Wx, n
				   );
  
	//	zinhart::serial::print_matrix_row_major(current_layer_Wx, 1, total_layers[current_layer].second, "pre bias and activation vector");

		// add in bias
		for(i = current_threads_activation_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		{
		  total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		  // apply activation functions
		  total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
		  // save outputs
		  if(current_layer == total_layers.size() - 1)
			outputs_ptr_test[j] = total_activations_ptr_test[i];

		}
		//zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "pre activation vector");
  	//	std::cout<<"ACTIVATION: "<<zinhart::activation::get_activation_name(total_layers[current_layer].first)<<"\n"; 

	//	zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "activation vector");
		/**/
	
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
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);

	  multi_layer_perceptron<double> mlp;
	  mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr,
					  n_threads,
					  thread_id
					 );
	  // validate
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";
/*
	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr_test, 1, total_activations_length, "serial hidden inputs vector");
	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr, 1, total_activations_length, "parallel hidden inputs vector");
	  zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
	  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");*/
	}
	// clear futures
	results.clear();

	// set up for calculating output layer gradients
	current_layer = 1;
	previous_layer = 0;
	// adjust to output layer
	current_layer = output_layer;
	previous_layer = output_layer - 1;
	// adjust pointer indices
	current_layer_index = output_layer_index;
	previous_layer_index = output_layer_index - total_layers[previous_layer].second;
	current_threads_activation_index = thread_id * thread_stride;
	current_threads_gradient_index = thread_id * gradient_stride;
	// set pointers
	current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index + current_layer_index;
	current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index +  current_layer_index;
	current_threads_delta_ptr = total_deltas_ptr_test + current_threads_activation_index + current_layer_index;
	current_threads_gradient_ptr = total_gradient_ptr_test + current_threads_gradient_index; 
	double * current_target{total_targets_ptr + (ith_case * total_layers[output_layer].second)};
    // calculate error 
	//error  = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::OBJECTIVE, outputs, targets, n_elements);
	// calculate error derivative
	error = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::DERIVATIVE,  current_threads_activation_ptr, current_target, total_layers[output_layer].second);
	// calculate output layer deltas
	// calculate output layer gradient 
	// for gemm
	m = total_layers[current_layer].second;
	n = total_layers[previous_layer].second;
	k = 1;
	// calc hidden layer deltas

  }
  // END FORWARD & BACKPROP PROP
  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_deltas_ptr);
  mkl_free(total_deltas_ptr_test);
  mkl_free(total_hidden_input_ptr);
  mkl_free(total_hidden_input_ptr_test);
  mkl_free(outputs_ptr);
  mkl_free(outputs_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_gradient_ptr);
  mkl_free(total_gradient_ptr_test);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
  mkl_free(total_targets_ptr);
}
