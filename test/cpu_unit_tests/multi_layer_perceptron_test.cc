#include <ann/ann.hh>
#include <concurrent_routines/concurrent_routines.hh>
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
#include <vector>
#include <iomanip>
using namespace zinhart::models;
using namespace zinhart::activation;

TEST(multi_layer_perceptron, forward_propagate_thread_safety)
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
  std::uint32_t i{0}, j{0}, ith_layer{0},ith_case{0}, thread_id{0}, activation_stride{0}, n_layers{layer_dist(mt)};
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
  multi_layer_perceptron<connection::dense, double> model;
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
  activation_stride = total_activations_length;// important!
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
					     multi_layer_perceptron<connection::dense, double> mlp;
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
	  
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * activation_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_hidden_input_ptr, n
				 );

	  // add in bias
	  for(i = current_threads_activation_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
	  {
		total_hidden_input_ptr_test[i] += total_bias_ptr[previous_layer];
		// apply activation functions
		total_activations_ptr_test[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_input_ptr_test[i]);
	  }
		

	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer].second;

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		double * current_layer_Wx{total_hidden_input_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 
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
		
		// update weight matrix index	
		weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer].second;
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
	   }

	  // synchronize w.r.t the current thread 
	  results[thread_id].get();

	  // validate
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
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

TEST(multi_layer_perceptron, get_results_thread_safety)
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
  std::uint32_t i{0}, j{0}, ith_layer{0},ith_case{0}, thread_id{0}, activation_stride{0}, n_layers{layer_dist(mt)};
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
   multi_layer_perceptron<connection::dense, double> model;
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
  activation_stride = total_activations_length;// important!
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
					      multi_layer_perceptron<connection::dense, double> mlp;
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
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * activation_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_hidden_input_ptr, n
				 );

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
		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[previous_layer].second;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_Wx, n
				   );
  
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
		// update weight matrix index	
		weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer].second;
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
	   }

	  // synchronize w.r.t the current thread 
	  results[thread_id].get();

	  // validate
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);

	   multi_layer_perceptron<connection::dense, double> mlp;
	  mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr,
					  n_threads,
					  thread_id
					 );
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";
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

TEST(multi_layer_perceptron, gradient_check_thread_safety)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types() - 1);// does not include input layer
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t input_layer{0};
  std::uint32_t output_layer{0};
  const std::uint32_t alignment = 64;
  std::uint32_t i{0}, j{0}, k{0}, ith_case{0}, thread_id{0}, ith_layer{0}, n_layers{layer_dist(mt)}, total_cases{0}, total_cases_length{0}, total_targets_length{0}, total_activations_length, total_hidden_weights_length{0}, total_gradient_length{0};
  const zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME name{zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE};
  
  double * total_cases_ptr{nullptr};
  double * total_targets{nullptr};
  double * total_hidden_inputs{nullptr};
  double * total_activations{nullptr};
  double * total_deltas{nullptr};
  double * total_hidden_weights{nullptr};
  double * total_hidden_weights_copy{nullptr};
  double * numerically_approx_gradients_parallel{nullptr};
  double * numerically_approx_gradients_serial{nullptr};
  double * bias{nullptr};
  double * d_error{nullptr};
  double * current_threads_output_layer_ptr{nullptr};
  double * current_threads_gradient_ptr{nullptr};


  // the thread pool & futures
  zinhart::parallel::thread_pool pool(n_threads);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;

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
  output_layer = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer].second;
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_cases_length = total_layers[input_layer].second * case_dist(mt);
  total_cases = total_cases_length / total_layers[input_layer].second;
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  total_gradient_length = total_hidden_weights_length * n_threads;
  const std::uint32_t bias_length{total_layers.size() -1};

  total_cases_ptr = (double*) mkl_malloc( total_cases_length * sizeof( double ), alignment );
  total_targets = (double*) mkl_malloc( total_targets_length * sizeof( double ), alignment );
  total_hidden_inputs = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_weights = {(double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment )};
  total_hidden_weights_copy = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  numerically_approx_gradients_parallel = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  numerically_approx_gradients_serial = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  bias = (double*) mkl_malloc( bias_length * sizeof( double ), alignment );
  current_threads_output_layer_ptr = (double*) mkl_malloc( total_targets_length * sizeof( double ), alignment );
  d_error = (double*) mkl_malloc(bias_length * sizeof( double ), alignment );

  // initialize model params


  // set random training data 
  for(i = 0; i < total_cases_length; ++i)
	total_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_targets_length; ++i)
	total_targets[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations[i] = 0.0;
	total_hidden_inputs[i] = 0.0;
	total_deltas[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
  {
	total_hidden_weights[i] = real_dist(mt);
	total_hidden_weights_copy[i] = total_hidden_weights[i];
  }
  for(i = 0; i < total_gradient_length; ++i)
  {
   numerically_approx_gradients_parallel[i] = 0.0;
   numerically_approx_gradients_serial[i] = 0.0;
  }
  for(i = 0; i < bias_length; ++i)
	bias[i] = real_dist(mt);

  
  multi_layer_perceptron<connection::dense, double> mlp;
  zinhart::function_space::error_metrics::loss_function loss;

  std::cout.precision(10);
  // for gradient checking
  const double limit_epsilon = 1.e-4;
  double right{0}, left{0}, original{0};





  auto gradient_check_lamda = [](zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME name_p,
								 const std::vector<zinhart::activation::LAYER_INFO> & total_layers_p,
								 const double * total_training_cases_p, const double * total_targets_p, const std::uint32_t case_index_p,
								 double * total_hidden_inputs_p, double * total_activations_p, const std::uint32_t total_activations_length_p,
								 double * const total_hidden_weights_p, const std::uint32_t total_hidden_weights_length_p,
								 const double * total_bias_p, 
								 double * numerically_approx_gradient_p, 
								 const double limit_epsilon_p, 
								 const std::uint32_t n_threads_p, const std::uint32_t thread_id_p)
                              {
							  	multi_layer_perceptron<connection::dense, double> mlp_p;
								mlp_p.gradient_check(name_p, 
												  total_layers_p,
												  total_training_cases_p, total_targets_p, case_index_p,
												  total_hidden_inputs_p, total_activations_p, total_activations_length_p,
												  total_hidden_weights_p, total_hidden_weights_length_p,
												  total_bias_p,
												  numerically_approx_gradient_p,
												  limit_epsilon_p,
												  n_threads_p, thread_id_p
												  );
							  };
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	// for each thread perform a gradient check on the same values, ideally each thread would come to the same conclusion (return the same values)
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
	  current_threads_gradient_ptr = numerically_approx_gradients_serial + (thread_id * total_hidden_weights_length);
	  for(i = 0; i < total_hidden_weights_length ; ++i)
	  {
		original = total_hidden_weights_copy[i];
		// right side
		total_hidden_weights_copy[i] += limit_epsilon;
		mlp.forward_propagate(total_layers, 
						  total_cases_ptr, ith_case, 
						  total_hidden_inputs, total_activations, total_activations_length,
						  total_hidden_weights_copy, total_hidden_weights_length, 
						  bias,
						  n_threads, thread_id
						 );
		mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
		right = loss(name, zinhart::function_space::OBJECTIVE(), current_threads_output_layer_ptr, total_targets, total_targets_length, 2);

		// set back
		total_hidden_weights_copy[i] = original; 

		// left side
		total_hidden_weights_copy[i] -= limit_epsilon;
		mlp.forward_propagate(total_layers, 
					  total_cases_ptr, ith_case, 
					  total_hidden_inputs, total_activations, total_activations_length,
					  total_hidden_weights_copy, total_hidden_weights_length, 
					  bias,
					  n_threads, thread_id
					);
		mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
		left = loss(name, zinhart::function_space::OBJECTIVE(), current_threads_output_layer_ptr, total_targets, total_targets_length, 2);
		// calc numerically derivative for the ith_weight, save it, increment the pointer to the next weight
		*(current_threads_gradient_ptr + i) = (right - left) / (2 * limit_epsilon);
		// set back
		total_hidden_weights_copy[i] = original; 
	  }
	  // forward prop
	  mlp.forward_propagate(std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_inputs, total_activations, total_activations_length, total_hidden_weights, total_hidden_weights_length, bias, n_threads, thread_id);
	  // error derivative
	  mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
	  
	  results.push_back(pool.add_task(gradient_check_lamda,
									  name, 
									  total_layers,
									  total_cases_ptr, total_targets, ith_case,
									  total_hidden_inputs, total_activations, total_activations_length,
									  total_hidden_weights, total_hidden_weights_length,
									  bias,
									  numerically_approx_gradients_parallel,
									  limit_epsilon,
									  n_threads, thread_id )
					   );
	  results[thread_id].get();
	  for(i = 0; i < total_hidden_weights_length; ++i)
		EXPECT_DOUBLE_EQ(total_hidden_weights[i], total_hidden_weights_copy[i]);
	  // compare gradients
	  for(i = 0; i < total_gradient_length; ++i)
	  {
		EXPECT_NEAR(numerically_approx_gradients_parallel[i], numerically_approx_gradients_serial[i], limit_epsilon)<<"i: "<<i<<"\n";/**/
	  }
	}
	results.clear();
  }
     
  
  mkl_free(total_cases_ptr);
  mkl_free(total_hidden_inputs);
  mkl_free(total_deltas);
  mkl_free(total_activations);
  mkl_free(total_hidden_weights);
  mkl_free(total_hidden_weights_copy);
  mkl_free(numerically_approx_gradients_parallel);
  mkl_free(numerically_approx_gradients_serial);
  mkl_free(bias);
  mkl_free(current_threads_output_layer_ptr);
  mkl_free(d_error);
}



TEST(multi_layer_perceptron, backward_propagate_thread_safety)
{

  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types() - 1);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> loss_function_dist(0, 1);
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-1, 1);
  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_gradient_length{0}, total_bias_length{0}, total_case_length{0}, total_targets_length{0}, total_cases{0}, total_error_length{0};
  
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
  double * d_error{nullptr};
  double * current_target{nullptr};
  double * gradient_approx{nullptr};

  // loop counters misc vars
  std::uint32_t i{0}, j{0}, ith_layer{0},ith_case{0}, thread_id{0}, activation_stride{0}, gradient_stride{0}, n_layers{layer_dist(mt)};
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
  const double limit_epsilon = 1.e-4;
  const zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME name{zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE};


  // the thread pool & futures
  zinhart::parallel::thread_pool pool(n_threads);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;

  // the model
  multi_layer_perceptron<connection::dense, double> model;
  zinhart::activation::activation_function af;
  zinhart::function_space::error_metrics::loss_function loss;


  // set layers
  std::vector<LAYER_INFO> total_layers;
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = neuron_dist(mt);
  total_layers.push_back(a_layer);
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	a_layer.first = zinhart::activation::ACTIVATION_NAME::SIGMOID;//ACTIVATION_NAME(layer_dist(mt));
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
  activation_stride = total_activations_length;// important!
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  gradient_stride = total_hidden_weights_length;
  total_gradient_length = total_hidden_weights_length * n_threads;// important!
  total_bias_length = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer].second * total_cases;


  const std::uint32_t alignment{64};
  const std::uint32_t output_layer_nodes{total_layers[output_layer].second};
  total_error_length = total_layers[output_layer].second * n_threads;


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
  d_error = (double*) mkl_malloc(total_error_length * sizeof(double), alignment );
  gradient_approx = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );


  // set random training data 
  for(i = 0; i < total_case_length; ++i)
	total_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_targets_length; ++i)
	total_targets_ptr[i] = real_dist(mt);
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
  for(i = 0; i < total_gradient_length; ++i)
  {
	total_gradient_ptr[i] = 0.0;
	total_gradient_ptr_test[i] = 0.0;
	gradient_approx[i] = 0.0;
  }

  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);
  for(i = 0; i < output_layer_nodes; ++i)
  {
	outputs_ptr[i] = 0.0;
	outputs_ptr_test[i] = 0.0;
  }

  for(i = 0; i < total_error_length; ++i)
	d_error[i] = 0.0;

  // lambda to call member function multi_layer_perceptron.forward propagate
  auto fprop_model = [](std::vector<LAYER_INFO> & layers,
						double * total_cases_ptr_init, const std::uint32_t ith_training_case,
						double * total_hidden_inputs_init, double * total_activations_ptr_init, const std::uint32_t activations_length,
						double * total_hidden_weights_ptr_init, const std::uint32_t weights_length,
						double * total_bias_ptr_init,
						const std::uint32_t total_threads, const std::uint32_t thread_index
					   )
					   {
					      multi_layer_perceptron<connection::dense, double> mlp;
					     mlp.forward_propagate(layers,
											   total_cases_ptr_init, ith_training_case,
											   total_hidden_inputs_init, total_activations_ptr_init, activations_length,
											   total_hidden_weights_ptr_init, weights_length,
											   total_bias_ptr_init,
											   total_threads,
											   thread_index
										      );
					  }; 
  auto bprop_model = [](const std::vector<zinhart::activation::LAYER_INFO> & total_layers_init, 
						const double * const total_training_cases_init, const double * const total_targets_init, const double * const d_error_init, const std::uint32_t case_index_init,
						const double * const total_hidden_inputs_init, const double * const total_activations_init, double * total_deltas_init, const std::uint32_t total_activations_length_init,
						const double * const total_hidden_weights_init, double * total_gradient_init, const std::uint32_t total_hidden_weights_length_init,
						const double * const total_bias_init,
						const std::uint32_t n_threads_init,
					    const std::uint32_t thread_id_init
					)
					{
					  multi_layer_perceptron<connection::dense, double> mlp;
					  mlp.backward_propagate(total_layers_init,
						                     total_training_cases_init, total_targets_init, d_error_init, case_index_init,
											 total_hidden_inputs_init, total_activations_init, total_deltas_init, total_activations_length_init,
											 total_hidden_weights_init, total_gradient_init, total_hidden_weights_length_init,
											 total_bias_init,
											 n_threads_init,
											 thread_id_init
						                    );
					};

  auto gradient_check_lamda = [](zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME name_p,
								 const std::vector<zinhart::activation::LAYER_INFO> & total_layers_p,
								 const double * total_training_cases_p, const double * total_targets_p, const std::uint32_t case_index_p,
								 double * total_hidden_inputs_p, double * total_activations_p, const std::uint32_t total_activations_length_p,
								 double * const total_hidden_weights_p, const std::uint32_t total_hidden_weights_length_p,
								 const double * total_bias_p, 
								 double * numerically_approx_gradient_p, 
								 const double limit_epsilon_p, 
								 const std::uint32_t n_threads_p, const std::uint32_t thread_id_p)
                              {
							  	multi_layer_perceptron<connection::dense, double> mlp_p;
								mlp_p.gradient_check(name_p, 
												  total_layers_p,
												  total_training_cases_p, total_targets_p, case_index_p,
												  total_hidden_inputs_p, total_activations_p, total_activations_length_p,
												  total_hidden_weights_p, total_hidden_weights_length_p,
												  total_bias_p,
												  numerically_approx_gradient_p,
												  limit_epsilon_p,
												  n_threads_p, thread_id_p
												  );
							  };
  // BEGIN FORWARD & BACKWARD PROP
  for(ith_case = 0; ith_case < /*total_cases*/1; ++ith_case)
  {
	const double * current_training_case{total_cases_ptr + (ith_case * total_layers[input_layer].second)};
	current_target = total_targets_ptr + (ith_case * total_layers[output_layer].second);
	std::uint32_t error_stride{total_layers[output_layer].second};
	for(thread_id = 0; thread_id < /*n_threads*/1; ++thread_id)
	{
	  results.push_back(pool.add_task(fprop_model, std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;
	  m = total_layers[current_layer].second;
	  n = 1;
	  k = total_layers[previous_layer].second;
	  current_threads_activation_index = thread_id * activation_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_hidden_input_ptr, n
				 );

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
	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer].second;

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		double * current_layer_Wx{total_hidden_input_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 
		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[previous_layer].second;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_Wx, n
				   );
  
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
		// update weight matrix index	
		weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer].second;
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
	  }
	  
	  // synchronize w.r.t the current thread, forward prop for the current thread ends here
	  results[thread_id].get();

	  // validate forward prop outputs
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i])<<"i: "<<i<<"\n";
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i])<<"i: "<<i<<"\n";

	  multi_layer_perceptron<connection::dense, double> mlp;
	  mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr,
					  n_threads,
					  thread_id
					 );
	  // validate output_layer
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";

	  // calculate error 
	  error = loss(name, zinhart::function_space::OBJECTIVE(), outputs_ptr, current_target, output_layer_nodes, 2);
	
	 current_layer_index = 0; 
	 for(i = 1; i < total_layers.size() - 1; ++i)
	   current_layer_index += total_layers[i].second;// the start of the output layer
	 previous_layer_index = 0;
	 for(i = 1; i < total_layers.size() - 2; ++i)
	   previous_layer_index += total_layers[i].second;// the start of the layer right behind the output layer

	 std::uint32_t current_gradient_index{0};
   	 // calc number of hidden weights
	 for(i = 0 ; i < total_layers.size() - 2; ++i)
	   current_gradient_index += total_layers[i + 1].second * total_layers[i].second;
	 
	 current_layer = output_layer;
	 previous_layer = current_layer - 1; 

//	 current_threads_activation_index = thread_id * activation_stride; //set earlier
	 current_threads_gradient_index = thread_id * gradient_stride;
	 std::cout<<current_threads_gradient_index<<" ctgi\n";

//	 current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;// set earlier
//	 current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;// set earlier
	 current_threads_delta_ptr = total_deltas_ptr_test + current_threads_activation_index;
	 current_threads_gradient_ptr = total_gradient_ptr_test + current_threads_gradient_index;

	 // set pointers
	 double * current_layers_hidden_input_ptr{current_threads_hidden_input_ptr + current_layer_index};
	 double * current_layer_activation_ptr{current_threads_activation_ptr + current_layer_index};
	 // if this is a 2 layer model then the prior activations are essentially the inputs to the model
	 const double * prior_layer_activation_ptr{(total_layers.size() > 2) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case}; 
	 double * current_layer_deltas_ptr{current_threads_delta_ptr + current_layer_index};
	 double * current_gradient_ptr{current_threads_gradient_ptr + current_gradient_index};

	 double * current_error_matrix = d_error + (thread_id * error_stride);
	 // calculate error derivative
	 loss(name, zinhart::function_space::DERIVATIVE(), outputs_ptr, current_target, current_error_matrix, output_layer_nodes, 2);
	  // begin backprop 
	  results[thread_id] = pool.add_task(bprop_model,std::ref(total_layers), total_cases_ptr, total_targets_ptr, d_error, ith_case, 
										 total_hidden_input_ptr, total_activations_ptr, total_deltas_ptr, total_activations_length, 
										 total_hidden_weights_ptr, total_gradient_ptr, total_hidden_weights_length, 
										 total_bias_ptr, 
										 n_threads, thread_id 
									    );
/**/
/*
	 // calculate output layer deltas
	 for(i = current_threads_activation_index + current_layer_index, j = thread_error_stride, k = 0; k < total_layers[current_layer].second; ++i, ++j, ++k)
	 {
	   total_deltas_ptr_test[i] = d_error[j] * af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::DERIVATIVE, total_activations_ptr_test[i]);
	 }
	 


	 // set up to calculate output layer gradient 
	 m = total_layers[current_layer].second;
	 n = total_layers[previous_layer].second;
	 k = 1; 

	 // calc output layer gradient
	 cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, current_layer_deltas_ptr, k,
				  prior_layer_activation_ptr, n, beta, 
				  current_gradient_ptr, n
				 );

	  std::uint32_t next_weight_matrix_index{total_hidden_weights_length};
	  std::uint32_t next_layer_index{current_layer_index};
	  std::uint32_t next_layer{current_layer};
	  --current_layer;
	  --previous_layer;

	  // calc hidden layer gradients
	  while(current_layer > 0)
	  {
		next_weight_matrix_index -= total_layers[next_layer].second * total_layers[current_layer].second;	
		current_layer_index = previous_layer_index;
		previous_layer_index -= total_layers[previous_layer].second; 
		current_gradient_index -= total_layers[current_layer].second * total_layers[previous_layer].second;
		current_gradient_ptr = current_threads_gradient_ptr + current_gradient_index;

   		double * weight_ptr{total_hidden_weights_ptr + current_threads_gradient_index + next_weight_matrix_index};
		double * next_layer_delta_ptr{current_threads_delta_ptr + next_layer_index};
		current_layer_deltas_ptr = current_threads_delta_ptr + current_layer_index ;
		const double * previous_layer_activation_ptr{ (current_layer > 1) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case};//= current_threads_activation_ptr + previous_layer_index;

		m = total_layers[current_layer].second;
	    n = 1;
	    k = total_layers[next_layer].second;

   		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				  m, n, k,
				  alpha, weight_ptr, m,
				  next_layer_delta_ptr, n, beta, 
				  current_layer_deltas_ptr, n
				 );

   		// calculate current layer deltas
		for(i = current_threads_activation_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		  total_deltas_ptr_test[i] *= af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::DERIVATIVE, total_activations_ptr_test[i]);
		 
		m = total_layers[current_layer].second;
   		n = total_layers[previous_layer].second;
   		k = 1;
   
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, current_layer_deltas_ptr, k,
				  previous_layer_activation_ptr, n, beta, 
				  current_gradient_ptr, n
				 );
		next_layer_index = current_layer_index;
		--next_layer;
		--current_layer;
		--previous_layer;
	  }	  
*/	
	  // serial backprop done
	  // synchronize w.r.t the current thread, back prop ends here
	  results[thread_id].get();

	  // validate bprop outputs
/*	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_deltas_ptr[i], total_deltas_ptr_test[i]);
	  	for(i = 0; i < total_gradient_length; ++i)
		EXPECT_NEAR(total_gradient_ptr[i], total_gradient_ptr_test[i], std::numeric_limits<double>::epsilon())<< "i: "<<i<<"\n";
		*/

	  // gradient check
	  results[thread_id] = pool.add_task(gradient_check_lamda,
									  name, 
									  total_layers,
									  total_cases_ptr, total_targets_ptr, ith_case,
									  total_hidden_input_ptr_test, total_activations_ptr_test, total_activations_length,
									  total_hidden_weights_ptr, total_hidden_weights_length,
									  total_bias_ptr,
									  gradient_approx,
									  limit_epsilon,
									  n_threads, thread_id );
	  results[thread_id].get();
	  for(i = 0; i < total_gradient_length; ++i)
	  {
		EXPECT_NEAR(gradient_approx[i], total_gradient_ptr[i], limit_epsilon)<<" ith_case: "<<ith_case<<" thread_id: "<<thread_id<<" i: "<<i<<"\n";
	  }

	}
	// clear futures
	results.clear();
  }
  // END FORWARD & BACKPROP PROP
/*  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length,  "total_activations");
  zinhart::serial::print_matrix_row_major(d_error, 1, total_error_length,  "d_error");
  zinhart::serial::print_matrix_row_major(total_deltas_ptr, 1, total_activations_length,  "total_deltas");
   zinhart::serial::print_matrix_row_major(total_gradient_ptr, 1, total_gradient_length,  "total_gradient");*/
  std::cout<<total_hidden_weights_length<<"\n"; 
  for(i = 0; i < total_layers.size(); ++i)
  {
	std::cout<<"Layer: "<<i<<" "<<total_layers[i].second<<"\n";
  }
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
  mkl_free(d_error);
  mkl_free(gradient_approx);
}

