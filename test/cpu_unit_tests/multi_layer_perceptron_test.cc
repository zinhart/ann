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
//	  std::string s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
// 	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
//	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
	  
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

//	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr_test, 1, total_activations_length, "serial hidden inputs vector");
//	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr, 1, total_activations_length, "parallel hidden inputs vector");
//	  zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
//	  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");
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
//	  std::string s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
//	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
//	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
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

	   multi_layer_perceptron<connection::dense, double> mlp;
	  mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr,
					  n_threads,
					  thread_id
					 );
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";

//	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr_test, 1, total_activations_length, "serial hidden inputs vector");
//	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr, 1, total_activations_length, "parallel hidden inputs vector");
//	  zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
//	  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");
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
/*
TEST(multi_layer_perceptron, backward_propagate)
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


  // the thread pool & futures
  zinhart::parallel::thread_pool pool(n_threads);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;

  // the model
   multi_layer_perceptron<connection::dense, double> model;
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
  const std::uint32_t output_layer_nodes{total_layers[total_layers.size() - 1].second};


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
  auto bprop_model = [](const std::vector<zinhart::activation::LAYER_INFO> & layers, 
					 const double error,
					 const double * total_training_cases_init, const double * total_targets_init, const std::uint32_t ith_training_case,
					 double * total_hidden_inputs_init, double * total_activations_init, double * total_deltas_init, const std::uint32_t activations_length,
					 const double * total_hidden_weights_init, double * total_gradient_init, const std::uint32_t weights_length,
					 const double * const total_bias_init,
					 const std::uint32_t total_threads,
					 const std::uint32_t thread_index
					)
					{
					   multi_layer_perceptron<connection::dense, double> mlp;
				 	  mlp.backward_propagate(layers,
											error,
											total_training_cases_init, total_targets_init, ith_training_case,
											total_hidden_inputs_init, total_activations_init, total_deltas_init, activations_length,
											total_hidden_weights_init, total_gradient_init, weights_length,
											total_bias_init,
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
	  std::string s;
//	  s = "Weight matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
// 	  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);
//	  zinhart::serial::print_matrix_row_major(current_training_case, total_layers[input_layer].second, 1, "Inputs");
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

//		zinhart::serial::print_matrix_row_major(current_layer_ptr, 1, total_layers[current_layer].second, "activation vector");
	
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

	  // synchronize w.r.t the current thread, forward prop for the current thread ends here
	  results[thread_id].get();



	  // validate forward prop outputs
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
	  // validate output_layer
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";

//	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr_test, 1, total_activations_length, "serial hidden inputs vector");
//	  zinhart::serial::print_matrix_row_major(total_hidden_input_ptr, 1, total_activations_length, "parallel hidden inputs vector");
//	  zinhart::serial::print_matrix_row_major(total_activations_ptr_test, 1, total_activations_length, "serial activation vector");
//	  zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");
	  
	  double * current_target{total_targets_ptr + (ith_case * total_layers[output_layer].second)};
	  // calculate error 
	  // error  = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::OBJECTIVE, outputs, targets, n_elements);
	  // calculate error derivative
	  
//	  zinhart::serial::print_matrix_row_major(outputs_ptr_test, 1, total_layers[output_layer].second, "output layer ptr");
	  error = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::DERIVATIVE, outputs_ptr_test, current_target, total_layers[output_layer].second);
	  //std::cout<<"error: "<<error<<"\n";
	  // begin backprop 
	  results[thread_id] = pool.add_task(bprop_model,std::ref(total_layers), error, total_cases_ptr, total_targets_ptr, ith_case, 
									  total_hidden_input_ptr, total_activations_ptr, total_deltas_ptr, total_activations_length, 
									  total_hidden_weights_ptr, total_gradient_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id );
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

	 current_threads_activation_index = thread_id * activation_stride;
	 current_threads_gradient_index = thread_id * gradient_stride;

	 current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	 current_threads_hidden_input_ptr = total_hidden_input_ptr_test + current_threads_activation_index;
	 current_threads_delta_ptr = total_deltas_ptr_test + current_threads_activation_index;
	 current_threads_gradient_ptr = total_gradient_ptr_test + current_threads_gradient_index;

	 // set pointers
	 double * current_layers_hidden_input_ptr{current_threads_hidden_input_ptr + current_layer_index};
	 double * current_layer_activation_ptr{current_threads_activation_ptr + current_layer_index};
	 // if this is a 2 layer model then the prior activations are essentially the inputs to the model
	 const double * prior_layer_activation_ptr{(total_layers.size() > 2) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case}; 
	 double * current_layer_deltas_ptr{current_threads_delta_ptr + current_layer_index};
	 double * current_gradient_ptr{current_threads_gradient_ptr + current_gradient_index};

	 // calculate output layer deltas
	 for(i = current_threads_activation_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
	   total_deltas_ptr_test[i] = error * af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::DERIVATIVE, total_hidden_input_ptr_test[i]);


	 // set up to calculate output layer gradient 
	 m = total_layers[current_layer].second;
	 n = total_layers[previous_layer].second;
	 k = 1; 

	 // calc output layer gradient
	 cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				  m, n, k,
				  alpha, current_layer_deltas_ptr, k,
				  prior_layer_activation_ptr, n, beta, 
				  current_gradient_ptr, n
				 );



//	 zinhart::serial::print_matrix_row_major(current_layer_activation_ptr, 1, total_layers[current_layer].second, "current_layers activation");
//	 zinhart::serial::print_matrix_row_major(current_layer_deltas_ptr, total_layers[current_layer].second, 1, "current_layers deltas ");
//	 zinhart::serial::print_matrix_row_major(prior_layer_activation_ptr, 1, total_layers[previous_layer].second, "prior_layers activation");


//	  s = "Gradient matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second);
 //	  zinhart::serial::print_matrix_row_major(current_gradient_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s);

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


//		s = "weight matrix between layers: " + std::to_string(next_layer) + " " + std::to_string(current_layer) + " dimensions: " + std::to_string(total_layers[next_layer].second) + " " + std::to_string(total_layers[current_layer].second);
//	  	zinhart::serial::print_matrix_row_major(weight_ptr, total_layers[current_layer].second, total_layers[next_layer].second, s);
//	    zinhart::serial::print_matrix_row_major(next_layer_delta_ptr, total_layers[next_layer].second, 1, "next layers deltas");

		m = total_layers[current_layer].second;
	    n = 1;
	    k = total_layers[next_layer].second;

   		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				  m, n, k,
				  alpha, weight_ptr, m,
				  next_layer_delta_ptr, n, beta, 
				  current_layer_deltas_ptr, n
				 );/

//	    zinhart::serial::print_matrix_row_major(current_layer_deltas_ptr, total_layers[current_layer].second, 1, "current layers deltas");

   		// calculate current layer deltas
		for(i = current_threads_activation_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		  total_deltas_ptr_test[i] *= af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::DERIVATIVE, total_hidden_input_ptr_test[i]);

//	    zinhart::serial::print_matrix_row_major(current_layer_deltas_ptr, total_layers[current_layer].second, 1, "current layers deltas");
//	    zinhart::serial::print_matrix_row_major(previous_layer_activation_ptr, 1, total_layers[previous_layer].second, "prior layers activation");

		m = total_layers[current_layer].second;
   		n = total_layers[previous_layer].second;
   		k = 1;
   
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				  m, n, k,
				  alpha, current_layer_deltas_ptr, k,
				  previous_layer_activation_ptr, n, beta, 
				  current_gradient_ptr, n
				 );
//
//		s = (current_layer > 0) ? 
//		  "current gradient matrix between layers: " + std::to_string(current_layer) + " " + std::to_string(previous_layer) + " dimensions: " + std::to_string(total_layers[current_layer].second) + " " + std::to_string(total_layers[previous_layer].second) 
//		  :"current gradient matrix between layers: " + std::to_string(next_layer) + " " + std::to_string(current_layer) + " dimensions: " + std::to_string(total_layers[next_layer].second) + " " + std::to_string(total_layers[current_layer].second) 
//		  ;
//		(current_layer > 0) ? zinhart::serial::print_matrix_row_major(current_gradient_ptr, total_layers[current_layer].second, total_layers[previous_layer].second, s)
//		                    : zinhart::serial::print_matrix_row_major(current_gradient_ptr, total_layers[next_layer].second, total_layers[current_layer].second, s);
//
		next_layer_index = current_layer_index;
		--next_layer;
		--current_layer;
		--previous_layer;
//		
//		std::cout<<"current_layer: "<<current_layer<<"\n";
//		std::cout<<"previous_layer: "<<previous_layer<<"\n";
//		std::cout<<"current_layer_index: "<<current_layer_index<<"\n";
//		std::cout<<"previous_layer_index: "<<previous_layer_index<<"\n";
//		
	  }
//	  std::cout<<"total_layers: "<<total_layers.size()<<"\n";	
//	  std::cout<<"Total threads: "<<n_threads<<"\n";
//	  std::cout<<"total cases: "<<total_cases<<"\n";
	  // serial backprop done
	  // synchronize w.r.t the current thread, back prop ends here
	  results[thread_id].get();

	  // validate bprop outputs
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_hidden_input_ptr[i], total_hidden_input_ptr_test[i]);
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
//	  for(i = 0; i < total_activations_length; ++i)
//		EXPECT_DOUBLE_EQ(total_deltas_ptr[i], total_deltas_ptr_test[i]);
//	  	for(i = 0; i < total_gradient_length; ++i)
//		EXPECT_NEAR(total_gradient_ptr[i], total_gradient_ptr_test[i], std::numeric_limits<double>::epsilon())<< "i: "<<i<<"\n";
//		
	}
	// clear futures
	results.clear();
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

TEST(multi_layer_perceptron, gradient_check)
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
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_gradient_length{0}, total_bias_length{0}, total_case_length{0}, total_targets_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_deltas_ptr{nullptr};
  double * total_hidden_input_ptr{nullptr};
  double * total_hidden_weights_ptr_numeric{nullptr};
  double * total_hidden_weights_ptr_analytic{nullptr};
  double * total_gradient_ptr_numeric{nullptr};
  double * total_gradient_ptr_analytic{nullptr};
  double * total_bias_ptr{nullptr};
  double * total_targets_ptr{nullptr};
  double * current_inputs_ptr{nullptr};
  double * total_cases_ptr{nullptr};
  double * outputs_ptr{nullptr};
  double * outputs_ptr_plus{nullptr};
  double * outputs_ptr_minus{nullptr};


  std::uint32_t i{0}, j{0}, ith_layer{0},ith_case{0}, thread_id{0}, activation_stride{0}, n_layers{layer_dist(mt)};
  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t input_layer{0};
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

  const std::uint32_t output_layer{total_layers.size() - 1};
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_case_length = total_layers[input_layer].second *  n_threads; //case_dist(mt);
  total_cases = total_case_length / total_layers[input_layer].second;
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  activation_stride = total_activations_length;// important!
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  total_gradient_length = total_hidden_weights_length * n_threads;

  total_bias_length = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer].second * total_cases;
  std::uint32_t alignment = 64;
  const std::uint32_t output_layer_nodes{total_layers[total_layers.size() - 1].second};

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_input_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_weights_ptr_analytic = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_hidden_weights_ptr_numeric = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_gradient_ptr_analytic = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  total_gradient_ptr_numeric = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  outputs_ptr = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  outputs_ptr_plus = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  outputs_ptr_minus = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_cases_ptr = (double*) mkl_malloc( total_case_length * sizeof( double ), alignment );
  total_targets_ptr = (double*) mkl_malloc(total_targets_length * sizeof(double), alignment );

  // set random training data 
  for(i = 0; i < total_case_length; ++i)
	total_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_targets_length; ++i)
	total_targets_ptr[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = 0.0;
	total_hidden_input_ptr[i] = 0.0;
	total_deltas_ptr[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
  {
	total_hidden_weights_ptr_numeric[i] = real_dist(mt);
	total_hidden_weights_ptr_analytic[i] = real_dist(mt);
  }
   for(i = 0; i < total_gradient_length; ++i)
  {
	total_gradient_ptr_analytic[i] = 0.0;
	total_gradient_ptr_numeric[i] = 0.0;
  }
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);
  for(i = 0; i < output_layer_nodes; ++i)
  {
	outputs_ptr[i] = 0.0;
	outputs_ptr_plus[i] = 0.0;
	outputs_ptr_minus[i] = 0.0;
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
 
  auto bprop_model = [](const std::vector<zinhart::activation::LAYER_INFO> & layers, 
					 const double error,
					 const double * total_training_cases_init, const double * total_targets_init, const std::uint32_t ith_training_case,
					 double * total_hidden_inputs_init, double * total_activations_init, double * total_deltas_init, const std::uint32_t activations_length,
					 const double * total_hidden_weights_init, double * total_gradient_init, const std::uint32_t weights_length,
					 const double * const total_bias_init,
					 const std::uint32_t total_threads,
					 const std::uint32_t thread_index
					)
					{
					   multi_layer_perceptron<connection::dense, double> mlp;
				 	  mlp.backward_propagate(layers,
											error,
											total_training_cases_init, total_targets_init, ith_training_case,
											total_hidden_inputs_init, total_activations_init, total_deltas_init, activations_length,
											total_hidden_weights_init, total_gradient_init, weights_length,
											total_bias_init,
											total_threads,
											thread_index
										   );
					};
  ith_case = 0;
  zinhart::error_metrics::loss_function loss;
  double gradient_epsilon = 1.e-4;
   multi_layer_perceptron<connection::dense, double> mlp;
  double * current_target{total_targets_ptr + (ith_case * total_layers[output_layer].second)};

  for(i = 0; i < total_hidden_weights_length; ++i)
  {	  	
	double theta = total_hidden_weights_ptr_numeric[i];
	double theta_plus = total_hidden_weights_ptr_numeric[i] + gradient_epsilon;
	double theta_minus = total_hidden_weights_ptr_numeric[i] - gradient_epsilon;
	double error = 0;

	total_hidden_weights_ptr_numeric[i] = theta_plus;
	mlp.forward_propagate(std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr_numeric, total_hidden_weights_length, total_bias_ptr, 1, 0);
	mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr_plus,
					  1,
					  0
					 );
	error = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::DERIVATIVE, outputs_ptr_plus, current_target, total_layers[output_layer].second);

	mlp.backward_propagate(std::ref(total_layers), error, total_cases_ptr, total_targets_ptr, ith_case, 
									  total_hidden_input_ptr, total_activations_ptr, total_deltas_ptr, total_activations_length, 
									  total_hidden_weights_ptr_numeric, total_gradient_ptr_numeric, total_hidden_weights_length, total_bias_ptr, 1, 0);
	double gradient_plus = total_gradient_ptr_numeric[i];

	total_hidden_weights_ptr_numeric[i] = theta_minus;

	mlp.forward_propagate(std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr_numeric, total_hidden_weights_length, total_bias_ptr, 1, 0);

	mlp.get_outputs(total_layers,
					  total_activations_ptr, total_activations_length,
					  outputs_ptr_minus,
					  1,
					  0
					 );

	error = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::DERIVATIVE, outputs_ptr_minus, current_target, total_layers[output_layer].second);
	mlp.backward_propagate(std::ref(total_layers), error, total_cases_ptr, total_targets_ptr, ith_case, 
									  total_hidden_input_ptr, total_activations_ptr, total_deltas_ptr, total_activations_length, 
									  total_hidden_weights_ptr_numeric, total_gradient_ptr_numeric, total_hidden_weights_length, total_bias_ptr, 1, 0);
	double gradient_minus = total_gradient_ptr_numeric[i];


	total_hidden_weights_ptr_numeric[i] = theta;
	mlp.forward_propagate(std::ref(total_layers), total_cases_ptr, ith_case, total_hidden_input_ptr, total_activations_ptr, total_activations_length, total_hidden_weights_ptr_numeric, total_hidden_weights_length, total_bias_ptr, 1, 0);

	error = loss(zinhart::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::error_metrics::LOSS_FUNCTION_TYPE::DERIVATIVE, outputs_ptr, current_target, total_layers[output_layer].second);
	mlp.backward_propagate(std::ref(total_layers), error, total_cases_ptr, total_targets_ptr, ith_case, 
									  total_hidden_input_ptr, total_activations_ptr, total_deltas_ptr, total_activations_length, 
									  total_hidden_weights_ptr_numeric, total_gradient_ptr_numeric, total_hidden_weights_length, total_bias_ptr, 1, 0);
	double numeric_gradient = total_gradient_ptr_numeric[i];
	double analytic_gradient = (gradient_plus - gradient_minus) / (2 * gradient_epsilon);
	//EXPECT_DOUBLE_EQ(analytic_gradient, numeric_gradient )<< "i: "<<i<<"\n";

	std::cout<<analytic_gradient<< " "<< numeric_gradient<<"\n"; 

  }

  // gradient check 

  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_deltas_ptr);
  mkl_free(total_hidden_input_ptr);
  mkl_free(total_hidden_weights_ptr_numeric);
  mkl_free(total_hidden_weights_ptr_analytic);
  mkl_free(total_gradient_ptr_analytic);
  mkl_free(total_gradient_ptr_numeric);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
  mkl_free(total_targets_ptr);
  mkl_free(outputs_ptr);
  mkl_free(outputs_ptr_plus);
  mkl_free(outputs_ptr_minus);
}
*/

/*
 * I've taken the vales here of a known neural network architecture with a very small data set to check my gradient check
 * */
TEST(mlp, pre_gradient_check_mazur)
{
  const std::uint32_t input_layer{0};
  const std::uint32_t output_layer{2};
  const std::uint32_t activations_length{4};
  const std::uint32_t weights_length{8};
  std::uint32_t i{0}, j{0}, k{0};
  // set layers
  std::vector<LAYER_INFO> total_layers;
  LAYER_INFO a_layer;

  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = 2;

  total_layers.push_back(a_layer);

  a_layer.first = ACTIVATION_NAME::SIGMOID;
  a_layer.second = 2;
 
  total_layers.push_back(a_layer);

  a_layer.first = ACTIVATION_NAME::SIGMOID;
  a_layer.second = 2;

  total_layers.push_back(a_layer);

  const std::uint32_t bias_length{total_layers.size() -1};
  std::uint32_t alignment = 64;
  double * inputs{(double*) mkl_malloc( total_layers[input_layer].second * sizeof( double ), alignment )};
  double * targets{(double*) mkl_malloc( total_layers[output_layer].second * sizeof( double ), alignment )};
  double * deltas{(double*) mkl_malloc( activations_length * sizeof( double ), alignment )};
  double * hidden_input{(double*) mkl_malloc( activations_length * sizeof( double ), alignment )};
  double * activations{(double*) mkl_malloc( activations_length * sizeof( double ), alignment )};
  double * weights{(double*) mkl_malloc( weights_length * sizeof( double ), alignment )};
  double * weights1{(double*) mkl_malloc( weights_length * sizeof( double ), alignment )};
  double * gradients{(double*) mkl_malloc( weights_length * sizeof( double ), alignment )};
  double * bias{(double*) mkl_malloc( bias_length * sizeof( double ), alignment )};
  double * d_error{(double*) mkl_malloc(bias_length * sizeof( double ), alignment )};



  inputs[0] = 0.05;
  inputs[1] = 0.1;

  weights[0] = .15;
  weights[1] = .20;
  weights[2] = .25;
  weights[3] = .30;

  weights[4] = .40;
  weights[5] = .45;
  weights[6] = .50;
  weights[7] = .55;



  weights1[0] = .15;
  weights1[1] = .20;
  weights1[2] = .25;
  weights1[3] = .30;


  weights1[4] = .40;
  weights1[5] = .45;
  weights1[6] = .50;
  weights1[7] = .55;

  bias[0] = .35;
  bias[1] = .60;

  targets[0] = .01;
  targets[1] = 0.99;
  for(i = 0; i < 8; ++i)
	gradients[i] = 0.0;

  
  multi_layer_perceptron<connection::dense, double> mlp;
  zinhart::function_space::error_metrics::loss_function loss;


  mlp.forward_propagate(std::ref(total_layers), inputs, 0, hidden_input, activations, 4, weights, 8, bias, 1, 0);

  zinhart::serial::print_matrix_row_major(hidden_input, 1, 4 , "hidden_input");
  zinhart::serial::print_matrix_row_major(activations, 1, 4 , "activations");
  zinhart::serial::print_matrix_row_major(targets, 1, 2 , "total_targets");
  
  double error = loss(zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::function_space::OBJECTIVE(), activations + 2, targets, total_layers[2].second, 2);

  std::cout<<"error: "<<error<<"\n";

  loss(zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::function_space::DERIVATIVE(), activations + 2, targets, d_error, total_layers[2].second, 2);
  mlp.backward_propagate(std::ref(total_layers), inputs, targets, d_error, 0, 
									  hidden_input, activations, deltas, 4, 
									  weights, gradients, 8, bias);
  std::cout.precision(10);
  zinhart::serial::print_matrix_row_major(d_error, 1, 2, "d_error");
  zinhart::serial::print_matrix_row_major(gradients, 1, 8, "gradients");
  zinhart::serial::print_matrix_row_major(deltas, 1, 4 , "deltas");
	/* */
	  for(i = 0; i < weights_length; ++i)
		weights[i] -= 0.5 * gradients[i];

  zinhart::serial::print_matrix_row_major(weights, 1, 8, "weights");
  // gradient check
  const double gradient_epsilon = 1.e-4;
  for(i = 0; i < 1; ++i)
  {
	for(j = 0; j < /*weights_length*/1; ++j)
	{
	  double theta = weights1[j];
	  double theta_plus =  weights1[j] + gradient_epsilon;
	  double theta_minus = weights1[j] - gradient_epsilon;
	  double numerical_gradient{0};
	  double analytic_gradient{0};
	  double gradient_plus{0};
	  double gradient_minus{0};

	  for(k = 0; k < total_layers[output_layer].second; ++k)
		d_error[k] = 0;
	  for(k = 0; k < activations_length; ++k)
	  {
		hidden_input[k] = 0;
		activations[k] = 0;
		deltas[k] = 0;
	  }
	  for(k = 0; k < weights_length; ++k)
		gradients[k] = 0;

	  weights1[j] = theta;
	  mlp.forward_propagate(std::ref(total_layers), inputs, 0, hidden_input, activations, 4, weights1, 8, bias, 1, 0);
	  loss(zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::function_space::DERIVATIVE(), activations + 2, targets, d_error, total_layers[2].second, 2);
	  mlp.backward_propagate(std::ref(total_layers), inputs, targets, d_error, 0, 
										  hidden_input, activations, deltas, 4, 
										  weights1, gradients, 8, bias);
	  numerical_gradient = gradients[j];  

	  for(k = 0; k < total_layers[output_layer].second; ++k)
		d_error[k] = 0;
	  for(k = 0; k < activations_length; ++k)
	  {
		hidden_input[k] = 0;
		activations[k] = 0;
		deltas[k] = 0;
	  }
	  for(k = 0; k < weights_length; ++k)
		gradients[k] = 0;

	  
	  weights1[j] = theta_plus;
	  std::cout<<weights1[j]<<"\n";
	  mlp.forward_propagate(std::ref(total_layers), inputs, 0, hidden_input, activations, 4, weights1, 8, bias, 1, 0);
	  loss(zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::function_space::DERIVATIVE(), activations + 2, targets, d_error, total_layers[2].second, 2);
	  mlp.backward_propagate(std::ref(total_layers), inputs, targets, d_error, 0, 
										  hidden_input, activations, deltas, 4, 
										  weights1, gradients, 8, bias);
	  gradient_plus = gradients[j];

	  for(k = 0; k < total_layers[output_layer].second; ++k)
		d_error[k] = 0;
	  for(k = 0; k < activations_length; ++k)
	  {
		hidden_input[k] = 0;
		activations[k] = 0;
		deltas[k] = 0;
	  }
	  for(k = 0; k < weights_length; ++k)
		gradients[k] = 0;
	  weights1[j] = theta_minus;
	  mlp.forward_propagate(std::ref(total_layers), inputs, 0, hidden_input, activations, 4, weights1, 8, bias, 1, 0);
	  loss(zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE, zinhart::function_space::DERIVATIVE(), activations + 2, targets, d_error, total_layers[2].second, 2);
	  mlp.backward_propagate(std::ref(total_layers), inputs, targets, d_error, 0, 
										  hidden_input, activations, deltas, 4, 
										  weights1, gradients, 8, bias);
	  gradient_minus = gradients[j];
	
  	  analytic_gradient = (gradient_plus - gradient_minus) / (2 * gradient_epsilon);
	  EXPECT_DOUBLE_EQ(analytic_gradient, numerical_gradient )<< "i: "<<i<<"\n";


	  std::cout<<"analytic_gradient: "<<analytic_gradient<< " numerical_gradient: "<< numerical_gradient <<"\n"; 
	  //for(i = ; i < weights_length; ++i)
		//weights[i] -= 0.5 * gradients[i];

	}

  }
    
  
  mkl_free(inputs);
  mkl_free(hidden_input);
  mkl_free(deltas);
  mkl_free(activations);
  mkl_free(weights);
  mkl_free(weights1);
  mkl_free(gradients);
  mkl_free(bias);

}

