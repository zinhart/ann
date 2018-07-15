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
// note that this test assumes one training case per thread, multiply training cases per threads = activations * n_threads * n_cases_per_thread
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
  double * activation_ptr{nullptr};

  // loop counters misc vars
  std::uint32_t i{0}, j{0},ith_layer{0}, ith_case{0}, thread_id{0}, activation_offset{0}, thread_stride{0}, n_layers{layer_dist(mt)};
  const std::uint32_t n_threads{thread_dist(mt)};
  // variables necessary for forward_propagation
  const std::uint32_t input_layer{0};
  std::uint32_t output_layer{0};
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
  thread_stride = total_activations_length;
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
  auto forward_propagate_lambda = [&model, &total_layers, &total_cases_ptr, &total_activations_ptr, &total_hidden_weights_ptr, &total_bias_ptr]
								  (const std::uint32_t & ith_training_case, const std::uint32_t & activations, const std::uint32_t & weights, const std::uint32_t total_threads, const std::uint32_t & thread_index)
								  {
									model.forward_propagate(total_layers,
															total_cases_ptr, ith_training_case,
															total_activations_ptr, activations,
															total_hidden_weights_ptr, weights,
															total_bias_ptr,
															total_threads,
															thread_index
										                   );
								  }; 


  
  // FORWARD PROPAGATE BEGIN
  ith_case = 0; 
  thread_id = 0;
  activation_offset = total_layers[input_layer].second * thread_id;
 
  std::cout<<"Total layers: "<<total_layers.size()<<"\n";
  for(ith_layer = 0; ith_layer < total_layers.size(); ++ith_layer)
  {
	std::cout<<"NEURONS IN LAYER: "<<ith_layer<<" "<<total_layers[ith_layer].second<<"\n";
  }
  std::cout<<"Thread stride: "<<thread_stride << "\n";
  // first hidden layer and input layer for all threads
  for(ith_case = 0; ith_case < /*total_cases*/1; ++ith_case)
  {
	current_inputs_ptr = total_cases_ptr + ( ith_case * total_layers[input_layer].second );
	for(thread_id = 0; thread_id < /*n_threads*/1; ++thread_id)
	{
	  results.push_back(pool.add_task(forward_propagate_lambda, ith_case, total_activations_length, total_hidden_weights_length, n_threads, thread_id));
	  std::cout<<"Thread_id: "<<thread_id<<" activation_offset: "<<activation_offset<<"\n";
	  activation_offset = thread_id * thread_stride;
	  activation_ptr = total_activations_ptr_test + activation_offset;
	  std::uint32_t layer_stride{0};
	  std::uint32_t prior_layer_stride{0};
	  std::uint32_t next_layer{0};
	  std::uint32_t weight_stride{0};
	  for(ith_layer = 0, next_layer = ith_layer + 1; ith_layer < /*total_layers.size() - 1*/1; ++ith_layer, ++next_layer)
	  {
		std::cout<<"CURRENT LAYER "<< next_layer <<" NEURONS: "<<total_layers[next_layer].second<<"\n";
		double * weight_ptr = total_hidden_weights_ptr;
		if(ith_layer == 0)
		{
		  std::cout<<"layer_stride: "<<layer_stride<<"\n";
		  std::cout<<"weight_stride: "<<weight_stride<<"\n";
  		  std::string s = "Weight matrix between layers: " + std::to_string(next_layer) + " " + std::to_string(ith_layer) 
		                + " dimensions " + std::to_string(total_layers[next_layer].second) + " " + std::to_string(total_layers[ith_layer].second);
  		  zinhart::serial::print_matrix_row_major(weight_ptr, total_layers[next_layer].second, total_layers[ith_layer].second, s);
		  zinhart::serial::print_matrix_row_major(current_inputs_ptr, total_layers[input_layer].second, 1, "Inputs");
		  m = total_layers[next_layer].second;
		  n = 1;
		  k = total_layers[input_layer].second;
		  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					  m, n, k,
					  alpha, weight_ptr, k,
					  current_inputs_ptr, n, beta, 
					  activation_ptr, n
				 );
		  zinhart::serial::print_matrix_row_major(activation_ptr, 1, total_layers[next_layer].second, "parallel activation vector");
		  for(i = layer_stride, j = 0; j < total_layers[next_layer].second; ++i, ++j)
			activation_ptr[i] += total_bias_ptr[input_layer];
		  for(i = layer_stride, j = 0; j < total_layers[next_layer].second; ++i, ++j)
			af(total_layers[next_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, activation_ptr[i]);
		  zinhart::serial::print_matrix_row_major(activation_ptr, 1, total_layers[next_layer].second, "parallel(add in bias + activation) activation vector");
		  prior_layer_stride = layer_stride;
		  layer_stride += total_layers[next_layer].second;
		  weight_stride += total_layers[next_layer].second * total_layers[ith_layer].second;
		}
		else
		{
		  std::cout<<"layer_stride: "<<layer_stride<<"\n";
		  std::cout<<"weight_stride: "<<weight_stride<<"\n";
		  weight_ptr = total_hidden_weights_ptr + weight_stride;
  		  std::string s = "Weight matrix between layers: " + std::to_string(next_layer) + " " + std::to_string(ith_layer) 
		                + " dimensions " + std::to_string(total_layers[next_layer].second) + " " + std::to_string(total_layers[ith_layer].second);
  		  zinhart::serial::print_matrix_row_major(weight_ptr, total_layers[next_layer].second, total_layers[ith_layer].second, s);
		  m = total_layers[next_layer].second;
		  n = 1;
		  k = total_layers[ith_layer].second;
		  if(ith_layer == 1)
		  {
  			zinhart::serial::print_matrix_row_major(activation_ptr, total_layers[ith_layer].second, 1, "Inputs");
  			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						m, n, k,
						alpha, weight_ptr, k,
						activation_ptr, n, beta, 
						activation_ptr + layer_stride, n
					   );
			zinhart::serial::print_matrix_row_major(activation_ptr + layer_stride, 1, total_layers[next_layer].second, "parallel activation vector");
			for(i = layer_stride, j = 0; j < total_layers[next_layer].second; ++i, ++j)
			  activation_ptr[i] += total_bias_ptr[ith_layer];
			for(i = layer_stride, j = 0; j < total_layers[next_layer].second; ++i, ++j)
			  af(total_layers[next_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, activation_ptr[i]);
			zinhart::serial::print_matrix_row_major(activation_ptr + layer_stride, 1, total_layers[next_layer].second, "parallel(add in bias + activation) activation vector");

		  }
		  else
		  {
			zinhart::serial::print_matrix_row_major(activation_ptr + prior_layer_stride, total_layers[ith_layer].second, 1, "Inputs");
  			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						m, n, k,
						alpha, weight_ptr, k,
						activation_ptr + prior_layer_stride, n, beta, 
						activation_ptr + layer_stride, n
					   );
			zinhart::serial::print_matrix_row_major(activation_ptr + layer_stride, 1, total_layers[next_layer].second, "parallel activation vector");
			for(i = layer_stride, j = 0; j < total_layers[next_layer].second; ++i, ++j)
			  activation_ptr[i] += total_bias_ptr[ith_layer];
			for(i = layer_stride, j = 0; j < total_layers[next_layer].second; ++i, ++j)
			  af(total_layers[next_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, activation_ptr[i]);
			zinhart::serial::print_matrix_row_major(activation_ptr + layer_stride, 1, total_layers[next_layer].second, "parallel(add in bias + activation) activation vector");
		  }
		  prior_layer_stride = layer_stride;
		  layer_stride += total_layers[next_layer].second;
		  weight_stride += total_layers[next_layer].second * total_layers[ith_layer].second;
		}
		std::cout<<"\n";
	  }
	  results[thread_id].get();
	  for(i = 0; i < thread_stride; ++i)
		ASSERT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
	}
  }
  // SERIAL FORWARD PROPAGATE END
  // hope fully by the time the serial forward prop has finished all the threads will be done as well
  for(thread_id = 0; thread_id < results.size(); ++thread_id)
  {
  }
  // PARALLEL FORWARD PROPAGATE END
  
  //  parallel forward propagation
/*  ith_case = 0;

  for (ith_layer = 0; ith_layer < 1; ++ith_layer)
  {
	for(thread_id = 0, activation_offset = total_layers[input_layer].second * thread_id; thread_id < n_threads; ++thread_id, activation_ptr+= thread_id * total_layers[ith_layer].second)
	{
	  std::cout<<"thread_id " + std::to_string(thread_id) + " layer: " + std::to_string(ith_layer) +"\n";
	  zinhart::serial::print_matrix_row_major(total_activations_ptr + activation_offset, 1, total_layers[1].second, "parallel activation vector");
	}
  }
  //zinhart::serial::print_matrix_row_major(total_activations_ptr, 1, total_activations_length, "parallel activation vector");
  zinhart::serial::print_matrix_row_major(total_hidden_weights_ptr, total_layers[input_layer + 1].second, total_layers[input_layer].second, "Weight matrix layers 0 -> 1");
  zinhart::serial::print_matrix_row_major(total_cases_ptr, total_layers[input_layer].second, 1, "Inputs");*/


  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_cases_ptr);
}
