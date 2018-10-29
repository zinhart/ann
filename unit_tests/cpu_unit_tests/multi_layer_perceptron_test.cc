#include <ann/ann.hh>
#include <multi_core/multi_core.hh>
#include <ann/function_space.hh>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
#include <vector>
#include <iomanip>
using namespace zinhart::models;

void random_layer(std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > & total_layers, std::uint32_t layer_id, std::uint32_t layer_size)
{
  if(layer_id == 1)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::identity_layer<double>>());
	total_layers.back()->set_size(layer_size);
  } 
  else if(layer_id == 2)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::sigmoid_layer<double>>());
	total_layers.back()->set_size(layer_size);
  }
  else if(layer_id == 3)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::softplus_layer<double>>());
	total_layers.back()->set_size(layer_size);
  }
  else if(layer_id == 4)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::tanh_layer<double>>());
	total_layers.back()->set_size(layer_size);
  }
  else if(layer_id == 5)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::relu_layer<double>>());
	total_layers.back()->set_size(layer_size);
  }
  else if(layer_id == 6)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::leaky_relu_layer<double>>());
	total_layers.back()->set_size(layer_size);
  }
  else if(layer_id == 7)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::exp_leaky_relu_layer<double>>());
	total_layers.back()->set_size(layer_size);
  }
  else if(layer_id == 8)
  {
	total_layers.push_back(std::make_shared<zinhart::models::layers::softmax_layer<double>>(layer_size));
	total_layers.back()->set_size(layer_size);
  }
 /* else if(layer_id == 9)
  {
  }*/
}

TEST(multi_layer_perceptron, forward_propagate_thread_safety)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, /*total_activation_types() - 1*/7);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_bias_length{0}, total_training_cases_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  //double * current_inputs_ptr{nullptr};
  double * total_training_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};

  // loop counters misc vars
  std::uint32_t i{0}, ith_layer{0},ith_case{0}, thread_id{0}, activation_stride{0}, n_layers{layer_dist(mt)};
  const std::uint32_t n_threads{thread_dist(mt)};
  // variables necessary for forward_propagation
  const std::uint32_t input_layer{0};
  std::uint32_t current_layer{0};
  std::uint32_t previous_layer{0};
  std::uint32_t current_layer_index{0};
  std::uint32_t previous_layer_index{0};
  std::uint32_t weight_index{0};
  std::uint32_t current_threads_activation_index{0};
//  std::uint32_t case_index{0};
  std::uint32_t m{0}, n{0}, k{0};
  double alpha{1.0}, beta{0.0};


  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

  // the model
  multi_layer_perceptron<double> model;
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  zinhart::function_space::objective objective_function{};

  // set layers
  total_layers.push_back(std::make_shared< zinhart::models::layers::input_layer<double> >());
  total_layers[input_layer]->set_size(neuron_dist(mt));
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	random_layer(total_layers, layer_dist(mt), neuron_dist(mt));
  }
  
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_training_cases_length = total_layers[input_layer]->get_size() * case_dist(mt);
  total_cases = total_training_cases_length / total_layers[input_layer]->get_size();
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
  activation_stride = total_activations_length;// important!
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 
  total_bias_length = total_layers.size() - 1;

  std::uint32_t alignment = 64;

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_weights_ptr = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_training_cases_ptr = (double*) mkl_malloc( total_training_cases_length * sizeof( double ), alignment );

  // set random training data 
  for(i = 0; i < total_training_cases_length; ++i)
	total_training_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = 0.0;
	total_activations_ptr_test[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
	total_hidden_weights_ptr[i] = real_dist(mt);
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = real_dist(mt);


  // BEGIN FORWARD PROP
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	const double * current_training_case{total_training_cases_ptr + (ith_case * total_layers[input_layer]->get_size())};
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
	  
	  results.push_back(pool.add_task(fprop_mlp<double>, std::ref(total_layers), total_training_cases_ptr, ith_case, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
	  
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;

	  m = total_layers[current_layer]->get_size();
	  n = 1;
	  k = total_layers[previous_layer]->get_size();

	  current_threads_activation_index = thread_id * activation_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_activation_ptr, n
				 );

	  // activate
	  total_layers[current_layer]->activate(objective_function, current_threads_activation_ptr, total_layers[current_layer]->get_size(), total_bias_ptr[previous_layer]);
		

	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer]->get_size();

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;

	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 

		m = total_layers[current_layer]->get_size();
		n = 1;
		k = total_layers[previous_layer]->get_size();

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_ptr, n
				   );
  
	    // activate
		total_layers[current_layer]->activate(objective_function, current_layer_ptr, total_layers[current_layer]->get_size(), total_bias_ptr[previous_layer]);
		
		// update weight matrix index	
		weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer]->get_size();
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
	   }

	  // synchronize w.r.t the current thread 
	  results[thread_id].get();

	  // validate
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);
	  
	}
	results.clear();
  }

  // END FORWARD PROP
  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_training_cases_ptr);
}

TEST(multi_layer_perceptron, get_results_thread_safety)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, 7);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_bias_length{0}, total_training_cases_length{0}, total_cases{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  //double * current_inputs_ptr{nullptr};
  double * total_training_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};
  //double * current_threads_hidden_input_ptr{nullptr};
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
  std::uint32_t m{0}, n{0}, k{0};
  double alpha{1.0}, beta{0.0};


  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

  // the model
  multi_layer_perceptron<double> model;
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  zinhart::function_space::objective objective_function{};

  // set layers
  total_layers.push_back(std::make_shared< zinhart::models::layers::input_layer<double> >());
  total_layers[input_layer]->set_size(neuron_dist(mt));
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	random_layer(total_layers, layer_dist(mt), neuron_dist(mt));
  }
  
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_training_cases_length = total_layers[input_layer]->get_size() * case_dist(mt);
  total_cases = total_training_cases_length / total_layers[input_layer]->get_size();
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
  activation_stride = total_activations_length;// important!
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 
  total_bias_length = total_layers.size() - 1;

  const std::uint32_t alignment{64};
  const std::uint32_t output_layer_nodes{total_layers[total_layers.size() - 1]->get_size()};
  std::uint32_t output_layer_index{0};
  for(i = 1; i < total_layers.size(); ++i)
	output_layer_index += total_layers[i]->get_size();

  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  outputs_ptr = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  outputs_ptr_test = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  total_hidden_weights_ptr = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_training_cases_ptr = (double*) mkl_malloc( total_training_cases_length * sizeof( double ), alignment );

  // set random training data 
  for(i = 0; i < total_training_cases_length; ++i)
	total_training_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = 0.0;
	total_activations_ptr_test[i] = 0.0;
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


  // BEGIN FORWARD PROP
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	const double * current_training_case{total_training_cases_ptr + (ith_case * total_layers[input_layer]->get_size())};
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
	  results.push_back(pool.add_task(fprop_mlp<double>, std::ref(total_layers), total_training_cases_ptr, ith_case, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;

	  m = total_layers[current_layer]->get_size();
	  n = 1;
	  k = total_layers[previous_layer]->get_size();

	  current_threads_activation_index = thread_id * activation_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_activation_ptr, n
				 );

	  // activate
	  total_layers[current_layer]->activate(objective_function, current_threads_activation_ptr, total_layers[current_layer]->get_size(), total_bias_ptr[previous_layer]);

	  // save outputs if a two layer model
	  if(current_layer == total_layers.size() - 1)
		for(j = 0; j < total_layers[current_layer]->get_size(); ++j)
  		  outputs_ptr_test[j] = *(current_threads_activation_ptr + j);
			
	  // f(Wx + b complete) for first hidden layer and input layer
	  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer]->get_size();

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;
	  //std::cout<<"\n";
	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 

		m = total_layers[current_layer]->get_size();
		n = 1;
		k = total_layers[previous_layer]->get_size();

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_ptr, n
				   );

	    // activate
		total_layers[current_layer]->activate(objective_function, current_layer_ptr, total_layers[current_layer]->get_size(), total_bias_ptr[previous_layer]);

  		// save outputs if > a two layer model
  	    if(current_layer == total_layers.size() - 1)
		  for(j = 0; j < total_layers[current_layer]->get_size(); ++j)
		   	outputs_ptr_test[j] = *(current_layer_ptr + j);

		// update weight matrix index	
		weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer]->get_size();
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
	   }

	  // synchronize w.r.t the current thread 
	  results[thread_id].get();

	  // validate
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i]);

	  pool.add_task(get_outputs_mlp<double>, total_layers,
					total_activations_ptr, total_activations_length,
					outputs_ptr,
					n_threads,
					thread_id
				   ).get();

	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";
	}
	results.clear();
  }

  // END FORWARD PROP
  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(outputs_ptr);
  mkl_free(outputs_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_training_cases_ptr);
}

TEST(multi_layer_perceptron, gradient_check_thread_safety)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, 7);// does not include input layer
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t input_layer{0};
  std::uint32_t output_layer{0};
  const std::uint32_t alignment = 64;
  std::uint32_t i{0}, ith_case{0}, thread_id{0}, ith_layer{0}, n_layers{layer_dist(mt)}, total_cases{0}, total_cases_length{0}, total_targets_length{0}, total_activations_length, total_hidden_weights_length{0}, total_gradient_length{0};

  zinhart::loss_functions::loss_function<double> * loss = new zinhart::loss_functions::mean_squared_error<double>();
//  zinhart::loss_functions::loss_function<double> * loss = new zinhart::loss_functions::cross_entropy_multi_class<double>();
  
  double * total_training_cases_ptr{nullptr};
  double * total_targets{nullptr};
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
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

  // the model
   multi_layer_perceptron<double> model;
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  zinhart::function_space::objective objective_function{};

  // set layers
  total_layers.push_back(std::make_shared< zinhart::models::layers::input_layer<double> >());
  total_layers[input_layer]->set_size(neuron_dist(mt));
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	random_layer(total_layers, layer_dist(mt), neuron_dist(mt));
  }

  output_layer = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer]->get_size();
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_cases_length = total_layers[input_layer]->get_size() * case_dist(mt);
  total_cases = total_cases_length / total_layers[input_layer]->get_size();
  total_targets_length = total_layers[output_layer]->get_size() * total_cases;
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 
  total_gradient_length = total_hidden_weights_length * n_threads;
  const std::uint32_t bias_length{total_layers.size() -1};

  total_training_cases_ptr = (double*) mkl_malloc( total_cases_length * sizeof( double ), alignment );
  total_targets = (double*) mkl_malloc( total_targets_length * sizeof( double ), alignment );
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
	total_training_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_targets_length; ++i)
	total_targets[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations[i] = 0.0;
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

  
  multi_layer_perceptron<double> mlp;

  std::cout.precision(10);
  // for gradient checking
  const double limit_epsilon = 1.e-6;
  double right{0}, left{0}, original{0};




  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {

    const double * current_target{total_targets + (ith_case * total_layers[output_layer]->get_size())};
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
						  total_training_cases_ptr, ith_case, 
						  total_activations, total_activations_length,
						  total_hidden_weights_copy, total_hidden_weights_length, 
						  bias,
						  n_threads, thread_id
						 );

		mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);

		right = loss->error(objective_function, current_threads_output_layer_ptr, current_target, total_layers[output_layer]->get_size());
		// set back
		total_hidden_weights_copy[i] = original; 

		// left side
		total_hidden_weights_copy[i] -= limit_epsilon;
		mlp.forward_propagate(total_layers, 
					  total_training_cases_ptr, ith_case, 
					  total_activations, total_activations_length,
					  total_hidden_weights_copy, total_hidden_weights_length, 
					  bias,
					  n_threads, thread_id
					);
		mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
		left = loss->error(objective_function, current_threads_output_layer_ptr, current_target, total_layers[output_layer]->get_size());

		// calc numerically derivative for the ith_weight, save it, increment the pointer to the next weight
		*(current_threads_gradient_ptr + i) = (right - left) / (double{2} * limit_epsilon);
		// set back
		total_hidden_weights_copy[i] = original; 
	  }
	  // gradient check
	  results.push_back(pool.add_task(gradient_check_mlp<double>,
									  loss,
									  total_layers,
									  total_training_cases_ptr, total_targets, ith_case,
									  total_activations, total_activations_length,
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
		EXPECT_NEAR(numerically_approx_gradients_parallel[i], numerically_approx_gradients_serial[i], limit_epsilon)<<"i: "<<i<<"\n";
	  }
	}
	results.clear();
  }
 
  delete loss;
  mkl_free(total_training_cases_ptr);
  mkl_free(total_deltas);
  mkl_free(total_activations);
  mkl_free(total_hidden_weights);
  mkl_free(total_hidden_weights_copy);
  mkl_free(numerically_approx_gradients_parallel);
  mkl_free(numerically_approx_gradients_serial);
  mkl_free(bias);
  mkl_free(current_threads_output_layer_ptr);
  mkl_free(d_error);
  mkl_free(total_targets);
}

TEST(multi_layer_perceptron, backward_propagate_thread_safety)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, 7);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> loss_function_dist(0, 1);
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(0, 1);
  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_gradient_length{0}, total_bias_length{0}, total_training_cases_length{0}, total_targets_length{0}, total_cases{0}, total_error_length{0};
  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_activations_ptr_check{nullptr};
  double * total_deltas_ptr{nullptr};
  double * total_deltas_ptr_test{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_gradient_ptr{nullptr};
  double * total_gradient_ptr_test{nullptr};
  double * total_bias_ptr{nullptr};
  double * total_targets_ptr{nullptr};
  double * total_training_cases_ptr{nullptr};
  double * current_threads_activation_ptr{nullptr};
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
  const std::uint32_t input_layer{0};
  std::uint32_t current_layer{0};
  std::uint32_t previous_layer{0};
  std::uint32_t current_layer_index{0};
  std::uint32_t previous_layer_index{0};
  std::uint32_t weight_index{0};
  std::uint32_t current_threads_activation_index{0};
  std::uint32_t current_threads_gradient_index{0};
  std::uint32_t m{0}, n{0}, k{0};
  double alpha{1.0}, beta{0.0}, error{0.0};
  const double limit_epsilon = 1.e-4;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

  // the model
  multi_layer_perceptron<double> model;
  zinhart::loss_functions::loss_function<double> * loss = new zinhart::loss_functions::mean_squared_error<double>();
//  zinhart::loss_functions::loss_function<double> * loss = new zinhart::loss_functions::cross_entropy_multi_class<double>();

  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  zinhart::function_space::objective objective_function{};
  zinhart::function_space::derivative derivative_function{};

  // set layers
  total_layers.push_back(std::make_shared< zinhart::models::layers::input_layer<double> >());
  total_layers[input_layer]->set_size(neuron_dist(mt));
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
    //layer_dist(mt)
	random_layer(total_layers, 2, neuron_dist(mt));
  }
  
  const std::uint32_t output_layer{total_layers.size() - 1};
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_training_cases_length = total_layers[input_layer]->get_size() * case_dist(mt);
  total_cases = total_training_cases_length / total_layers[input_layer]->get_size();
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
  activation_stride = total_activations_length;// important!
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 
  gradient_stride = total_hidden_weights_length;
  total_gradient_length = total_hidden_weights_length * n_threads;// important!
  total_bias_length = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer]->get_size() * total_cases;


  const std::uint32_t alignment{64};
  const std::uint32_t output_layer_nodes{total_layers[output_layer]->get_size()};
  total_error_length = total_layers[output_layer]->get_size() * n_threads;


  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_check = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  outputs_ptr = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  outputs_ptr_test = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  total_hidden_weights_ptr = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_gradient_ptr = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  total_gradient_ptr_test = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_training_cases_ptr = (double*) mkl_malloc( total_training_cases_length * sizeof( double ), alignment );
  total_targets_ptr = (double*) mkl_malloc(total_targets_length * sizeof(double), alignment );
  d_error = (double*) mkl_malloc(total_error_length * sizeof(double), alignment );
  gradient_approx = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );


  // set random training data 
  for(i = 0; i < total_training_cases_length; ++i)
	total_training_cases_ptr[i] = real_dist(mt);
  for(i = 0; i < total_targets_length; ++i)
	total_targets_ptr[i] = real_dist(mt);
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = 0.0;
	total_activations_ptr_test[i] = 0.0;
	total_activations_ptr_check[i] = 0.0;
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


  // BEGIN FORWARD & BACKWARD PROP
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	const double * current_training_case{total_training_cases_ptr + (ith_case * total_layers[input_layer]->get_size())};
	current_target = total_targets_ptr + (ith_case * total_layers[output_layer]->get_size());
	std::uint32_t error_stride{total_layers[output_layer]->get_size()};
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
	  
	  results.push_back(pool.add_task(fprop_mlp<double>, std::ref(total_layers), total_training_cases_ptr, ith_case, total_activations_ptr, total_activations_length, total_hidden_weights_ptr, total_hidden_weights_length, total_bias_ptr, n_threads, thread_id));
	  current_layer = 1;
	  previous_layer = 0; 
	  current_layer_index = 0;
	  previous_layer_index = 0;
	  weight_index = 0;

	  m = total_layers[current_layer]->get_size();
	  n = 1;
	  k = total_layers[previous_layer]->get_size();

	  current_threads_activation_index = thread_id * activation_stride;
	  current_threads_activation_ptr = total_activations_ptr_test + current_threads_activation_index;
	  // Wx for first hidden layer and input layer
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, total_hidden_weights_ptr, k,
				  current_training_case, n, beta, 
				  current_threads_activation_ptr, n
				 );
	
      // activate
	  total_layers[current_layer]->activate(objective_function, current_threads_activation_ptr, total_layers[current_layer]->get_size(), total_bias_ptr[previous_layer]);
	  // f(Wx + b complete) for first hidden layer and input layer
	  
	  // save outputs if a two layer model
	  if(current_layer == total_layers.size() - 1)
		for(j = 0; j < total_layers[current_layer]->get_size(); ++j)
  		  outputs_ptr_test[j] = *(current_threads_activation_ptr + j);  

	  // update weight matrix index	
	  weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

	  // update layer indices
	  previous_layer_index = current_layer_index;
	  current_layer_index = total_layers[current_layer]->get_size();

	  //increment layer counters
	  ++current_layer;
	  ++previous_layer;

	  while( current_layer < total_layers.size() )
	  {
		const double * current_weight_matrix{total_hidden_weights_ptr + weight_index};
		double * current_layer_ptr{total_activations_ptr_test + current_threads_activation_index + current_layer_index};
		const double * prior_layer_ptr = total_activations_ptr_test + current_threads_activation_index + previous_layer_index; 

		m = total_layers[current_layer]->get_size();
		n = 1;
		k = total_layers[previous_layer]->get_size();

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_weight_matrix, k,
					prior_layer_ptr, n, beta, 
					current_layer_ptr, n
				   );

        // activate
		total_layers[current_layer]->activate(objective_function, current_layer_ptr, total_layers[current_layer]->get_size(), total_bias_ptr[previous_layer]);

  		// save outputs if a two layer model
  	    if(current_layer == total_layers.size() - 1)
		  for(j = 0; j < total_layers[current_layer]->get_size(); ++j)
		   	outputs_ptr_test[j] = *(current_layer_ptr + j);

		// update weight matrix index	
		weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index += total_layers[current_layer]->get_size();
		
		// increment layer counters 
		++current_layer; 
		++previous_layer;
	  }
	  
	  // synchronize w.r.t the current thread, forward prop for the current thread ends here
	  results[thread_id].get();

	  // validate forward prop outputs
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i])<<"i: "<<i<<"\n";

	   pool.add_task(get_outputs_mlp<double>, total_layers,
	 				 total_activations_ptr, total_activations_length,
					 outputs_ptr,
					 n_threads,
					 thread_id
				    ).get();

	  // validate output_layer
	  for(i = 0; i < output_layer_nodes; ++i)
		EXPECT_DOUBLE_EQ(outputs_ptr[i], outputs_ptr_test[i])<<"total_layers: "<<total_layers.size()<<"\n";

	  // calculate error 
	  error = loss->error(zinhart::function_space::objective(), outputs_ptr, current_target, output_layer_nodes);
	
	 current_layer_index = 0; 
	 for(i = 1; i < total_layers.size() - 1; ++i)
	   current_layer_index += total_layers[i]->get_size();// the start of the output layer
	 previous_layer_index = 0;
	 for(i = 1; i < total_layers.size() - 2; ++i)
	   previous_layer_index += total_layers[i]->get_size();// the start of the layer right behind the output layer

	 std::uint32_t current_gradient_index{0};
   	 // calc number of hidden weights
	 for(i = 0 ; i < total_layers.size() - 2; ++i)
	   current_gradient_index += total_layers[i + 1]->get_size() * total_layers[i]->get_size();
	 
	 current_layer = output_layer;
	 previous_layer = current_layer - 1; 

	 current_threads_gradient_index = thread_id * gradient_stride;

	 current_threads_delta_ptr = total_deltas_ptr_test + current_threads_activation_index;
	 current_threads_gradient_ptr = total_gradient_ptr_test + current_threads_gradient_index;

	 // set pointers
	 // if this is a 2 layer model then the prior activations are essentially the inputs to the model
	 const double * prior_layer_activation_ptr{(total_layers.size() > 2) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case}; 
	 double * current_layer_deltas_ptr{current_threads_delta_ptr + current_layer_index};
	 double * current_gradient_ptr{current_threads_gradient_ptr + current_gradient_index};

	 // calculate error derivative
	 double * current_error_matrix = d_error + (thread_id * error_stride);
     loss->error(zinhart::function_space::derivative(), outputs_ptr, current_target, current_error_matrix, output_layer_nodes);
 
	 // begin backprop 
	 results[thread_id] = pool.add_task(bprop_mlp<double>, std::ref(total_layers), total_training_cases_ptr, total_targets_ptr, d_error, ith_case, 
										 total_activations_ptr, total_deltas_ptr, total_activations_length, 
										 total_hidden_weights_ptr, total_gradient_ptr, total_hidden_weights_length, 
										 total_bias_ptr, 
										 n_threads, thread_id 
									    );

	  // calculate output layer deltas
	  total_layers[current_layer]->activate(zinhart::models::layers::layer_info::output_layer(), derivative_function, current_layer_deltas_ptr, current_error_matrix, 
			                                 (total_activations_ptr_test + current_threads_activation_index + current_layer_index), total_layers[current_layer]->get_size());

	 // set up to calculate output layer gradient 
	 m = total_layers[current_layer]->get_size();
	 n = total_layers[previous_layer]->get_size();
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
	  std::uint32_t output_layer_gradient_index{current_gradient_index};
	  
	  // calc hidden layer gradients
	  while(current_layer > 0)
	  {
		next_weight_matrix_index -= total_layers[next_layer]->get_size() * total_layers[current_layer]->get_size();	
		current_layer_index = previous_layer_index;
		previous_layer_index -= total_layers[previous_layer]->get_size(); 

		current_gradient_index -= total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();
		current_gradient_ptr = current_threads_gradient_ptr + current_gradient_index;

   		double * weight_ptr{total_hidden_weights_ptr +  next_weight_matrix_index};
		double * next_layer_delta_ptr{current_threads_delta_ptr + next_layer_index};
		current_layer_deltas_ptr = current_threads_delta_ptr + current_layer_index ;
		const double * previous_layer_activation_ptr{ (current_layer > 1) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case};

		m = total_layers[current_layer]->get_size();
	    n = 1;
	    k = total_layers[next_layer]->get_size();

   		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				  m, n, k,
				  alpha, weight_ptr, m,
				  next_layer_delta_ptr, n, beta, 
				  current_layer_deltas_ptr, n
				 );
  	    
		// calculate hidden layer deltas
	    total_layers[current_layer]->activate(zinhart::models::layers::layer_info::hidden_layer(), derivative_function, current_layer_deltas_ptr, 
			                                 (total_activations_ptr_test + current_threads_activation_index + current_layer_index), total_layers[current_layer]->get_size());

		m = total_layers[current_layer]->get_size();
   		n = total_layers[previous_layer]->get_size();
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
	  // serial backprop done
	  // synchronize w.r.t the current thread, back prop ends here
	  results[thread_id].get();

	  // validate bprop outputs
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_activations_ptr[i], total_activations_ptr_test[i])<< "case: "<<ith_case<<" thread_id: "<<thread_id<<" i: "<<i<<"\n";
	  for(i = 0; i < total_activations_length; ++i)
		EXPECT_DOUBLE_EQ(total_deltas_ptr[i], total_deltas_ptr_test[i])<< "case: "<<ith_case<<" thread_id: "<<thread_id<<" i: "<<i<<"\n";
	  for(i = 0; i < total_gradient_length; ++i)
		EXPECT_NEAR(total_gradient_ptr[i], total_gradient_ptr_test[i], std::numeric_limits<double>::epsilon())<< "case: "<<ith_case<<" thread_id: "<<thread_id<<" i: "<<i<<"\n";
	  
	  // gradient check
	  results[thread_id] = pool.add_task(gradient_check_mlp<double>,
									  loss,
									  total_layers,
									  total_training_cases_ptr, total_targets_ptr, ith_case,
									  total_activations_ptr_check, total_activations_length,
									  total_hidden_weights_ptr, total_hidden_weights_length,
									  total_bias_ptr,
									  gradient_approx,
									  limit_epsilon,
									  n_threads, thread_id );
	  
	  results[thread_id].get();
	  
	  // output layer gradient
	  for(i = output_layer_gradient_index; i < total_hidden_weights_length; ++i)
	  {
		EXPECT_NEAR( *(gradient_approx + current_threads_gradient_index + i), *(total_gradient_ptr + current_threads_gradient_index + i), limit_epsilon)<<" ith_case: "<<ith_case<<" thread_id: "<<thread_id<<" i: "<<i<<"\n";
	  }
	
	  for(i = 0; i < total_gradient_length; ++i)
	  {
		EXPECT_NEAR(gradient_approx[i], total_gradient_ptr[i], limit_epsilon)<<" ith_case: "<<ith_case<<" thread_id: "<<thread_id<<" i: "<<i<<"\n";
	  }
	}
	// clear futures
	results.clear();
  }
  // END FORWARD & BACKPROP PROP
  
  
  // release memory
  delete loss;
  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_activations_ptr_check);
  mkl_free(total_deltas_ptr);
  mkl_free(total_deltas_ptr_test);
  mkl_free(outputs_ptr);
  mkl_free(outputs_ptr_test);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_gradient_ptr);
  mkl_free(total_gradient_ptr_test);
  mkl_free(total_bias_ptr);
  mkl_free(total_training_cases_ptr);
  mkl_free(total_targets_ptr);
  mkl_free(d_error);
  mkl_free(gradient_approx);
}

TEST(multi_layer_perceptron, train)
{
  /* Learning A toy data set with 5 classes. 4 of  which is a quadrant plane.
   * E.g 
   * (0.1, 0.1)   -> class 1 (quadrant 1)
   * (-0.1, 0.1)  -> class 2 (quadrant 2)
   * (-0.1, -0.1) -> class 3 (quadrant 3)
   * (0.1, -0.1)  -> class 4 (quadrant 4)
   * So each example of the data set has 2 features, it's x and y coordinate.
   * The fifth class is the point (0.0, 0.0) which of geometrically would be a member of each class so it is treated as it's own unique class
   * */

  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, 7);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> loss_function_dist(0, 1);
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-1.0, 1.0);
  std::uniform_real_distribution<float> neg_real_dist(-1.0, 0);
  std::uniform_real_distribution<float> pos_real_dist(0, 1.0);
  // declarations for vector lengths
  std::uint32_t total_activations_length{0}, total_hidden_weights_length{0}, total_gradient_length{0}, total_bias_length{0}, total_training_cases_length{0}, total_targets_length{0}, total_cases{0}, total_error_length{0};

  
  // declarations for pointers
  double * total_activations_ptr{nullptr};
  double * total_deltas_ptr{nullptr};
  double * total_hidden_weights_ptr{nullptr};
  double * total_gradient_ptr{nullptr};
  double * total_bias_ptr{nullptr};
  double * total_targets_ptr{nullptr};
  double * total_training_cases_ptr{nullptr};
  double * outputs_ptr{nullptr};
  double * error_matrix{nullptr};

  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function;
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer;

  // loop counters misc vars
  std::uint32_t i{0}, j{0}, ith_layer{0}, ith_case{0}, n_layers{1/*layer_dist(mt)*/};
  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t input_layer{0};

  // set layers
  total_layers.push_back(std::make_shared< zinhart::models::layers::input_layer<double> >());
  total_layers[input_layer]->set_size(2);
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
    //layer_dist(mt)
	random_layer(total_layers, 2, /*neuron_dist(mt)*/5);
  }
  
  const std::uint32_t output_layer{total_layers.size() - 1};
  // To ensure their are atleast as many cases as threads 
  std::uniform_int_distribution<std::uint32_t> case_dist(n_threads, 50);
  // set total case length 
  total_training_cases_length = total_layers[input_layer]->get_size() * /*case_dist(mt)*/ 500; // 100 of each class
  total_cases = total_training_cases_length / total_layers[input_layer]->get_size();
 
  
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
  total_activations_length *= n_threads;
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 
  total_gradient_length = total_hidden_weights_length * n_threads;// important!
  total_bias_length = total_layers.size() - 1;
  total_targets_length = total_layers[output_layer]->get_size() * total_cases;


  const std::uint32_t alignment{64};
  const std::uint32_t output_layer_nodes{total_layers[output_layer]->get_size()};
  total_error_length = total_layers[output_layer]->get_size() * n_threads;


  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  outputs_ptr = (double*) mkl_malloc( output_layer_nodes * sizeof( double ), alignment );
  total_hidden_weights_ptr = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  total_gradient_ptr = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  total_bias_ptr = (double*) mkl_malloc( total_bias_length * sizeof( double ), alignment );
  total_training_cases_ptr = (double*) mkl_malloc( total_training_cases_length * sizeof( double ), alignment );
  total_targets_ptr = (double*) mkl_malloc(total_targets_length * sizeof(double), alignment );
  error_matrix = (double*) mkl_malloc(total_error_length * sizeof(double), alignment );


  double x_coordinate{0}, y_coordinate{0};

  // initialize training data matrix
  for(i = 0; i < total_training_cases_length; ++i)
	total_training_cases_ptr[i] = 0;
  // set training data
  
  // initialize target matrix
  for(i = 0; i < total_targets_length; ++i)
	total_targets_ptr[i] = 0;

  // set target vectors
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = 0.0;
	total_deltas_ptr[i] = 0.0;
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
  {
	total_hidden_weights_ptr[i] = real_dist(mt);
	total_gradient_ptr[i] = 0.0;
  }
  for(i = 0; i < total_bias_length; ++i)
	total_bias_ptr[i] = 0.0;
  for(i = 0; i < output_layer_nodes; ++i)
	outputs_ptr[i] = 0.0;

  for(i = 0; i < total_error_length; ++i)
	error_matrix[i] = 0.0;


  train(total_layers,
	    loss_function,
		optimizer,
		total_training_cases_ptr, total_training_cases_length,
		total_targets_ptr, error_matrix,
		total_activations_ptr, total_deltas_ptr, total_activations_length,
		total_hidden_weights_ptr, total_gradient_ptr, total_hidden_weights_length,
		total_bias_ptr,
		10, // batchsize
		n_threads,
		true
	   );


  // release memory
  mkl_free(total_activations_ptr);
  mkl_free(total_deltas_ptr);
  mkl_free(outputs_ptr);
  mkl_free(total_hidden_weights_ptr);
  mkl_free(total_gradient_ptr);
  mkl_free(total_bias_ptr);
  mkl_free(total_training_cases_ptr);
  mkl_free(total_targets_ptr);
  mkl_free(error_matrix);
}
