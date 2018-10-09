#include <ann/ann.hh>
#include <ann/activation.hh>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
using namespace zinhart::models;
using namespace zinhart::activation;

TEST(ann_test, get_layer_add_layer_clear_layers)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, std::numeric_limits<std::uint16_t>::max() );// random number of neurons
  std::uniform_int_distribution<std::uint32_t> layer_dist(0, total_activation_types());// random activation function
  std::uint32_t i, n_layers{layer_dist(mt)};
  ann<architecture::mlp_dense, double> model;
  std::vector<LAYER_INFO> total_layers, total_layers_copy;
  for(i = 0; i < n_layers; ++i)
  {
	LAYER_INFO a_layer;
	a_layer.first = ACTIVATION_NAME(layer_dist(mt));
	a_layer.second = neuron_dist(mt);  
	total_layers.push_back(a_layer);
	model.add_layer(a_layer);
  }
  total_layers_copy = model.get_total_layers();
  ASSERT_EQ(total_layers.size(), total_layers_copy.size());
  ASSERT_EQ(total_layers_copy.size(), n_layers);
  for(std::uint32_t i = 0; i < total_layers.size(); ++i)
  {
	ASSERT_EQ(total_layers[i].first, total_layers_copy[i].first);
	ASSERT_EQ(total_layers[i].second, total_layers_copy[i].second);
  }
  model.clear_layers();
  total_layers_copy = model.get_total_layers();
  ASSERT_EQ(total_layers_copy.size(), 0);
}
/*
TEST(ann_test, initialize_model_cleanup_model)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(0,5000);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uint32_t total_activations, total_deltas, total_hidden_weights, total_gradients, total_bias;
  std::uint32_t ith_layer, n_layers{layer_dist(mt)}, n_threads{thread_dist(mt)};
  ann<architecture::mlp_dense, double> model;
  std::vector<LAYER_INFO> total_layers, total_layers_copy;
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	LAYER_INFO a_layer;
	a_layer.first = ACTIVATION_NAME(layer_dist(mt));
	a_layer.second = neuron_dist(mt);  
	total_layers.push_back(a_layer);
	model.add_layer(a_layer);
  }
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  total_activations *= n_threads;
  total_deltas = total_activations;
  
  //calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  total_gradients = total_hidden_weights;
  total_bias = total_layers.size() - 1;
  
  model.init(n_threads);

  ASSERT_EQ(total_activations, model.get_total_activations());
  ASSERT_EQ(total_deltas, model.get_total_deltas());
  ASSERT_EQ(total_hidden_weights, model.get_total_hidden_weights());
  ASSERT_EQ(total_gradients, model.get_total_gradients());
  ASSERT_EQ(total_bias, model.get_total_bias());

  model.cleanup();


  ASSERT_EQ(0, model.get_total_activations());
  ASSERT_EQ(0, model.get_total_deltas());
  ASSERT_EQ(0, model.get_total_hidden_weights());
  ASSERT_EQ(0, model.get_total_gradients());
  ASSERT_EQ(0, model.get_total_bias());
} 


TEST(ann_test, forward_propagate)
{
}

TEST(ann_test, ann_train)
{
}*/

 /*
 * I've taken the vales here of a known neural network architecture with a very small data set to check my gradient check
 * */
/*
TEST(mlp, pre_gradient_check_mazur)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t input_layer{0};
  const std::uint32_t output_layer{2};
  const std::uint32_t total_inputs_length{2};
  const std::uint32_t total_targets_length{2};
  const std::uint32_t total_activations_length{n_threads * 4};
  const std::uint32_t total_hidden_weights_length{8};
  const std::uint32_t total_gradient_length{n_threads * total_hidden_weights_length};
  const std::uint32_t alignment = 64;
  std::uint32_t i{0}, j{0}, k{0}, thread_id{0};
  const zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME name{zinhart::function_space::error_metrics::LOSS_FUNCTION_NAME::MSE};
  
  double * total_inputs{nullptr};
  double * total_targets{nullptr};
  double * total_hidden_inputs{nullptr};
  double * total_activations{nullptr};
  double * total_deltas{nullptr};
  double * total_hidden_weights{nullptr};
  double * total_hidden_weights_copy{nullptr};
  double * analytic_gradients{nullptr};
  double * numerically_approx_gradients{nullptr};
  double * bias{nullptr};
  double * d_error{nullptr};
  double * current_threads_output_layer_ptr{nullptr};
  double * current_threads_gradient_ptr{nullptr};


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

  total_inputs = (double*) mkl_malloc( total_inputs_length * sizeof( double ), alignment );
  total_targets = (double*) mkl_malloc( total_targets_length * sizeof( double ), alignment );
  total_hidden_inputs = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_hidden_weights = {(double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment )};
  total_hidden_weights_copy = (double*) mkl_malloc( total_hidden_weights_length * sizeof( double ), alignment );
  analytic_gradients = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  numerically_approx_gradients = (double*) mkl_malloc( total_gradient_length * sizeof( double ), alignment );
  bias = (double*) mkl_malloc( bias_length * sizeof( double ), alignment );
  current_threads_output_layer_ptr = (double*) mkl_malloc( total_targets_length * sizeof( double ), alignment );
  d_error = (double*) mkl_malloc(bias_length * sizeof( double ), alignment );

  // initialize model params

  total_inputs[0] = 0.05;
  total_inputs[1] = 0.1;

  total_hidden_weights[0] = .15;
  total_hidden_weights[1] = .20;
  total_hidden_weights[2] = .25;
  total_hidden_weights[3] = .30;

  total_hidden_weights[4] = .40;
  total_hidden_weights[5] = .45;
  total_hidden_weights[6] = .50;
  total_hidden_weights[7] = .55;


  total_hidden_weights_copy[0] = .15;
  total_hidden_weights_copy[1] = .20;
  total_hidden_weights_copy[2] = .25;
  total_hidden_weights_copy[3] = .30;


  total_hidden_weights_copy[4] = .40;
  total_hidden_weights_copy[5] = .45;
  total_hidden_weights_copy[6] = .50;
  total_hidden_weights_copy[7] = .55;

  bias[0] = .35;
  bias[1] = .60;

  total_targets[0] = .01;
  total_targets[1] = 0.99;

  for(i = 0; i < total_gradient_length; ++i)
  {
   analytic_gradients[i] = 0.0;
   numerically_approx_gradients[i] = 0.0;
  }
  
  multi_layer_perceptron<connection::dense, double> mlp;
  zinhart::function_space::error_metrics::loss_function loss;

  std::cout.precision(10);
  // for gradient checking
  const double limit_epsilon = 1.e-4;
  double right{0}, left{0}, original{0};

 // for each thread perform a gradient check on the same values, ideally each thread would come to the same conclusion (return the same values)
 for(thread_id = 0; thread_id < n_threads; ++thread_id)
 {
   //zinhart::serial::print_matrix_row_major(numerically_approx_gradients,1,total_gradient_length,"num approx");
   current_threads_gradient_ptr = numerically_approx_gradients + (thread_id * total_hidden_weights_length);
   for(i = 0; i < total_hidden_weights_length ; ++i)
   {
	 original = total_hidden_weights_copy[i];
	 // right side
	 total_hidden_weights_copy[i] += limit_epsilon;
	 mlp.forward_propagate(total_layers, 
	   				total_inputs, 0, 
	   				total_hidden_inputs, total_activations, total_activations_length,
	   				total_hidden_weights_copy, total_hidden_weights_length, 
	   				bias,
	   				n_threads, thread_id
	   			  );
//	 zinhart::serial::print_matrix_row_major(total_activations, total_activations_length, 1, "activations " + std::to_string(i));
	 mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
	 zinhart::serial::print_matrix_row_major(current_threads_output_layer_ptr,2,1,"outputs");
	 right = loss(name, zinhart::function_space::OBJECTIVE(), current_threads_output_layer_ptr, total_targets, total_targets_length, 2);
//	 std::cout<<total_hidden_weights_copy[j]<<"\n";

	 // set back
	 total_hidden_weights_copy[i] = original; 

	 // left side
	 total_hidden_weights_copy[i] -= limit_epsilon;
	 mlp.forward_propagate(total_layers, 
	   				total_inputs, 0, 
	   				total_hidden_inputs, total_activations, total_activations_length,
	   				total_hidden_weights_copy, total_hidden_weights_length, 
	   				bias,
	   				n_threads, thread_id
	   			  );
	 mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
//	 zinhart::serial::print_matrix_row_major(current_threads_output_layer_ptr,2,1,"outputs");
	 left = loss(name, zinhart::function_space::OBJECTIVE(), current_threads_output_layer_ptr, total_targets, total_targets_length, 2);
//	 std::cout<<total_hidden_weights_copy[j]<<"\n";
	 // calc numerically derivative for the ith_weight, save it, increment the pointer to the next weight

	 *(current_threads_gradient_ptr + i) = (right - left) / (2 * limit_epsilon);
//	 std::cout<<right<<" "<<left<<"\n"; 
	 std::cout<<(right - left) / (2 * limit_epsilon)<<" a\n";
	 // set back
	 total_hidden_weights_copy[i] = original; 
	 
   }
   
   // forward prop
   mlp.forward_propagate(std::ref(total_layers), total_inputs, 0, total_hidden_inputs, total_activations, total_activations_length, total_hidden_weights, total_hidden_weights_length, bias, n_threads, thread_id);
   // error derivative
   mlp.get_outputs(total_layers, total_activations, total_activations_length, current_threads_output_layer_ptr, n_threads, thread_id);
   loss(name, zinhart::function_space::DERIVATIVE(), current_threads_output_layer_ptr, total_targets, d_error, total_targets_length, 2);
   // back prop
   mlp.backward_propagate(std::ref(total_layers), total_inputs, total_targets, d_error, 0, 
									  total_hidden_inputs, total_activations, total_deltas, total_activations_length, 
									  total_hidden_weights, analytic_gradients, total_hidden_weights_length, bias, n_threads, thread_id);
//   zinhart::serial::print_matrix_row_major(numerically_approx_gradients, 2,2 , "approx");
   // compare gradients
 //  for(i = 0; i < total_gradient_length; ++i)
//	 EXPECT_NEAR(analytic_gradients[i], numerically_approx_gradients[i], limit_epsilon)<<"i: "<<i<<"\n";
   // update weights
   // 
 }   
  
  mkl_free(total_inputs);
  mkl_free(total_hidden_inputs);
  mkl_free(total_deltas);
  mkl_free(total_activations);
  mkl_free(total_hidden_weights);
  mkl_free(total_hidden_weights_copy);
  mkl_free(analytic_gradients);
  mkl_free(numerically_approx_gradients);
  mkl_free(bias);
  mkl_free(current_threads_output_layer_ptr);
  mkl_free(d_error);

}
 */


