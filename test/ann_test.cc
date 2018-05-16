#include "ann/ann.hh"
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
using namespace zinhart;

TEST(ann_test,ann_test_constructor)
{
  ann< ffn > model;
}
TEST(ann_test, add_layer)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(1, std::numeric_limits<std::uint16_t>::max() );
  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::IDENTITY;
  a_layer.second = dist(mt);  
  add_layer(model, a_layer);
}

TEST(ann_test, initialize_model)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uint32_t total_num_targets = dist(mt);

  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = ACTIVATION_NAME::SIGMOID;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = ACTIVATION_NAME::RELU;
  a_layer.second = total_num_targets; 
  add_layer(model,a_layer);
  
  std::pair<std::uint32_t, std::shared_ptr<double>> total_observations;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_targets;
  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t total_hidden_weights, total_activations, ith_layer;
  
  total_observations.first = dist(mt);//number of observations
  total_observations.second = std::shared_ptr<double> ( new double[total_observations.first * total_layers[0].second], std::default_delete<double[]>() );//observations themselves 
 
  total_targets.first = total_num_targets; // number of targets
  total_targets.second = std::shared_ptr<double> ( new double[total_targets.first], std::default_delete<double[]>() );//targets themselves 
    
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  
  //calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  }
  ASSERT_EQ(initialize_model(model, total_observations, total_targets), 0);
  ASSERT_EQ(total_activations, get_total_activations(model).first);
  ASSERT_EQ(total_hidden_weights, get_total_hidden_weights(model).first);
  ASSERT_EQ(cleanup(model), 0);
}
TEST(ann_test, forward_propagate)
{
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uint32_t total_num_targets = dist(mt);

  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = ACTIVATION_NAME::SIGMOID;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = ACTIVATION_NAME::RELU;
  a_layer.second = total_num_targets; 
  add_layer(model,a_layer);
  
  std::pair<std::uint32_t, std::shared_ptr<double>> total_observations;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_targets;
  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t total_hidden_weights, total_activations, ith_layer, total_bias;
  
  total_observations.first = dist(mt);//number of observations
  total_observations.second = std::shared_ptr<double> ( new double[total_observations.first * total_layers[0].second], std::default_delete<double[]>() );//observations themselves 
  total_targets.first = total_num_targets; // number of targets
  total_targets.second = std::shared_ptr<double> ( new double[total_targets.first], std::default_delete<double[]>() );//targets themselves 
    
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  
  //calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  }  
  total_bias = total_layers.size() - 1;
  cudaError_t error_id;
  // device vectors
  double * device_total_observations, * device_total_activations, * device_total_bias, * device_total_hidden_weights;

  error_id = cudaMalloc( (void **) &device_total_observations, total_observations.first * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_observations allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMalloc( (void **) &device_total_activations, total_activations * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_activations allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  error_id = cudaMalloc( (void **) &device_total_bias, total_bias * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_bias allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
	
  error_id = cudaMalloc( (void **) &device_total_hidden_weights, total_hidden_weights * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_hidden_weights allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";




  //ASSERT_EQ(initialize_model(model, total_observations, total_targets), 0);
  //ASSERT_EQ(total_activations, get_total_activations(model).first);
  //ASSERT_EQ(total_hidden_weights, get_total_hidden_weights(model).first);
  //call forward_propagate wrapper here
  //ASSERT_EQ(cleanup(model), 0);
  
  // deallocate device memory and check for errors  
  error_id = cudaFree(device_total_observations);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_observations deallocation (In forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaFree(device_total_activations);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_activations deallocation (In forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaFree(device_total_bias);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_bias deallocation (In forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaFree(device_total_hidden_weights);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_hidden_weights deallocation (In forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";


}

TEST(ann_test, ann_train)
{
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uint32_t total_num_targets = dist(mt);

  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = ACTIVATION_NAME::RELU;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = ACTIVATION_NAME::SOFTMAX;
  a_layer.second = total_num_targets; 
  add_layer(model,a_layer);
  
  std::pair<std::uint32_t, std::shared_ptr<double>> total_observations;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_targets;
  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t total_hidden_weights, total_activations, ith_layer;
  
  total_observations.first = dist(mt);//number of observations
  total_observations.second = std::shared_ptr<double> ( new double[total_observations.first * total_layers[0].second], std::default_delete<double[]>() );//observations themselves 
  total_targets.first = total_num_targets; // number of targets
  total_targets.second = std::shared_ptr<double> ( new double[total_targets.first], std::default_delete<double[]>() );//targets themselves 
    
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  
  //calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  }  
  ASSERT_EQ(initialize_model(model, total_observations, total_targets), 0);
  ASSERT_EQ(total_activations, get_total_activations(model).first);
  ASSERT_EQ(total_hidden_weights, get_total_hidden_weights(model).first);
  ASSERT_EQ(train(model,dist(mt),dist(mt), dist(mt)), 0);
  ASSERT_EQ(cleanup(model), 0);
}
