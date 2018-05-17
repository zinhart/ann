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
#if CUDA_ENABLED == 1
/*
TEST(ann_test, forward_propagate)
{
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
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

  // host vectors 
  std::pair<std::uint32_t, std::shared_ptr<double>> total_observations;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_targets;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_activations;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_bias;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_hidden_weights;

  // host validation vectors
  
  std::pair<std::uint32_t, std::shared_ptr<double>> total_observations_copy;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_targets_copy;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_activations_copy;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_bias_copy;
  std::pair<std::uint32_t, std::shared_ptr<double>> total_hidden_weights_copy;

  // device vectors
  double * device_total_observations, * device_total_activations, * device_total_bias, * device_total_hidden_weights;


  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t  i, ith_layer, ith_observation, total_observations_length;
  std::int32_t error;
  total_observations_length = total_observations.first * total_layers[0].second;
  
  total_observations.first = dist(mt);//number of observations
  total_observations.second = std::shared_ptr<double> ( new double[total_observations_length], std::default_delete<double[]>() );//observations themselves 
  total_targets.first = total_num_targets; // number of targets
  total_targets.second = std::shared_ptr<double> ( new double[total_targets.first], std::default_delete<double[]>() );//targets themselves 
    
  // calc number of activations
  for(ith_layer = 1, total_activations.first = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations.first += total_layers[ith_layer].second;// accumulate neurons in the hidden layers and output layer
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights.first = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights.first += total_layers[ith_layer + 1].second * total_layers[ith_layer].second;
  
  // allocate host activation and weight activations
  total_activations.second = std::shared_ptr<double> ( new double[total_activations.first], std::default_delete<double[]>() );
  total_hidden_weights.second = std::shared_ptr<double> ( new double[total_hidden_weights.first], std::default_delete<double[]>() );

  // calc bias neurons
  total_bias.first = total_layers.size() - 1;
  total_bias.second = std::shared_ptr<double> ( new double[total_bias.first], std::default_delete<double[]>() );
  cudaError_t error_id;

  // deal with validation vectors
  total_observations_copy.first = total_observations.first;//number of observations
  total_observations_copy.second = std::shared_ptr<double> ( new double[total_observations_length], std::default_delete<double[]>() );//observations themselves 
  total_targets_copy.first = total_num_targets; // number of targets
  total_targets_copy.second = std::shared_ptr<double> ( new double[total_targets.first], std::default_delete<double[]>() );//targets themselves 
  total_activations_copy.first = total_activations.first;
  total_activations_copy.second = std::shared_ptr<double> ( new double[total_activations_copy.first], std::default_delete<double[]>() );
  total_bias_copy.first = total_bias.first;
  total_bias_copy.second = std::shared_ptr<double> ( new double[total_bias_copy.first], std::default_delete<double[]>() );
  total_hidden_weights_copy.first = total_hidden_weights.first;
  total_hidden_weights_copy.second = std::shared_ptr<double> ( new double[total_hidden_weights_copy.first], std::default_delete<double[]>() );

  // extra caution because if these assertions aren't true this test case breaks
  ASSERT_EQ(total_observations.first, total_observations_copy.first);
  ASSERT_EQ(total_targets.first, total_targets_copy.first);
  ASSERT_EQ(total_activations.first, total_activations_copy.first);
  ASSERT_EQ(total_bias.first, total_bias_copy.first);
  ASSERT_EQ(total_hidden_weights.first, total_hidden_weights_copy.first);
   
  // initialize host vectors with dummy data  
  for(i = 0; i < total_observations_length; ++i)
	total_observations.second.get()[i] = real_dist(mt); // random training set
  for(i = 0; i < total_activations.first; ++i)
	total_activations.second.get()[i] = 0.0; // start at zero since these values have not been calculated yet
  for(i = 0; i < total_hidden_weights.first; ++i)
	total_hidden_weights.second.get()[i] = real_dist(mt); // random weights
  for(i = 0; i < total_bias.first; ++i)
	total_bias.second.get()[i] = real_dist(mt); // random bias (which is an oxymoron?)

  // allocate device vectors
  error_id = cudaMalloc( (void **) &device_total_observations, total_observations_length * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_observations allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMalloc( (void **) &device_total_activations, total_activations.first * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_activations allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  error_id = cudaMalloc( (void **) &device_total_bias, total_bias.first * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_bias allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
	
  error_id = cudaMalloc( (void **) &device_total_hidden_weights, total_hidden_weights.first * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_hidden_weights allocation (in forward_propagate) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  // copy host memory to device
  error_id = cudaMemcpy(device_total_observations, total_observations.second.get(), total_observations_length * sizeof(double), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_observations (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy(device_total_activations, total_activations.second.get(), total_activations.first * sizeof(double), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_activations (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy(device_total_bias, total_bias.second.get(), total_bias.first *  sizeof(double), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_bias (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy(device_total_hidden_weights, total_hidden_weights.second.get(), total_hidden_weights.first * sizeof(double), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_hidden_weights (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //ASSERT_EQ(initialize_model(model, total_observations, total_targets), 0);
  //ASSERT_EQ(total_activations, get_total_activations(model).first);
  //ASSERT_EQ(total_hidden_weights, get_total_hidden_weights(model).first);
  //call forward_propagate wrapper here
  //ASSERT_EQ(cleanup(model), 0);
  
  // copy device memory back to host
  error_id = cudaMemcpy( total_observations_copy.second.get(), device_total_observations, total_observations_length * sizeof(double), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"device_observation (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy( total_activations_copy.second.get(), device_total_activations, total_activations.first * sizeof(double), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"device_activations (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy( total_bias_copy.second.get(), device_total_bias, total_bias.first * sizeof(double), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"device_bias (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy( total_hidden_weights_copy.second.get(), device_total_hidden_weights, total_hidden_weights.first * sizeof(double), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"device_total_hidden_weights (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";


  // cublas initialization and error check
  cublasStatus_t cublas_error_id;
  cublasHandle_t handle;
  cublas_error_id = cublasCreate(&handle);
  if(cublas_error_id != CUBLAS_STATUS_SUCCESS)
  {
	std::cerr<<"CublasHandle creation failed with error: "<<cublasGetErrorString(cublas_error_id)<<"\n";
  }

  // This is the forward propagation loop that is to be validated
  // The only matrix that should not be the same is device activations
    

  for(ith_observation = 0; ith_observation < total_observations.first / total_observations.first; ++ith_observation)
  {
	error = forward_propagate(model, false, handle, ith_observation, total_layers, total_targets, total_hidden_weights, total_activations, device_total_observations, device_total_activations, device_total_bias, device_total_hidden_weights);

	if(error == 1)
	{
	  std::cerr<<"An error occured in forward_propagate during the "<<ith_observation<<"th iterator\n";
	  std::abort();
	}	
  }
  
  // release cublas resources and check for errors
  cublas_error_id = cublasDestroy(handle);
  if(cublas_error_id != CUBLAS_STATUS_SUCCESS)
  {
	std::cerr<<"cublas handle destruction failed with error: "<<cublasGetErrorString(cublas_error_id)<<"\n";
  }
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

}*/
#endif
/*
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
}*/
