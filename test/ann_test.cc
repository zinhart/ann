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
  
  std::pair<std::uint32_t, double *> total_observations;
  std::pair<std::uint32_t, double *> total_targets;
  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t total_hidden_weights, total_activations, ith_layer;
  
  total_observations.first = dist(mt);//number of observations
  total_observations.second = new double[total_observations.first * total_layers[0].second];//observations themselves 
 
  total_targets.first = total_num_targets; // number of targets
  total_targets.second = new double[total_targets.first];//targets themselves 
    
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

TEST(ann_test, forward_propagate)
{
  std::uint32_t total_pinned_memory = 0;
  const std::uint32_t n_cuda_streams = MAX_CPU_THREADS;
  // set device properties
  cudaDeviceProp properties;
  zinhart::check_cuda_api(cudaGetDeviceProperties(&properties,0),__FILE__, __LINE__);
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  std::uniform_int_distribution<std::uint8_t> layer_num_dist(1,10/*std::numeric_limits<std::uint8_t>::max()*/);
  std::uniform_int_distribution<std::uint8_t> activation_dist(1,8);// currently there are 8 different activation functions not counting the input layer
  //std::uniform_int_distribution<std::uint8_t> cuda_stream_dist(1, MAX_CPU_THREADS); 
  std::uint32_t total_num_targets = neuron_dist(mt);
  std::vector<LAYER_INFO> total_layers(layer_num_dist(mt));
  cudaStream_t * streams; 

  // host vectors 
  std::pair<std::uint32_t, double *> total_observations;
  std::pair<std::uint32_t, double *> total_targets;
  std::pair<std::uint32_t, double *> total_activations;
  std::pair<std::uint32_t, double *> total_bias;
  std::pair<std::uint32_t, double *> total_hidden_weights;

  // host validation vectors
  double * total_observations_copy;
  double * total_targets_copy;
  double * total_activations_copy;
  double * total_bias_copy;
  double * total_hidden_weights_copy;

  // device vectors
  double * device_total_observations, * device_total_activations, * device_total_bias, * device_total_hidden_weights;

  std::uint32_t  i, ith_layer, ith_observation, total_observations_length;
  std::int32_t error;

  //first layer is always input layer
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = neuron_dist(mt);
  total_layers[0] = a_layer;  
  for(i = 1; i < total_layers.size(); ++i) // this for loop is for the hiddent layers and output layers
  {
	a_layer.first = ACTIVATION_NAME(activation_dist(mt));// random layer
	a_layer.second = neuron_dist(mt);// random amount of neurons
	total_layers[i] = a_layer;
  }

  // of course should always be the input layer
  ASSERT_EQ(total_layers[0].first, ACTIVATION_NAME::INPUT);

  // calculate array sizes
  total_observations.first = neuron_dist(mt);// number of observations
  total_targets.first = total_num_targets; // number of targets
  total_observations_length = total_observations.first * total_layers[0].second;
 
  // calc number of activations
  for(ith_layer = 1, total_activations.first = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations.first += total_layers[ith_layer].second;// accumulate neurons in the hidden layers and output layer
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights.first = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights.first += total_layers[ith_layer + 1].second * total_layers[ith_layer].second;
  }
  // calc bias neurons
  total_bias.first = total_layers.size() - 1;

  total_pinned_memory += total_observations.first;
  total_pinned_memory += total_targets.first;
  total_pinned_memory += total_activations.first;
  total_pinned_memory += total_hidden_weights.first;
  total_pinned_memory += total_bias.first;
  total_pinned_memory *= 2;
  if(total_pinned_memory * sizeof(double) > 4000000 * 1000)
	std::cout<<"total pinned memory: "<<total_pinned_memory<<"\n";

  // allocate host vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_observations.second, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_targets.second, sizeof(double) * total_targets.first, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_activations.second, sizeof(double) * total_activations.first, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_hidden_weights.second, sizeof(double) * total_hidden_weights.first,cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_bias.second, sizeof(double) * total_bias.first,cudaHostAllocDefault),__FILE__,__LINE__));

  // allocate validation vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_observations_copy, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_targets_copy, sizeof(double) * total_targets.first, cudaHostAllocDefault),__FILE__,__LINE__));
 /* ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_activations_copy, sizeof(double) * total_activations.first, cudaHostAllocDefault),__FILE__,__LINE__));
  zinhart::check_cuda_api(cudaHostAlloc((void**)&total_hidden_weights_copy, sizeof(double) * total_hidden_weights.first,cudaHostAllocDefault),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaHostAlloc((void**)&total_bias_copy, sizeof(double) * total_bias.first,cudaHostAllocDefault),__FILE__,__LINE__);
*/

  total_activations_copy = new double[total_observations_length];
  total_hidden_weights_copy = new double[total_hidden_weights.first];
  total_bias_copy = new double[total_bias.first];
  // allocate device vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_observations, total_observations_length * sizeof(double) ),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_activations, total_activations.first * sizeof(double) ) ,__FILE__,__LINE__ ));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_bias, total_bias.first * sizeof(double) ) ,__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_hidden_weights, total_hidden_weights.first * sizeof(double) ) ,__FILE__,__LINE__));

  // allocate streams 
  streams = new cudaStream_t[n_cuda_streams];

  // create streams
  for (i = 0; i < n_cuda_streams; ++i)
  {
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaStreamCreate(&streams[i]),__FILE__,__LINE__));
  }

  // initialize host vectors
  for(i = 0; i < total_observations_length; ++i)
	total_observations.second[i] = real_dist(mt); // random training set
    for(i = 0; i < total_activations.first; ++i)
	total_activations.second[i] = 0.0f; // start at zero since these values have not been calculated yet
  for(i = 0; i < total_hidden_weights.first; ++i)
	total_hidden_weights.second[i] = real_dist(mt); // random weights
  for(i = 0; i < total_bias.first; ++i)
	total_bias.second[i] = real_dist(mt); // random bias (which is an oxymoron?)

  //ASSERT_EQ(0,zinhart::check_cuda_api(cudaMemcpy(device_total_observations, total_observations.second, total_observations_length * sizeof(double), cudaMemcpyHostToDevice))); 
  
  // copy host memory to device 

  // cublas initialization and error check
 
  // release cublas resources and check for errors
  
  // copy device memory back to host
  
  // validate  

  // deallocate host memory and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_observations.second),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_targets.second),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_activations.second),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_hidden_weights.second),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_bias.second),__FILE__,__LINE__));

  // deallocate host validation memory and check for errors
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_observations_copy),__FILE__,__LINE__));
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_targets_copy),__FILE__,__LINE__));
 delete []total_activations_copy;
 delete []total_hidden_weights_copy;
 delete []total_bias_copy;
 /*ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_activations_copy),__FILE__,__LINE__));
  zinhart::check_cuda_api(cudaFreeHost(total_hidden_weights_copy),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFreeHost(total_bias_copy),__FILE__,__LINE__);
*/
  // deallocate device memory and check for errors  
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_observations), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_activations), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_bias), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_hidden_weights), __FILE__, __LINE__));

  // destroy cuda streams
   for (i = 0; i < n_cuda_streams; ++i)
  {
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaStreamDestroy(streams[i]),__FILE__,__LINE__));
  }

  // deallocate cuda streams
  delete [] streams;

  ASSERT_EQ(0,zinhart::check_cuda_api(cudaDeviceReset(), __FILE__, __LINE__));
}
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
  
  std::pair<std::uint32_t, double *> total_observations;
  std::pair<std::uint32_t, double *> total_targets;
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
*/
