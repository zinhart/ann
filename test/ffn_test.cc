#include "ann/ann.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
using namespace zinhart;
#if CUDA_ENABLED == 1

TEST(ffn_test, async_forward_propagate)
{
  const std::uint32_t n_cuda_streams = MAX_CPU_THREADS;
  // set device properties
  cudaDeviceProp properties;
  zinhart::check_cuda_api(cudaGetDeviceProperties(&properties,0),__FILE__, __LINE__);
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  std::uniform_int_distribution<std::uint8_t> layer_num_dist(1,3/*std::numeric_limits<std::uint8_t>::max()*/);
  std::uniform_int_distribution<std::uint8_t> activation_dist(1,8);// currently there are 8 different activation functions not counting the input layer
  //std::uniform_int_distribution<std::uint8_t> cuda_stream_dist(1, MAX_CPU_THREADS); 
  std::uint32_t total_num_targets = neuron_dist(mt);
  std::vector<LAYER_INFO> total_layers(layer_num_dist(mt));
  cudaStream_t * streams; 

  // host vectors 
  double * total_observations = nullptr;
  double * total_targets = nullptr;
  double * total_activations = nullptr;
  double * total_bias = nullptr;
  double * total_hidden_weights = nullptr;

  // host validation vectors
  double * total_observations_copy = nullptr;
  double * total_targets_copy = nullptr;
  double * total_activations_copy = nullptr;
  double * total_bias_copy = nullptr;
  double * total_hidden_weights_copy = nullptr;

  // device vectors
  double * device_total_observations = nullptr; 
  double * device_total_activations = nullptr;
  double * device_total_bias = nullptr;
  double * device_total_hidden_weights = nullptr;

  std::uint32_t  i, ith_layer, ith_observation, total_observations_length, total_targets_length, total_activations_length, total_hidden_weights_length, total_bias_length;
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
  total_observations_length = neuron_dist(mt) * a_layer.second;// number of observations matrix 
  total_targets_length = total_num_targets; // number of targets
 
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer].second;// accumulate neurons in the hidden layers and output layer
  
  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights_length += total_layers[ith_layer + 1].second * total_layers[ith_layer].second;
  }
  // calc bias neurons
  total_bias_length = total_layers.size() - 1;

  // allocate host vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_observations, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_targets, sizeof(double) * total_targets_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_activations, sizeof(double) * total_activations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_hidden_weights, sizeof(double) * total_hidden_weights_length,cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_bias, sizeof(double) * total_bias_length, cudaHostAllocDefault),__FILE__,__LINE__));

  // allocate validation vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_observations_copy, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_targets_copy, sizeof(double) * total_targets_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_activations_copy, sizeof(double) * total_activations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_hidden_weights_copy, sizeof(double) * total_hidden_weights_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&total_bias_copy, sizeof(double) * total_bias_length, cudaHostAllocDefault),__FILE__,__LINE__));
/**/

/*  total_activations_copy = new double[total_observations_length];
  total_hidden_weights_copy = new double[total_hidden_weights.first];
  total_bias_copy = new double[total_bias.first];*/
  // allocate device vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_observations, total_observations_length * sizeof(double) ),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_activations, total_activations_length * sizeof(double) ) ,__FILE__,__LINE__ ));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_bias, total_bias_length * sizeof(double) ) ,__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_hidden_weights, total_hidden_weights_length * sizeof(double) ) ,__FILE__,__LINE__));

  // allocate streams 
  streams = new cudaStream_t[n_cuda_streams];

  // create streams
  for (i = 0; i < n_cuda_streams; ++i)
  {
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaStreamCreate(&streams[i]),__FILE__,__LINE__));
  }

  // initialize host vectors
  for(i = 0; i < total_observations_length; ++i)
	total_observations[i] = real_dist(mt); // random training set
    for(i = 0; i < total_activations_length; ++i)
	total_activations[i] = 0.0f; // start at zero since these values have not been calculated yet
  for(i = 0; i < total_hidden_weights_length; ++i)
	total_hidden_weights[i] = real_dist(mt); // random weights
  for(i = 0; i < total_bias_length; ++i)
	total_bias[i] = real_dist(mt); // random bias (which is an oxymoron?)

  //ASSERT_EQ(0,zinhart::check_cuda_api(cudaMemcpy(device_total_observations, total_observations.second, total_observations_length * sizeof(double), cudaMemcpyHostToDevice))); 
  
  // copy host memory to device for each stream
  for(i = 0; i < n_cuda_streams; ++i )
  {
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_observations, total_observations, total_observations_length * sizeof(double), cudaMemcpyHostToDevice, streams[i]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_activations, total_activations, total_activations_length * sizeof(double), cudaMemcpyHostToDevice, streams[i]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_bias, total_bias, total_bias_length * sizeof(double), cudaMemcpyHostToDevice, streams[i]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_hidden_weights, total_hidden_weights, total_hidden_weights_length * sizeof(double), cudaMemcpyHostToDevice, streams[i]), __FILE__, __LINE__)); /**/
  }

  // synchronize the host thread wrt each stream to ensure the memory transactions (HostToDevice) above have been completed 
  for(i = 0; i < n_cuda_streams; ++i )
  {
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaStreamSynchronize(streams[i]), __FILE__, __LINE__));
  } 

  // cublas initialization and error check
  cublasHandle_t context;
  ASSERT_EQ(0, zinhart::check_cublas_api(cublasCreate(&context),__FILE__, __LINE__)); 

  for(i = 0; i < n_cuda_streams; ++i)
  {
	// call dgem wrapper to get column major info for the ith stream
	
	// set the stream for the current iteration
	ASSERT_EQ(0, zinhart::check_cublas_api(cublasSetStream(context, streams[i]),__FILE__,__LINE__));
	
	// call dgemm and check for errors
  }
  // release cublas resources and check for errors
  ASSERT_EQ(0,zinhart::check_cublas_api(cublasDestroy(context),__FILE__, __LINE__));

  // copy device memory back to host
  for(i = 0; i < n_cuda_streams; ++i )
  {
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(total_observations, device_total_observations, total_observations_length * sizeof(double), cudaMemcpyDeviceToHost, streams[i]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(total_activations, device_total_activations, total_activations_length * sizeof(double), cudaMemcpyDeviceToHost, streams[i]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(total_bias, device_total_bias, total_bias_length * sizeof(double), cudaMemcpyDeviceToHost, streams[i]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(total_hidden_weights, device_total_hidden_weights, total_hidden_weights_length * sizeof(double), cudaMemcpyDeviceToHost, streams[i]), __FILE__, __LINE__));
  }

  // synchronize the host thread wrt each stream to ensure the asynchronous memory transactions (DeviceToHost) above have been completed 
  for(i = 0; i < n_cuda_streams; ++i )
  {
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaStreamSynchronize(streams[i]), __FILE__, __LINE__));
  }

  // do serial forward propagate operation for each layer for each stream on a different host thread
  
  // get serial results 
  
  // validate cpu and cpu activation vectors
  
  // validate output vector

  // deallocate host memory and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_observations),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_targets),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_activations),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_hidden_weights),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_bias),__FILE__,__LINE__));

  // deallocate host validation memory and check for errors
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_observations_copy),__FILE__,__LINE__));
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_targets_copy),__FILE__,__LINE__));
 /*delete []total_activations_copy;
 delete []total_hidden_weights_copy;
 delete []total_bias_copy;*/
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_activations_copy),__FILE__,__LINE__));
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_hidden_weights_copy),__FILE__,__LINE__));
 ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(total_bias_copy),__FILE__,__LINE__));
/**/
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
