#include "ann/ann.hh"
#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
#include <list>
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
  std::uniform_int_distribution<std::uint32_t> case_dist(1,25);
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  std::uniform_int_distribution<std::uint8_t> layer_num_dist(2,5/*std::numeric_limits<std::uint8_t>::max()*/);// at least an input and output layer
  std::uniform_int_distribution<std::uint8_t> activation_dist(1,8);// currently there are 8 different activation functions not counting the input layer
  //std::uniform_int_distribution<std::uint8_t> cuda_stream_dist(1, MAX_CPU_THREADS); 
  std::vector<LAYER_INFO> total_layers(layer_num_dist(mt));
  cudaStream_t * streams{nullptr}; 
  std::list<zinhart::thread_pool::task_future<std::int32_t>> tasks;

  // host vectors 
  double * host_total_observations{nullptr};
  double * host_total_targets{nullptr};
  double * host_total_activations{nullptr};
  double * host_total_bias{nullptr};
  double * host_total_hidden_weights{nullptr};

  // host validation vectors
  double * host_total_observations_validation{nullptr};
  double * host_total_targets_validation{nullptr};
  double * host_total_activations_validation{nullptr};
  double * host_total_bias_validation{nullptr};
  double * host_total_hidden_weights_validation{nullptr};

  // device vectors
  double * device_total_observations{nullptr}; 
  double * device_total_activations{nullptr};
  double * device_total_bias{nullptr};
  double * device_total_hidden_weights{nullptr};

  // layer counters
  std::uint32_t current_layer{1}, prior_layer{0};// layer counters start at 1 and 0 respectively because we start with the hidden layer and input layer
  const std::uint32_t input_layer{0};
  const std::uint32_t output_layer{total_layers.size() - 1};

  // declarations for dgemm and dgeam
  std::int32_t  m{0}, n{0}, k{0}, lda{0}, ldb{0}, ldc{0};// note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1

  std::uint32_t total_cases{0};

  // array lengths
  std::uint32_t total_observations_length{0};
  std::uint32_t total_targets_length{0};
  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
  std::uint32_t total_bias_length{0};

  std::uint32_t  i, ith_stream, ith_case; 


  //first layer is always input layer
  LAYER_INFO a_layer;
  a_layer.first = ACTIVATION_NAME::INPUT;
  a_layer.second = neuron_dist(mt);
  total_layers[input_layer] = a_layer;  

  for(current_layer = 1; current_layer < total_layers.size(); ++current_layer) // this for loop is for the hidden layers and output layers
  {
	a_layer.first = ACTIVATION_NAME(activation_dist(mt));// random layer
	a_layer.second = neuron_dist(mt);// random amount of neurons
	total_layers[current_layer] = a_layer;
  }

  // of course should always be the input layer
  ASSERT_EQ(total_layers[input_layer].first, ACTIVATION_NAME::INPUT);

  // total training cases
  total_cases = case_dist(mt);

  // calculate array sizes
  total_observations_length = total_cases * a_layer.second;// number of observations matrix 
  total_targets_length = total_layers[output_layer].second; // number of targets
 
  // calc number of activations
  for(current_layer = 1, total_activations_length = 0; current_layer < total_layers.size(); ++current_layer )
	total_activations_length += total_layers[current_layer].second;// accumulate neurons in the hidden layers and output layer
  
  // calc number of hidden weights
  for(current_layer = 1, prior_layer = 0, total_hidden_weights_length = 0; prior_layer < total_layers.size() - 1; ++current_layer, ++prior_layer)
  {
	total_hidden_weights_length += total_layers[current_layer].second * total_layers[prior_layer].second;
  }
  // calc bias neurons
  total_bias_length = total_layers.size() - 1;

  std::cout<<"In test\n";
  std::cout<<"Total cases: "<<total_cases<<"\n";
  std::cout<<"Total layers size: "<<total_layers.size()<<"\n";
  for(current_layer = 0; current_layer < total_layers.size() ; ++current_layer)
	std::cout<<"Neurons in layer "<<current_layer + 1<<": "<<total_layers[current_layer].second<<"\n";
  std::cout<<"Total hidden weights: "<<total_hidden_weights_length<<"\n";
  for(current_layer = 1, prior_layer = 0; prior_layer < total_layers.size() - 1; ++current_layer, ++prior_layer)
	std::cout<<"weight matrix between layer "<<current_layer + 1<<" and "<<prior_layer + 1<<" dimensions: "<<total_layers[current_layer].second<<" by "<<total_layers[prior_layer].second<<"\n";

  // allocate host vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_observations, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_targets, sizeof(double) * total_targets_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_activations, sizeof(double) * total_activations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_hidden_weights, sizeof(double) * total_hidden_weights_length,cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_bias, sizeof(double) * total_bias_length, cudaHostAllocDefault),__FILE__,__LINE__));

  // allocate validation vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_observations_validation, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_targets_validation, sizeof(double) * total_targets_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_activations_validation, sizeof(double) * total_activations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_hidden_weights_validation, sizeof(double) * total_hidden_weights_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_bias_validation, sizeof(double) * total_bias_length, cudaHostAllocDefault),__FILE__,__LINE__));

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
	host_total_observations[i] = real_dist(mt); // random training set
    for(i = 0; i < total_activations_length; ++i)
	host_total_activations[i] = 0.0f; // start at zero since these values have not been calculated yet
  for(i = 0; i < total_hidden_weights_length; ++i)
	host_total_hidden_weights[i] = real_dist(mt); // random weights
  for(i = 0; i < total_bias_length; ++i)
	host_total_bias[i] = real_dist(mt); // random bias (which is an oxymoron?)

  // copy host memory to device for each stream
  for(ith_stream = 0; ith_stream < n_cuda_streams; ++ith_stream )
  {
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_observations, host_total_observations, total_observations_length * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_activations, host_total_activations, total_activations_length * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_bias, host_total_bias, total_bias_length * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_hidden_weights, host_total_hidden_weights, total_hidden_weights_length * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__)); /**/
  }

  // synchronize the host thread wrt each stream to ensure the memory transactions (HostToDevice) above have been completed 
  for(ith_stream = 0; ith_stream < n_cuda_streams; ++ith_stream )
  {
	ASSERT_EQ(0, zinhart::check_cuda_api( cudaStreamSynchronize(streams[ith_stream]), __FILE__, __LINE__));
  } 

  // cublas initialization and error check
  cublasHandle_t context;
  ASSERT_EQ(0, zinhart::check_cublas_api(cublasCreate(&context),__FILE__, __LINE__)); 

  // prepare for forward propagate
  ffn net;
  bool copy_to_host = false;
  // for each case for each stream forward propogate
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	for (ith_stream = 0; ith_stream < /*n_cuda_streams*/1; ++ith_stream)
	{
	  // assert proper return codes
	  tasks.push_back(zinhart::default_thread_pool::push_task([&](){return net.forward_propagate_async(copy_to_host, 
															   streams[ith_stream], context, 
																ith_case, total_layers,
																total_targets_length, host_total_targets,
																total_hidden_weights_length, host_total_hidden_weights,
																device_total_observations, device_total_bias, device_total_hidden_weights,
																total_activations_length, host_total_activations, device_total_activations);
                                              }
						                )
					 );
	  // this is necesarry to ensure that for example all async forward propagates return before the cuda context they are using is destroyed
	  // auto res = tasks.back().get();
	  // ASSERT_EQ(0, res);
	}
  }

  // validation loop will check the results of forward propagate for each training case in each stream (in each thread)
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	for(ith_stream = 0; ith_stream < 1/*n_cuda_streams*/; ++ith_stream )
	{

	  // block the main thread until forward propagate has been completed for all training cases, will start at the front since this is the most likely task to be completed
	  ASSERT_EQ(0, tasks.front().get());

	  // copy device memory back to host at each iteration
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_observations, device_total_observations, total_observations_length * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_activations, device_total_activations, total_activations_length * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_bias, device_total_bias, total_bias_length * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_hidden_weights, device_total_hidden_weights, total_hidden_weights_length * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  
	  // synchronize the host thread wrt each stream to ensure the asynchronous memory transactions (DeviceToHost) above have been completed
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaStreamSynchronize(streams[ith_stream]), __FILE__, __LINE__));


	  // do serial forward propagate operation for each layer for each stream on a different host thread
	  
	  // get serial results 
	  
	  // validate cpu and gpu activation vectors
	  
	  // validate output vector
	  
	  // setup for the next iteration i.e the future has been consumed by this point 
	  tasks.pop_front();

	}
  }


 

  // release cublas resources and check for errors
  ASSERT_EQ(0,zinhart::check_cublas_api(cublasDestroy(context),__FILE__, __LINE__));

  // deallocate host memory and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_observations),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_targets),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_activations),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_hidden_weights),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_bias),__FILE__,__LINE__));

  // deallocate host validation memory and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_observations_validation),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_targets_validation),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_activations_validation),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_hidden_weights_validation),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_bias_validation),__FILE__,__LINE__));

  // deallocate device memory and check for errors  
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_observations), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_activations), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_bias), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_hidden_weights), __FILE__, __LINE__));

  // destroy cuda streams
  for (ith_stream = 0; ith_stream < n_cuda_streams; ++ith_stream)
  {
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaStreamDestroy(streams[ith_stream]),__FILE__,__LINE__));
  }

  // deallocate cuda streams
  delete [] streams;

  ASSERT_EQ(0,zinhart::check_cuda_api(cudaDeviceReset(), __FILE__, __LINE__));
}
#endif
