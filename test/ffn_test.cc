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
TEST(ffn_test, async_forward_propagate_one_host_thread_multiple_cuda_streams)
{
  // set device properties
  cudaDeviceProp properties;
  zinhart::check_cuda_api(cudaGetDeviceProperties(&properties,0),__FILE__, __LINE__);
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1,10);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  std::uniform_int_distribution<std::uint32_t> case_dist(1,1);
  /*std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );*/
  std::uniform_int_distribution<int> real_dist(std::numeric_limits<std::uint8_t>::min(), std::numeric_limits<std::uint8_t>::max() );

  std::uniform_int_distribution<std::uint8_t> layer_num_dist(2,5/*std::numeric_limits<std::uint8_t>::max()*/);// at least an input and output layer
  std::uniform_int_distribution<std::uint8_t> activation_dist(1,8);// currently there are 8 different activation functions not counting the input layer
  const std::uint32_t n_cuda_streams = 5/*MAX_CPU_THREADS*/;
  std::vector<LAYER_INFO> total_layers(layer_num_dist(mt));
  cudaStream_t * streams{nullptr}; 
  std::list<zinhart::thread_pool::task_future<std::int32_t>> forward_propagate_tasks;

  // host vectors 
  double * host_total_observations{nullptr};
  double * host_total_targets{nullptr};
  double * host_total_activations{nullptr};
  double * host_total_bias{nullptr};
  double * host_total_hidden_weights{nullptr};

  // host validation vectors
  double * host_total_observations_val{nullptr};
  double * host_total_targets_val{nullptr};
  double * host_total_activations_val{nullptr};
  double * host_total_bias_val{nullptr};
  double * host_total_hidden_weights_val{nullptr};

  // device vectors
  double * device_total_observations{nullptr}; 
  double * device_total_activations{nullptr};
  double * device_total_hidden_weights{nullptr};

  // layer counters
  std::uint32_t current_layer{1}, prior_layer{0};// layer counters start at 1 and 0 respectively because we start with the hidden layer and input layer
  const std::uint32_t input_layer{0};
  const std::uint32_t output_layer{total_layers.size() - 1};

  // declarations for dgemm and dgeam
  std::int32_t  m{0}, n{0}, k{0}, lda{0}, ldb{0}, ldc{0}, start{0}, stop{0};// note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1

  std::uint32_t total_cases{0};

  // array lengths
  std::uint32_t total_observations_length{0};
  std::uint32_t total_targets_length{0};
  std::uint32_t total_activations_length{0};
  std::uint32_t total_activations_val_length{0};
  std::uint32_t total_hidden_weights_length{0};
  std::uint32_t total_bias_length{0};

  std::uint32_t  i, j, ith_stream, ith_case; 


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
  
  total_activations_val_length = total_activations_length;
  total_activations_length  *= n_cuda_streams; // so that each stream has it's own workspace

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
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_hidden_weights, sizeof(double) * total_hidden_weights_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_bias, sizeof(double) * total_bias_length, cudaHostAllocDefault),__FILE__,__LINE__));

  // allocate validation vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_observations_val, sizeof(double) * total_observations_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_targets_val, sizeof(double) * total_targets_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_activations_val, sizeof(double) * total_activations_val_length, cudaHostAllocDefault),__FILE__,__LINE__));// for each host thread
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_hidden_weights_val, sizeof(double) * total_hidden_weights_length, cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&host_total_bias_val, sizeof(double) * total_bias_length * n_cuda_streams, cudaHostAllocDefault),__FILE__,__LINE__));

  // allocate device vectors
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_observations, total_observations_length * sizeof(double) ),__FILE__,__LINE__));
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaMalloc( (void **) &device_total_activations, total_activations_length * sizeof(double) ) ,__FILE__,__LINE__ ));
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
  {
	host_total_observations[i] = real_dist(mt); // random training set
	host_total_observations_val[i] = host_total_observations[i];
  }
  for(i = 0; i < total_activations_length; ++i)
  {
	host_total_activations[i] = 0.0f; // start at zero since these values have not been calculated yet
  }
  for(i = 0; i < total_activations_val_length; ++i)
  {
	host_total_activations_val[i] = 0.0f; // start at zero since these values have not been calculated yet
  }
  for(i = 0; i < total_hidden_weights_length; ++i)
  {
	host_total_hidden_weights[i] = real_dist(mt); // random weights
	host_total_hidden_weights_val[i] = host_total_hidden_weights[i];
  }
  for(i = 0; i < total_bias_length; ++i)
  {
	host_total_bias[i] = 1/*real_dist(mt)*/; // random bias (which is an oxymoron?!)
	host_total_bias_val[i] = host_total_bias[i];
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
	// when training a model each iteration of forward propagate will have the same observations, weights, and biases, since the point of forward prop is to calculate activation vectors given weights, observations, and biases
	for (ith_stream = 0; ith_stream < n_cuda_streams; ++ith_stream)
	{
	  start = ith_stream * total_layers[1].second;
	  std::cout<<"start: "<<start<<"\n";
	  // copy host memory to device for each stream
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_observations, host_total_observations, total_observations_length * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__));
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_hidden_weights, host_total_hidden_weights, total_hidden_weights_length * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__)); 
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(device_total_activations + start, host_total_activations + start, total_layers[1].second * sizeof(double), cudaMemcpyHostToDevice, streams[ith_stream]), __FILE__, __LINE__));
	  	  // synchronize the host thread wrt each stream to ensure the memory transactions (HostToDevice) above have been completed  
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaStreamSynchronize(streams[ith_stream]), __FILE__, __LINE__));

	  // forward propagate on the with the net (gpu)
	  forward_propagate_tasks.push_back(zinhart::default_thread_pool::push_task(
													  [&](std::uint32_t s)
													  {
														return net.forward_propagate_async(copy_to_host, 
														streams[ith_stream], context, 
														ith_case, total_layers,
														total_targets_length, host_total_targets,
														total_hidden_weights_length, host_total_hidden_weights,
														total_activations_length, host_total_activations,
														host_total_bias,
														device_total_observations, device_total_hidden_weights, device_total_activations + s);
                                                      }, start
						                                                      )
					                   );
	}
  }

  // validation loop will check the results of forward propagate for each training case in each stream (in each thread)
  for(ith_case = 0; ith_case < total_cases; ++ith_case)
  {
	for(ith_stream = 0; ith_stream < n_cuda_streams; ++ith_stream )
	{   
	  // block the main thread until forward propagate has been completed for all training cases, will start at the front since this is the most likely task to be completed
	  ASSERT_EQ(0, forward_propagate_tasks.front().get());
	  start = ith_stream * total_layers[1].second;
	  stop = (ith_stream + 1) * total_layers[1].second;
	  std::cout<<"start: "<<start<<" stop: "<<stop<<"\n";
	  // copy device memory back to host at each iteration
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_observations, device_total_observations, total_observations_length * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_activations + start, device_total_activations + start, total_layers[1].second * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaMemcpyAsync(host_total_hidden_weights, device_total_hidden_weights, total_hidden_weights_length * sizeof(double), cudaMemcpyDeviceToHost, streams[ith_stream]), __FILE__, __LINE__));
	  
	  //SERIAL FORWARD PROPAGATE BEGIN
	  current_layer = 1;
	  prior_layer = 0;
	  m = total_layers[current_layer].second;
	  n = total_layers[prior_layer].second;
	  k = 1;

	  // do Wx for input and first hidden layer
	  zinhart::serial_matrix_product(host_total_hidden_weights_val, host_total_observations_val, host_total_activations_val, m, n, k);
	  // add bias for input and first hidden layer
	  for(i = 0; i < total_layers[1].second; ++i)
	  {
		host_total_activations_val[i] += host_total_bias_val[0];
	  }
	  // Wx + b is complete
	  
	  // call activation
	  
	  for(i = 0; i < total_layers[1].second; ++i)
	  {
		//total_layers[1].second = 
	  }
	  //SERIAL FORWARD PROPAGATE END

	  // synchronize the host thread wrt each stream to ensure the asynchronous memory transactions (DeviceToHost) above have been completed
	  ASSERT_EQ(0, zinhart::check_cuda_api( cudaStreamSynchronize(streams[ith_stream]), __FILE__, __LINE__));

	  zinhart::print_matrix_row_major(host_total_hidden_weights, total_layers[1].second, total_layers[0].second, "total_hidden_weights (cuda)");
	  zinhart::print_matrix_row_major(host_total_observations, total_layers[0].second, 1, "total_observations (cuda and x)");
	  zinhart::print_matrix_row_major(host_total_activations, total_activations_length, 1, "total_activations (cuda and Wx)");
	  zinhart::print_matrix_row_major(host_total_activations_val, total_activations_val_length, 1, "host_total_activations_val (validation)");

	  
	  // validate cpu and gpu activation vectors
	  for(i = 0; i < total_targets_length; ++i)
	  {
		ASSERT_EQ(host_total_observations[i], host_total_observations_val[i]);
	  }		
	  for(i = 0; i < total_bias_length; ++i)
	  {
		ASSERT_EQ(host_total_bias[i], host_total_bias_val[i]);
	  }
	  for(i = 0; i < total_layers[1].second/*total_activations_val_length*/; ++i)// for now just the first hidden layer
	  {
		ASSERT_EQ( host_total_activations[i], host_total_activations_val[i])<<"stream: "<<ith_stream<<" i: "<<i<<"\n";
	  }
	  for(i = 0; i < total_hidden_weights_length; ++i)
	  {
		ASSERT_EQ(host_total_hidden_weights[i], host_total_hidden_weights_val[i])<<"iteration: "<<ith_stream<<"\n";
	  }
	  // validate output vector
	  
	  // setup for the next iteration i.e the future has been consumed by this point 
	  forward_propagate_tasks.pop_front();

	  for(i = 0; i < total_activations_val_length; ++i)
		host_total_activations_val[i] = 0.0f;
	}
  }
  std::cout<<"total cases: "<<total_cases<<"\n";

 

  // release cublas	context and check for errors
  ASSERT_EQ(0,zinhart::check_cublas_api(cublasDestroy(context),__FILE__, __LINE__));

  // deallocate host memory and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_observations),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_targets),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_activations),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_hidden_weights),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_bias),__FILE__,__LINE__));

  // deallocate host validation memory and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_observations_val),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_targets_val),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_activations_val),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_hidden_weights_val),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(host_total_bias_val),__FILE__,__LINE__));

  // deallocate device memory and check for errors  
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_observations), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_total_activations), __FILE__, __LINE__));
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
