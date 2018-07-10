#include <cassert>
namespace zinhart
{
  namespace models
  {
  	template <class model_type, class precision_type>
  	  HOST void ann<model_type, precision_type>::add_layer(const zinhart::activation::LAYER_INFO & ith_layer)
  	  { total_layers.push_back(ith_layer); }
	template <class model_type, class precision_type>
	  HOST const std::vector<zinhart::activation::LAYER_INFO> ann<model_type, precision_type>::get_total_layers()const
	  {return total_layers; }
	template <class model_type, class precision_type>
	  HOST void ann<model_type, precision_type>::clear_layers()
	  {total_layers.clear(); }
	template <class model_type, class precision_type>
	  HOST const std::uint32_t ann<model_type, precision_type>::get_total_activations()const
	  {return total_activations_length;}
	template <class model_type, class precision_type>
	  HOST const std::uint32_t ann<model_type, precision_type>::get_total_deltas()const
	  {return total_deltas_length;}
	template <class model_type, class precision_type>
	  HOST const std::uint32_t ann<model_type, precision_type>::get_total_hidden_weights()const
	  {return total_hidden_weights_length;}
	template <class model_type, class precision_type>
	  HOST const std::uint32_t ann<model_type, precision_type>::get_total_gradients()const
	  {return total_gradient_length;}
	template <class model_type, class precision_type>
	  HOST const std::uint32_t ann<model_type, precision_type>::get_total_bias()const
	  {return total_bias_length;}
	/* 
	// partial cpu and gpu functions here
  	template <class model_type, class precision_type>
	  HOST std::int32_t ann<model_type, precision_type>::train(const std::uint16_t & max_epochs, const std::uint32_t & batch_size, const double & weight_penalty)
		{
		  int error = 0;
		  std::uint32_t ith_epoch, ith_observation;
		  std::uint32_t batch_count;
#if CUDA_ENABLED == 1
		  printf("CUDA ENABLED TRAIN\n");
		  cublasStatus_t error_id;
		  cublasHandle_t handle;
		  error_id = cublasCreate(&handle);
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"CublasHandle creation failed with error: "<<cublas_get_error_string(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
#else
		  //cpu multi-threaded code will go here
		  printf("CUDA DISABLED TRAIN\n");
#endif
		  //debugging lines
		  std::cout<<"max_epochs: "<<max_epochs<<" total training cases: "<<total_observations.first<<" Training case size: "<<total_layers[0].second
		  		   <<" total_hidden_weights: "<<total_hidden_weights.first<<"\n";
		  for(unsigned int ith_layer = 0; ith_layer < total_layers.size() ; ++ith_layer)
			std::cout<<"Neurons in layer "<<ith_layer + 1<<": "<<total_layers[ith_layer].second<<"\n";
		  for(unsigned int ith_layer = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
			std::cout<<"weight matrix between layer "<<ith_layer + 2<<" and "<<ith_layer + 1<<" dimensions: "<<total_layers[ith_layer + 1].second<<" by "<<total_layers[ith_layer].second<<"\n";
		  //end debug lines
		  
		  for(ith_epoch = 0; ith_epoch < max_epochs / max_epochs; ++ith_epoch)
		  {
			std::cout<<"Epoch: "<<ith_epoch + 1<<"\n";
			for(ith_observation = 0, batch_count = 0; ith_observation < total_observations.first; ++ith_observation, ++batch_count)
			{
			  std::cout<<"Case: "<<ith_observation + 1<<"\n";
#if CUDA_ENABLED == 1 
			  if(batch_count == batch_size)
			  {

  				error = forward_propagate(true, handle, ith_observation, total_layers, total_targets, total_hidden_weights, total_activations, global_device_total_observations, global_device_total_activations, global_device_total_bias, global_device_total_hidden_weights);
				batch_count = 0;//reset the count
			  }	  
			  else
				error = forward_propagate(false, handle, ith_observation, total_layers, total_targets, total_hidden_weights, total_activations, global_device_total_observations, global_device_total_activations, global_device_total_bias, global_device_total_hidden_weights);
			  //do something with the error code
			  if(error == 1)
			  {
				std::cerr<<"An error occured in forward_propagate\n";
				std::abort();
			  }
			  //call back_propagate
#else
			  std::cout<<"Apples\n";
			  static_cast<model_type*>(this)->forward_propagate(total_layers, ith_observation, total_observations, total_targets, total_hidden_weights, total_activations);
#endif
			}
		  }

#if CUDA_ENABLED == 1
		  error_id = cublasDestroy(handle);
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"cublas handle destruction failed with error: "<<cublas_get_error_string(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
#else
		  //cpu multi-threaded code will go here
		  
#endif
		  return error;
		}
*/
// gpu only functions here
#if CUDA_ENABLED == 1
  	template <class model_type, class precision_type>
  	  HOST std::int32_t ann<model_type, precision_type>::init(std::pair<std::uint32_t, double *> & total_observations,
						   std::pair<std::uint32_t, double *> & total_targets,
						   std::uint32_t device_id = 0
				          )
		{
		
		  if ( check_cuda_api( cudaSetDevice(device_id),__FILE__, __LINE__) == 1) // set device
		      return 1;
		  std::uint32_t ith_layer;
		  this->total_observations.first = total_observations.first;
		  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_observations.second, sizeof(double) * this->total_observations.first, cudaHostAllocDefault),__FILE__,__LINE__);
		  this->total_targets.first = total_targets.first;
		  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_targets.second, sizeof(double) * this->total_targets.first, cudaHostAllocDefault),__FILE__,__LINE__);

		  //of course the last layer should have the same number of neurons as their are targets,
		  //additionally the user may make the mistake on to doing this and the correction is not the
		  //responsibility of ann

		  //calc number of activations, number of deltas is the same
		  for(ith_layer = 1, this->total_activations.first = 0; ith_layer < total_layers.size(); ++ith_layer )
			this->total_activations.first += total_layers[ith_layer].second;// accumulate neurons in the hidden layers and output layers
	  	  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_activations.second, sizeof(double) * this->total_activations.first, cudaHostAllocDefault),__FILE__,__LINE__);// allocate activations

          this->total_deltas.first = this->total_activations.first;
		  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_deltas.second, sizeof(double) * this->total_deltas.first,cudaHostAllocDefault),__FILE__,__LINE__);// allocate deltas


		  //calc number of hidden weights, number of gradients is the same
		  for(ith_layer = 0, this->total_hidden_weights.first = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
			this->total_hidden_weights.first += this->total_layers[ith_layer + 1].second * this->total_layers[ith_layer].second;
		  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_hidden_weights.second, sizeof(double) * this->total_hidden_weights.first,cudaHostAllocDefault),__FILE__,__LINE__);

		  this->total_gradient.first = this->total_hidden_weights.first;
		  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_gradient.second, sizeof(double) * this->total_gradient.first,cudaHostAllocDefault),__FILE__,__LINE__);// allocate gradients

		  //the error
		  this->total_error.first = this->total_targets.first;
		  zinhart::check_cuda_api(cudaHostAlloc((void**)&this->total_error.second, sizeof(double) * this->total_targets.first, cudaHostAllocDefault),__FILE__,__LINE__);
		  return cuda_init();
		}
	template <class model_type, class precision_type>
  	  HOST std::int32_t ann<model_type, precision_type>::cuda_init()
		{
		  // allocate space for observations		
  		  if( check_cuda_api(cudaMalloc( (void **) &global_device_total_observations, total_observations.first * total_layers[0].second * sizeof(double)),__FILE__, __LINE__) == 1)
  			return 1;	
		  // copy allocations from host to device
		  else if(check_cuda_api(cudaMemcpy(global_device_total_observations, total_observations.second, total_observations.first * sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1) 
			return 1;			
		  // allocate space for targets	
		  else if ( check_cuda_api(cudaMalloc((void **) &global_device_total_targets, total_targets.first * sizeof(double)),__FILE__, __LINE__) == 1) 
			return 1;	  
		  //copy targets from host to device
		  else if ( check_cuda_api(cudaMemcpy(global_device_total_targets, total_targets.second, total_targets.first * sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1 )
			return 1;
		  //allocate space for hidden weights
		  else if ( check_cuda_api(cudaMalloc((void **) &global_device_total_hidden_weights, total_hidden_weights.first * sizeof(double)),__FILE__, __LINE__) == 1)
			return 1;
		 //copy hidden_weights from host to device
		 else if ( check_cuda_api(cudaMemcpy(global_device_total_hidden_weights, total_hidden_weights.second, total_hidden_weights.first * sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1)
		   return 1;
		 //allocate space for activations
		 else if ( check_cuda_api(cudaMalloc((void **) &global_device_total_activations, total_activations.first * sizeof(double)),__FILE__, __LINE__) == 1 )
		   return 1;
		 //copy activations from host to device
		 else if( check_cuda_api(cudaMemcpy(global_device_total_activations, total_activations.second, total_activations.first * sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1) 
		   return 1;
		 //allocate space for error
		 else if ( check_cuda_api(cudaMalloc((void **) &global_device_total_error, total_error.first * sizeof(double)),__FILE__, __LINE__) == 1)
		   return 1;
   		 //copy error from host to device
		 else if ( check_cuda_api(cudaMemcpy(global_device_total_error, total_error.second, total_error.first * sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1)
		   return 1;
	  	 //allocate space for gradients
		 else if ( check_cuda_api(cudaMalloc((void **) &global_device_total_gradient, total_gradient.first * sizeof(double)),__FILE__, __LINE__) == 1)
		   return 1;
		 //copy gradients from host to device
		 else if ( check_cuda_api(cudaMemcpy(global_device_total_gradient, total_gradient.second, total_gradient.first * sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1) 
			 return 1;
	     //allocate space for deltas
		 else if (check_cuda_api(cudaMalloc((void **) &global_device_total_deltas, total_deltas.first * sizeof(double)),__FILE__, __LINE__) == 1)
  		   return 1;
  		 //copy deltas from host to device
		 else if (check_cuda_api(cudaMemcpy(global_device_total_deltas, total_deltas.second, sizeof(double), cudaMemcpyHostToDevice),__FILE__, __LINE__) == 1)
		   return 1;
		 //allocate space for bias
		 else if(check_cuda_api(cudaMalloc((void**)&global_device_total_bias, total_activations.first * sizeof(double)),__FILE__, __LINE__) == 1)
		   return 1;
		 //set bias from host to device
		 else if(check_cuda_api(cudaMemset(global_device_total_bias, 1, total_activations.first * sizeof(double)),__FILE__, __LINE__) == 1)
		   return 1;
		 return 0; 
		}
	template <class model_type, class precision_type>
		HOST std::int32_t ann<model_type, precision_type>::cuda_cleanup()
		{

		  if(check_cuda_api(cudaFreeHost(total_observations.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if(check_cuda_api(cudaFreeHost(total_targets.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if(check_cuda_api(cudaFreeHost(total_hidden_weights.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if(check_cuda_api(cudaFreeHost(total_activations.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if(check_cuda_api(cudaFreeHost(total_error.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if (check_cuda_api(cudaFreeHost(total_gradient.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if (check_cuda_api(cudaFreeHost(total_deltas.second),__FILE__,__LINE__) == 1)
			return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_observations),__FILE__, __LINE__) == 1)
		   return 1;	
		  else if (check_cuda_api(cudaFree(global_device_total_targets),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_hidden_weights),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_activations),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_error),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_gradient),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_deltas),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaFree(global_device_total_bias),__FILE__, __LINE__) == 1)
		   return 1;
		  else if(check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__) == 1)
		   return 1;
		  return 0;
		}
	template <class model_type, class precision_type>
  	  HOST std::int32_t ann<model_type, precision_type>::forward_propagate(const bool & copy_device_to_host, cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, double *> & total_targets, 
			                   const std::pair<std::uint32_t, double *> & total_hidden_weights,
							   const std::pair<std::uint32_t, double *> & total_activations,
							   double * device_total_observation, double * device_total_activation, double * device_total_bia, double * device_total_hidden_weight)
		{ return static_cast<model_type*>(this)->forward_propagate(copy_device_to_host, context, ith_observation_index, total_layers, total_targets, total_hidden_weights, total_activations
			,device_total_observation, device_total_activation, device_total_bia, device_total_hidden_weight); }
	//cpu only functions here
#else
	template <class model_type, class precision_type>
		HOST void ann<model_type, precision_type>::init(const std::uint32_t & n_threads)
		{
		  assert(this->total_layers.size() != 0);
		  assert(n_threads > 0);
		  const std::uint32_t input_layer{0};
		  const std::uint32_t output_layer{total_layers.size() - 1};
		  std::uint32_t ith_layer;

		  // calculate vector lengths
		  for(ith_layer = 1; ith_layer < this->total_layers.size(); ++ith_layer)
			this->total_activations_length += this->total_layers[ith_layer].second;// accumulate neurons from the first hidden layer to the output layer
		  this->total_activations_length *= n_threads;// lengthen this vector by the number of threads so that each thread can have its own workspace
		  this->total_deltas_length = total_activations_length;
		  
		  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < this->total_layers.size() - 1; ++ith_layer)
			this->total_hidden_weights_length += this->total_layers[ith_layer + 1].second * this->total_layers[ith_layer].second;
		  this->total_gradient_length = this->total_hidden_weights_length;

		  this->total_bias_length = this->total_layers.size() - 1;

		  // memory alignment
		  std::uint32_t alignment = ( sizeof(precision_type) == 8) ? 64 : 32;
		  
		  // allocate vectors 
		  this->total_activations = (precision_type*) mkl_malloc( this->total_activations_length * sizeof( precision_type ), alignment );
		  this->total_deltas = (precision_type*) mkl_malloc( this->total_deltas_length * sizeof( precision_type ), alignment );
		  this->total_hidden_weights = (precision_type*) mkl_malloc( this->total_hidden_weights_length * sizeof( precision_type ), alignment );
		  this->total_gradient = (precision_type*) mkl_malloc( this->total_gradient_length * sizeof( precision_type ), alignment );
		  this->total_bias = (precision_type*) mkl_malloc( this->total_bias_length * sizeof( precision_type ), alignment );
		}
	template <class model_type, class precision_type>
	  HOST std::int32_t ann<model_type, precision_type>::forward_propagate(const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
																		   const precision_type * total_training_cases, const std::uint32_t & case_index,
																		   precision_type * total_activations, const std::uint32_t & total_activations_length,
																		   const precision_type * total_hidden_weights, const std::uint32_t & total_hidden_weights_length,
																	  	   const precision_type * total_bias,
																		   const std::uint32_t & thread_id 
		                                                                  )
	  {
		static_cast<model_type*>(this)->forward_propagate(this->total_layers, 
			                                              total_training_cases, case_index, 
			                                              total_activations, total_activations_length, 
														  total_hidden_weights, total_hidden_weights_length,
														  total_bias,
														  thread_id
														 );
	  }
	template <class model_type, class precision_type>
	  HOST void ann<model_type, precision_type>::cleanup()
	  {
		mkl_free(this->total_activations);
		mkl_free(this->total_deltas);
		mkl_free(this->total_hidden_weights);
		mkl_free(this->total_gradient);
		mkl_free(this->total_bias);

		total_activations_length = 0;
		total_deltas_length = 0;	
		total_hidden_weights_length = 0;
		total_gradient_length = 0;
		total_bias_length = 0;
	  }
#endif

  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
