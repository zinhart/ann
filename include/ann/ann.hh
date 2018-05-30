#ifndef ANN_HH
#define ANN_HH
#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "loss_function.hh"
#include "activation.hh"
#include "optimizer.hh"
#include <memory>
//#include <zinhart/vector_space>
#include <vector>
#if CUDA_ENABLED == true
#define ERROR_CUDA_ERROR 1
#include <cublas_v2.h>
//#include <thrust>
#endif
#include <cassert>
namespace zinhart
{

  template <class model_type>
	class ann
	{
#if CUDA_ENABLED == true
		 double * global_device_total_observations;
		 double * global_device_total_targets;
		 double * global_device_total_hidden_weights;
		 double * global_device_total_activations;
		 double * global_device_total_bias;
		 double * global_device_total_error;
		 double * global_device_total_gradient;
		 double * global_device_total_deltas;
#endif

	  protected:

		std::vector<LAYER_INFO> total_layers;//Layer types(intput relu sigmoid etc) and the number of inputs of the respective layer 
        //number of training cases, trainint case size, the training cases themselves
		std::pair<std::uint32_t, double *> total_observations;
		std::pair<std::uint32_t, double *> total_targets; // output layer size and the complete set of targets for each input
		std::pair<std::uint32_t, double *> total_hidden_weights;// the number of hidden weights for a layer and the weights themselves
		std::pair<std::uint32_t, double *> total_activations;//this is the sum of all the hidden layers and the output layer neurons
		std::pair<std::uint32_t, double *> total_error;
		std::pair<std::uint32_t, double *> total_gradient;
		std::pair<std::uint32_t, double *> total_deltas;
	  public:
		ann() = default;
		ann(const ann<model_type> &) = delete;
		ann(ann<model_type> &&) = delete;
		ann<model_type> & operator = (const ann<model_type>&) = delete;
		ann<model_type> & operator = (ann<model_type> &&) = delete;
		~ann() = default;

		//debugging functions
		const std::vector<LAYER_INFO> & get_total_layers()const
		{return total_layers; }
		const std::pair<std::uint32_t, double *> & get_total_observations()const
		{return total_observations;}
		const std::pair<std::uint32_t, double *> & get_total_hidden_weights()const
		{return total_hidden_weights;}
		const std::pair<std::uint32_t, double *> & get_total_activations()const
		{return total_activations;}
		const std::pair<std::uint32_t, double *> & get_total_error()const
		{return total_error;}
	    const std::pair<std::uint32_t, double *> & get_total_gradient()const
		{return total_gradient;}
	    const std::pair<std::uint32_t, double *> & get_total_deltas()const
		{return total_deltas;}
		//end debugging functions

		//model manipulating functions
		//I assume the first layer will be an input layer
		HOST void add_layer(const LAYER_INFO & ith_layer)
		{ total_layers.push_back(ith_layer); }

#if CUDA_ENABLED == 1
		HOST std::int32_t init(
		         std::pair<std::uint32_t, double *> & total_observations,
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
#else
		HOST std::int32_t init(
		         std::pair<std::uint32_t, double *> & total_observations,
				 std::pair<std::uint32_t, double *> & total_targets
				)
		{
		  std::uint32_t ith_layer;
		  std::swap(this->total_observations,total_observations);
		  std::swap(this->total_targets, total_targets);

		  //of course the last layer should have the same number of neurons as their are targets,
		  //additionally the user may make the mistake on to doing this and the correction is not the
		  //responsibility of ann

		  //calc number of activations, number of deltas is the same
		  for(ith_layer = 1, this->total_activations.first = 0; ith_layer < total_layers.size(); ++ith_layer )
			this->total_activations.first += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layers
	  	  this->total_activations.second  = new double[this->total_activations.first];//allocate activations
          this->total_deltas.first = this->total_activations.first;
		  this->total_deltas.second = new double[this->total_deltas.first];//allocate deltas

		  //calc number of hidden weights, number of gradients is the same
		  for(ith_layer = 0, this->total_hidden_weights.first = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
			this->total_hidden_weights.first += this->total_layers[ith_layer + 1].second * this->total_layers[ith_layer].second;

 		  this->total_hidden_weights.second = new double[this->total_hidden_weights.first];//allocate weights
		  this->total_gradient.first = this->total_hidden_weights.first;
		  this->total_gradient.second = new double[this->total_gradient.first];//allocate gradients

		  //the error
		  this->total_error.first = this->total_targets.first;
		  this->total_error.second = new double[this->total_targets.first];//allocate gradients
		  return 0;
#endif
		}

#if CUDA_ENABLED == 1
		HOST std::int32_t cuda_init()
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
		HOST std::int32_t cuda_cleanup()
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
		

#endif
#if CUDA_ENABLED == 1
		HOST std::int32_t forward_propagate(const bool & copy_device_to_host, cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, double *> & total_targets, 
			                   const std::pair<std::uint32_t, double *> & total_hidden_weights,
							   const std::pair<std::uint32_t, double *> & total_activations,
							   double * device_total_observation, double * device_total_activation, double * device_total_bia, double * device_total_hidden_weight)
		{ return static_cast<model_type*>(this)->forward_propagate(copy_device_to_host, context, ith_observation_index, total_layers, total_targets, total_hidden_weights, total_activations
			,device_total_observation, device_total_activation, device_total_bia, device_total_hidden_weight); }
#endif
		HOST int train(const std::uint16_t & max_epochs, const std::uint32_t & batch_size, const double & weight_penalty)
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
	};
  	class ffn : public ann< ffn >
	{
	  public:
		ffn() = default;
		ffn(const ffn &) = default;
		ffn(ffn &&) = default;
		ffn & operator = (const ffn &) = default;
		ffn & operator = (ffn &&) = default;
		~ffn() = default;
#if CUDA_ENABLED == 1
		HOST std::int32_t forward_propagate(const bool & copy_device_to_host, cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, double *> & total_targets, 
			                   const std::pair<std::uint32_t, double *> & total_hidden_weights,
							   const std::pair<std::uint32_t, double *> & total_activations,
							   double * device_total_observations, double * device_total_activations, double * device_total_bias, double * device_total_hidden_weights
							  )

		{
		  std::cout<<"Total layers size: "<<total_layers.size()<<"\n";
		  for(unsigned int ith_layer = 0; ith_layer < total_layers.size() ; ++ith_layer)
			std::cout<<"Neurons in layer "<<ith_layer + 1<<": "<<total_layers[ith_layer].second<<"\n";
		  for(unsigned int ith_layer = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
			std::cout<<"weight matrix between layer "<<ith_layer + 2<<" and "<<ith_layer + 1<<" dimensions: "<<total_layers[ith_layer + 1].second<<" by "<<total_layers[ith_layer].second<<"\n";

		  cublasStatus_t error_id;
		  std::int32_t  m, n, k, lda, ldb,ldc;// note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1
		  std::uint32_t ith_layer;// from the first hidden layer to the output layer 
		  std::uint32_t weight_offset = 0;// number of weights connection between layer i and layer i + 1
		  std::uint32_t case_begin = total_layers[0].second * ith_observation_index;//where a case begins, when ith_obs_index is 0 this is the first case
		
		  const double alf = 1;
		  const double bet_mult = 0, bet_add = 1;
		  const double *alpha = &alf;
		  const double *beta1 = &bet_mult;
		  const double *beta2 = &bet_add;

		  m = 1; // cols of x
		  n = total_layers[1].second; // rows of weight matrix 
		  k = total_layers[0].second; // rows of x

		  lda = m;
		  ldb = k;
		  ldc = m;

		  
		  error_id = cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, 
					  m, n, k, 
			          alpha, 
					  device_total_observations + case_begin, lda,
					  device_total_hidden_weights, ldb, 
					  beta1,
					  device_total_activations, ldc
					  );
/*		  //do  first hidden layer and input layer
		  m = total_layers[1].second; // the weight matrix's rows are the number of neurons in the first hidden layer, it's columns are the number of neurons in  the input layer
		  n = 1; // the result is a row vector
		  k = total_layers[0].second; // the number of columns of the weight matrix and rows of the input layer

		  lda = m;
		  ldb = k;
		  ldc = m;
	 
		  //perform Wx  original
		  error_id = cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, 
					  m, n, k, 
			          alpha, 
					  device_total_hidden_weights, lda,
					  device_total_observations + case_begin, ldb, 
					  beta1,
					  device_total_activations, ldc
					  );*/
		  
		  //check for errors
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"cublas dgemm on first hidden layer and input layer failed with error: "<<cublas_get_error_string(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
/*
		  //update dimensions of matrix multiplication
		  lda = total_layers[1].second;
		  ldb = lda;
		  ldc = lda;
		  //add in bias
		  error_id = cublasDgeam(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[1].second, 1,
			                     alpha, device_total_activations, lda,
								 beta2, device_total_bias, ldb,
								 device_total_activations, ldc
			                    );
		  //check for error on dgemm
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"cublas dgeam on first hidden layer and input layer failed with error: "<< cublas_get_error_string(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  //Wx + b complete
		  //call activation function kernel here
		  call_activation(total_layers[1].first, ACTIVATION_TYPE::OBJECTIVE, device_total_activations, total_layers[1].second);

		  //f(Wx + b) complete for first hidden layer and input layer
		  
		  //second hidden layer to output layer, see above for why weight offset = lda * ldb
		  for(ith_layer = 1, weight_offset = total_layers[1].second * total_layers[0].second; ith_layer < total_layers.size() - 1; ++ith_layer )
		  {
			//std::cout<<"ith_layer: "<<ith_layer<<" weight offset: "<<weight_offset<<" "
			//<<total_layers[2].second<<" "<<total_layers[1].second<<" "<<std::uint32_t(total_layers[ith_layer + 1].second * total_layers[ith_layer].second)<<"\n";

			//perform Wx
			lda = total_layers[ith_layer + 1].second;//Neurons in the current layer(Rows of A) m
			ldb = total_layers[ith_layer].second;//Neurons in the prior layer(Rows of B and by definitions of the matrix product Columns of A)
			ldc = lda;//output vector of the matrix product of the current layer times the output of the prior layer
			error_id = cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[ith_layer + 1].second, 1, total_layers[ith_layer].second,
								   alpha, device_total_hidden_weights + weight_offset, lda, 		
								   device_total_activations + total_layers[ith_layer].second, ldb, beta1,
								   device_total_activations + total_layers[ith_layer + 1].second, ldc 
								  );
		    if(error_id != CUBLAS_STATUS_SUCCESS)
		    {
			  std::cerr<<"cublas dgemm failed with error: "<<cublas_get_error_string(error_id)<<"\n";
		 	  return ERROR_CUDA_ERROR;
		    }
		    ldb = total_layers[ith_layer + 1].second;	
			//add in bias
			error_id = cublasDgeam(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[ith_layer + 1].second, 1,
			                     alpha, device_total_activations + total_layers[ith_layer + 1].second, lda,
								 beta2, device_total_bias + total_layers[ith_layer + 1].second, ldb,
								 device_total_activations + total_layers[ith_layer + 1].second, ldc
			                    );
			if(error_id != CUBLAS_STATUS_SUCCESS)
			{
			  std::cerr<<"cublas dgeam on failed with error: "<< cublas_get_error_string(error_id)<<"\n";
			  return ERROR_CUDA_ERROR;
			}
			//Wx + b complete
			//call activation function kernel here	
			call_activation(total_layers[ith_layer + 1].first, ACTIVATION_TYPE::OBJECTIVE, device_total_activations, total_layers[ith_layer + 1].second);	
			//f(Wx + b) complete for all hidden layers after the first till the output layer
			weight_offset += total_layers[ith_layer + 1].second * total_layers[ith_layer].second;//update weight pointer
		  } 
		  //copy activations back to host
		  if(copy_device_to_host == true)
		  {
		    
			//copy activations from device to host
		    error_id = cudaMemcpy(device_total_targets, total_targets.second.get(), total_targets.first * sizeof(double), cudaMemcpyDeviceToHost);
			if(error_id != cudaSuccess)
			{
			  std::cerr<<"device target copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			  return ERROR_CUDA_ERROR;
			}
			
		  }
*/		
		  return 0;
		}

		HOST std::int32_t forward_propagate_async(const bool & copy_device_to_host, 
							   const cudaStream_t & stream, const cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<LAYER_INFO> & total_layers,
							   const std::uint32_t & total_targets, const double * host_total_targets, 
			                   const std::uint32_t & total_hidden_weights, const double * host_total_hidden_weights,
							   const double * device_total_observations,  const double * device_total_bias, const double * device_total_hidden_weights,
							   const std::uint32_t & total_activations, double * host_total_activations, double * device_total_activations
							  )

		{
		  // layer counters
		  std::uint32_t current_layer{1}, prior_layer{0};// layer counters start at 1 and 0 respectively because we start with the hidden layer and input layer
		  const std::uint32_t input_layer{0};
		  const std::uint32_t output_layer{total_layers.size() - 1};
		  
		  // declarations for dgemm and dgeam
		  std::int32_t  m{0}, n{0}, k{0}, lda{0}, ldb{0},ldc{0};// note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1

		  std::uint32_t weight_offset = {0};// number of weights connection between layer i and layer i + 1
		  std::uint32_t case_begin = {total_layers[input_layer].second * ith_observation_index};// where a case begins, when ith_obs_index is 0 this is the first case

	 	  // coefficients for gemm and geam
		  const double alpha{1};
		  const double beta_mult{0};
		  const double beta_add{1};


		  // get col major coordinates without explicitly transposing 
		  // ( total_layers[1].second is rows of the weight matrix and the number of neurons in the first hidden layer )
		  // ( total_layers[0].second is the columns of the weight matrix and the number of neurons in the input layer )
		  zinhart::gemm_wrapper(m, n, k, lda, ldb, ldc, total_layers[current_layer].second, total_layers[input_layer].second, total_layers[input_layer].second, 1);

		  // set cublasStream
		  if(zinhart::check_cublas_api(cublasSetStream(context, stream), __FILE__, __LINE__) != 0 )
			return 1;


		  /*std::cout<<"In forward_propagate_async Total layers size: "<<total_layers.size()<<"\n";
		  for(current_layer = 0; current_layer < total_layers.size() ; ++current_layer)
		  	std::cout<<"Neurons in layer "<<current_layer + 1<<": "<<total_layers[current_layer].second<<"\n";
  		  for(current_layer = 1, prior_layer = 0; prior_layer < total_layers.size() - 1; ++current_layer, ++prior_layer)
			std::cout<<"weight matrix between layer "<<current_layer + 1<<" and "<<prior_layer + 1<<" dimensions: "<<total_layers[current_layer].second<<" by "<<total_layers[prior_layer].second<<"\n";*/

		  // do Wx for first hidden layer and input layer
		  if(zinhart::check_cublas_api(cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, 
					  m, n, k, 
			          &alpha, 
					  device_total_observations + case_begin, lda,
					  device_total_hidden_weights, ldb, 
					  &beta_mult,
					  device_total_activations, ldc
					  ),__FILE__, __LINE__) != 0)
		  {
			return 1;
		  }
		  // The dimensions of Wx is m x n, parameters updated for dgeam
		  current_layer = 1;
		  lda = total_layers[current_layer].second;
		  ldb = total_layers[current_layer].second;
		  ldc = total_layers[current_layer].second;
		  //add in bias
		  /*if(zinhart::check_cublas_api(cublasDgeam(context, CUBLAS_OP_N, CUBLAS_OP_N, 
								 m, n,
			                     &alpha, 
								 device_total_activations, lda,
								 &beta_add, 
								 device_total_bias, ldb,
								 device_total_activations, ldc
			                    ),__FILE__, __LINE__) != 0)
		  {
			return 1;
		  }*/		
		  // call activation
		  // f(Wx + b) complete for first hidden layer and input layer
		  


		  //do  first hidden layer and input layer

/*		  		  //copy activations back to host
		  if(copy_device_to_host == true)
		  {
		    
			//copy activations from device to host
		    error_id = cudaMemcpy(device_total_targets, total_targets.second.get(), total_targets.first * sizeof(double), cudaMemcpyDeviceToHost);
			if(error_id != cudaSuccess)
			{
			  std::cerr<<"device target copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			  return ERROR_CUDA_ERROR;
			}
			
		  }
*/		
		  return 0;
		}
		HOST void backward_propagate(cublasHandle_t & context)
		{
		  //cublas gemm here
		}

#else
		//cpu multi-threaded code will go here
		void forward_propagate(const std::uint16_t & case_size, const std::uint32_t & ith_observation_index,
			                   const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, std::shared_ptr<float>> & total_observations, 
							   const std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets, 
			                   const std::pair<std::uint32_t, double *> & total_hidden_weights
					          )
		{
		  std::cout<<"IN CPU\n";
		  //lapacke gemm etc
		}
		void backward_propagate(LAYER_INFO & info)
		{
		  //lapacke gemm etc
		}

#endif
	};

	//Begin External Interface
	//debugging functions
	template <class T>
	  HOST std::vector<LAYER_INFO> get_total_layers(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, double *> get_total_observations(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, double *> get_total_hidden_weights(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, double *> get_total_activations(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, double *> get_total_error(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, double *> get_total_gradient(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, double *> get_total_deltas(const ann<T> & model);
	//End debugging functions
	template<class T>
  	  HOST void add_layer(ann<T> & model, const LAYER_INFO & ith_layer);

#if CUDA_ENABLED == 1
	template<class T>
	  HOST std::int32_t initialize_model(ann<T> & model,  
							 std::pair<std::uint32_t, double *> & total_observations,
							 std::pair<std::uint32_t, double *> & total_targets, std::uint32_t device_id = 0 );
	template<class T>
	  HOST std::int32_t forward_propagate(ann<T> & model,const bool & copy_device_to_host, cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, double *> & total_targets, 
			                   const std::pair<std::uint32_t, double *> & total_hidden_weights,
							   const std::pair<std::uint32_t, double *> & total_activations,
							   double * device_total_observations, double * device_total_activations, double * device_total_bias, double * device_total_hidden_weights);
#endif
	template<class T>
	  HOST int train(ann<T> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);
	template<class T>
	  HOST int cleanup(ann<T> & model);
}
#endif
