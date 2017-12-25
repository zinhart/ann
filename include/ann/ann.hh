#ifndef ANN_HH
#define ANN_HH
#include "loss_function.hh"
#include "layer.hh"
#include "optimizer.hh"
#include <memory>
//#include <zinhart/vector_space>
#include <vector>
#define MAXPOSNUM 2137483647
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#if CUDA_ENABLED == 1
#define ERROR_CUDA_ERROR 1
#include <cublas_v2.h>
//#include <thrust>
#endif
namespace zinhart
{
#if CUDA_ENABLED == 1
		__constant__ double * device_total_observations;
		__constant__ double * device_total_targets;
		__constant__ double * device_total_hidden_weights;
		__constant__ double * device_total_activations;
		__constant__ double * device_total_bias;
		__constant__ double * device_total_error;
		__constant__ double * device_total_gradient;
		__constant__ double * device_total_deltas;
#endif
  template <class model_type>
	class ann
	{
	  protected:
		std::vector<LAYER_INFO> total_layers;//Layer types(intput relu sigmoid etc) and the number of inputs of the respective layer 
        //number of training cases, trainint case size, the training cases themselves
		std::pair<std::uint32_t, std::shared_ptr<double>> total_observations;
		std::pair<std::uint32_t, std::shared_ptr<double>> total_targets; // output layer size and the complete set of targets for each input
		std::pair<std::uint32_t, std::shared_ptr<double>> total_hidden_weights;// the number of hidden weights for a layer and the weights themselves
		std::pair<std::uint32_t, std::shared_ptr<double>> total_activations;//this is the sum of all the hidden layers and the output layer neurons
		std::pair<std::uint32_t, std::shared_ptr<double>> total_error;
		std::pair<std::uint32_t, std::shared_ptr<double>> total_gradient;
		std::pair<std::uint32_t, std::shared_ptr<double>> total_deltas;
	  public:
		ann() = default;
		ann(const ann<model_type> &) = default;
		ann(ann<model_type> &&) = default;
		ann<model_type> & operator = (const ann<model_type>&) = default;
		ann<model_type> & operator = (ann<model_type> &&) = default;
		~ann() = default;

		//debugging functions
		const std::vector<LAYER_INFO> & get_total_layers()const
		{return total_layers; }
		const std::pair<std::uint32_t, std::shared_ptr<double>> & get_total_observations()const
		{return total_observations;}
		const std::pair<std::uint32_t, std::shared_ptr<double>> & get_total_hidden_weights()const
		{return total_hidden_weights;}
		const std::pair<std::uint32_t, std::shared_ptr<double>> & get_total_activations()const
		{return total_activations;}
		const std::pair<std::uint32_t, std::shared_ptr<double>> & get_total_error()const
		{return total_error;}
	    const std::pair<std::uint32_t, std::shared_ptr<double>> & get_total_gradient()const
		{return total_gradient;}
	    const std::pair<std::uint32_t, std::shared_ptr<double>> & get_total_deltas()const
		{return total_deltas;}
		//end debugging functions

		//model manipulating functions
		//I assume the first layer will be an input layer
		HOST void add_layer(const LAYER_INFO & ith_layer)
		{ total_layers.push_back(ith_layer); }
		HOST int init(
		         std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
				 std::pair<std::uint32_t, std::shared_ptr<double>> & total_targets
				)
		{
		  std::uint32_t ith_layer, prior_layer_neurons;
		  std::swap(this->total_observations,total_observations);
		  std::swap(this->total_targets, total_targets);

		  //of course the last layer should have the same number of neurons as their are targets,
		  //additionally the user may make the mistake on to doing this and the correction is not the
		  //responsibility of ann
		  //total_layers[total_layers.size() - 1].second = total_targets.first;

		  for(ith_layer = 0; ith_layer < total_layers.size(); ++ith_layer )
			std::cout<<"layer "<<ith_layer + 1<<" rows: "<<total_layers[ith_layer].second<<" columns: "<<1<<"\n";

		  //calc number of activations, number of deltas is the same
		  for(ith_layer = 1, this->total_activations.first = 0; ith_layer < total_layers.size(); ++ith_layer )
			this->total_activations.first += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layers
	  	  this->total_activations.second  = std::shared_ptr<double>(new double[this->total_activations.first], std::default_delete<double[]>());//allocate activations
          this->total_deltas.first = this->total_activations.first;
		  this->total_deltas.second = std::shared_ptr<double>(new double[this->total_deltas.first], std::default_delete<double[]>());//allocate deltas

		  //calc number of hidden weights, number of gradients is the same
		  for(ith_layer = 0, this->total_hidden_weights.first = 0, prior_layer_neurons = total_layers[0].second; ith_layer < total_layers.size()-1; ++ith_layer)
		  {
			std::cout<<"weight matrix connecting layers "<<ith_layer + 1<<" to "<<ith_layer + 2<<" rows: "<<this->total_layers[ith_layer].second<<" columns: "<< (prior_layer_neurons)<<"\n";
			this->total_hidden_weights.first += this->total_layers[ith_layer].second * (prior_layer_neurons );
			prior_layer_neurons = this->total_layers[ith_layer].second;
		  }
 		  this->total_hidden_weights.second = std::shared_ptr<double> ( new double[this->total_hidden_weights.first], std::default_delete<double[]>() );//allocate weights
		  this->total_gradient.first = this->total_gradient.first;
		  this->total_gradient.second = std::shared_ptr<double>(new double[this->total_gradient.first], std::default_delete<double[]>());//allocate gradients

		  //the error
		  this->total_error.first = this->total_targets.first;
		  this->total_error.second = std::shared_ptr<double>(new double[this->total_targets.first], std::default_delete<double[]>());//allocate gradients
		
#if CUDA_ENABLED == 1
		  return cuda_init();
#else
		  return 0;
#endif
		}

#if CUDA_ENABLED == 1
		HOST int cuda_init()
		{
		  cudaError_t error_id;
		  error_id = cudaSetDevice(0);

		  //it seems that when their are no memory leaks cuda-memcpy reports 0 allocations along with 0 leaks, so don't trip
		  /*cudaMalloc((void**)&test, total_observations.first * sizeof(float));
		  cudaMemcpy(test, total_observations.second.get(), total_observations.first * sizeof(float),cudaMemcpyHostToDevice);
		  cudaFree(test);*/
		  
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"cuda init setDevice failed: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for observations
		  error_id = cudaMalloc( (void **) &device_total_observations, total_observations.first * total_layers[0].second * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device case allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy observations from host to device
		  error_id = cudaMemcpyToSymbol(device_total_observations, &(*(total_observations.second.get()) ), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device case copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  //allocate space for targets
		  error_id = cudaMalloc((void **) &device_total_targets, total_targets.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device target allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  //copy targets from host to device
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*total_targets.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device target copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for hidden weights
		  error_id = cudaMalloc((void **) &device_total_hidden_weights, total_hidden_weights.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device weight allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy hidden weights from host to device
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*total_hidden_weights.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device weight copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for activations
		  error_id = cudaMalloc((void **) &device_total_activations, total_activations.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy activations from host to device
		  error_id = cudaMemcpyToSymbol(device_total_activations, &(*total_activations.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device activation copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for error
		  error_id = cudaMalloc((void **) &device_total_error, total_error.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device error allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy error from host to device
		  error_id = cudaMemcpyToSymbol(device_total_error, &(*total_error.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device error copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for gradient
		  error_id = cudaMalloc((void **) &device_total_gradient, total_gradient.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device gradient allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy gradient from host to device
		  error_id = cudaMemcpyToSymbol(device_total_gradient, &(*total_gradient.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device gradient copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for deltas
		  error_id = cudaMalloc((void **) &device_total_deltas, total_deltas.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device deltas allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy deltas from host to device
		  error_id = cudaMemcpyToSymbol(device_total_deltas, &(*total_deltas.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device deltas copy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //bias matrixs must be the same size as the matrix they are being added to b/c matrix addition rules
		  error_id = cudaMalloc((void**)&device_total_bias, total_activations.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"device bias allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //set bias vector to 1's
		  cudaMemset(device_total_bias, 1, total_activations.first * sizeof(double));
		  return cudaSuccess;
		}
		HOST int cuda_cleanup()
		{
		  cudaError_t error_id;
		  error_id  = cudaFree(device_total_observations);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device observations deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_targets);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device target deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_hidden_weights);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device hidden weight deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_activations);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device hidden activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_error);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device hidden error deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_gradient);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device hidden gradient deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_deltas);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device deltas deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_bias);
		  if(error_id != cudaSuccess)
		  {
			std::cerr<<"Device bias deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  cudaDeviceReset();
		  return cudaSuccess;
		}
		//later move this to an appropriate place
		HOST const char* cublasGetErrorString(cublasStatus_t status)
		{
	  	  switch(status)
	  	  {
			case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	 		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
			case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
			case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
			case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	  	  }
	  	  return "unknown error";
		}

#endif
		HOST int train(const std::uint16_t & max_epochs, const std::uint32_t & batch_size, const double & weight_penalty)
		{
		  int error = 0;
		  std::uint32_t ith_epoch, ith_observation;
#if CUDA_ENABLED == 1
		  printf("CUDA ENABLED TRAIN\n");
		  cublasStatus_t error_id;
		  cublasHandle_t handle;
		  error_id = cublasCreate(&handle);
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"CublasHandle creation failed with error:\t"<<cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
#else
		  //cpu multi-threaded code will go here
		  printf("CUDA DISABLED TRAIN\n");
#endif
		  std::cout<<"max_epochs: "<<max_epochs<<" total training cases: "<<std::get<0>(total_observations)<<" Training case size: "<<total_layers[0].second<<"\n";
		  for(ith_epoch = 0; ith_epoch < max_epochs/max_epochs; ++ith_epoch)
		  {
			for(ith_observation = 0; ith_observation < total_observations.first / total_observations.first; ++ith_observation)
			{
#if CUDA_ENABLED == 1 
			  error = static_cast<model_type*>(this)->forward_propagate(handle, ith_observation, total_layers, total_targets, total_hidden_weights, total_activations);
			  //do something with the error code
			  if(error == 1)
				std::cerr<<"An unknown error occured in forward_propagate\n";
			  //call back_propagate
#else
			  static_cast<model_type*>(this)->forward_propagate(total_layers, ith_observation, total_observations, total_targets, total_hidden_weights, total_activations);
#endif
			}
		  }

#if CUDA_ENABLED == 1
		  error_id = cublasDestroy(handle);
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"cublas handle destruction failed with error: "<<cublasGetErrorString(error_id)<<"\n";
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
		HOST int forward_propagate(cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, std::shared_ptr<double>> & total_targets, 
			                   const std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights,
							   const std::pair<std::uint32_t, std::shared_ptr<double>> & total_activations
							  )

		{
		  //cublas gemm here
		  cublasStatus_t error_id;
		  std::int32_t m, n, k, lda, ldb,ldc;//note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1
		  std::uint32_t ith_layer;//from the first hidden layer to the output layer 
		  std::uint32_t weight_offset;//number of weights connection between layer i and layer i + 1
		  std::uint32_t activation_offset;//number of activations(input) to a layer
		  std::uint32_t case_begin = total_layers[0].second * ith_observation_index;//where a case begins, when ith_obs_index is 0 this is the first case
		 

		  //do  first hidden layer and input layer
		  lda = total_layers[1].second;
		  ldb = total_layers[0].second;
		  ldc = lda;//obviously
		  std::cout<<"Matrix A rows(m): "<<total_layers[1].second<<" columns(k): "<<total_layers[0].second<<"\n";
		  std::cout<<"Matrix B rows(k): "<<total_layers[0].second<<" columns(n): "<<1<<"\n";
		  std::cout<<"Matrix C rows(m): "<<total_layers[1].second<<" columns(n): "<<1<<"\n";
		  const double alf = 1;
		  const double bet_mult = 0, bet_add = 1;
		  const double *alpha = &alf;
		  const double *beta1 = &bet_mult;
		  const double *beta2 = &bet_add;
		 
		 //perform Wx 
		  error_id = cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[1].second, 1, total_layers[0].second, 
			          alpha, device_total_hidden_weights, lda,
					  device_total_observations + case_begin, ldb, beta1,
					  device_total_activations,ldc
					  );
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"cublas dgemm on first hidden layer and input layer  failed with error: "<<cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  lda = total_layers[1].second;
		  ldb = lda;
		  ldc = lda;
		  //add in bias
		  error_id = cublasDgeam(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[1].second, 1,
			                     alpha, device_total_activations, lda,
								 beta2, device_total_bias, ldb,
								 device_total_activations, ldc
			                    );
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cerr<<"cublas dgeam on first hidden layer and input layer failed with error: "<< cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //f(Wx + b) complete
		  
		  //second hidden layer to output layer, see above for why weight offset = lda * ldb
		  for(ith_layer = 1, weight_offset = lda * ldb ; ith_layer < total_layers.size() - 1; ++ith_layer )
		  {
			//perform Wx
			/*lda = total_layers[ith_layer + 1].second;//Neurons in the current layer(Rows of A) m
			ldb = total_layers[ith_layer].second;//Neurons in the prior layer(Rows of B and by definitions of the matrix product Columns of A)
			ldc = lda;//output vector of the matrix product of the current layer times the output of the prior layer
			error_id = cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[ith_layer + 1].second, 1, total_layers[ith_layer].second,
								   alpha, device_total_hidden_weights + weight_offset, lda, 		
								   device_total_activations + total_layers[ith_layer].second, ldb, beta1,
								   device_total_activations + total_layers[ith_layer + 1].second,ldc 
								  );
		    if(error_id != CUBLAS_STATUS_SUCCESS)
		    {
			  std::cerr<<"cublas dgemm failed with error: "<<cublasGetErrorString(error_id)<<"\n";
		 	  return ERROR_CUDA_ERROR;
		    }
			weight_offset = total_layers[ith_layer + 1].second * total_layers[ith_layer].second;
			//add in bias
			error_id = cublasDgeam(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[ith_layer + 1].second, 1,
			                     alpha, device_total_activations, lda,
								 beta2, device_total_bias, ldb,
								 device_total_activations, ldc
			                    );
			if(error_id != CUBLAS_STATUS_SUCCESS)
			{
			  std::cerr<<"cublas dgeam on failed with error: "<< cublasGetErrorString(error_id)<<"\n";
			  return ERROR_CUDA_ERROR;
			}*/
		  } 
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
			                   const std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights
					          )
		{
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
	 HOST  std::vector<LAYER_INFO> get_total_layers(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_observations(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_hidden_weights(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_activations(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_error(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_gradient(const ann<T> & model);
	template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_deltas(const ann<T> & model);
	//End debugging functions
	template<class T>
  	  HOST void add_layer(ann<T> & model, const LAYER_INFO & ith_layer);
	template<class T>
	  HOST int initialize_model(ann<T> & model,  
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_targets
							);
	template<class T>
	  HOST int train(ann<T> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);
	template<class T>
	  HOST int cleanup(ann<T> & model);
}
#endif
