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
#endif
namespace zinhart
{
#if CUDA_ENABLED == 1
		__constant__ double * device_total_observations;
		__constant__ float * device_total_targets;
		__constant__ double * device_total_hidden_weights;
		__constant__ double * device_total_activations;
		//__constant__ short * test;
#endif
  template <class model_type>
	class ann
	{
	  protected:
		std::uint16_t case_size;// input layer size max dimensions 2^16
		std::vector<LAYER_INFO> total_layers;//Layer types(Relu sigmoid etc) and the number of inputs of the respective layer 
		std::pair<std::uint32_t, std::shared_ptr<double>> total_observations;// number of training cases and the training cases themselves
		std::pair<std::uint32_t, std::shared_ptr<float>> total_targets; // output layer size and the complete set of targets for each input
		std::pair<std::uint32_t, std::shared_ptr<double>> total_hidden_weights;// the number of hidden weights for a layer and the weights themselves
		std::pair<std::uint32_t, std::shared_ptr<double>> total_activations;//this is the sum of all the hidden layers and the output layer neurons
	  public:
		ann() = default;
		ann(const ann<model_type> &) = default;
		ann(ann<model_type> &&) = default;
		ann<model_type> & operator = (const ann<model_type>&) = default;
		ann<model_type> & operator = (ann<model_type> &&) = default;
		~ann() = default;

		void add_layer(const LAYER_INFO & ith_layer)
		{ total_layers.push_back(ith_layer); }
		int init(const std::uint16_t & case_size,
		         std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
				 std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets
				)
		{
		  std::uint32_t ith_layer, prior_layer_neurons;
		  this->case_size = case_size; // input layer size essentially
		  std::swap(this->total_observations,total_observations);
		  std::swap(this->total_targets, total_targets);

		  this->total_activations.first = 0;

		  prior_layer_neurons = case_size;
		  //calc number of hidden weights
		  for(ith_layer = 0; ith_layer < total_layers.size(); ++ith_layer)
		  {
			this->total_hidden_weights.first += this->total_layers[ith_layer].second * (prior_layer_neurons + 1);//+ 1 for bias input
			prior_layer_neurons = this->total_layers[ith_layer].second;
		  }
 		  this->total_hidden_weights.second = std::shared_ptr<double> ( new double[this->total_hidden_weights.first], std::default_delete<double[]>() );//allocate weights
		  //calc number of activations
		  for(ith_layer = 0, this->total_activations.first = 0; ith_layer < total_layers.size(); ++ith_layer )
			this->total_activations.first += total_layers[ith_layer].second;//accumulate neurons in the hidden layers
		  this->total_activations.first += this->total_targets.first;//add in output layer 
		  this->total_activations.second  = std::shared_ptr<double>(new double[this->total_activations.first], std::default_delete<double[]>());//allocate activations
#if CUDA_ENABLED == 1
		  return cuda_init();
#else
		  return 0;
#endif
		}
		
#if CUDA_ENABLED == 1
		int cuda_init()
		{
		  cudaError_t error_id;
		  error_id = cudaSetDevice(0);

		  //it seems that when their are no memory leaks cuda-memcpy reports 0 allocations along with 0 leaks, so don't trip
		  /*cudaMalloc((void**)&test, total_observations.first * sizeof(float));
		  cudaMemcpy(test, total_observations.second.get(), total_observations.first * sizeof(float),cudaMemcpyHostToDevice);
		  cudaFree(test);*/
		  
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Cuda init setDevice failed:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for observations
		  error_id = cudaMalloc( (void **) &device_total_observations, total_observations.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device case allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy observations from host to device
		  error_id = cudaMemcpyToSymbol(device_total_observations, &(*total_observations.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device case copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  //allocate space for targets
		  error_id = cudaMalloc((void **) &device_total_targets, total_targets.first * sizeof(float));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device target allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  //copy targets from host to device
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*total_targets.second.get()), sizeof(float*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device target copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for hidden weights
		  error_id = cudaMalloc((void **) &device_total_hidden_weights, total_hidden_weights.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device weight allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy hidden weights from host to device
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*total_hidden_weights.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device weight copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  
		  //allocate space for activations
		  error_id = cudaMalloc((void **) &device_total_activations, total_activations.first * sizeof(double));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device activation allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy activations from host to device
		  error_id = cudaMemcpyToSymbol(device_total_activations, &(*total_activations.second.get()), sizeof(double*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device activation copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  return cudaSuccess;
		}
		int cuda_cleanup()
		{
		  cudaError_t error_id;
		  error_id  = cudaFree(device_total_observations);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device observations deallocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_targets);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device target deallocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_hidden_weights);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device hidden weight deallocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaFree(device_total_activations);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device hidden activation deallocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  cudaDeviceReset();
		  return cudaSuccess;
		}
		//later move this to an appropriate place
		const char* cublasGetErrorString(cublasStatus_t status)
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
		int train(const std::uint16_t & max_epochs, const std::uint32_t & batch_size, const double & weight_penalty)
		{
		  std::uint32_t ith_epoch, ith_observation;
#if CUDA_ENABLED == 1
		  printf("CUDA ENABLED TRAIN\n");
		  cublasStatus_t error_id;
		  cublasHandle_t handle;
		  error_id = cublasCreate(&handle);
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cout<<"CublasHandle creation failed with error:\t"<<cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
#else
		  //cpu multi-threaded code will go here
		  printf("CUDA DISABLED TRAIN\n");
#endif
		  for(ith_epoch = 0; ith_epoch < max_epochs; ++ith_epoch)
		  {
			for(ith_observation = 0; ith_observation < total_observations.first; ++ith_observation)
			{
#if CUDA_ENABLED == 1 
			  static_cast<model_type*>(this)->forward_propagate(handle, case_size, ith_observation, total_layers, total_targets, total_hidden_weights, total_activations);
#else
			  static_cast<model_type*>(this)->forward_propagate(case_size, total_layers, ith_observation, total_observations, total_targets, total_hidden_weights, total_activations);
#endif
			}
		  }

#if CUDA_ENABLED == 1
		  error_id = cublasDestroy(handle);
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cout<<"CublasHandle destruction failed with error:\t"<<cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
#else
		  //cpu multi-threaded code will go here
		  
#endif
		  return 0;
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
		void forward_propagate(cublasHandle_t & context, 
			                   const std::uint16_t & case_size, const std::uint32_t & ith_observation_index, 
							   const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets, 
			                   const std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights,
							   const std::pair<std::uint32_t, std::shared_ptr<double>> & total_activations
							  )

		{
		  //cublas gemm here
		  std::int32_t lda, ldb,ldc;//note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1
		  std::uint32_t ith_layer;//from the first hidden layer to the output layer 
		  std::uint32_t case_begin = case_size * ith_observation_index;//where a case begins, when ith_obs_index is 0 this is the first case
		  const double alf = 1;
		  const double bet = 0;
		  const double *alpha = &alf;
		  const double *beta = &bet;

		  //do  input layer
		  lda = total_layers[0].second;//neurons in the first hidden layer
		  ldb = case_size;//input layer is has case_size many neurons which is also the number of columns of the input layer matrix
		  ldc = lda;//obviously
		  cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, total_layers[0].second, case_size, case_size, 
			          alpha, device_total_hidden_weights, lda,
					  device_total_observations + case_begin, ldb, beta,
					  device_total_activations,ldc
					  );

		  //to do hidden to output layers
		  /*for(ith_layer = 0; ith_layer < total_layers.size(); ++ith_layer )
		  {
			lda = total_layers[ith_layer].second;
			ldb = 
		  }*/
		  
		  
		}
		void backward_propagate(cublasHandle_t & context)
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
	template<class T>
  	  void add_layer(ann<T> & model, const LAYER_INFO & ith_layer);
	template<class T>
	  int initialize_network(ann<T> & model,  
						     const std::uint16_t & case_size,
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
							 std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets
							);
	template<class T>
	  int train(ann<T> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);
	template<class T>
	  int cleanup(ann<T> & model);
}
#endif
