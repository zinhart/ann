#ifndef ANN_HH
#define ANN_HH
#include "loss_function.hh"
#include "layer.hh"
#include "optimizer.hh"
#include <memory>
//#include <zinhart/vector_space>
#include <vector>
#define MAXPOSNUM 2137483647
#if CUDA_ENABLED == 1
#define ERROR_CUDA_ERROR 1
#include <cublas_v2.h>
#endif
namespace zinhart
{
#if CUDA_ENABLED == 1
		__constant__ float * device_total_observations;
		__constant__ float * device_total_targets;
		__constant__ double * device_total_hidden_weights;
		__constant__ short * test;
#endif
  template <class model_type>
	class ann
	{
	  private:
		std::uint16_t case_size;// max dimensions 2^16
		std::vector<LAYER_INFO> total_layers; 
		std::pair<std::uint32_t, std::shared_ptr<float>> total_observations; // input layer size
		std::pair<std::uint32_t, std::shared_ptr<float>> total_targets; // output layer size
		std::pair<std::uint32_t, std::shared_ptr<double>> total_hidden_weights;

  public:
		ann() = default;
		ann(const ann<model_type> &) = default;
		ann(ann<model_type> &&) = default;
		ann<model_type> & operator = (const ann<model_type>&) = default;
		ann<model_type> & operator = (ann<model_type> &&) = default;
		~ann() = default;

		void add_layer(LAYER_INFO & ith_layer)
		{ total_layers.push_back(ith_layer); }

		int init(const std::uint16_t & case_size,
		         std::pair<std::uint32_t, std::shared_ptr<float>> & total_observations,
				 std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets,
				 std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights
				)
		{
		  this->case_size = case_size; // input layer size essentially
		  std::swap(this->total_observations,total_observations);
		  std::swap(this->total_targets, total_targets);
		  std::swap(this->total_hidden_weights, total_hidden_weights);
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
		  error_id = cudaMalloc( (void **) &device_total_observations, total_observations.first * sizeof(float));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device case allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy observations from host to device
		  error_id = cudaMemcpyToSymbol(device_total_observations, &(*total_observations.second.get()), sizeof(float*), 0, cudaMemcpyHostToDevice);
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
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*total_targets.second.get()), sizeof(float), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device target copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //allocate space for hidden weights
		  error_id = cudaMalloc((void **) &device_total_hidden_weights, total_hidden_weights.first * sizeof(float));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<"Device weight allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  //copy hidden weights from host to device
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*total_hidden_weights.second.get()), sizeof(float), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device weight copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
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
		int train(const std::uint32_t & epochs, const std::uint32_t & batch_size, const double & weight_penalty)
		{
		  std::uint32_t max_observations = total_observations.first * epochs;
		  std::uint32_t ith_observation, ith_layer;
#if CUDA_ENABLED == 1
		  printf("CUDA ENABLED TRAIN");
		  cublasStatus_t error_id;
		  cublasHandle_t handle;
		  error_id = cublasCreate(&handle);
		  //not sure if this is statement is necesarry
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cout<<"CublasHandle creation failed with error:\t"<<cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
#else
	  printf("CUDA DISABLED TRAIN");

#endif
		  for(ith_observation = 0; ith_observation < max_observations; ++ith_observation)
		  {
			//all relevant info is already copied to device 
			for(ith_layer = 1; ith_layer < total_layers.size(); ++ith_layer)
			{
			  //removing layer_type from layer so that a layer can be made with just enum,
			  //and hence determined @ compile time avoid an if/switch statement;
#if CUDA_ENABLED == 1 
			 // forward_propagate(handle, total_layers[ith_layer], total_layers[ith_layer - 1].first, total_hidden_weights, device_total_targets, case_size, ith_observation); 
#else
			  //forward_propagate(total_layers[ith_layer], total_layers[ith_layer - 1].first, total_hidden_weights, total_observations.second.get(), case_size, ith_observation); 
#endif
			}
		  }

#if CUDA_ENABLED == 1
		  error_id = cublasDestroy(handle);
		  //not sure if this is statement is necesarry
		  if(error_id != CUBLAS_STATUS_SUCCESS)
		  {
			std::cout<<"CublasHandle destruction failed with error:\t"<<cublasGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  
#endif
		  return 0;
		}
/*
#if CUDA_ENABLED == 1
		void forward_propagate(cublasHandle_t & context, LAYER_INFO & info, std::uint32_t & n_prev_layer_outputs, 
							  std::pair<std::uint32_t, std::shared_ptr<double>> & ith_layer_weights,
							  float * total_observations, std::uint32_t & case_size, 
							  std::uint32_t & ith_observation_index)
		{ static_cast<model_type*>(this)->forward(context, info, n_prev_layer_outputs, ith_layer_weights, total_observations, case_size, ith_observation_index); };
#else
		void forward_propagate(LAYER_INFO & info, std::uint32_t & n_prev_layer_outputs, 
							  std::pair<std::uint32_t, std::shared_ptr<double>> & ith_layer_weights,
							  float * total_observations, std::uint32_t & case_size, 
							  std::uint32_t & ith_observation_index)
		{ static_cast<model_type*>(this)->forward(info, n_prev_layer_outputs, ith_layer_weights, total_observations, case_size, ith_observation_index); };
#endif

		void backward_propagate(std::uint32_t n_cases, std::uint32_t n_inputs, int istart, int istop, int ntarg)// output layer eventually remove ilayer from call list
		{ static_cast<model_type*>(this)->backward(num_cases, n_inputs, istart, istop, ntarg); };
  	 */
	  private:
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
/*
#if CUDA_ENABLED == 1
		void forward(cublasHandle_t & context, LAYER_INFO & info, std::uint32_t & n_prev_layer_outputs, std::pair<std::uint32_t, std::shared_ptr<double>> & ith_layer_weights,
					float * total_observations, std::uint32_t & case_size, std::uint32_t & ith_observation_index	)

		{
		  printf("CUDA_ENABLED - ffn::forward ");
		  float * ith_observation = total_observations + (case_size * ith_observation_index);
		 //pass in cublas_handle, perform weight * input , add bias and perform activation function in kernel
		  Layer ith_hidden_layer;
		  std::cout<<"Here 1:\n";
		 // cuda_hidden_activation (int istart, int istop, int nhid, int ilayer );
		}

#else
		void forward(LAYER_INFO & info, std::uint32_t & n_prev_layer_outputs, std::pair<std::uint32_t, std::shared_ptr<double>> & ith_layer_weights,
					float * total_observations, std::uint32_t & case_size, std::uint32_t & ith_observation_index	)

		{
		  printf("CUDA_DISABLED - ffn::forward ");
		  float * ith_observation = total_observations + (case_size * ith_observation_index);
		 //pass in cublas_handle, perform weight * input , add bias and perform activation function in kernel
		  Layer ith_hidden_layer;
		 // cuda_hidden_activation (int istart, int istop, int nhid, int ilayer );
		}
#endif
		void backward(std::uint32_t n_cases, std::uint32_t n_inputs, int istart, int istop, int ntarg)
		{
		 // cuda_output_delta (int istart, int istop, int ntarg );
		 // cuda_output_gradient (n_cases, n_inputs, ntarg);
		}

		void backward(std::uint32_t n_cases, std::uint32_t ilayer, int nhid_this, int nhid_prior, int last_hidden)
		{
		  cuda_subsequent_hidden_gradient(n_cases, ilayer, nhid_this, nhid_prior, int last_hidden);
		}

		void backward(int istart, int istop, int nin, int nhid, int only_hidden)
		{
		  cuda_first_hidden_gradient(istart, istop, nin, nhid, only_hidden);
		}*/
	};

	//Begin Cuda Wrappers
	template<class T>
	  int initialize_network(ann<T> & model,  
						     const std::uint16_t & case_size,
							 std::pair<std::uint32_t, std::shared_ptr<float>> & total_observations,
							 std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets,
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights
							);
	template<class T>
	  int train(ann<T> & model, const std::uint32_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);
	template<class T>
	  int cleanup(ann<T> & model);
}
#endif
