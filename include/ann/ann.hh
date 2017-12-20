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
		__device__ float * device_total_observations;
		__device__ float * device_total_targets;
		__device__ double * device_total_hidden_weights;
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

		int set_case_info(const std::uint32_t & n_observations, 
						   const std::uint32_t & n_targets, 
						   const std::uint32_t & n_hidden_weights, 
						   const std::uint16_t & case_size)
		{
		  this->case_size = case_size; // input layer size essentially

		  this->total_observations.first = n_observations; //number of observations
		  this->total_observations.second = std::shared_ptr<float> ( new float[n_observations], std::default_delete<float[]>() );//observations themselves 

		  this->total_targets.first = n_targets; // number of targets
		  this->total_targets.second = std::shared_ptr<float> ( new float[n_targets], std::default_delete<float[]>() );//targets themselves 

		  this->total_hidden_weights.first = n_hidden_weights; // number of weights
		  this->total_hidden_weights.second = std::shared_ptr<double> ( new double[n_hidden_weights], std::default_delete<double[]>() );// weights themselves 
#if CUDA_ENABLED == 1
		  return cuda_init(total_observations, total_targets, total_hidden_weights, case_size);
#else
		  return 0;
#endif
		}

#if CUDA_ENABLED == 1
		int cuda_init(std::pair<std::uint32_t, std::shared_ptr<float>> & tot_cases, 
					  std::pair<std::uint32_t, std::shared_ptr<float>> & tot_targs,
		              std::pair<std::uint32_t, std::shared_ptr<double>> & tot_hidden_weights,
					  const std::uint16_t & case_sz	)
		{
		  cudaError_t error_id;
		  error_id = cudaMalloc( (void **) &device_total_observations, tot_cases.first * sizeof(float));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device case allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }

		  error_id = cudaMemcpyToSymbol(device_total_observations, &(*tot_cases.second.get()), sizeof(float*), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device case copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaMalloc((void **) &device_total_targets, tot_targs.first * sizeof(float));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device target allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*tot_targs.second.get()), sizeof(float), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device target copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaMalloc((void **) &device_total_hidden_weights, tot_hidden_weights.first * sizeof(float));
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device weight allocation failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }
		  error_id = cudaMemcpyToSymbol(device_total_targets, &(*tot_targs.second.get()), sizeof(float), 0, cudaMemcpyHostToDevice);
		  if(error_id != cudaSuccess)
		  {
			std::cout<<" Device weight copy failed with error:\t"<<cudaGetErrorString(error_id)<<"\n";
			return ERROR_CUDA_ERROR;
		  }/**/
		  return cudaSuccess;
		}
#endif
/*		int train(std::uint32_t & epochs, std::uint32_t & batch_size, double weight_penalty = 1.0 )
		{
		  std::uint32_t max_observations = total_observations.first * epochs;
		  std::uint32_t ith_observation, ith_layer;
#if CUDA_ENABLED == 1
		  printf("CUDA ENABLED TRAIN");
		  cublasHandle_t handle;
		  cublasCreate(&handle);
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
		  cublasDestroy(handle);
#endif
		  return 0;
		}

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
	  int set_case_info(ann<T> & model, std::uint32_t & n_observations, 
						   std::uint32_t & n_targets,  
						   std::uint32_t & n_hidden_weights, 
						   std::uint16_t & case_size);
}
#endif
