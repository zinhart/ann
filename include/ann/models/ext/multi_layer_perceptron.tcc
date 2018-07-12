#include "mkl.h"
#include "concurrent_routines/concurrent_routines.hh"
namespace zinhart
{
  namespace models
  {
#if CUDA_ENABLED == 1
	template <class precision_type>
	  template <class LOSS_FUNCTION>
	  HOST std::int32_t multi_layer_perceptron<precision_type>::forward_propagate_async(const bool & copy_device_to_host, 
								 const cudaStream_t & stream, const cublasHandle_t & context, LOSS_FUNCTION error_metric,
								 const std::uint32_t & ith_observation_index, const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
								 const std::uint32_t & total_targets, const double * host_total_targets, const double * device_total_targets,
								 const std::uint32_t & total_hidden_weights, const double * host_total_hidden_weights, const double * device_total_hidden_weights, 
								 const std::uint32_t & total_activations, double * host_total_activations, double * device_total_activations,
								 const double * device_total_observations, double * device_total_outputs,
								 const double * host_total_bias, std::uint32_t device_id = 0
								)

		  {
			// layer counters
			std::uint32_t current_layer{1}, prior_layer{0}, ith_layer{0};// layer counters start at 1 and 0 respectively because we start with the hidden layer and input layer
			const std::uint32_t input_layer{0};
			const double * d_weight{device_total_hidden_weights};
			const double * d_obs{device_total_observations};
			double * d_act{device_total_activations};
			// declarations for gemm
			std::int32_t  m{0}, n{0}, k{0}, lda{0}, ldb{0},ldc{0};// note that for a weight matrix with dimensions m, n: m = neurons in layer i & n = neurons in layer i - 1
			std::uint32_t current_activation_offset{0};
			std::uint32_t prior_activation_offset{0};
			std::uint32_t weight_offset = {0};// number of weights connection between layer i and layer i + 1
			std::uint32_t case_begin = {total_layers[input_layer].second * ith_observation_index};// where a case begins, when ith_obs_index is 0 this is the first case

			// coefficients for gemm
			const double alpha{1};
			const double beta_mult{0};
			const double beta_add{1};


			// get col major coordinates without explicitly transposing 
			// ( total_layers[1].second is rows of the weight matrix and the number of neurons in the first hidden layer )
			// ( total_layers[0].second is the columns of the weight matrix and the number of neurons in the input layer )
			zinhart::serial::gemm_wrapper(m, n, k, lda, ldb, ldc, total_layers[current_layer].second, total_layers[input_layer].second, total_layers[input_layer].second, 1);

			// set cublasStream
			if(zinhart::check_cublas_api(cublasSetStream(context, stream), __FILE__, __LINE__) != 0 )
			  return 1;

			// do Wx for first hidden layer and input layer
			if(zinhart::check_cublas_api(cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, 
						m, n, k, 
						&alpha, 
						(d_obs + case_begin), lda,
						d_weight, ldb, 
						&beta_mult,
						d_act, ldc
						),__FILE__, __LINE__) != 0)
			{
			  return 1;
			}

			// add in bias
			if(call_axps_async(1.0, d_act, host_total_bias[0], total_layers[current_layer].second, stream) != 0)
			  return 1;		

			// call activation
			call_activation(total_layers[current_layer].first, ACTIVATION_TYPE::OBJECTIVE, d_act, total_layers[current_layer].second);
			// f(Wx + b) complete for first hidden layer and input layer
			
			// update layer counters
			current_layer = 2;
			prior_layer = 1;

			//second hidden layer to output layer, see above for why weight offset = lda * ldb
			for(ith_layer = 1, prior_activation_offset = 0; ith_layer < total_layers.size() - 1; ++ith_layer, ++current_layer, ++prior_layer, prior_activation_offset += total_layers[prior_layer].second)
			{
			  // set offsets
			  current_activation_offset += total_layers[prior_layer].second;
			  weight_offset += total_layers[prior_layer].second * total_layers[prior_layer - 1].second;
  /*
			  double *test_weights = new double[total_layers[current_layer].second * total_layers[prior_layer].second];
			  double *test_activations_prior= new double[total_layers[prior_layer].second];
			  double *test_activations= new double[total_layers[current_layer].second];*/
  //		    zinhart::check_cuda_api( cudaMemcpyAsync(test_weights, (d_weight + weight_offset), total_layers[current_layer].second * total_layers[prior_layer].second * sizeof(double), cudaMemcpyDeviceToHost, stream), __FILE__, __LINE__); 
  //		    zinhart::check_cuda_api( cudaMemcpyAsync(test_activations_prior, d_act/*+ prior_activation_offset*/, total_layers[prior_layer].second *  sizeof(double), cudaMemcpyDeviceToHost, stream), __FILE__, __LINE__); 
  /*		    zinhart::check_cuda_api( cudaMemcpyAsync(test_activations, (d_act + current_activation_offset), total_layers[current_layer].second *  sizeof(double), cudaMemcpyDeviceToHost, stream), __FILE__, __LINE__); 
			  cudaStreamSynchronize(stream);*/



			  // get col major coordinates without explicitly transposing 
			  // ( total_layers[current_layer].second is rows of the weight matrix)
			  // ( total_layers[prior_layers].second is the columns of the weight matrix)
			  zinhart::serial::gemm_wrapper(m, n, k, lda, ldb, ldc, total_layers[current_layer].second, total_layers[prior_layer].second, total_layers[prior_layer].second, 1);
  /*			std::cout<<"F ith_layer: "<<ith_layer<<"\n";
			  std::cout<<"F current_activation_offset: "<<current_activation_offset<<"\n";
			  std::cout<<"F weight_offset: "<<weight_offset<<"\n";
			  std::cout<<"F prior_activation_offset: "<<prior_activation_offset<<"\n";
			  zinhart::print_matrix_row_major(test_weights, total_layers[current_layer].second, total_layers[prior_layer].second, "F weight matrix");
			  zinhart::print_matrix_row_major(test_activations, total_layers[prior_layer].second, 1, "F prior activation matrix");
			  zinhart::print_matrix_row_major(test_activations, total_layers[current_layer].second, 1, "F activation matrix");*/
			  if(zinhart::check_cublas_api(cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N, 
						  m, n, k, 
						  &alpha, 
						  (d_act + prior_activation_offset), lda,
						  (d_weight + weight_offset), ldb, 
						  &beta_mult,
						  (d_act + current_activation_offset), ldc
						  ),__FILE__, __LINE__) != 0)
			  {
				return 1;
			  }



			  // add in bias
			  if(call_axps_async(1.0, (d_act + current_activation_offset), host_total_bias[ith_layer], total_layers[current_layer].second, stream) != 0)
				return 1;		

			  // call activation
			  if(call_activation(total_layers[current_layer].first, ACTIVATION_TYPE::OBJECTIVE, (d_act + current_activation_offset), total_layers[current_layer].second) != 0)
				return 1;

		  }
		  // f(Wx + b) complete for second hidden layer to output layer
		  
		  // calc error
		  
		  // call loss function kernel and write to outputs
			  


  /*		  //copy activations back to host
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
	template <class precision_type>
	  //template <class LOSS_FUNCTION>
		  HOST void multi_layer_perceptron<precision_type>::backward_propagate(cublasHandle_t & context)
		  {
			//cublas gemm here
			
			// CALCULATION OF OUTPUT LAYER GRADIENTS
			
			// axpy for calculation of error
			
			// axpy for calculation of hadamard product
			
			// dgemm for calculation of hadamard product with error
			
			// axpy for deltas
			
			// dgemm for delta times previos layer activations 
		  }

	//cpu multi-threaded code will go here
#else
	template <class precision_type>
	  void multi_layer_perceptron<precision_type>::forward_propagate(const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
																	 const precision_type * total_training_cases, const std::uint32_t & case_index,
																	 precision_type * total_activations, const std::uint32_t & total_activations_length,
																	 const precision_type * total_hidden_weights, const std::uint32_t & total_hidden_weights_length,
																	 const precision_type * total_bias,
								                                     precision_type * outputs,
																	 const std::uint32_t & thread_id
								                                    )
		  {
			std::cout<<"IN CPU\n";
			//mkl gemm etc
			const std::uint32_t input_layer{0};
			const std::uint32_t output_layer{total_layers.size() - 1};
			std::uint32_t i, activation_offset, ith_layer;
			precision_type * current_inputs{total_training_cases + case_index}, activation_ptr;
			precision_type alpha{1.0}, beta{0.0};
			std::uint32_t m{ total_layers[input_layer + 1].second }, n{1}, k{ total_layers[input_layer].second };
			zinhart::activation::activation_function af;

			// set activation_offset in the case that their are multiple threads, for the first hidden layer this is the thread_id * neurons in the first hidden_layer
			activation_offset = thread_id * total_layers[input_layer + 1].second;
			activation_ptr = total_activations + activation_offset;

		    // do input layer and the first input layer, aka Wx
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				        m, n, k,
						alpha, total_hidden_weights, k,
						current_inputs, n, beta, 
						activation_ptr, n
				       );

			// add in bias, consider using neaumaer sum
			for(i = activation_offset; i < total_layers[1].second; ++i)
			  activation_ptr[i] += total_bias[0];
			
			
			// apply activation functions
			for(i = activation_offset; i < total_layers[1].second; ++i)
			  //first_hidden_layer_activation(activation_ptr[i]);
			  af(total_layers[1].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, activation_ptr[i]);

			// f(Wx + b complete) 
			
			// repeat till output layer 
			
		  }
	template <class precision_type>
	  template <class LOSS_FUNCTION>
	  void multi_layer_perceptron<precision_type>::backward_propagate(const std::vector<zinhart::activation::LAYER_INFO> & total_layers, LOSS_FUNCTION error_metric, 
																	  const std::uint32_t & ith_observation_index)
	  {
		//mkl gemm etc
	  }

#endif

  }
}
