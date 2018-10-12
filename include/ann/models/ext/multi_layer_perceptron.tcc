#include "mkl.h"
#include "concurrent_routines/concurrent_routines.hh"
#include "cassert"
namespace zinhart
{
  namespace models
  {
	template <class precision_type>
  	  HOST void multi_layer_perceptron<connection::dense, precision_type>::gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
																						  const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
																						  const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
																						  precision_type * total_hidden_inputs, precision_type * total_activations, const std::uint32_t total_activations_length,
																						  precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																						  const precision_type * total_bias, 
																						  precision_type * numerically_approx_gradient, 
																						  const precision_type limit_epsilon, 
																						  const std::uint32_t n_threads, const std::uint32_t thread_id
								                                                         )
	  { 
		std::uint32_t i{0};
		const std::uint32_t input_layer{0};
		const std::uint32_t output_layer{total_layers.size() - 1};

		std::uint32_t output_layer_index{0};
		// start at 1 to skip the input layer
		for(i = 1; i < total_layers.size() - 1; ++i)
		  output_layer_index += total_layers[i].second;

		// a ptr to the current training case, 
		// the number of nodes in the input layer is the same length as the length of the current training case so we move case_index times forward in the total_training_cases ptr, 
		// -> case_index = 0 is the first training case, case_index = 1 the second case, case_index = n the nth case.
		const precision_type * current_training_case{total_training_cases + (case_index * total_layers[input_layer].second)};
		const precision_type * current_target{total_targets + (case_index * total_layers[output_layer].second)};

		// variables for the thread calling this method, to determine it's workspace
		std::uint32_t current_threads_activation_workspace_index{0}, current_threads_gradient_workspace_index{0}, thread_activation_stride{0}, thread_gradient_stride{0}, thread_output_layer_stride{0};
		precision_type * current_threads_gradient_ptr{nullptr};
		precision_type * current_threads_output_layer_ptr{nullptr};

		// Assumes the numerically_approx_gradient vector is partitioned into equally size chucks, 1 for each thread
		thread_gradient_stride = total_hidden_weights_length;
		thread_activation_stride = total_activations_length / n_threads; 
		thread_output_layer_stride = (thread_id * thread_activation_stride) + output_layer_index;

		// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the relevant vector
		current_threads_gradient_workspace_index = thread_id * thread_gradient_stride;
		current_threads_output_layer_ptr = total_activations + thread_output_layer_stride;

		// and finally the beginning of gradient for the current thread
		current_threads_gradient_ptr = numerically_approx_gradient + current_threads_gradient_workspace_index;

		// As in the left and right side derivatives of the error function
		precision_type right{0};
		precision_type left{0};

		// to set the parameter back to its original value
		precision_type original{0};
	//	precision_type outputs[total_layers[output_layer].second];
		// gradient_check loop
		for(i = 0; i < total_hidden_weights_length; ++i)
		{
		  
		  // save original
		  original = total_hidden_weights[i];
		  
		  // right side
		  total_hidden_weights[i] += limit_epsilon;
		  forward_propagate(total_layers, 
							total_training_cases, case_index, 
							total_hidden_inputs, total_activations, total_activations_length,
							total_hidden_weights, total_hidden_weights_length, 
							total_bias,
							n_threads, thread_id
						  );

	      right = loss->error(zinhart::function_space::objective(), current_threads_output_layer_ptr, current_target, total_layers[output_layer].second);

		  // set back
		  total_hidden_weights[i] = original; 

		  // left side
		  total_hidden_weights[i] -= limit_epsilon;
		  forward_propagate(total_layers, 
							total_training_cases, case_index, 
							total_hidden_inputs, total_activations, total_activations_length,
							total_hidden_weights, total_hidden_weights_length, 
							total_bias,
							n_threads, thread_id
						  );

	      left = loss->error(zinhart::function_space::objective(), current_threads_output_layer_ptr, current_target, total_layers[output_layer].second);

		  // calc numerically derivative for the ith_weight, save it, increment the pointer to the next weight
		  *(current_threads_gradient_ptr + i) = (right - left) / (precision_type{2.0} * limit_epsilon);

		  // set back
		  total_hidden_weights[i] = original;
		}
	  }


#if CUDA_ENABLED == 1
	template <class precision_type>
	  template <class LOSS_FUNCTION>
	  HOST std::int32_t multi_layer_perceptron<connection::dense, precision_type>::forward_propagate_async(const bool & copy_device_to_host, 
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
			call_activation(total_layers[current_layer].first, ACTIVATION_TYPE::objective, d_act, total_layers[current_layer].second);
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
			  if(call_activation(total_layers[current_layer].first, ACTIVATION_TYPE::objective, (d_act + current_activation_offset), total_layers[current_layer].second) != 0)
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
		  HOST void multi_layer_perceptron<connection::dense, precision_type>::backward_propagate(cublasHandle_t & context)
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
	  void multi_layer_perceptron<connection::dense, precision_type>::forward_propagate(const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
																						const precision_type * total_training_cases, const std::uint32_t case_index,
																						precision_type * total_hidden_inputs, precision_type * total_activations, const std::uint32_t total_activations_length,
																						const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																						const precision_type * total_bias,
																						const std::uint32_t n_threads,
																						const std::uint32_t thread_id
																					   )
		  {
			std::uint32_t i{0}, j{0};

			const std::uint32_t input_layer{0};
			const std::uint32_t output_layer{total_layers.size() - 1};

			// All layer counters
			std::uint32_t current_layer{1}, previous_layer{input_layer}, current_layer_index{0}, previous_layer_index{0};
			
		    // The index of the beginning of the weight matrix between two layers 
			std::uint32_t weight_index{0};

			// a ptr to the current training case, 
			// the number of nodes in the input layer is the same length as the length of the current training case so we move case_index times forward in the total_training_cases ptr, 
			// -> case_index = 0 is the first training case, case_index = 1 the second case, case_index = n the nth case.
			const precision_type * current_training_case{total_training_cases + (case_index * total_layers[input_layer].second)};
			
			// variables for the thread calling this method, to determine it's workspace
		    std::uint32_t current_threads_workspace_index{0}, thread_stride{0};
			precision_type * current_threads_hidden_input_ptr{nullptr};
		    precision_type * current_threads_output_ptr{nullptr};

			// variables for gemm
			precision_type alpha{1.0}, beta{0.0};
			std::uint32_t m{ total_layers[current_layer].second }, n{1}, k{ total_layers[previous_layer].second };

			// the activation function of each layer
			zinhart::activation::activation_function af;

			// Assumes the activation vector is partitioned into equally size chucks, 1 for each thread
			thread_stride = total_activations_length / n_threads;

			// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the relevant vector
			current_threads_workspace_index = thread_id * thread_stride;
			current_threads_hidden_input_ptr = total_hidden_inputs + current_threads_workspace_index;
			current_threads_output_ptr = total_activations + current_threads_workspace_index;

		    // do input layer and the first hidden layer -> Wx
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				        m, n, k,
						alpha, total_hidden_weights, k,
						current_training_case, n, beta, 
						current_threads_hidden_input_ptr, n
				       );

			// add in bias, calc output of this layer
			for(i = current_threads_workspace_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
			{
			  total_hidden_inputs[i] += total_bias[previous_layer];
  			  // apply activation functions
			  total_activations[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_inputs[i]);
			}
			// f(Wx + b complete) for first hidden layer and input layer
			

			// update weight matrix index	
			weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

			// update layer indices
			previous_layer_index = current_layer_index;
			current_layer_index = total_layers[current_layer].second;

			//increment layer counters
			++current_layer;
			++previous_layer;

			while( current_layer < total_layers.size() )
			{
			  const precision_type * current_weight_matrix{total_hidden_weights + weight_index};
			  precision_type * current_layer_inputs_ptr{total_hidden_inputs + current_threads_workspace_index + current_layer_index};
			  precision_type * current_layer_outputs_ptr{total_activations + current_threads_workspace_index + current_layer_index};
			  const precision_type * prior_layer_ptr{total_activations + current_threads_workspace_index + previous_layer_index}; 
			  m = total_layers[current_layer].second;
			  n = 1;
			  k = total_layers[previous_layer].second;
			  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						  m, n, k,
						  alpha, current_weight_matrix, k,
						  prior_layer_ptr, n, beta, 
						  current_layer_inputs_ptr, n
						 );

			  // add in bias, calc output of this layer
			  for(i = current_threads_workspace_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
			  {
				total_hidden_inputs[i] += total_bias[previous_layer];
				// apply activation functions
				total_activations[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_inputs[i]);
			  }

			  // update weight matrix index	
			  weight_index += total_layers[current_layer].second * total_layers[previous_layer].second;

			  // update layer indices
			  previous_layer_index = current_layer_index;
			  current_layer_index += total_layers[current_layer].second;
			  
			  // increment layer counters 
			  ++current_layer; 
			  ++previous_layer;
			 }	  

			// calculate the error for this case
		  }
	// new frop to replace one above
	template <class precision_type>
	  void multi_layer_perceptron<connection::dense, precision_type>::forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<double> > > & total_layers,
																						const precision_type * total_training_cases, const std::uint32_t  case_index,
																						precision_type * total_hidden_inputs, precision_type * total_activations, const std::uint32_t total_activations_length,
																						const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																						const precision_type * total_bias,
																						const std::uint32_t n_threads,
																						const std::uint32_t thread_id
																					  )
	  {
		std::uint32_t i{0}, j{0};

		const std::uint32_t input_layer{0};
		const std::uint32_t output_layer{total_layers.size() - 1};

		// All layer counters
		std::uint32_t current_layer{1}, previous_layer{input_layer}, current_layer_index{0}, previous_layer_index{0};
		
		// The index of the beginning of the weight matrix between two layers 
		std::uint32_t weight_index{0};

		// a ptr to the current training case, 
		// the number of nodes in the input layer is the same length as the length of the current training case so we move case_index times forward in the total_training_cases ptr, 
		// -> case_index = 0 is the first training case, case_index = 1 the second case, case_index = n the nth case.
		const precision_type * current_training_case{total_training_cases + (case_index * total_layers[input_layer]->get_size())};
		
		// variables for the thread calling this method, to determine it's workspace
		std::uint32_t current_threads_workspace_index{0}, thread_stride{0};
		precision_type * current_threads_hidden_input_ptr{nullptr};
		precision_type * current_threads_output_ptr{nullptr};

		// variables for gemm
		precision_type alpha{1.0}, beta{0.0};
		std::uint32_t m{ total_layers[current_layer]->get_size() }, n{1}, k{ total_layers[previous_layer]->get_size() };

		// the activation function of each layer
		zinhart::activation::activation_function af;

		// Assumes the activation vector is partitioned into equally size chucks, 1 for each thread
		thread_stride = total_activations_length / n_threads;

		// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the relevant vector
		current_threads_workspace_index = thread_id * thread_stride;
		current_threads_hidden_input_ptr = total_hidden_inputs + current_threads_workspace_index;
		current_threads_output_ptr = total_activations + current_threads_workspace_index;

		// do input layer and the first hidden layer -> Wx
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, total_hidden_weights, k,
					current_training_case, n, beta, 
					current_threads_hidden_input_ptr, n
				   );

		// add in bias, calc output of this layer
		for(i = current_threads_workspace_index, j = 0; j < total_layers[current_layer]->get_size(); ++i, ++j)
		{
//		  total_hidden_inputs[i] += total_bias[previous_layer];
		  // apply activation functions
//		  total_activations[i] = af(total_layers[current_layer]->get_size(), zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_inputs[i]);
		}
		// f(Wx + b complete) for first hidden layer and input layer
		

		// update weight matrix index	
		weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

		// update layer indices
		previous_layer_index = current_layer_index;
		current_layer_index = total_layers[current_layer]->get_size();

		//increment layer counters
		++current_layer;
		++previous_layer;

		while( current_layer < total_layers.size() )
		{
		  const precision_type * current_weight_matrix{total_hidden_weights + weight_index};
		  precision_type * current_layer_inputs_ptr{total_hidden_inputs + current_threads_workspace_index + current_layer_index};
		  precision_type * current_layer_outputs_ptr{total_activations + current_threads_workspace_index + current_layer_index};
		  const precision_type * prior_layer_ptr{total_activations + current_threads_workspace_index + previous_layer_index}; 

		  m = total_layers[current_layer]->get_size();
		  n = 1;
		  k = total_layers[previous_layer]->get_size();

		  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					  m, n, k,
					  alpha, current_weight_matrix, k,
					  prior_layer_ptr, n, beta, 
					  current_layer_inputs_ptr, n
					 );

		  // add in bias, calc output of this layer
		  for(i = current_threads_workspace_index + current_layer_index, j = 0; j < total_layers[current_layer]->get_size(); ++i, ++j)
		  {
//			total_hidden_inputs[i] += total_bias[previous_layer];
			// apply activation functions
//			total_activations[i] = af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::OBJECTIVE, total_hidden_inputs[i]);
		  }

		  // update weight matrix index	
		  weight_index += total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();

		  // update layer indices
		  previous_layer_index = current_layer_index;
		  current_layer_index += total_layers[current_layer]->get_size();
		  
		  // increment layer counters 
		  ++current_layer; 
		  ++previous_layer;
		 }
	  }

	template <class precision_type>
  	  void multi_layer_perceptron<connection::dense, precision_type>::get_outputs(const std::vector<zinhart::activation::LAYER_INFO> & total_layers, 
																				  const precision_type * total_hidden_outputs, const std::uint32_t total_hidden_outputs_length, 
																				  precision_type * model_outputs, 
																				  const std::uint32_t n_threads, 
																				  const std::uint32_t thread_id
																				 )
  	  {
		std::uint32_t i{0}, j{0}, output_layer{total_layers.size()-1}, output_layer_index{0};
		std::uint32_t thread_stride{0}, current_threads_workspace_index{0}; 

		// Assumes the activation vector is partitioned into equally size chucks, 1 for each thread
		thread_stride = total_hidden_outputs_length / n_threads;

		// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the activation vector
		current_threads_workspace_index = thread_id * thread_stride;

		for(i = 1; i < total_layers.size() - 1; ++i)
		  output_layer_index += total_layers[i].second;
		
		for(i = current_threads_workspace_index + output_layer_index, j = 0; j < total_layers[output_layer].second; ++i, ++j)
		  *(model_outputs + j) = *(total_hidden_outputs + i);
	  }
	template <class precision_type>
	  void multi_layer_perceptron<connection::dense, precision_type>::backward_propagate(const std::vector<zinhart::activation::LAYER_INFO> & total_layers, 
																			const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
																			const precision_type * const total_hidden_inputs, const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
																			const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
																			const precision_type * const total_bias,
																			const std::uint32_t n_threads,
																			const std::uint32_t thread_id
																			)
	  {
		std::uint32_t i{0}, j{0};

		const std::uint32_t input_layer{0}, output_layer{total_layers.size() - 1};
		std::uint32_t current_layer_index{0}, previous_layer_index{0}, current_gradient_index{0};
		// the start of the output layer
 		for(i = 1; i < total_layers.size() - 1; ++i)
 		  current_layer_index += total_layers[i].second;
		// the start of the layer right behind the output layer
 		for(i = 1; i < total_layers.size() - 2; ++i)
 		  previous_layer_index += total_layers[i].second;	   
		// The index of the beginning of the gradient matrix between the last hidden layer and the output layer 
	   	for(i = 0 ; i < total_layers.size() - 2; ++i)
		  current_gradient_index += total_layers[i + 1].second * total_layers[i].second;

		// All layer counters
		std::uint32_t current_layer{output_layer}, previous_layer{output_layer - 1}; 

		// variables for gemm
		precision_type alpha{1.0}, beta{0.0};
		std::uint32_t m{0}, n{0}, k{0};

		// a ptr to the current training case, 
		// the number of nodes in the input layer is the same length as the length of the current training case so we move case_index times forward in the total_training_cases ptr, 
		// -> case_index = 0 is the first training case, case_index = 1 the second case, case_index = n the nth case.
		std::uint32_t input_stride{case_index * total_layers[input_layer].second};
		std::uint32_t error_stride{thread_id * total_layers[output_layer].second};
		const precision_type * const current_training_case{total_training_cases + input_stride};
	//	const precision_type * const current_error_matrix{d_error + error_stride};
		
		// variables for the thread calling this method, to determine it's workspace
		std::uint32_t current_threads_activation_workspace_index{0}, current_threads_gradient_workspace_index{0}, current_threads_output_workspace_index{0};
		std::uint32_t thread_activation_stride{0}, thread_gradient_stride{0}, thread_output_stride{0};
		
		// the activation function of each layer
		zinhart::activation::activation_function af;

		// Assumes the activation vector is partitioned into equally size chucks, 1 for each thread
		thread_activation_stride = total_activations_length / n_threads;
		thread_gradient_stride = total_hidden_weights_length;
		thread_output_stride = total_layers[output_layer].second;

		// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the relevant vector
		current_threads_activation_workspace_index = thread_id * thread_activation_stride;
		current_threads_output_workspace_index = thread_id * thread_output_stride;
		current_threads_gradient_workspace_index = thread_id * thread_gradient_stride;

		// set pointers for output layer gradient for the current thread
	    const precision_type * current_threads_hidden_input_ptr{total_hidden_inputs + current_threads_activation_workspace_index};
		const precision_type * current_threads_activation_ptr{total_activations + current_threads_activation_workspace_index};
		precision_type * current_threads_delta_ptr{total_deltas + current_threads_activation_workspace_index};
		precision_type * current_threads_gradient_ptr{total_gradient + current_threads_gradient_workspace_index};

		const precision_type * output_layer_hidden_inputs_ptr{current_threads_hidden_input_ptr + current_layer_index};
   		// if this is a 2 layer model then the prior activations are essentially the inputs to the model, i.e the while loop does not activate
		const precision_type * prior_activation_ptr{ (total_layers.size() > 2) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case };
		precision_type * current_layer_deltas_ptr{current_threads_delta_ptr + current_layer_index};
		precision_type * current_gradient_ptr{current_threads_gradient_ptr + current_gradient_index};

		// calc output layer deltas
		for(i = current_threads_activation_workspace_index + current_layer_index, j = error_stride, k = 0; k < total_layers[output_layer].second; ++i, ++j, ++k)
		{
		  total_deltas[i] = d_error[j] * af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::DERIVATIVE, total_activations[i]);
		}

		// for gemm
		m = total_layers[current_layer].second;
		n = total_layers[previous_layer].second;
	   	k = 1;

		// calc output layer gradient
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, current_layer_deltas_ptr, k,
					prior_activation_ptr, n, beta, 
					current_gradient_ptr, n
				   );

	  // set up for hidden layer gradients
	  std::uint32_t next_weight_matrix_index{total_hidden_weights_length};
	  std::uint32_t next_layer_index{current_layer_index};
	  std::uint32_t next_layer{current_layer};
	  --current_layer;
	  --previous_layer;
	  while(current_layer > 0)
	  {
		next_weight_matrix_index -= total_layers[next_layer].second * total_layers[current_layer].second;
		current_layer_index = previous_layer_index;
		previous_layer_index -= total_layers[previous_layer].second;
		current_gradient_index -= total_layers[current_layer].second * total_layers[previous_layer].second;
		current_gradient_ptr = current_threads_gradient_ptr + current_gradient_index;
		// the weight matrix one layer in front of the current gradient matrix
		const precision_type * weight_ptr{total_hidden_weights + next_weight_matrix_index};
		precision_type * next_layer_delta_ptr{current_threads_delta_ptr + next_layer_index};
		current_layer_deltas_ptr = current_threads_delta_ptr + current_layer_index;
		const precision_type * previous_layer_activation_ptr{ (current_layer > 1) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case };

		m = total_layers[current_layer].second;
		n = 1;
		k = total_layers[next_layer].second;

   		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				  m, n, k,
				  alpha, weight_ptr, m,
				  next_layer_delta_ptr, n, beta, 
				  current_layer_deltas_ptr, n
				 );
		for(i = current_threads_activation_workspace_index + current_layer_index, j = 0; j < total_layers[current_layer].second; ++i, ++j)
		{
		  total_deltas[i] *= af(total_layers[current_layer].first, zinhart::activation::ACTIVATION_TYPE::DERIVATIVE, total_activations[i]);
		}
	
		m = total_layers[current_layer].second;
   		n = total_layers[previous_layer].second;
   		k = 1;

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				  m, n, k,
				  alpha, current_layer_deltas_ptr, k,
				  previous_layer_activation_ptr, n, beta, 
				  current_gradient_ptr, n
				 );

		next_layer_index = current_layer_index;
		--next_layer;
		--current_layer;
		--previous_layer;
	  }
	}	  
#endif
  }
}
