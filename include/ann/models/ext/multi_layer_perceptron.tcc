#include <mkl.h>
#include <multi_core/multi_core.hh>
#include <cassert>
#include <exception>
#include <random>
#include <algorithm>
namespace zinhart
{
  namespace models
  {


#if CUDA_ENABLED == MULTI_CORE_DISABLED
	//cpu multi-threaded code

	template <class precision_type>
	  HOST void multi_layer_perceptron<precision_type>::forward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
																		  const precision_type * total_training_cases, const std::uint32_t case_index,
																		  precision_type * total_activations, const std::uint32_t total_activations_length,
																		  const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																		  const precision_type * total_bias,
																		  const std::uint32_t n_threads,
																		  const std::uint32_t thread_id
																		 )
	  {
		zinhart::function_space::objective objective_function{};

		const std::uint32_t input_layer{0};

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
		precision_type * current_threads_output_ptr{nullptr};

		// variables for gemm
		precision_type alpha{1.0}, beta{0.0};
		std::uint32_t m{ total_layers[current_layer]->get_size() }, n{1}, k{ total_layers[previous_layer]->get_size() };

		// Assumes the activation vector is partitioned into equally size chucks, 1 for each thread
		thread_stride = total_activations_length / n_threads;

		// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the relevant vector
		current_threads_workspace_index = thread_id * thread_stride;
		current_threads_output_ptr = total_activations + current_threads_workspace_index;

		// do input layer and the first hidden layer -> Wx
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k,
					alpha, total_hidden_weights, k,
					current_training_case, n, beta, 
					current_threads_output_ptr, n
				   );

		// add in bias, calc output of this layer e.g. activative
		total_layers[current_layer]->activate(objective_function, current_threads_output_ptr, total_layers[current_layer]->get_size(), total_bias[previous_layer] );


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
		  precision_type * current_layer_outputs_ptr{total_activations + current_threads_workspace_index + current_layer_index};
		  const precision_type * prior_layer_ptr{total_activations + current_threads_workspace_index + previous_layer_index}; 

		  m = total_layers[current_layer]->get_size();
		  n = 1;
		  k = total_layers[previous_layer]->get_size();

		  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					  m, n, k,
					  alpha, current_weight_matrix, k,
					  prior_layer_ptr, n, beta, 
					  current_layer_outputs_ptr, n
					 );

		  // add in bias, calc output of this layer e.g. activative
  		  total_layers[current_layer]->activate(objective_function, current_layer_outputs_ptr, total_layers[current_layer]->get_size(), total_bias[previous_layer]);

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
  	  void multi_layer_perceptron<precision_type>::get_outputs_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
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
		  output_layer_index += total_layers[i]->get_size();
		
		for(i = current_threads_workspace_index + output_layer_index, j = 0; j < total_layers[output_layer]->get_size(); ++i, ++j)
		  *(model_outputs + j) = *(total_hidden_outputs + i);
	  }

  	template <class precision_type>
  	  HOST void multi_layer_perceptron<precision_type>::gradient_check_impl(zinhart::loss_functions::loss_function<precision_type> * loss,
																		    const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
																		    const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
																		    precision_type * total_activations, const std::uint32_t total_activations_length,
																		    precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																		    const precision_type * total_bias, 
																		    precision_type * numerically_approx_gradient, 
																		    const precision_type limit_epsilon, 
																		    const std::uint32_t n_threads, 
																			const std::uint32_t thread_id
																		   )
	  { 
		zinhart::function_space::objective objective_function{};
		std::uint32_t i{0};
		const std::uint32_t output_layer{total_layers.size() - 1};

		std::uint32_t output_layer_index{0};
		// start at 1 to skip the input layer
		for(i = 1; i < total_layers.size() - 1; ++i)
		  output_layer_index += total_layers[i]->get_size();

		// a ptr to the current training case, 
		// the number of nodes in the input layer is the same length as the length of the current training case so we move case_index times forward in the total_training_cases ptr, 
		// -> case_index = 0 is the first training case, case_index = 1 the second case, case_index = n the nth case.
		const precision_type * current_target{total_targets + (case_index * total_layers[output_layer]->get_size())};

		// variables for the thread calling this method, to determine it's workspace
		std::uint32_t /*current_threads_activation_workspace_index{0},*/ current_threads_gradient_workspace_index{0}, thread_activation_stride{0}, thread_gradient_stride{0}, thread_output_layer_stride{0};
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
		  forward_propagate_impl(total_layers, 
								 total_training_cases, case_index, 
								 total_activations, total_activations_length,
								 total_hidden_weights, total_hidden_weights_length, 
								 total_bias,
								 n_threads, thread_id
							    );

	      right = loss->error(objective_function, current_threads_output_layer_ptr, current_target, total_layers[output_layer]->get_size());

		  // set back
		  total_hidden_weights[i] = original; 

		  // left side
		  total_hidden_weights[i] -= limit_epsilon;
		  forward_propagate_impl(total_layers, 
								 total_training_cases, case_index, 
								 total_activations, total_activations_length,
								 total_hidden_weights, total_hidden_weights_length, 
								 total_bias,
								 n_threads, thread_id
							    );

	      left = loss->error(objective_function, current_threads_output_layer_ptr, current_target, total_layers[output_layer]->get_size());

		  // calc numerically derivative for the ith_weight, save it, increment the pointer to the next weight
		  *(current_threads_gradient_ptr + i) = (right - left) / (precision_type{2.0} * limit_epsilon);

		  // set back
		  total_hidden_weights[i] = original;
		}
	  }  

	template <class precision_type>
	  void multi_layer_perceptron<precision_type>::backward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
																		   const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
																	       const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
																	       const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
																	       const precision_type * const total_bias,
																	       const std::uint32_t n_threads,
																	       const std::uint32_t thread_id
																	      )
	  {
		zinhart::function_space::derivative derivative_function{};
		std::uint32_t i{0};

		const std::uint32_t input_layer{0}, output_layer{total_layers.size() - 1};
		std::uint32_t current_layer_index{0}, previous_layer_index{0}, current_gradient_index{0};
		// the start of the output layer
 		for(i = 1; i < total_layers.size() - 1; ++i)
 		  current_layer_index += total_layers[i]->get_size();
		// the start of the layer right behind the output layer
 		for(i = 1; i < total_layers.size() - 2; ++i)
 		  previous_layer_index += total_layers[i]->get_size();	   
		// The index of the beginning of the gradient matrix between the last hidden layer and the output layer 
	   	for(i = 0 ; i < total_layers.size() - 2; ++i)
		  current_gradient_index += total_layers[i + 1]->get_size() * total_layers[i]->get_size();

		// All layer counters
		std::uint32_t current_layer{output_layer}, previous_layer{output_layer - 1}; 

		// variables for gemm
		precision_type alpha{1.0}, beta{0.0};
		std::uint32_t m{0}, n{0}, k{0};

		// a ptr to the current training case, 
		// the number of nodes in the input layer is the same length as the length of the current training case so we move case_index times forward in the total_training_cases ptr, 
		// -> case_index = 0 is the first training case, case_index = 1 the second case, case_index = n the nth case.
		std::uint32_t input_stride{case_index * total_layers[input_layer]->get_size()};
		std::uint32_t error_stride{thread_id * total_layers[output_layer]->get_size()};
		const precision_type * const current_training_case{total_training_cases + input_stride};
		const precision_type * const current_error_matrix{d_error + error_stride};
		
		// variables for the thread calling this method, to determine it's workspace
		std::uint32_t current_threads_activation_workspace_index{0}, current_threads_gradient_workspace_index{0}, current_threads_output_workspace_index{0};
		std::uint32_t thread_activation_stride{0}, thread_gradient_stride{0}, thread_output_stride{0};
		
		// Assumes the activation vector is partitioned into equally size chucks, 1 for each thread
		thread_activation_stride = total_activations_length / n_threads;
		thread_gradient_stride = total_hidden_weights_length;
		thread_output_stride = total_layers[output_layer]->get_size();

		// with the assumption above the index of where the current chunk begins is the length of each case thread_id chunks forward in the relevant vector
		current_threads_activation_workspace_index = thread_id * thread_activation_stride;
		current_threads_output_workspace_index = thread_id * thread_output_stride;
		current_threads_gradient_workspace_index = thread_id * thread_gradient_stride;

		// set pointers for output layer gradient for the current thread
	    // const precision_type * current_threads_hidden_input_ptr{total_hidden_inputs + current_threads_activation_workspace_index};
		const precision_type * current_threads_activation_ptr{total_activations + current_threads_activation_workspace_index};
		precision_type * current_threads_delta_ptr{total_deltas + current_threads_activation_workspace_index};
		precision_type * current_threads_gradient_ptr{total_gradient + current_threads_gradient_workspace_index};

		const precision_type * output_layer_activation_ptr{current_threads_activation_ptr + current_layer_index};
   		// if this is a 2 layer model then the prior activations are essentially the inputs to the model, i.e the while loop does not activate
		const precision_type * prior_activation_ptr{ (total_layers.size() > 2) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case };
		precision_type * current_layer_deltas_ptr{current_threads_delta_ptr + current_layer_index};
		precision_type * current_gradient_ptr{current_threads_gradient_ptr + current_gradient_index};

		// calc output layer deltas
		total_layers[current_layer]->activate(zinhart::models::layers::layer_info::output_layer(), derivative_function, current_layer_deltas_ptr, current_error_matrix, output_layer_activation_ptr, total_layers[current_layer]->get_size());

		// for gemm
		m = total_layers[current_layer]->get_size();
		n = total_layers[previous_layer]->get_size();
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
		next_weight_matrix_index -= total_layers[next_layer]->get_size() * total_layers[current_layer]->get_size();
		current_layer_index = previous_layer_index;
		previous_layer_index -= total_layers[previous_layer]->get_size();
		current_gradient_index -= total_layers[current_layer]->get_size() * total_layers[previous_layer]->get_size();
		current_gradient_ptr = current_threads_gradient_ptr + current_gradient_index;
		// the weight matrix one layer in front of the current gradient matrix
		const precision_type * weight_ptr{total_hidden_weights + next_weight_matrix_index};
		precision_type * next_layer_delta_ptr{current_threads_delta_ptr + next_layer_index};
		current_layer_deltas_ptr = current_threads_delta_ptr + current_layer_index;
		const precision_type * current_layer_activation_ptr{current_threads_activation_ptr + current_layer_index};
		const precision_type * previous_layer_activation_ptr{ (current_layer > 1) ? (current_threads_activation_ptr + previous_layer_index) : current_training_case };

		m = total_layers[current_layer]->get_size();
		n = 1;
		k = total_layers[next_layer]->get_size();

   		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				  m, n, k,
				  alpha, weight_ptr, m,
				  next_layer_delta_ptr, n, beta, 
				  current_layer_deltas_ptr, n
				 );

		// calc hidden layer deltas
		total_layers[current_layer]->activate(zinhart::models::layers::layer_info::hidden_layer(), derivative_function, current_layer_deltas_ptr, current_layer_activation_ptr, total_layers[current_layer]->get_size());

		m = total_layers[current_layer]->get_size();
   		n = total_layers[previous_layer]->get_size();
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
	template <class precision_type>
	  HOST void multi_layer_perceptron<precision_type>::forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
																		  const precision_type * total_training_cases, const std::uint32_t  case_index,
																		  precision_type * total_activations, const std::uint32_t total_activations_length,
																		  const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																		  const precision_type * total_bias,
																		  const std::uint32_t n_threads,
																		  const std::uint32_t thread_id
																		 )
	  {	forward_propagate_impl(total_layers, total_training_cases, case_index, total_activations, total_activations_length, total_hidden_weights, total_hidden_weights_length, total_bias, n_threads, thread_id); }
	template <class precision_type>
  	  HOST void multi_layer_perceptron<precision_type>::get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
																	const precision_type * total_activations, const std::uint32_t total_activations_length,
																	precision_type * model_outputs, 
																	const std::uint32_t n_threads, 
																	const std::uint32_t thread_id
																   )
	  { get_outputs_impl(total_layers, total_activations, total_activations_length, model_outputs, n_threads, thread_id); }

	template <class precision_type>
	  HOST void multi_layer_perceptron<precision_type>::gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
																	   const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
																	   const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
																	   precision_type * total_activations, const std::uint32_t total_activations_length,
																	   precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
																	   const precision_type * total_bias, 
																	   precision_type * num_approx_grad, 
																	   const precision_type limit_epsilon, 
																	   const std::uint32_t n_threads, const std::uint32_t thread_id
																	  )
	  {	gradient_check_impl(loss, total_layers, total_training_cases, total_targets, case_index, total_activations, total_activations_length, total_hidden_weights, total_hidden_weights_length, total_bias, num_approx_grad, limit_epsilon, n_threads, thread_id); }

	template <class precision_type>
	  HOST  void multi_layer_perceptron<precision_type>::backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
																			const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
																			const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
																			const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
																			const precision_type * const total_bias,
																			const std::uint32_t n_threads,
																			const std::uint32_t thread_id
																		   )
	  { backward_propagate_impl(total_layers, total_training_cases, total_targets, d_error, case_index, total_activations, total_deltas, total_activations_length, total_hidden_weights, total_gradient, total_hidden_weights_length, total_bias, n_threads, thread_id); }
	
	template<class precision_type>
  	  void fprop_mlp(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
					 precision_type * total_cases_ptr, const std::uint32_t ith_training_case,
					 precision_type * total_activations_ptr, const std::uint32_t activations_length,
					 precision_type * total_hidden_weights_ptr, const std::uint32_t weights_length,
					 precision_type * total_bias_ptr,
					 const std::uint32_t n_threads, 
					 const std::uint32_t thread_id
					)
	  {
		multi_layer_perceptron<precision_type> mlp;
		mlp.forward_propagate(total_layers,
							  total_cases_ptr, ith_training_case,
							  total_activations_ptr, activations_length,
							  total_hidden_weights_ptr, weights_length,
							  total_bias_ptr,
							  n_threads,
							  thread_id
							 );

	  }

	template<class precision_type>
	  HOST void get_outputs_mlp(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
  		                        const precision_type * total_hidden_outputs, const std::uint32_t total_hidden_outputs_length,
							    precision_type * model_outputs, 
							    const std::uint32_t n_threads, 
							    const std::uint32_t thread_id
						       )
	  {
		multi_layer_perceptron<precision_type> mlp;
		mlp.get_outputs(total_layers, total_hidden_outputs, total_hidden_outputs_length, model_outputs, n_threads, thread_id);
	  }
	template<class precision_type>
	  void gradient_check_mlp(zinhart::loss_functions::loss_function<precision_type> * loss,
							  std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
							  const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
							  precision_type * total_activations, const std::uint32_t total_activations_length,
							  precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
							  const precision_type * total_bias, 
							  precision_type * numerically_approx_gradient, 
							  const precision_type limit_epsilon, 
							  const std::uint32_t n_threads, 
							  const std::uint32_t thread_id
							)
	  {
		multi_layer_perceptron<precision_type> mlp;
		mlp.gradient_check(loss, 
						   total_layers,
						   total_training_cases, total_targets, case_index,
						   total_activations, total_activations_length,
						   total_hidden_weights, total_hidden_weights_length,
						   total_bias,
						   numerically_approx_gradient,
						   limit_epsilon,
						   n_threads, 
						   thread_id
						  );
						
	  }
	template<class precision_type>
	  void bprop_mlp(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers, 
					 const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
					 precision_type * total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
					 const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
					 const precision_type * const total_bias,
					 const std::uint32_t n_threads,
					 const std::uint32_t thread_id
					)
	  {
		multi_layer_perceptron<precision_type> mlp;
		mlp.backward_propagate(total_layers,
							   total_training_cases, total_targets, d_error, case_index,
							   total_activations, total_deltas, total_activations_length,
							   total_hidden_weights, total_gradient, total_hidden_weights_length,
							   total_bias,

							   n_threads,
							   thread_id
							  );
	  }

	template<class precision_type>
	  void online_train(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
					   std::shared_ptr< zinhart::loss_functions::loss_function<precision_type> > loss_function,
					   std::shared_ptr< zinhart::optimizers::optimizer<precision_type> > optimizer,
					   precision_type * total_training_cases_ptr,  const std::uint32_t total_training_cases_length,
					   const precision_type * const total_targets_ptr, precision_type * total_error_ptr,
					   precision_type * total_activations_ptr, precision_type * total_deltas_ptr, const std::uint32_t total_activations_length,
					   precision_type * total_hidden_weights_ptr, precision_type * total_gradient_ptr, const std::uint32_t total_hidden_weights_length,
					   precision_type * total_bias_ptr,
					   const std::uint32_t max_epochs,
					   bool verbose,
					   std::ostream & output_stream
					  )
	  {
	  }
	
	template<class precision_type>
	  void batch_train(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
					   std::shared_ptr< zinhart::loss_functions::loss_function<precision_type> > loss_function,
					   std::shared_ptr< zinhart::optimizers::optimizer<precision_type> > optimizer,
					   precision_type * total_training_cases_ptr,  const std::uint32_t total_training_cases_length,
					   const precision_type * const total_targets_ptr, precision_type * total_error_ptr,
					   precision_type * total_activations_ptr, precision_type * total_deltas_ptr, const std::uint32_t total_activations_length,
					   precision_type * total_hidden_weights_ptr, precision_type * total_gradient_ptr, const std::uint32_t total_hidden_weights_length,
					   precision_type * total_bias_ptr,
					   const std::uint32_t max_epochs, 
					   const std::uint32_t batch_size,
					   bool verbose,
					   std::ostream & output_stream
					  )
	  {
		// task futures
		std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> tasks;
		std::vector<zinhart::multi_core::thread_pool::tasks::task_future<precision_type>> error_tasks;

		auto this_batch_error = [&loss_function](const precision_type * o, const precision_type * t, std::uint32_t len)
		{
		  return loss_function->error(zinhart::function_space::objective{}, o, t, len);
		};


		auto this_batch_error_derivative = [&loss_function](const precision_type * o, const precision_type * t, precision_type * r, std::uint32_t len)
		{
		  loss_function->error(zinhart::function_space::derivative{}, o, t, r, len);
		};

		auto weight_update = [&optimizer](precision_type * weights, const precision_type * const gradient, const std::uint32_t len, const std::uint32_t nthreads, const std::uint32_t thread_id)
		{
		  optimizer->update(weights, gradient, len, nthreads, thread_id);
		};
		std::uint32_t   * case_ids{nullptr};
		precision_type ** model_outputs_ptr{nullptr};
	    std::uint32_t     thread_id{0};
		try
		{
		  if(batch_size < 2 )
			throw std::logic_error("batch_size much be greater than 1, you may also use the online training function");
		  zinhart::multi_core::thread_pool::resize(batch_size);// consider moving this to scope of function that calls train

		  // to shuffle the case id_s;
		  std::random_device rd;
		  std::mt19937       mt(rd());
		  const std::uint32_t input_layer{0};
		  const std::uint32_t output_layer{total_layers.size() - 1};
		  const std::uint32_t total_training_cases{ total_training_cases_length / total_layers[input_layer]->get_size() };
		  const std::uint32_t full_batches{ total_training_cases / batch_size  };
		  const std::uint32_t remaining_cases{ total_training_cases % batch_size  };
		  const std::uint32_t training_loop_stop{ total_training_cases - remaining_cases };
		  precision_type batch_error{0.0};
		  std::uint32_t ith_batch{0}, ith_epoch{0}, current_batch_begin{0}, current_batch_end{0}, ith_training_case{0}, i{0}, case_count{0};

		  // allocate 
		  case_ids          = new std::uint32_t[total_training_cases];
		  for(i = 0; i < total_training_cases; ++i)
			case_ids[i]     = i;

		  model_outputs_ptr = new precision_type*[batch_size];
		  for(thread_id = 0; thread_id < batch_size; ++thread_id)
		  {
			model_outputs_ptr[thread_id] = new precision_type[total_layers[output_layer]->get_size()];
			for(i = 0; i < total_layers[output_layer]->get_size(); ++i)
			  model_outputs_ptr[thread_id][i] = precision_type{};
		  }


		  if(verbose == true) 
		  {
			output_stream<<"max epochs: "               << max_epochs                            << "\n";
			output_stream<<"full batches: "             << full_batches                          << "\n";
			output_stream<<"left over cases: "          << remaining_cases                       << "\n";
			output_stream<<"cases per batch: "          << batch_size                            << "\n";
			output_stream<<"total training cases: "     << total_training_cases                  << "\n";
			output_stream<<"case dimensions: "          << total_layers[input_layer]->get_size() << "\n";
			output_stream<<"model layers: "             << total_layers.size()                   << "\n";
			output_stream<<"\n";
		  }
		  for(ith_epoch = 1; ith_epoch <= max_epochs; ++ith_epoch)
		  {
			if(verbose == true)
			{
			  output_stream<<"current epoch: "<< ith_epoch<<"\n";
			}
			// randomly shuffle the case id's
			std::shuffle(case_ids, case_ids + total_training_cases, mt);
			for(ith_batch = 1, current_batch_begin = 0, current_batch_end = batch_size; ith_batch <= full_batches; ++ith_batch, current_batch_begin+=batch_size, current_batch_end += batch_size)
  			{
			  if(verbose == true)			
  			  {
				output_stream<<"batches left: "<< full_batches - ith_batch + 1<<"\n";
				output_stream<<"current batch: "<< ith_batch <<"\n";
				output_stream<<"ith_training_case: "<< ith_training_case <<"\n";
			  }
			  // forward propagate for each cases in this batch
			  for(ith_training_case = current_batch_begin, thread_id = 0; ith_training_case < current_batch_end; ++ith_training_case, ++thread_id)
			  {
				// forward propagate
				tasks.push_back(zinhart::multi_core::thread_pool::push_task(fprop_mlp<precision_type>,
																					total_layers,
																					total_training_cases_ptr, case_ids[ith_training_case],
																					total_activations_ptr, total_activations_length,
																					total_hidden_weights_ptr, total_hidden_weights_length,
																					total_bias_ptr,
																					batch_size,
																					thread_id
																				   )
							   );
			  } 

			  // synchronize forward prop
			  for(thread_id = 0; thread_id < batch_size; ++thread_id)
				tasks[thread_id].get();

			  tasks.clear();

			  // get_outputs
			  for(thread_id = 0; thread_id < batch_size; ++thread_id)
				tasks[thread_id] = zinhart::multi_core::thread_pool::push_task(get_outputs_mlp<precision_type>, std::ref(total_layers), total_activations_ptr, total_activations_length, model_outputs_ptr[thread_id], batch_size, thread_id);

			  // synchronize outputs
			  for(thread_id = 0; thread_id < batch_size; ++thread_id)
				tasks[thread_id].get();

			  tasks.clear();
			
  			  // calculate error and error derivative for each case in this batch
			  for(ith_training_case = current_batch_begin, thread_id = 0; ith_training_case < current_batch_end; ++ith_training_case, ++thread_id)
			  {
				// clean this up
				const precision_type * current_target_ptr{total_targets_ptr + (case_ids[ith_training_case] * total_layers[output_layer]->get_size())};
				const precision_type * current_outputs_ptr{model_outputs_ptr[thread_id]};
				precision_type       * current_error_ptr{total_error_ptr + (thread_id * total_layers[output_layer]->get_size())};
				const std::uint32_t length{ total_layers[output_layer]->get_size() };
				error_tasks.push_back(zinhart::multi_core::thread_pool::push_task(this_batch_error, current_outputs_ptr, current_target_ptr, length) );
				tasks[thread_id] = zinhart::multi_core::thread_pool::push_task(this_batch_error_derivative, current_outputs_ptr, current_target_ptr, current_error_ptr, length);
			  }


			  // synchronize error calculation for this batch
			  for(thread_id = 0, batch_error = 0; thread_id < batch_size; ++thread_id)
			  {
				batch_error += error_tasks[thread_id].get();
				tasks[thread_id].get();
			  }
			
  			  if(verbose == true) 
			  {
				output_stream<<"batch error: "<< batch_error <<"\n";
			  }

  			  error_tasks.clear();
			  tasks.clear();

			  // backpropagate for each case in this batch
			  for(ith_training_case = current_batch_begin, thread_id = 0; ith_training_case < current_batch_end; ++ith_training_case, ++thread_id)
			  {
				tasks[thread_id] = zinhart::multi_core::thread_pool::push_task(bprop_mlp<precision_type>,
																					   total_layers,
																					   total_training_cases_ptr, total_targets_ptr, total_error_ptr, case_ids[ith_training_case],
																					   total_activations_ptr, total_deltas_ptr, total_activations_length,
																					   total_hidden_weights_ptr, total_gradient_ptr, total_hidden_weights_length,
																					   total_bias_ptr,
																					   batch_size,
																					   thread_id
																					  ); 

			  } 	  

			  // synchronize gradient calculation for this batch
			  for(thread_id = 0; thread_id < batch_size; ++thread_id)
				tasks[thread_id].get();

			  /* sketch
			   * instead of allocating memory for a space to cumulate the gradient in,
  			   * get a pointer to the gradient vector computed by the first thread.
			   * add all other vectors to it
			   * pass the pointer to this first vector to the optimizer
			   * */
	  		  precision_type * initial_gradient_ptr{total_gradient_ptr};
			  precision_type * next_gradient_ptr{nullptr};

			  for(thread_id = 1; thread_id < batch_size; ++thread_id)
			  {
				next_gradient_ptr = total_gradient_ptr + (thread_id * total_hidden_weights_length); 
				for(i = 0; i < total_hidden_weights_length; ++i)
				  *(initial_gradient_ptr + i) += *(next_gradient_ptr + i);
			  }

			  // update weights for this batch
			  if(total_hidden_weights_length >= batch_size)
			  {
				for(thread_id = 0; thread_id < batch_size; ++thread_id)
				  tasks[thread_id] = zinhart::multi_core::thread_pool::push_task( weight_update, total_hidden_weights_ptr, initial_gradient_ptr, total_hidden_weights_length, batch_size, thread_id );	
			  }
			  else
			  {
				weight_update( total_hidden_weights_ptr, initial_gradient_ptr, total_hidden_weights_length, 1, 0);
			  }
			  // conditionally synchronize weight updates
			  if(total_hidden_weights_length >= batch_size)	
				for(thread_id = 0; thread_id < batch_size; ++thread_id)
				  tasks[thread_id].get();

			  tasks.clear();
			}// end full batch loop
			output_stream<<"\n";
		  }// end epoch loop
		  
		  output_stream<<"\n\nith_training_case: "<< ith_training_case <<"\n";


		}
		catch(std::logic_error)
		{

		  for(thread_id = 0; thread_id < batch_size; ++thread_id)
		  {
			delete [] model_outputs_ptr[thread_id];
		  }
		  delete [] model_outputs_ptr;
		  delete [] case_ids;

		  case_ids          = nullptr;
		  model_outputs_ptr = nullptr;

		}
		catch(std::exception & e)
		{
		  output_stream<<e.what()<<"\n";
		  // deallocate memory
		  
		  for(thread_id = 0; thread_id < batch_size; ++thread_id)
		  {
			delete [] model_outputs_ptr[thread_id];
		  }
		  delete [] model_outputs_ptr;
		  delete [] case_ids;

		  case_ids          = nullptr;
		  model_outputs_ptr = nullptr;

		}
		if(case_ids != nullptr)
		  delete [] case_ids;

		if(model_outputs_ptr != nullptr)
		{
		  for(thread_id = 0; thread_id < batch_size; ++thread_id)
		  {
			delete [] model_outputs_ptr[thread_id];
		  }
		  delete [] model_outputs_ptr;
		}
	  }
#endif
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
