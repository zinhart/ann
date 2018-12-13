#ifndef MULTI_LAYER_PERCEPTRON_HH
#define MULTI_LAYER_PERCEPTRON_HH
namespace zinhart
{
  namespace models
  {	
	
	template<class precision_type>
  	  class multi_layer_perceptron : public ann_mlp<precision_type>
	  {
		private:
		  HOST virtual void forward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
												   const precision_type * total_training_cases, const std::uint32_t case_index,
												   precision_type * total_activations, const std::uint32_t total_activations_length,
												   const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
												   const precision_type * total_bias,
												   const std::uint32_t n_threads,
												   const std::uint32_t thread_id) override;

		  HOST virtual void get_outputs_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
										     const precision_type * total_activations, const std::uint32_t total_activations_length, 
										     precision_type * model_outputs, 
										     const std::uint32_t n_threads, 
										     const std::uint32_t thread_id
									        ) override;

		  HOST virtual void gradient_check_impl(zinhart::loss_functions::loss_function<precision_type> * loss,
										        const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
										        const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
										        precision_type * total_activations, const std::uint32_t total_activations_length,
										        precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
										        const precision_type * total_bias, 
										        precision_type * numerically_approx_gradient, 
										        const precision_type limit_epsilon, 
										        const std::uint32_t n_threads, 
												const std::uint32_t thread_id
										       ) override;

		  HOST virtual void backward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
											        const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
											        const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
											        const precision_type * const total_hidden_weights, precision_type * tot_grad, const std::uint32_t total_hidden_weights_length,
											        const precision_type * const total_bias,
											        const std::uint32_t n_threads,
											        const std::uint32_t thread_id
			                                       ) override;


		public:
		  multi_layer_perceptron() = default;
		  multi_layer_perceptron(const multi_layer_perceptron &) = default;
		  multi_layer_perceptron(multi_layer_perceptron &&) = default;
		  multi_layer_perceptron & operator = (const multi_layer_perceptron &) = default;
		  multi_layer_perceptron & operator = (multi_layer_perceptron &&) = default;
		  ~multi_layer_perceptron() = default;

#if CUDA_ENABLED == MULTI_CORE_DISABLED
		// Defaults to single-threaded
		HOST void forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
								    const precision_type * total_training_cases, const std::uint32_t  case_index,
								    precision_type * total_activations, const std::uint32_t total_activations_length,
								    const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
								    const precision_type * total_bias,
							        const std::uint32_t n_threads = 1,
								    const std::uint32_t thread_id = 0
								   );

		// Defaults to single-threaded
		HOST void get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
							  const precision_type * total_hidden_outputs, const std::uint32_t total_hidden_outputs_length,
							  precision_type * model_outputs, 
							  const std::uint32_t n_threads = 1, 
							  const std::uint32_t thread_id = 0
							 );

		// Defaults to single threaded
		HOST void gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
								 const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
								 const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
								 precision_type * total_activations, const std::uint32_t total_activations_length,
								 precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
								 const precision_type * total_bias, 
								 precision_type * numerically_approx_gradient, 
								 const precision_type limit_epsilon = 1.e-4, 
								 const std::uint32_t n_threads = 1, const std::uint32_t thread_id = 0
								);

		 // Defaults to single threaded
		 HOST void backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
									  const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
									  const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
									  const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
									  const precision_type * const total_bias,
									  const std::uint32_t n_threads = 1,
									  const std::uint32_t thread_id = 0
									 );

#endif
		};// END CLASS MULTI_LAYER_PERCEPTRON

	// allocate everything based on information in total_layers
	template <class precision_type>
	  void init(const std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
				std::uint32_t & total_activations_length,
				std::uint32_t & total_hidden_weights_length,
				const std::uint32_t n_threads
			   )
	  {
		std::uint32_t ith_layer{0};
		// calc number of activations
		for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
		  total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
		total_activations_length *= n_threads;

		// calc number of hidden weights
		for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
		  total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 

	  }
	template<class precision_type>
  	  HOST void fprop_mlp(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
						  precision_type * total_cases_ptr, const std::uint32_t ith_training_case,
						  precision_type * total_activations_ptr, const std::uint32_t activations_length,
						  precision_type * total_hidden_weights_ptr, const std::uint32_t weights_length,
						  precision_type * total_bias_ptr,
						  const std::uint32_t total_threads, 
						  const std::uint32_t thread_id
						 );

	template<class precision_type>
	  HOST void get_outputs_mlp(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
  		                        const precision_type * total_hidden_outputs, const std::uint32_t total_hidden_outputs_length,
							    precision_type * model_outputs, 
							    const std::uint32_t n_threads, 
							    const std::uint32_t thread_id
						       );

	template<class precision_type>
	  HOST void gradient_check_mlp(zinhart::loss_functions::loss_function<precision_type> * loss,
					    std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
					    const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
					    precision_type * total_activations, const std::uint32_t total_activations_length,
					    precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
					    const precision_type * total_bias, 
					    precision_type * numerically_approx_gradient, 
					    const precision_type limit_epsilon, 
					    const std::uint32_t n_threads, const std::uint32_t thread_id);

	template<class precision_type>
	  void bprop_mlp(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers, 
					 const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
					 precision_type * total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
					 const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
					 const precision_type * const total_bias,
					 const std::uint32_t n_threads,
					 const std::uint32_t thread_id);

	template<class precision_type>
	  void online_train(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
					   std::shared_ptr< zinhart::loss_functions::loss_function<precision_type> > loss_function,
					   std::shared_ptr< zinhart::optimizers::optimizer<precision_type> > optimizer,
					   precision_type * total_training_cases_ptr,  const std::uint32_t total_training_cases_length,
					   const precision_type * const total_targets_ptr, precision_type * total_error_ptr,
					   precision_type * total_activations_ptr, precision_type * total_deltas_ptr, const std::uint32_t total_activations_length,
					   precision_type * total_hidden_weights_ptr, precision_type * total_gradient_ptr, const std::uint32_t total_hidden_weights_length,
					   precision_type * total_bias_ptr,
					   const std::uint32_t max_epochs = 1,
					   bool verbose = true,
					   std::ostream & output_stream = std::cout
					  );

	// defaults to single threaded and online training
	template<class precision_type>
	  void batch_train(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
					   std::shared_ptr< zinhart::loss_functions::loss_function<precision_type> > loss_function,
					   std::shared_ptr< zinhart::optimizers::optimizer<precision_type> > optimizer,
					   precision_type * total_training_cases_ptr,  const std::uint32_t total_training_cases_length,
					   const precision_type * const total_targets_ptr, precision_type * total_error_ptr,
					   precision_type * total_activations_ptr, precision_type * total_deltas_ptr, const std::uint32_t total_activations_length,
					   precision_type * total_hidden_weights_ptr, precision_type * total_gradient_ptr, const std::uint32_t total_hidden_weights_length,
					   precision_type * total_bias_ptr,
					   const std::uint32_t max_epochs = 1,
					   const std::uint32_t batch_size = 2,
					   bool verbose = true,
					   std::ostream & output_stream = std::cout
					  );

  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
#include "ext/multi_layer_perceptron.tcc"
#endif
