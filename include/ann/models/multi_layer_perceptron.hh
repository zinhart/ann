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
	template<class precision_type>
  	  HOST void fprop_mlp(std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
			   precision_type * total_cases_ptr, const std::uint32_t ith_training_case,
			   precision_type * total_activations_ptr, const std::uint32_t activations_length,
			   precision_type * total_hidden_weights_ptr, const std::uint32_t weights_length,
			   precision_type * total_bias_ptr,
			   const std::uint32_t total_threads, const std::uint32_t thread_index
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

  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
#include "ext/multi_layer_perceptron.tcc"
#endif
