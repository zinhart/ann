namespace zinhart
{
  namespace models
  {
// outer interface	
#if CUDA_ENABLED == /*MULTI_CORE_DISABLED*/ false
	template <class precision_type>
	  HOST void ann<precision_type>::forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
													   const precision_type * total_training_cases, const std::uint32_t case_index,
													   precision_type * total_activations, const std::uint32_t total_activations_length,
													   const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
													   const precision_type * total_bias,
													   const std::uint32_t n_threads,
													   const std::uint32_t thread_id
			  						                   )
	  {
	  }

	template <class precision_type>
	  HOST void ann<precision_type>::get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
												 const precision_type * total_hidden_outputs, const std::uint32_t total_hidden_outputs_length, 
												 precision_type * model_outputs, 
												 const std::uint32_t n_threads, 
												 const std::uint32_t thread_id
											    )
	  {
	  }


	template <class precision_type>
	  HOST void ann<precision_type>::gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
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
	  }

	template <class precision_type>
	   HOST void ann<precision_type>::backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
													     const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
													     const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
													     const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
													     const precision_type * const total_bias,
													     const std::uint32_t n_threads,
													     const std::uint32_t thread_id
													    )
	   {
	   }		 

	template <class precision_type>
  	  HOST void ann<precision_type>::add_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer)
	  {
	  }

	template <class precision_type>
	  HOST void ann<precision_type>::remove_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer)
	  {
	  }

	template <class precision_type>
  	  HOST void ann<precision_type>::set_optimizer(const std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> & op)
	  {
	  }

	template <class precision_type>
	  HOST void ann<precision_type>::set_loss_function(const std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> & loss_function)
	  {
	  }
		 
#endif
	
	template <class precision_type>	  
	   HOST void ann<precision_type>::init()
	   {
	   }

	template <class precision_type>
	   HOST std::uint32_t ann<precision_type>::get_total_hidden_weights()const
	   {
	   }

	template <class precision_type>
	  HOST std::uint32_t ann<precision_type>::get_total_activations()const
	  {
	  }
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
