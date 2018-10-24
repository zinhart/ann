namespace zinhart
{
  namespace models
  {
// outer interface	
#if CUDA_ENABLED == /*MULTI_CORE_DISABLED*/ false
	template <class precision_type>
	  HOST void ann<precision_type>::forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
													   const precision_type * total_training_cases, const std::uint32_t case_index,
													   precision_type * tot_activations, const std::uint32_t tot_activations_length,
													   const precision_type * total_hidden_wts, const std::uint32_t total_hidden_wts_length,
													   const precision_type * total_bias,
													   const std::uint32_t n_threads,
													   const std::uint32_t thread_id
			  						                   )
	  {	forward_propagate_impl(total_layers, total_training_cases, case_index, tot_activations, tot_activations_length, total_hidden_wts, total_hidden_wts_length, total_bias, n_threads, thread_id); }

	template <class precision_type>
	  HOST void ann<precision_type>::get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
												 const precision_type * tot_activations, const std::uint32_t tot_activations_length, 
												 precision_type * model_outputs, 
												 const std::uint32_t n_threads, 
												 const std::uint32_t thread_id
											    )
	  { get_outputs_impl(total_layers, tot_activations, tot_activations_length, model_outputs, n_threads, thread_id); }


	template <class precision_type>
	  HOST void ann<precision_type>::gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
												    const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
												    const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
												    precision_type * tot_activations, const std::uint32_t tot_activations_length,
												    precision_type * const total_hidden_wts, const std::uint32_t total_hidden_wts_length,
												    const precision_type * total_bias, 
												    precision_type * num_approx_grad, 
												    const precision_type limit_epsilon, 
												    const std::uint32_t n_threads, 
													const std::uint32_t thread_id
												  )
	  {	gradient_check_impl(loss, total_layers, total_training_cases, total_targets, case_index, tot_activations, tot_activations_length, total_hidden_wts, total_hidden_wts_length, total_bias, num_approx_grad, limit_epsilon, n_threads, thread_id); }

	template <class precision_type>
	   HOST void ann<precision_type>::backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
													     const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
													     const precision_type * const tot_activations, precision_type * total_deltas, const std::uint32_t tot_activations_length,
													     const precision_type * const total_hidden_wts, precision_type * tot_grad, const std::uint32_t total_hidden_wts_length,
													     const precision_type * const total_bias,
													     const std::uint32_t n_threads,
													     const std::uint32_t thread_id
													    )
	   { backward_propagate_impl(total_layers, total_training_cases, total_targets, d_error, case_index, tot_activations, total_deltas, tot_activations_length, total_hidden_wts, tot_grad, total_hidden_wts_length, total_bias, n_threads, thread_id);}		 

	template <class precision_type>
  	  HOST void ann<precision_type>::add_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer)
	  { add_layer_impl(layer); }

	template <class precision_type>
	  HOST void ann<precision_type>::remove_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer)
	  { remove_layer(layer); }

	template <class precision_type>
  	  HOST void ann<precision_type>::set_optimizer(const std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> & op)
	  { set_optimizer_impl(op); }

	template <class precision_type>
	  HOST void ann<precision_type>::set_loss_function(const std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> & loss_function)
	  { set_loss_function_inpl(loss_function); }
		 
#endif
	
	template <class precision_type>	  
	   HOST void ann<precision_type>::init()
	   { init_impl(); }

	template <class precision_type>
	   HOST std::uint32_t ann<precision_type>::get_total_hidden_weights()const
	   { return get_total_hidden_weights(); }

	template <class precision_type>
	  HOST std::uint32_t ann<precision_type>::get_total_activations()const
	  {return get_total_activations_impl(); }
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
