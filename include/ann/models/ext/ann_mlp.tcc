namespace zinhart
{
  namespace models
  {
	template<class precision_type>
	  HOST void ann_mlp<precision_type>::add_layer_impl(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer)
	  {	total_layers.push_back(layer); }

	template<class precision_type>
	  HOST void ann_mlp<precision_type>::remove_layer_impl(std::uint32_t index)
	  {
		assert(index < total_layers.size());
		total_layers.erase(total_layers.begin() + index);
	  }
	template <class precision_type>
	  HOST std::uint32_t ann_mlp<precision_type>::size_impl()const
	  {	return total_layers.size(); }

	template<class precision_type>
  	  HOST void ann_mlp<precision_type>::set_optimizer_impl(const std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> & op)
	  { optimizer = op; }

	template<class precision_type>
	  HOST void ann_mlp<precision_type>::set_loss_function_impl(const std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> & loss_function)
	  { this->loss_function = loss_function; }

	template<class precision_type>
	  HOST void ann_mlp<precision_type>::init_impl()
	  {
	  }
	template<class precision_type>
	  HOST void ann_mlp<precision_type>::train_impl(bool verbose)
	  {
	  }
	template<class precision_type>
	  HOST std::uint32_t ann_mlp<precision_type>::get_total_hidden_weights()const
	  {
	  }
	template<class precision_type>
	  HOST std::uint32_t ann_mlp<precision_type>::get_total_activations()const 
	  {
	  }


#if CUDA_ENABLED == MULTI_CORE_DISABLED

	template <class precision_type>
	  HOST void ann_mlp<precision_type>::forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
													   const  precision_type * total_training_cases, const std::uint32_t case_index,
													   precision_type * tot_activations, const std::uint32_t tot_activations_length,
													   const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
													   const precision_type * total_bias,
													   const std::uint32_t n_threads,
													   const std::uint32_t thread_id
			  						                   )
	  {	forward_propagate_impl(total_layers, total_training_cases, case_index, tot_activations, tot_activations_length, total_hidden_weights, total_hidden_weights_length, total_bias, n_threads, thread_id); }

	template <class precision_type>
	  HOST void ann_mlp<precision_type>::get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
												 const precision_type * tot_activations, const std::uint32_t tot_activations_length, 
												 precision_type * model_outputs, 
												 const std::uint32_t n_threads, 
												 const std::uint32_t thread_id
											    )
	  { get_outputs_impl(total_layers, tot_activations, tot_activations_length, model_outputs, n_threads, thread_id); }


	template <class precision_type>
	  HOST void ann_mlp<precision_type>::gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
												    const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
												    const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
												    precision_type * tot_activations, const std::uint32_t tot_activations_length,
												    precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
												    const precision_type * total_bias, 
												    precision_type * num_approx_grad, 
												    const precision_type limit_epsilon, 
												    const std::uint32_t n_threads, 
													const std::uint32_t thread_id
												  )
	  {	gradient_check_impl(loss, total_layers, total_training_cases, total_targets, case_index, tot_activations, tot_activations_length, total_hidden_weights, total_hidden_weights_length, total_bias, num_approx_grad, limit_epsilon, n_threads, thread_id); }

	template <class precision_type>
	   HOST void ann_mlp<precision_type>::backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
													     const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
													     const precision_type * const tot_activations, precision_type * total_deltas, const std::uint32_t tot_activations_length,
													     const precision_type * const total_hidden_weights, precision_type * tot_grad, const std::uint32_t total_hidden_weights_length,
													     const precision_type * const total_bias,
													     const std::uint32_t n_threads,
													     const std::uint32_t thread_id
													    )
	   { backward_propagate_impl(total_layers, total_training_cases, total_targets, d_error, case_index, tot_activations, total_deltas, tot_activations_length, total_hidden_weights, tot_grad, total_hidden_weights_length, total_bias, n_threads, thread_id);}		 
#endif
	template <class precision_type>
	  HOST ann_mlp<precision_type>::ann_mlp()
	  {
	  }
	template <class precision_type>
	  HOST ann_mlp<precision_type>::~ann_mlp()
	  {
	  }
	template <class precision_type>
  	  HOST void ann_mlp<precision_type>::init()
	  {
	  }
	template <class precision_type>
	  HOST const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & ann_mlp<precision_type>::operator [] (std::uint32_t index)const
	  { return total_layers.at(index); } // .at throws exceptionw when a index is out of range
	template <class precision_type>
  	  HOST void ann_mlp<precision_type>::add_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer)
	  { add_layer_impl(layer); }
	template <class precision_type>
	  HOST std::uint32_t ann_mlp<precision_type>::size()const
	  {	return size_impl(); }
	template <class precision_type>
  	  HOST void ann_mlp<precision_type>::remove_layer(std::uint32_t index)
	  { remove_layer_impl(index); }
	
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
