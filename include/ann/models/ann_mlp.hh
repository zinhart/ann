#ifndef ANN_MLP_HH
#define ANN_MLP_HH
namespace zinhart
{
  namespace models
  {
	template<class precision_type>
	  class ann_mlp : public ann<precision_type>
	  {
		private:
		  using ann<precision_type>::optimizer;
		  using ann<precision_type>::loss_function;
		  using ann<precision_type>::total_layers;

		  // model pointers
		  precision_type * total_activations_ptr;
		  precision_type * total_deltas_ptr;
		  precision_type * total_hidden_weights_ptr;
		  precision_type * total_gradient_ptr;
		  precision_type * total_bias_ptr;
		  // vector length variables
		  std::uint32_t total_activations_length;
		  std::uint32_t total_hidden_weights_length;
		  std::uint32_t total_bias_length;
		  // loop counters
		  std::uint32_t ith_layer;
		  std::uint32_t ith_batch;
		  // thread variables
		  std::uint32_t thread_id;
		  std::uint32_t n_threads;
		  std::uint32_t batch_size;

		  std::uint32_t alignment;

		  HOST void safe_allocate(std::uint32_t total_activations_length, std::uint32_t total_hidden_weights_length);
		  HOST void safe_deallocate();

		  HOST virtual void add_layer_impl(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer) override;
		  HOST virtual void remove_layer_impl(std::uint32_t index) override;
		  HOST virtual std::uint32_t size_impl()const override;
		  HOST virtual void set_optimizer_impl(const std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> & op) override;
		  HOST virtual void set_loss_function_impl(const std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> & loss_function) override;
		  HOST virtual void init_impl(std::uint32_t n_threads) override;
		protected:
#if CUDA_ENABLED == MULTI_CORE_DISABLED

		  HOST virtual void forward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
												   const precision_type * total_training_cases, const std::uint32_t case_index,
												   precision_type * total_activations, const std::uint32_t total_activations_length,
												   const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
												   const precision_type * total_bias,
												   const std::uint32_t n_threads,
												   const std::uint32_t thread_id) override
		  {}

		  HOST virtual void get_outputs_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
										     const precision_type * total_activations, const std::uint32_t total_activations_length, 
										     precision_type * model_outputs, 
										     const std::uint32_t n_threads, 
										     const std::uint32_t thread_id
									        ) override
		  {}

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
										       ) override
		  {}

		  HOST virtual void backward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
											        const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
											        const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
											        const precision_type * const total_hidden_weights, precision_type * tot_grad, const std::uint32_t total_hidden_weights_length,
											        const precision_type * const total_bias,
											        const std::uint32_t n_threads,
											        const std::uint32_t thread_id
			                                       ) override
		  {}



#endif
		  // functions independant of cpu and gpu
		  HOST virtual void train_impl(bool verbose);
		public:
          ann_mlp(std::uint32_t batch_size = 1, std::uint32_t n_threads = 1);
          ann_mlp(const ann_mlp&)= delete;
          ann_mlp(ann_mlp &&) = delete;
          const ann_mlp & operator = (const ann_mlp &)= delete;
          const ann_mlp & operator = (ann_mlp&&) = delete;
		  ~ann_mlp();
		  HOST void add_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer);
		  HOST void remove_layer(std::uint32_t index);
		  HOST std::uint32_t size()const;
		  HOST void init(std::uint32_t n_threads = 1);
		  HOST std::uint32_t total_activations()const;
		  HOST std::uint32_t total_hidden_weights()const;
		  HOST void train(bool verbose);
		  HOST const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & operator [] (std::uint32_t index) const;
		  HOST void forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
									  const precision_type * total_training_cases, const std::uint32_t case_index,
									  precision_type * tot_activations, const std::uint32_t total_activations_length,
									  const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
									  const precision_type * total_bias,
									  const std::uint32_t n_threads,
									  const std::uint32_t thread_id
			                         );
		  HOST void get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
							    const precision_type * tot_activations, const std::uint32_t total_activations_length, 
							    precision_type * model_outputs, 
							    const std::uint32_t n_threads, 
							    const std::uint32_t thread_id
			   		           );
	
		  HOST void gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
								   const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
								   const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
								   precision_type * tot_activations, const std::uint32_t total_activations_length,
								   precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
								   const precision_type * total_bias, 
								   precision_type * num_approx_grad, 
								   const precision_type limit_epsilon, 
								   const std::uint32_t n_threads, 
								   const std::uint32_t thread_id
			   			          );

   		  HOST void backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
									   const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
									   const precision_type * const tot_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
									   const precision_type * const total_hidden_weights, precision_type * tot_grad, const std::uint32_t total_hidden_weights_length,
									   const precision_type * const total_bias,
									   const std::uint32_t n_threads,
									   const std::uint32_t thread_id
									  );


	  };

  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART 
#include <ann/models/ext/ann_mlp.tcc>
#include <ann/models/multi_layer_perceptron.hh>
#endif
