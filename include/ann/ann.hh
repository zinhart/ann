#ifndef ANN_HH
#define ANN_HH
#include <multi_core/multi_core.hh>
#include <multi_core/multi_core_error.hh>
#include <ann/loss_function.hh>
#include <ann/layer.hh>
#include <ann/optimizer.hh>
#include <vector>
#if CUDA_ENABLED == true
#define ERROR_CUDA_ERROR 1
#include <cublas_v2.h>
#else
#include "mkl.h"
#endif
namespace zinhart
{
  namespace models
  {
	namespace architecture
	{
	  enum multi_layer_perceptron          : std::uint32_t;
	  enum restricted_boltzman_machine     : std::uint32_t;
	  enum deep_belief_network             : std::uint32_t;
	  enum convolutional_neural_network    : std::uint32_t;
	  enum recurrent_neural_network        : std::uint32_t;
	  enum general_adversarial_network     : std::uint32_t;
	  enum echo_state_network              : std::uint32_t;
	  union architectures
	  {
		multi_layer_perceptron             mlp;     
		restricted_boltzman_machine        rbm;
		deep_belief_network                dbn;
		convolutional_neural_network       cnn;
		recurrent_neural_network           rnn;
		general_adversarial_network        gan;
		echo_state_network                 esn;
	  };
	  namespace connection
	  {
		enum dense  : std::uint32_t;
		enum sparse : std::uint32_t;
	  }
	}
	template <class precision_type>
	  class ann
	  {
		protected:
#if CUDA_ENABLED == /*MULTI_CORE_DISABLED*/ false
		  std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> optimizer;
		  std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> loss_function;
		  std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > total_layers;

		  HOST virtual void add_layer_impl(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer) = 0;
		  HOST virtual void remove_layer_impl(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer) = 0;
		  HOST virtual void set_optimizer_impl(const std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> & op) = 0;
		  HOST virtual void set_loss_function_inpl(const std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> & loss_function) = 0;
		  HOST virtual void init_impl() = 0;

		  HOST virtual void forward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
												   const precision_type * total_training_cases, const std::uint32_t case_index,
												   precision_type * tot_activations, const std::uint32_t tot_activations_length,
												   const precision_type * total_hidden_wts, const std::uint32_t total_hidden_wts_length,
												   const precision_type * total_bias,
												   const std::uint32_t n_threads,
												   const std::uint32_t thread_id);

		  HOST virtual void get_outputs_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
										     const precision_type * tot_activations, const std::uint32_t tot_activations_length, 
										     precision_type * model_outputs, 
										     const std::uint32_t n_threads, 
										     const std::uint32_t thread_id
									        );

		  HOST virtual void gradient_check_impl(zinhart::loss_functions::loss_function<precision_type> * loss,
										        const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
										        const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
										        precision_type * tot_activations, const std::uint32_t tot_activations_length,
										        precision_type * const total_hidden_wts, const std::uint32_t total_hidden_wts_length,
										        const precision_type * total_bias, 
										        precision_type * numerically_approx_gradient, 
										        const precision_type limit_epsilon, 
										        const std::uint32_t n_threads, 
												const std::uint32_t thread_id
										       );

		  HOST virtual void backward_propagate_impl(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
											        const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
											        const precision_type * const tot_activations, precision_type * total_deltas, const std::uint32_t tot_activations_length,
											        const precision_type * const total_hidden_wts, precision_type * tot_grad, const std::uint32_t total_hidden_wts_length,
											        const precision_type * const total_bias,
											        const std::uint32_t n_threads,
											        const std::uint32_t thread_id
			                                       );



#endif
		  // functions independant of cpu and gpu
		  HOST virtual void train_impl(bool verbose) = 0;
		  HOST virtual std::uint32_t get_total_hidden_weights_impl()const = 0;
		  HOST virtual std::uint32_t get_total_activations_impl()const = 0;
		public:
// outer interface	
#if CUDA_ENABLED == /*MULTI_CORE_DISABLED*/ false
		  HOST void forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
			  						  const precision_type * total_training_cases, const std::uint32_t case_index,
			  						  precision_type * tot_activations, const std::uint32_t tot_activations_length,
			  						  const precision_type * total_hidden_wts, const std::uint32_t total_hidden_wts_length,
			  						  const precision_type * total_bias,
			  						  const std::uint32_t n_threads = 1,
			  						  const std::uint32_t thread_id = 0
			  						 );
		  HOST void get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
			  					const precision_type * tot_activations, const std::uint32_t tot_activations_length, 
			  					precision_type * model_outputs, 
			  					const std::uint32_t n_threads = 1, 
			  					const std::uint32_t thread_id = 0
			  				   );

		  HOST void gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
			  					   const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers,
			  					   const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
			  					   precision_type * tot_activations, const std::uint32_t tot_activations_length,
			  					   precision_type * const total_hidden_wts, const std::uint32_t total_hidden_wts_length,
			  					   const precision_type * total_bias, 
			  					   precision_type * numerically_approx_gradient, 
			  					   const precision_type limit_epsilon, 
			  					   const std::uint32_t n_threads = 1,
								   const std::uint32_t thread_id = 0
			  					  );
		  HOST void backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<precision_type> > > & total_layers, 
									   const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
									   const precision_type * const tot_activations, precision_type * total_deltas, const std::uint32_t tot_activations_length,
									   const precision_type * const total_hidden_wts, precision_type * tot_grad, const std::uint32_t total_hidden_wts_length,
									   const precision_type * const total_bias,
									   const std::uint32_t n_threads = 1,
									   const std::uint32_t thread_id = 0
			                          );	  
		  HOST void add_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer);
		  HOST void remove_layer(const std::shared_ptr<zinhart::models::layers::layer<precision_type>> & layer);
		  HOST void set_optimizer(const std::shared_ptr<zinhart::optimizers::optimizer<precision_type>> & op);
		  HOST void set_loss_function(const std::shared_ptr<zinhart::loss_functions::loss_function<precision_type>> & loss_function);
		 
#endif
		  HOST void init();
		  HOST std::uint32_t get_total_hidden_weights()const;
		  HOST std::uint32_t get_total_activations()const;
	  };
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
#include <ann/ext/ann.tcc>
#include <ann/models/ann_mlp.hh>
#endif
