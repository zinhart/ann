#ifndef MULTI_LAYER_PERCEPTRON_HH
#define MULTI_LAYER_PERCEPTRON_HH
#include "ann_mlp.hh"
namespace zinhart
{
  namespace models
  {	
	
	template<class precision_type>
  	  class multi_layer_perceptron 
	  {
		public:
		  multi_layer_perceptron() = default;
		  multi_layer_perceptron(const multi_layer_perceptron &) = default;
		  multi_layer_perceptron(multi_layer_perceptron &&) = default;
		  multi_layer_perceptron & operator = (const multi_layer_perceptron &) = default;
		  multi_layer_perceptron & operator = (multi_layer_perceptron &&) = default;
		  ~multi_layer_perceptron() = default;
		  // defaults to single threaded
		  HOST void gradient_check(zinhart::loss_functions::loss_function<precision_type> * loss,
								   const std::vector< std::shared_ptr< zinhart::models::layers::layer<double> > > & total_layers,
								   const precision_type * total_training_cases, const precision_type * total_targets, const std::uint32_t case_index,
								   precision_type * total_activations, const std::uint32_t total_activations_length,
								   precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
								   const precision_type * total_bias, 
								   precision_type * numerically_approx_gradient, 
			                       const precision_type limit_epsilon = 1.e-4, 
								   const std::uint32_t n_threads = 1, const std::uint32_t thread_id = 0
								  );

#if CUDA_ENABLED == 1
#else 
			// Defaults to single-threaded
			void forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<double> > > & total_layers,
								   const precision_type * total_training_cases, const std::uint32_t  case_index,
								   precision_type * total_activations, const std::uint32_t total_activations_length,
								   const precision_type * total_hidden_weights, const std::uint32_t total_hidden_weights_length,
								   const precision_type * total_bias,
							       const std::uint32_t n_threads = 1,
								   const std::uint32_t thread_id = 0
								  );

			// Defaults to single-threaded
			void get_outputs(const std::vector< std::shared_ptr< zinhart::models::layers::layer<double> > > & total_layers, 
							 const precision_type * total_hidden_outputs, const std::uint32_t total_hidden_outputs_length,
							 precision_type * model_outputs, 
							 const std::uint32_t n_threads = 1, 
							 const std::uint32_t thread_id = 0
							);

			// Defaults to single threaded
			void backward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<double> > > & total_layers, 
									const precision_type * const total_training_cases, const precision_type * const total_targets, const precision_type * const d_error, const std::uint32_t case_index,
									const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
									const precision_type * const total_hidden_weights, precision_type * total_gradient, const std::uint32_t total_hidden_weights_length,
									const precision_type * const total_bias,
									const std::uint32_t n_threads = 1,
									const std::uint32_t thread_id = 0
								 );

#endif
		};// END CLASS MULTI_LAYER_PERCEPTRON
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
#include "ext/multi_layer_perceptron.tcc"
#endif
