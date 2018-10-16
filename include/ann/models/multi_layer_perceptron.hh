#ifndef MULTI_LAYER_PERCEPTRON_HH
#define MULTI_LAYER_PERCEPTRON_HH
#include "ann_mlp.hh"
namespace zinhart
{
  namespace models
  {	
	// base template
	template<connection c, class precision_type>
	  class multi_layer_perceptron;
	
	template<class precision_type>
  	  class multi_layer_perceptron<connection::dense, precision_type> : private ann< architecture::mlp_dense, precision_type>
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
								   precision_type * total_hidden_inputs, precision_type * total_activations, const std::uint32_t total_activations_length,
								   precision_type * const total_hidden_weights, const std::uint32_t total_hidden_weights_length,
								   const precision_type * total_bias, 
								   precision_type * numerically_approx_gradient, 
			                       const precision_type limit_epsilon = 1.e-4, 
								   const std::uint32_t n_threads = 1, const std::uint32_t thread_id = 0
								  );

#if CUDA_ENABLED == 1
		  template <class LOSS_FUNCTION>
			HOST std::int32_t forward_propagate_async(const bool & copy_device_to_host, 
								   const cudaStream_t & stream, const cublasHandle_t & context, LOSS_FUNCTION error_metric,
								   const std::uint32_t & ith_observation_index, const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
								   const std::uint32_t & total_targets, const double * host_total_targets, const double * device_total_targets,
								   const std::uint32_t & total_hidden_weights, const double * host_total_hidden_weights, const double * device_total_hidden_weights, 
								   const std::uint32_t & total_activations, double * host_total_activations, double * device_total_activations,
								   const double * device_total_observations, double * device_total_outputs,
								   const double * host_total_bias, std::uint32_t device_id = 0
								  );

			HOST void backward_propagate(cublasHandle_t & context);



#else 
			// Defaults to single-threaded
			void forward_propagate(const std::vector< std::shared_ptr< zinhart::models::layers::layer<double> > > & total_layers,
								   const precision_type * total_training_cases, const std::uint32_t  case_index,
								   precision_type * total_hidden_inputs, precision_type * total_activations, const std::uint32_t total_activations_length,
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
									const precision_type * const total_hidden_inputs, const precision_type * const total_activations, precision_type * total_deltas, const std::uint32_t total_activations_length,
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
