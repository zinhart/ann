#ifndef ANN_HH
#define ANN_HH
#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "loss_function.hh"
#include "activation.hh"
#include "optimizer.hh"
#include <memory>
//#include <zinhart/vector_space>
#include <vector>
#if CUDA_ENABLED == true
#define ERROR_CUDA_ERROR 1
#include <cublas_v2.h>
//#include <thrust>
#else
#include "mkl.h"
#endif
namespace zinhart
{
  namespace models
  {
	template <class model_type, class precision_type>
	  class ann
	  {
#if CUDA_ENABLED == true
		private:
  		  precision_type * global_device_total_observations;
		  precision_type * global_device_total_targets;
		  precision_type * global_device_total_hidden_weights;
		  precision_type * global_device_total_activations;
		  precision_type * global_device_total_bias;
		  precision_type * global_device_total_error;
		  precision_type * global_device_total_gradient;
		  precision_type * global_device_total_deltas;
#endif
		protected:
		  std::vector<zinhart::activation::LAYER_INFO> total_layers;//Layer types(intput relu sigmoid etc) and the number of inputs of the respective layer 

		  std::uint32_t total_activations_length{0};// this is the sum of all the hidden layers and the output layer neurons * the total number of threads in use
		  std::uint32_t total_deltas_length{0};// same as the number of total_activations_length in the case of one thread, for multiple it is total_activations_lengths/n_threads
		  std::uint32_t total_hidden_weights_length{0};// the number of hidden weights for a layer and the weights themselves
		  std::uint32_t total_gradient_length{0};// same as the total number of hidden weights
		  std::uint32_t total_bias_length{0};// equal to total_layers - 1

		  precision_type * total_activations{nullptr};// this is the sum of all the hidden layers and the output layer neurons * the total number of threads in use
		  precision_type * total_deltas{nullptr};// same as the number of total_activations_length, for multiple it is total_activations_lengths/n_threads
		  precision_type * total_hidden_weights{nullptr};// the number of hidden weights for a layer and the weights themselves
		  precision_type * total_gradient{nullptr};// same as the total number of hidden weights
		  precision_type * total_bias{nullptr};// equal to total_layers - 1
  /*
				   //number of training cases, training case size, the training cases themselves
		  std::pair<std::uint32_t, double *> total_observations;
		  std::pair<std::uint32_t, double *> total_targets; // output layer size and the complete set of targets for each input
		  std::pair<std::uint32_t, double *> total_hidden_weights;// the number of hidden weights for a layer and the weights themselves
		  std::pair<std::uint32_t, double *> total_activations;//this is the sum of all the hidden layers and the output layer neurons
		  std::pair<std::uint32_t, double *> total_error;
		  std::pair<std::uint32_t, double *> total_gradient;
		  std::pair<std::uint32_t, double *> total_deltas;
  */
		public:
		  ann() = default;
		  ann(const ann<model_type, precision_type> &) = delete;
		  ann(ann<model_type, precision_type> &&) = delete;
		  ann<model_type, precision_type> & operator = (const ann<model_type, precision_type>&) = delete;
		  ann<model_type, precision_type> & operator = (ann<model_type, precision_type> &&) = delete;
		  ~ann() = default;
  
		  //debugging functions
		  HOST const std::uint32_t get_total_activations()const;
		  HOST const std::uint32_t get_total_deltas()const;
		  HOST const std::uint32_t get_total_hidden_weights()const;
		  HOST const std::uint32_t get_total_gradients()const;
		  HOST const std::uint32_t get_total_bias()const;
		  //end debugging functions
  
		  //model manipulating functions
		  //I assume the first layer will be an input layer
		  HOST void add_layer(const zinhart::activation::LAYER_INFO & ith_layer);
		  HOST const std::vector<zinhart::activation::LAYER_INFO> get_total_layers()const;
		  HOST void clear_layers();
		  HOST int train(const std::uint16_t & max_epochs, const std::uint32_t & batch_size, const double & weight_penalty);
#if CUDA_ENABLED == 1
		  HOST std::int32_t init(
				   std::pair<std::uint32_t, double *> & total_observations,
				   std::pair<std::uint32_t, double *> & total_targets,
				   std::uint32_t device_id = 0
				  );
		  HOST std::int32_t cuda_init();
		  HOST std::int32_t cuda_cleanup();
		  HOST std::int32_t forward_propagate(const bool & copy_device_to_host, cublasHandle_t & context, 
								 const std::uint32_t & ith_observation_index, const std::vector<zinhart::activation::LAYER_INFO> & total_layers,
								 const std::pair<std::uint32_t, double *> & total_targets, 
								 const std::pair<std::uint32_t, double *> & total_hidden_weights,
								 const std::pair<std::uint32_t, double *> & total_activations,
								 double * device_total_observation, double * device_total_activation, double * device_total_bia, double * device_total_hidden_weight);

#else
		  HOST void init(const std::uint32_t & n_threads = 1);
		  HOST std::int32_t forward_propagate(const std::uint32_t & case_index, const precision_type * total_training_cases, const std::uint32_t & thread_id = 0);
		  HOST void get_model_outputs(precision_type * model_outputs, const std::uint32_t & thread_id = 0);
		  HOST void cleanup();
#endif
	  };
	//Begin External Interface
	//debugging functions
	template <class model_type, class precision_type>
	  HOST std::vector<zinhart::activation::LAYER_INFO> get_total_layers(const ann<model_type, precision_type> & model);
	template <class model_type, class precision_type>
	  HOST std::pair<std::uint32_t, double *> get_total_observations(const ann<model_type, precision_type> & model);
	template <class model_type, class precision_type>
	  HOST std::pair<std::uint32_t, double *> get_total_hidden_weights(const ann<model_type, precision_type> & model);
	template <class model_type, class precision_type>
	  HOST std::pair<std::uint32_t, double *> get_total_activations(const ann<model_type, precision_type> & model);
	template <class model_type, class precision_type>
	  HOST std::pair<std::uint32_t, double *> get_total_error(const ann<model_type, precision_type> & model);
	template <class model_type, class precision_type>
	  HOST std::pair<std::uint32_t, double *> get_total_gradient(const ann<model_type, precision_type> & model);
	template <class model_type, class precision_type>
	  HOST std::pair<std::uint32_t, double *> get_total_deltas(const ann<model_type, precision_type> & model);
	//End debugging functions
	template<class model_type, class precision_type>
  	  HOST void add_layer(ann<model_type, precision_type> & model, const zinhart::activation::LAYER_INFO & ith_layer);

#if CUDA_ENABLED == 1
	template <class model_type, class precision_type>
	  HOST std::int32_t initialize_model(ann<model_type, precision_type> & model,  
							 std::pair<std::uint32_t, double *> & total_observations,
							 std::pair<std::uint32_t, double *> & total_targets, std::uint32_t device_id = 0 );
	template <class model_type, class precision_type>
	  HOST std::int32_t forward_propagate(ann<model_type, precision_type> & model,const bool & copy_device_to_host, cublasHandle_t & context, 
			                   const std::uint32_t & ith_observation_index, const std::vector<LAYER_INFO> & total_layers,
							   const std::pair<std::uint32_t, double *> & total_targets, 
			                   const std::pair<std::uint32_t, double *> & total_hidden_weights,
							   const std::pair<std::uint32_t, double *> & total_activations,
							   double * device_total_observations, double * device_total_activations, double * device_total_bias, double * device_total_hidden_weights);
#endif
	template <class model_type, class precision_type>
	  HOST int train(ann<model_type, precision_type> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);
	template <class model_type, class precision_type>
	  HOST int cleanup(ann<model_type, precision_type> & model);
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
#include "ext/ann.tcc"
#endif
