#ifndef ANN_HH
#define ANN_HH
#include <multi_core/multi_core.hh>
#include <multi_core/multi_core_error.hh>
#include <ann/loss_function.hh>
#include <ann/layer.hh>
//#include <ann/activation.hh>
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
	enum architecture : std::uint32_t {mlp_dense = 0, mlp_sparse, rbm_dense, rbm_sparse, dbn_dense, dbn_sparse, cnn_dense, cnn_sparse, rnn_dense, rnn_sparse, gan_dense, gan_sparse, esn};
	enum connection : std::uint32_t {dense = 0, sparse = 1};
	template <architecture arch, class precision_type>
	  class ann;
	template <class precision_type>
	  class ann<architecture::mlp_dense, precision_type>;
	template <class precision_type>
	  class ann<architecture::mlp_sparse, precision_type>;
	template <class precision_type>
	  class ann<architecture::rbm_dense, precision_type>;
	template <class precision_type>
	  class ann<architecture::rbm_sparse, precision_type>;
	template <class precision_type>
	  class ann<architecture::dbn_dense, precision_type>;
	template <class precision_type>
	  class ann<architecture::dbn_sparse, precision_type>;
	template <class precision_type>
	  class ann<architecture::cnn_dense, precision_type>;
	template <class precision_type>
	  class ann<architecture::cnn_sparse, precision_type>;
	template <class precision_type>
	  class ann<architecture::rnn_dense, precision_type>;
	template <class precision_type>
	  class ann<architecture::rnn_sparse, precision_type>;
	template <class precision_type>
	  class ann<architecture::gan_dense, precision_type>;
	template <class precision_type>
	  class ann<architecture::gan_sparse, precision_type>;
	template <class precision_type>
	  class ann<architecture::esn, precision_type>;

/*
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
	  */
  }// END NAMESPACE MODELS
}// END NAMESPACE ZINHART
#include <ann/models/ann_mlp.hh>
#endif
