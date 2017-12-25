#include "ann/ann.hh"
#include "ann/typedefs.cuh"
#include <cstdio>
namespace zinhart
{
  //explicit instantiations
  template HOST std::vector<LAYER_INFO> get_total_layers(const ann<ffn> & model);
  template HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_observations(const ann<ffn> & model);
  template HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_hidden_weights(const ann<ffn> & model);
  template HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_activations(const ann<ffn> & model);
  template HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_error(const ann<ffn> & model);
  template HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_gradient(const ann<ffn> & model);
  template HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_deltas(const ann<ffn> & model);

  template HOST void add_layer(ann<ffn> & model, const LAYER_INFO & ith_layer);
  template HOST int initialize_model(ann<ffn> & model,  
						     std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_targets
							);
  template HOST int cleanup(ann<ffn> & model);
  template HOST int train(ann<ffn> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);

  //definitions
  template <class T>
	HOST std::vector<LAYER_INFO> get_total_layers(const ann<T> & model)
	{return model.get_total_layers(); }
  template <class T>
	HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_observations(const ann<T> & model)
	{return model.get_total_observations();}
  template <class T>
	HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_hidden_weights(const ann<T> & model)
	{return model.get_total_hidden_weights();}
  template <class T>
	HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_activations(const ann<T> & model)
	{return model.get_total_activations();}
  template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_error(const ann<T> & model)
	  {return model.get_total_error();}
  template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_gradient(const ann<T> & model)
	  {return model.get_total_gradient();}
  template <class T>
	  HOST std::pair<std::uint32_t, std::shared_ptr<double>> get_total_deltas(const ann<T> & model)
	  {return model.get_total_deltas();}





  template<class T>
	HOST void add_layer(ann<T> & model, const LAYER_INFO & ith_layer)
	{ model.add_layer(ith_layer);}
  template<class T>
	HOST int initialize_model(ann<T> & model,  
						   std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
						   std::pair<std::uint32_t, std::shared_ptr<double>> & total_targets
						  )
	  {
#if CUDA_ENABLED == 1 
		std::cerr<<"CUDA ENABLED INITIALIZE_NETWORK\n";
#else
		std::cerr<<"CUDA DISABLED INITIALIZE_NETWORK\n";
#endif
	  return model.init(total_observations, total_targets);
	  }
  template<class T>
	HOST int train(ann<T> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty)
	{
#if CUDA_ENABLED == 1 
	  std::cerr<<"CUDA ENABLED TRAIN\n";
#else
	  std::cerr<<"CUDA DISABLED TRAIN\n";
#endif
	  return model.train(epochs, batch_size, weight_penalty);
	}
  template<class T>
	HOST int cleanup(ann<T> & model)
	{
#if CUDA_ENABLED == 1 
	  std::cerr<<"CUDA ENABLED CLEANUP\n";
#else
	  std::cerr<<"CUDA DISABLED CLEANUP\n";
#endif
	  return model.cuda_cleanup();
	}
}
