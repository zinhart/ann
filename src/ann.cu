#include "ann/ann.hh"
#include "ann/typedefs.cuh"
#include <cstdio>
namespace zinhart
{
  //explicit instantiations
  template std::vector<LAYER_INFO> get_total_layers(const ann<ffn> & model);
  template std::pair<std::uint32_t, std::shared_ptr<double>> get_total_observations(const ann<ffn> & model);
  template std::pair<std::uint32_t, std::shared_ptr<double>> get_total_hidden_weights(const ann<ffn> & model);
  template std::pair<std::uint32_t, std::shared_ptr<double>> get_total_activations(const ann<ffn> & model);

  template void add_layer(ann<ffn> & model, const LAYER_INFO & ith_layer);
  template int initialize_network(ann<ffn> & model,  
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
							 std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets
							);
  template int cleanup(ann<ffn> & model);
  template int train(ann<ffn> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty);

  //definitions
  template <class T>
	std::vector<LAYER_INFO> get_total_layers(const ann<T> & model)
	{return model.get_total_layers(); }
  template <class T>
	std::pair<std::uint32_t, std::shared_ptr<double>> get_total_observations(const ann<T> & model)
	{return model.get_total_observations();}
  template <class T>
	std::pair<std::uint32_t, std::shared_ptr<double>> get_total_hidden_weights(const ann<T> & model)
	{return model.get_total_hidden_weights();}
  template <class T>
	std::pair<std::uint32_t, std::shared_ptr<double>> get_total_activations(const ann<T> & model)
	{return model.get_total_activations();}

  template<class T>
	void add_layer(ann<T> & model, const LAYER_INFO & ith_layer)
	{ model.add_layer(ith_layer);}
  template<class T>
	int initialize_network(ann<T> & model,  
						   std::pair<std::uint32_t, std::shared_ptr<double>> & total_observations,
						   std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets
						  )
	  {
#if CUDA_ENABLED == 1 
	  printf("CUDA ENABLED INITIALIZE_NETWORK\n");
#else
	  printf("CUDA DISABLED INITIALIZE_NETWORK\n");
#endif
	  return model.init(total_observations, total_targets);
	  }
  template<class T>
	int train(ann<T> & model, const std::uint16_t & epochs, const std::uint32_t & batch_size, const float & weight_penalty)
	{
#if CUDA_ENABLED == 1 
	  printf("CUDA ENABLED TRAIN\n");
#else
	  printf("CUDA DISABLED TRAIN\n");
#endif
	  return model.train(epochs, batch_size, weight_penalty);
	}
  template<class T>
	int cleanup(ann<T> & model)
	{
#if CUDA_ENABLED == 1 
	  printf("CUDA ENABLED CLEANUP\n");
#else
	  printf("CUDA DISABLED CLEANUP\n");
#endif
	  return model.cuda_cleanup();
	}
}
