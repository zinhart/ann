#include "ann/ann.hh"
#include "ann/typedefs.cuh"
#include <cstdio>
namespace zinhart
{
  //explicit instantiations
  template int initialize_network(ann<ffn> & model,  
						     const std::uint16_t & case_size,
							 std::pair<std::uint32_t, std::shared_ptr<float>> & total_observations,
							 std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets,
							 std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights
							);



  template int cleanup(ann<ffn> & model);

  //definitions
  template<class T>
	int initialize_network(ann<T> & model,  
                    	   const std::uint16_t & case_size,
						   std::pair<std::uint32_t, std::shared_ptr<float>> & total_observations,
						   std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets,
						   std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights
						  )
	  {
#if CUDA_ENABLED == 1 
	  printf("CUDA ENABLED SET_CASE_INFO\n");
#else
	  
	  printf("CUDA DISABLED SET_CASE_INFO\n");
#endif
	  return model.init(case_size, total_observations, total_targets, total_hidden_weights);
	  }
  template<class T>
	int cleanup(ann<T> & model)
	{
#if CUDA_ENABLED == 1 
	  printf("CUDA ENABLED CLEANUP\n");
#else
	  
	  printf("CUDA DISABLED CLEANUP\n");
#endif
	  model.cuda_cleanup();
	}

}
