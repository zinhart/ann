#include "ann/ann.hh"
#include "ann/typedefs.cuh"
#include <cstdio>
namespace zinhart
{
  //explicit instantiations
  template int set_case_info(ann<ffn> & model, std::uint32_t & n_observations, 
						   std::uint32_t & n_targets, 
						   std::uint32_t & n_hidden_weights, 
						   std::uint16_t & case_size);
  //definitions
  template <class T>
	  int set_case_info(ann<T> & model, std::uint32_t & n_observations,  
						   std::uint32_t & n_targets, 
						   std::uint32_t & n_hidden_weights,  
						   std::uint16_t & case_size)
	  {
#if CUDA_ENABLED == 1 
	  printf("CUDA ENABLED SET_CASE_INFO\n");
#else
	  
	  printf("CUDA DISABLED SET_CASE_INFO\n");
#endif
	  return model.set_case_info(n_observations, n_targets, n_hidden_weights, case_size);

	  }

}
