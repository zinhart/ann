#include "ann/ann.hh"
#include "ann/typedefs.cuh"
#include <cstdio>
namespace zinhart
{
  template <class model_type>
	void set_case_info(model_type m, std::pair<std::uint32_t, std::shared_ptr<float>> & total_observations, 
							std::pair<std::uint32_t, std::shared_ptr<float>> & total_targets, 
							std::pair<std::uint32_t, std::shared_ptr<double>> & total_hidden_weights, 
							std::uint32_t & case_size);
  /*void cuda_init(Host_Members & hm)
  {
#if CUDA_ENABLED == 1
//	std::cout<<"hello from cuda\n";
#else
//	std::cout<<"you fucked up\n";
#endif
  }*/

}
