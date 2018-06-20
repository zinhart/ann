#include "ann/loss_function.hh"
#include "ann/typedefs.cuh"

namespace zinhart
{
  // explicit template instantiations
  template HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, float * device_total_outputs, std::uint32_t device_id);

  template HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, double * device_total_outputs, std::uint32_t device_id);


	template HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, float * device_total_outputs, const float & epsilon, std::uint32_t device_id);


	template HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, double * device_total_outputs, const double & epsilon, std::uint32_t device_id);

  // template definitions
  template <class Precision_Type>
	HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, Precision_Type * device_total_outputs, std::uint32_t device_id)
	{
	  dim3 num_blocks;
	  dim3 threads_per_block;
	}


  template <class Precision_Type>
	HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, Precision_Type * device_total_outputs, const Precision_Type & epsilon, std::uint32_t device_id)
	{
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  if(loss_function_name == LOSS_FUNCTION_NAME::MSE)
	  {
	  }
	  else if (loss_function_name == LOSS_FUNCTION_NAME::CROSS_ENTROY_MULTI_CLASS)
	  {
	  }

	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()),__FILE__,__LINE__);
	}
}

