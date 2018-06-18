#include "ann/loss_function.hh"
#include "ann/typedefs.cuh"

namespace zinhart
{
  // explicit template instantiations
  template HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, float * device_total_outputs);

  // explicit template instantiations
  template HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, double * device_total_outputs);

  // template definitions
  template <class Precision_Type>
	HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, Precision_Type * device_total_outputs);
}

