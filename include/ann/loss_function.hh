#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H
#include "typedefs.cuh"
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  class loss_function
  {
	public:
	  loss_function() = default;
	  loss_function(const loss_function &) = default;
	  loss_function(loss_function &&) = default;
	  loss_function & operator = (const loss_function &) = default;
	  loss_function & operator = (loss_function &&) = default;
	  ~loss_function() = default;
	  enum F : char {OBJECTIVE = 0};
	  enum DF : char {DERIVATIVE = 1};
	  CUDA_CALLABLE_MEMBER double mse(pt kth_target, pt kth_output, F OBJECTIVE)
	  { return (kth_output - kth_target) * (kth_output - kth_target); }
	  CUDA_CALLABLE_MEMBER double mse(pt kth_target, pt kth_output, DF DERIVATIVE)
	  {	return double(2.0) * (kth_output - kth_target);}
	  CUDA_CALLABLE_MEMBER double cross_entropy_multi_class(pt kth_target, pt kth_output, F OBJECTIVE, pt epsilon = 1.e-30)
	  {	return kth_target * log(kth_output + epsilon); }
	  CUDA_CALLABLE_MEMBER double cross_entropy_multi_class(pt kth_target, pt kth_output, DF DERIVATIVE)
	  {return kth_output - kth_target;}
  };
}
#endif  

