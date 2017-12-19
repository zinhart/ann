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
  template<class error_metric>
	class loss_function
	{
	  public:
		enum F : char {OBJECTIVE = 0};
		enum DF : char {DERIVATIVE = 1};
		enum DEL : char {DELTA = 2};
		CUDA_CALLABLE_MEMBER pt operator()(pt kth_target, pt kth_output, F OBJECTIVE, pt epsilon = 1.e-30)
	    {
		  return static_cast<error_metric*>(this)->objective(kth_target,kth_output,epsilon);
		}
		CUDA_CALLABLE_MEMBER pt operator()(pt kth_target, pt kth_output, DF DERIVATIVE)
		{
		  return static_cast<error_metric*>(this)->derivative(kth_target, kth_output);
		}
		CUDA_CALLABLE_MEMBER pt operator()(pt kth_target, pt kth_output, DEL DELTA)
		{
		  return static_cast<error_metric*>(this)->derivative(kth_target, kth_output);
		}
		HOST std::string get_loss_function_label()
		{
		  return static_cast<error_metric*>(this)->get_loss_function_label();
		}
	};
/*
 * now in a templated class or function can have something like loss_function<T> & loss_func
 * template<class foo_metric>
 * void foo_propogate(loss_function<foo_metric> &) //foo metric determines what loss func is used!
 * {
 *	  //do stuff
 * }
*/ 
  class mse : public loss_function<mse>
  {
	public:
	  CUDA_CALLABLE_MEMBER pt objective(pt kth_target, pt kth_output, pt epsilon)
	  {
		return (kth_output - kth_target) * (kth_output - kth_target);
	  }
	  CUDA_CALLABLE_MEMBER pt derivative(pt kth_target, pt kth_output)
	  {
		return 2.0 * (kth_output - kth_target);
	  }
	  CUDA_CALLABLE_MEMBER pt delta(pt kth_target, pt kth_output)
	  {
		return 2.0 * (kth_target - kth_output);
	  }
  };
  class cross_entropy : public loss_function<cross_entropy>
  {
	public:
	  CUDA_CALLABLE_MEMBER pt objective(pt kth_target, pt kth_output, pt epsilon)
	  {
		return kth_target * log(kth_output + epsilon);
	  }
	  CUDA_CALLABLE_MEMBER pt derivative(pt kth_target, pt kth_output)
	  {
		return kth_output - kth_target;
	  }
	  CUDA_CALLABLE_MEMBER pt delta(pt kth_target, pt kth_output)
	  {
		return kth_target - kth_output;
	  }
	  HOST std::string get_loss_function_label() //host for now don't think it would change either
	  {
		return "cross_entropy-multi-class";
	  }
  };	
}
#endif  

