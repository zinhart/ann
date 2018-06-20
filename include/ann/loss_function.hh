#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H
#include "typedefs.cuh"
#include "activation.hh"
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  enum class LOSS_FUNCTION_NAME : std::uint32_t {MSE = std::uint32_t{1}, CROSS_ENTROY_MULTI_CLASS};
  enum class LOSS_FUNCTION_TYPE : std::uint32_t{OBJECTIVE = std::uint32_t{1}, DERIVATIVE};
  
  template <class LOSS_FUNCTION>
	class loss_function
	{
	  public:
		loss_function() = default;
		loss_function(const loss_function &) = default;
		loss_function(loss_function &&) = default;
		loss_function & operator = (const loss_function &) = default;
		loss_function & operator = (loss_function &&) = default;
		~loss_function() = default;
		template <class Precision_Type>
		  CUDA_CALLABLE_MEMBER Precision_Type operator()(Precision_Type kth_output, Precision_Type kth_target, LOSS_FUNCTION_TYPE type)
		  { return (type == LOSS_FUNCTION_TYPE::OBJECTIVE) ? static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target) : static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target) ;}

		template <class Precision_Type>
		  CUDA_CALLABLE_MEMBER Precision_Type operator()(Precision_Type kth_output, Precision_Type kth_target, LOSS_FUNCTION_TYPE type, Precision_Type epsilon)
		  { return (type == LOSS_FUNCTION_TYPE::OBJECTIVE) ? static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, epsilon) : static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target);}


/*		enum F : char {OBJECTIVE = 0};
		enum DF : char {DERIVATIVE = 1};
		CUDA_CALLABLE_MEMBER double mse(pt kth_target, pt kth_output, F OBJECTIVE)
		{ return (kth_output - kth_target) * (kth_output - kth_target); }
		CUDA_CALLABLE_MEMBER double mse(pt kth_target, pt kth_output, DF DERIVATIVE)
		{	return double(2.0) * (kth_output - kth_target);}
		CUDA_CALLABLE_MEMBER double cross_entropy_multi_class(pt kth_target, pt kth_output, F OBJECTIVE, pt epsilon = 1.e-30)
		{	return kth_target * log(kth_output + epsilon); }
		CUDA_CALLABLE_MEMBER double cross_entropy_multi_class(pt kth_target, pt kth_output, DF DERIVATIVE)
		{return kth_output - kth_target;}*/
	};


  class mean_squared_error : public loss_function<mean_squared_error>
  {
	public:
	  mean_squared_error() = default;
	  mean_squared_error(const mean_squared_error &) = default;
	  mean_squared_error(mean_squared_error &&) = default;
	  mean_squared_error & operator = (const mean_squared_error &) = default;
	  mean_squared_error & operator = (mean_squared_error &&) = default;
	  ~mean_squared_error() = default;
	  template <class Precision_Type>
		CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & kth_output, const Precision_Type & kth_target)
		{	return (kth_output - kth_target) * (kth_output - kth_target); }
	  template <class Precision_Type>
		CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & kth_output, const Precision_Type & kth_target)
		{ return Precision_Type{2.0} * (kth_output - kth_target); }
  };

  class cross_entropy_multi_class : public loss_function<cross_entropy_multi_class>
  {
	public:
	  cross_entropy_multi_class() = default;
	  cross_entropy_multi_class(const cross_entropy_multi_class &) = default;
	  cross_entropy_multi_class(cross_entropy_multi_class &&) = default;
	  cross_entropy_multi_class & operator = (const cross_entropy_multi_class &) = default;
	  cross_entropy_multi_class & operator = (cross_entropy_multi_class &&) = default;
	  ~cross_entropy_multi_class() = default;
	  template <class Precision_Type>
		CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & kth_output, const Precision_Type & kth_target, const Precision_Type & epsilon)
		{	return kth_target * log(kth_output + epsilon); }
	  template <class Precision_Type>
		CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & kth_output, const Precision_Type & kth_target)
		{return kth_output - kth_target;}
  };

  using mse = loss_function<mean_squared_error>;
  using ce_multi_class = loss_function<cross_entropy_multi_class>;

  template <class Precision_Type>
	HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, Precision_Type * device_total_outputs, std::uint32_t device_id = 0);

  template <class Precision_Type>
	HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<LAYER_INFO> & total_layers, Precision_Type * device_total_outputs, const Precision_Type & epsilon = 1.e-30, std::uint32_t device_id = 0);
}
#endif  

