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
  namespace error_metrics
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
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator()(precision_type kth_output, precision_type kth_target, LOSS_FUNCTION_TYPE type);
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator()(precision_type kth_output, precision_type kth_target, LOSS_FUNCTION_TYPE type, precision_type epsilon);
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
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & kth_output, const precision_type & kth_target);
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & kth_output, const precision_type & kth_target);
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
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon);
		//  {	return kth_target * log(kth_output + epsilon); }
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & kth_output, const precision_type & kth_target);
		  //{return kth_output - kth_target;}
	};

	using mse = loss_function<mean_squared_error>;
	using ce_multi_class = loss_function<cross_entropy_multi_class>;

	template <class precision_type>
	  HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<zinhart::activation::LAYER_INFO> & total_layers, precision_type * device_total_outputs, std::uint32_t device_id = 0);

	template <class precision_type>
	  HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<zinhart::activation::LAYER_INFO> & total_layers, precision_type * device_total_outputs, const precision_type & epsilon = 1.e-30, std::uint32_t device_id = 0);

  }// END NAMESPACE ERROR_METRICS
}// END NAMESPACE ZINHART
#include "ext/loss_function.tcc"
#endif  

