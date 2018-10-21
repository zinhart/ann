#include <multi_core/multi_core.hh>
namespace zinhart
{
  namespace loss_functions
  {

  template <class precision_type>
	HOST precision_type error_function<precision_type>::error(loss_attributes::mean_squared_error mse, zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
	  precision_type sum{0};
	  for(std::uint32_t i = 0; i < length; ++i)
		sum += objective(mse, *(outputs + i), *(targets + i));
	  return sum / length;
	}
  template <class precision_type>
	HOST void error_function<precision_type>::error(loss_attributes::mean_squared_error mse, zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length, const std::uint32_t & output_size)
	{
	  for(std::uint32_t i = 0; i < length; ++i)
		*(results + i) = derivative(mse, *(outputs + i), *(targets + i),  length);
	}
  template <class precision_type>
	HOST precision_type error_function<precision_type>::error(loss_attributes::cross_entropy_multi_class ce, zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length, const precision_type & epsilon)
	{
	  precision_type sum{0};
	  for(std::uint32_t i = 0; i < length; ++i)
		sum += -objective(ce, *(outputs + i), *(targets + i), epsilon);
	  return sum;
	}
  template <class precision_type>
	HOST void error_function<precision_type>::error(loss_attributes::cross_entropy_multi_class ce, zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length)
	{
	  for(std::uint32_t i = 0; i < length; ++i)
		*(results + i) = derivative(ce, *(outputs + i), *(targets + i));
	}


  template <class precision_type>
	CUDA_CALLABLE_MEMBER precision_type error_function<precision_type>::objective(loss_attributes::mean_squared_error mse, const precision_type & kth_output, const precision_type & kth_target)
	{ return (kth_output - kth_target) * (kth_output - kth_target); }
  template <class precision_type>
	CUDA_CALLABLE_MEMBER precision_type error_function<precision_type>::derivative(loss_attributes::mean_squared_error mse, const precision_type & kth_output, const precision_type & kth_target, const std::uint32_t & output_size)
	{ return precision_type{2.0} /  static_cast<precision_type>(output_size) * (kth_output - kth_target); }
  template <class precision_type>
	CUDA_CALLABLE_MEMBER precision_type error_function<precision_type>::objective(loss_attributes::cross_entropy_multi_class ce, const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon)
	{
#if CUDA_ENABLED == 1
#else
	  return kth_target * std::log(kth_output + epsilon);
#endif
   	}
  template <class precision_type>
	CUDA_CALLABLE_MEMBER precision_type error_function<precision_type>::derivative(loss_attributes::cross_entropy_multi_class ce, const precision_type & kth_output, const precision_type & kth_target)
	{ return -kth_target / kth_output; }

  template <class precision_type>
	HOST precision_type mean_squared_error<precision_type>::error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  return e.error(loss_attributes::mean_squared_error(), o, outputs, targets, length);
#endif
   	}
  template <class precision_type>
  	HOST void mean_squared_error<precision_type>::error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  e.error(loss_attributes::mean_squared_error(), d, outputs, targets, results, length, batch_size);
#endif
	}

  template <class precision_type>
	HOST precision_type cross_entropy_multi_class<precision_type>::error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  return e.error(loss_attributes::cross_entropy_multi_class(), o, outputs, targets, length, epsilon);
#endif
	}
  template <class precision_type>
	HOST void cross_entropy_multi_class<precision_type>::error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  e.error(loss_attributes::cross_entropy_multi_class(), d, outputs, targets, results, length);
#endif
	}
  }// END NAMESPACE LOSS_FUNCTIONS
}// END NAMESPACE ZINHART
