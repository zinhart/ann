#include <concurrent_routines/concurrent_routines.hh>
namespace zinhart
{
  namespace loss_functions
  {

  template <class precision_type>
	HOST precision_type error_function<precision_type>::error(loss_attributes::mean_squared_error mse, zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length, const std::uint32_t & batch_size)
	{
	  for(std::uint32_t i = 0; i < length; ++i)
		objective(mse, o, *(outputs + i), *(targets + i), batch_size);
	}
  template <class precision_type>
	HOST precision_type error_function<precision_type>::error(loss_attributes::mean_squared_error mse, zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length, const std::uint32_t & batch_size)
	{
	  for(std::uint32_t i = 0; i < length; ++i)
		derivative(mse, d, *(outputs + i), *(targets + i), batch_size);
	}
  template <class precision_type>
	HOST precision_type error_function<precision_type>::error(loss_attributes::cross_entropy_multi_class ce, zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length, const precision_type & epsilon)
	{
	  for(std::uint32_t i = 0; i < length; ++i)
		objective(ce, o, *(outputs + i), *(targets + i), epsilon);
	}
  template <class precision_type>
	HOST precision_type error_function<precision_type>::error(loss_attributes::cross_entropy_multi_class ce, zinhart::function_space::derivative d,const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
	  for(std::uint32_t i = 0; i < length; ++i)
		derivative(ce, d, *(outputs + i), *(targets + i));
	}


  template <class precision_type>
	CUDA_CALLABLE_MEMBER precision_type error_function<precision_type>::objective(loss_attributes::mean_squared_error mse, const precision_type & kth_output, const precision_type & kth_target, const std::uint32_t & batch_size)
	{ return precision_type{1.0} /  precision_type{batch_size}  * (kth_output - kth_target) * (kth_output - kth_target); }
  template <class precision_type>
	CUDA_CALLABLE_MEMBER precision_type error_function<precision_type>::derivative(loss_attributes::mean_squared_error mse, const precision_type & kth_output, const precision_type & kth_target, const std::uint32_t & batch_size)
	{ return precision_type{2.0} /  precision_type{batch_size} * (kth_output - kth_target); }
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
	{ return kth_output - kth_target; }

  template <class precision_type>
	HOST void mean_squared_error<precision_type>::error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  e.error(loss_attributes::mean_squared_error(), o, outputs, targets, length, batch_size);
#endif
   	}
  template <class precision_type>
  	HOST void mean_squared_error<precision_type>::error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  e.error(loss_attributes::mean_squared_error(), d, outputs, targets, length, batch_size);
#endif
	}

  template <class precision_type>
	HOST void cross_entropy_multi_class<precision_type>::error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  e.error(loss_attributes::cross_entropy_multi_class(), o, outputs, targets, length, epsilon);
#endif
	}
  template <class precision_type>
	HOST void cross_entropy_multi_class<precision_type>::error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length)
	{
#if CUDA_ENABLED == 1

#else
	  e.error(loss_attributes::cross_entropy_multi_class(), d, outputs, targets, length);
#endif
	}

  }// END NAMESPACE LOSS_FUNCTIONS
  namespace function_space
  {
	namespace error_metrics
	{
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type loss_function::operator()(LOSS_FUNCTION_NAME name, zinhart::function_space::objective label, const precision_type * outputs, const precision_type * targets, std::uint32_t vector_lengths, std::uint32_t batch_size)
		{

			if(name == LOSS_FUNCTION_NAME::MSE) 
			{
			  auto mse = [label, batch_size](const double & kth_output, const double & kth_target)
			  {
				loss_function_interface<mean_squared_error> loss;
				return loss(label, kth_output, kth_target, batch_size);
			  };
			  return zinhart::serial::neumaier_sum(outputs, targets, vector_lengths, mse);
			}
		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void loss_function::operator()(LOSS_FUNCTION_NAME name, zinhart::function_space::derivative label, precision_type * outputs, precision_type * targets, precision_type * results, std::uint32_t vector_lengths, std::uint32_t batch_size)
		{

			if(name == LOSS_FUNCTION_NAME::MSE) 
			{
			  loss_function_interface<mean_squared_error> loss;
			  for(std::uint32_t i = 0; i < vector_lengths; ++i)
				*(results + i) = loss(label, *(outputs + i), *(targets + i), batch_size);
			}
		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type loss_function::operator()(LOSS_FUNCTION_NAME name, zinhart::function_space::objective label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, precision_type epsilon)
		{
		  if(name == LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS)
		  {
			auto ce = [label, epsilon](const double & kth_output, const double & kth_target)
			{
			  loss_function_interface<cross_entropy_multi_class> loss;
			  return loss(label, kth_output, kth_target, epsilon);
			};
			return zinhart::serial::neumaier_sum(outputs, targets, vector_lengths, ce);
		  }
		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void loss_function::operator()(LOSS_FUNCTION_NAME name, zinhart::function_space::derivative label, precision_type * outputs, precision_type * targets, precision_type * results, std::uint32_t vector_lengths)
		{
		  if(name == LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS)
		  {
			loss_function_interface<cross_entropy_multi_class> loss;
			  for(std::uint32_t i = 0; i < vector_lengths; ++i)
				*(results + i) = loss(label, *(outputs + i), *(targets + i));
		  }
		}

	  // loss function
	  

	  template <class LOSS_FUNCTION>
		template <class precision_type>// mse
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( zinhart::function_space::objective label, precision_type kth_output, precision_type kth_target, std::uint32_t batch_size)
		{ return static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, batch_size);}


	  template <class LOSS_FUNCTION>
		template <class precision_type>// mse
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( zinhart::function_space::derivative label, precision_type kth_output, precision_type kth_target, std::uint32_t batch_size)
		{ return static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target, batch_size);}

	  template <class LOSS_FUNCTION>
		template <class precision_type>// ce
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( zinhart::function_space::objective label, precision_type kth_output, precision_type kth_target, precision_type epsilon)
		{ return static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, epsilon);}


	  template <class LOSS_FUNCTION>
		template <class precision_type>// ce
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( zinhart::function_space::derivative label, precision_type kth_output, precision_type kth_target)
		{ return static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target);}


	  // mean squared error
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type mean_squared_error::objective(const precision_type & kth_output, const precision_type & kth_target, std::uint32_t batch_size)
		{	return precision_type{1.0} /  precision_type{batch_size}  * (kth_output - kth_target) * (kth_output - kth_target); }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type mean_squared_error::derivative(const precision_type & kth_output, const precision_type & kth_target, std::uint32_t batch_size)
		{ return precision_type{2.0 } /  precision_type{batch_size} * (kth_output - kth_target); }

	  // cross entropy multi class
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type cross_entropy_multi_class::objective(const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon)
		{	return kth_target * log(kth_output + epsilon); }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type cross_entropy_multi_class::derivative(const precision_type & kth_output, const precision_type & kth_target)
		{return kth_output - kth_target;}
	}// END NAMESPACE ERROR_METRICS
  }// END NAMESPACE FUNCTION_SPACE
}// END NAMESPACE ZINHART
