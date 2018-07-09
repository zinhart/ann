namespace zinhart
{
  namespace error_metrics
  {
	// loss function
	template <class LOSS_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type loss_function<LOSS_FUNCTION>::operator()(precision_type kth_output, precision_type kth_target, LOSS_FUNCTION_TYPE type)
	  { return (type == LOSS_FUNCTION_TYPE::OBJECTIVE) ? static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target) : static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target) ;}
	template <class LOSS_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type loss_function<LOSS_FUNCTION>::operator()(precision_type kth_output, precision_type kth_target, LOSS_FUNCTION_TYPE type, precision_type epsilon)
	  { return (type == LOSS_FUNCTION_TYPE::OBJECTIVE) ? static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, epsilon) : static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target);}

	// mean squared error
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type mean_squared_error::objective(const precision_type & kth_output, const precision_type & kth_target)
	  {	return (kth_output - kth_target) * (kth_output - kth_target); }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type mean_squared_error::derivative(const precision_type & kth_output, const precision_type & kth_target)
	  { return precision_type{2.0} * (kth_output - kth_target); }

	// cross entropy multi class
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type cross_entropy_multi_class::objective(const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon)
	  {	return kth_target * log(kth_output + epsilon); }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type cross_entropy_multi_class::derivative(const precision_type & kth_output, const precision_type & kth_target)
	  {return kth_output - kth_target;}
  }// END NAMESPACE ERROR_METRICS
}// END NAMESPACE ZINHART
