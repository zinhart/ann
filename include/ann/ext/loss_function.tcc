namespace zinhart
{
  namespace function_space
  {
	namespace error_metrics
	{
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type loss_function::operator()(LOSS_FUNCTION_NAME name, OBJECTIVE label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, std::uint32_t batch_size)
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
		CUDA_CALLABLE_MEMBER void loss_function::operator()(LOSS_FUNCTION_NAME name, DERIVATIVE label, precision_type * outputs, precision_type * targets, precision_type * results, std::uint32_t vector_lengths, std::uint32_t batch_size)
		{

			if(name == LOSS_FUNCTION_NAME::MSE) 
			{
			  loss_function_interface<mean_squared_error> loss;
			  for(std::uint32_t i = 0; i < vector_lengths; ++i)
				*(results + i) = loss(label, *(outputs + i), *(targets + i), batch_size);
			}
		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type loss_function::operator()(LOSS_FUNCTION_NAME name, OBJECTIVE label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, precision_type epsilon)
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
		CUDA_CALLABLE_MEMBER void loss_function::operator()(LOSS_FUNCTION_NAME name, DERIVATIVE label, precision_type * outputs, precision_type * targets, precision_type * results, std::uint32_t vector_lengths)
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
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( OBJECTIVE label, precision_type kth_output, precision_type kth_target, std::uint32_t batch_size)
		{ return static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, batch_size);}


	  template <class LOSS_FUNCTION>
		template <class precision_type>// mse
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( DERIVATIVE label, precision_type kth_output, precision_type kth_target, std::uint32_t batch_size)
		{ return static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target, batch_size);}

	  template <class LOSS_FUNCTION>
		template <class precision_type>// ce
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( OBJECTIVE label, precision_type kth_output, precision_type kth_target, precision_type epsilon)
		{ return static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, epsilon);}


	  template <class LOSS_FUNCTION>
		template <class precision_type>// ce
		CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( DERIVATIVE label, precision_type kth_output, precision_type kth_target)
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
