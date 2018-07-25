namespace zinhart
{
  namespace error_metrics
  {
	/*
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type loss_function::operator()(LOSS_FUNCTION_NAME name, LOSS_FUNCTION_TYPE type, precision_type kth_output, precision_type kth_target)
	  {
		try
		{
		  if(name == LOSS_FUNCTION_NAME::MSE) return ( *static_cast< loss_function_interface<mean_squared_error>* >(this) )(type, kth_output, kth_target);
		  else
			  throw std::runtime_error("Their is no loss_function specified");

		}
		catch(std::runtime_error & e)
		{
		  std::cerr<<e.what()<<"\n";
		  throw e;
		}
		catch(...)
		{
		}
	  }*/
	template <class precision_type, class container>
	  HOST void loss_function::operator()(LOSS_FUNCTION_NAME name, LOSS_FUNCTION_TYPE type, 
										  precision_type & error,
										  precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths,
										  container & results,
										  precision_type epsilon,
										  zinhart::parallel::thread_pool & pool
										 )
	  {
		try
		{
		  error = precision_type{0};
		  if(name == LOSS_FUNCTION_NAME::MSE && type == LOSS_FUNCTION_TYPE::OBJECTIVE) 
		  {
			auto mse = [](const double & kth_output, const double & kth_target)
			{
			  loss_function_interface<mean_squared_error> loss;
			  return loss(LOSS_FUNCTION_TYPE::OBJECTIVE, kth_output, kth_target);
			};
			zinhart::parallel::async::neumaier_sum(outputs, targets, vector_lengths, error, mse, results, pool);
  		  }
		  else if(name == LOSS_FUNCTION_NAME::MSE && type == LOSS_FUNCTION_TYPE::DERIVATIVE)
		  {
			auto mse_derivative = [](const double & kth_output, const double & kth_target)
			{
			  loss_function_interface<mean_squared_error> loss;
			  return loss(LOSS_FUNCTION_TYPE::DERIVATIVE, kth_output, kth_target );
			};
			zinhart::parallel::async::neumaier_sum(outputs, targets, vector_lengths, error, mse_derivative, results, pool);
		  }
		  else if(name == LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS && type == LOSS_FUNCTION_TYPE::OBJECTIVE)
		  {
			//return ( *static_cast< loss_function_interface<cross_entropy_multi_class>* >(this) )(type, kth_output, kth_target, epsilon);
		  }
		  else if(name == LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS && type == LOSS_FUNCTION_TYPE::DERIVATIVE)
		  {
			//return ( *static_cast< loss_function_interface<cross_entropy_multi_class>* >(this) )(type, kth_output, kth_target, epsilon);
		  }
		  else
			  throw std::runtime_error("Their is no loss_function specified");

		}
		catch(std::runtime_error & e)
		{
		  std::cerr<<e.what()<<"\n";
		  throw e;
		}
		catch(...)
		{
		}
	  }
	// loss function
	template <class LOSS_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( LOSS_FUNCTION_TYPE type, precision_type kth_output, precision_type kth_target)
	  { return (type == LOSS_FUNCTION_TYPE::OBJECTIVE) ? static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target) : static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target) ;}
	template <class LOSS_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type loss_function_interface<LOSS_FUNCTION>::operator()( LOSS_FUNCTION_TYPE type, precision_type kth_output, precision_type kth_target, precision_type epsilon)
	  { return (type == LOSS_FUNCTION_TYPE::OBJECTIVE) ? static_cast<LOSS_FUNCTION*>(this)->objective(kth_output, kth_target, epsilon) : static_cast<LOSS_FUNCTION*>(this)->derivative(kth_output, kth_target, epsilon);}

	// mean squared error
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type mean_squared_error::objective(const precision_type & kth_output, const precision_type & kth_target)
	  {	return precision_type{0.5} * (kth_output - kth_target) * (kth_output - kth_target); }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type mean_squared_error::derivative(const precision_type & kth_output, const precision_type & kth_target)
	  { return precision_type{2.0} * (kth_output - kth_target); }

	// cross entropy multi class
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type cross_entropy_multi_class::objective(const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon)
	  {	return kth_target * log(kth_output + epsilon); }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type cross_entropy_multi_class::derivative(const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon)
	  {return kth_output - kth_target;}
  }// END NAMESPACE ERROR_METRICS
}// END NAMESPACE ZINHART
