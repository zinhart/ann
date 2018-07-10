#include <stdexcept>
#include <iostream>
namespace zinhart
{
  namespace activation
  {
	// activation_function
	template <class ACTIVATION_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type activation_function<ACTIVATION_FUNCTION>::operator ()(precision_type x, ACTIVATION_TYPE at)
	  { return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x); };

	template <class ACTIVATION_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type activation_function<ACTIVATION_FUNCTION>::operator ()(precision_type x, ACTIVATION_TYPE at, const precision_type coefficient)
	  { return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x, coefficient) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x, coefficient); };

    // identity
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type identity::objective(const precision_type & x)
	  {return x;};
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type identity::derivative(const precision_type & x)
	  {return precision_type{1};};
	
	// sigmoid
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type sigmoid::objective(const precision_type  x)
	  {
#if CUDA_ENABLED == 1
		return precision_type{1.0} / ( precision_type{1.0} + exp(-x) );
#else
		return precision_type{1.0} / ( precision_type{1.0} + std::exp(-x) );
#endif
	  };
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type sigmoid::derivative(const precision_type & x)
	  {
		return x * (precision_type(1.0) - x);
	  };

    // softplus
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type softplus::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return log(precision_type(1.0) + exp(x));
#else
  		return std::log(precision_type(1.0) + std::exp(x));
#endif
	  };
   	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type softplus::derivative(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return precision_type(1.0) / (precision_type(1.0) + exp(-x));
#else
		return precision_type(1.0) / (precision_type(1.0) + std::exp(-x));
#endif
	  };

	// hyperbolic tangent
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type hyperbolic_tangent::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return tanh(-x);
#else
		return std::tanh(-x);
#endif
	  };
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type hyperbolic_tangent::derivative(const precision_type & x)
	  { return precision_type(1.0) - (x * x); };

	// rectified linear unit
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type relu::objective(const precision_type & x)
	  { return (x >= precision_type{0.0} ) ? x : precision_type{0.0}; };
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type relu::derivative(const precision_type & x)
	  { return (x >= precision_type{0.0} ) ? precision_type{1.0} : precision_type{0.0};	};

	// leaky rectified linear unit
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type leaky_relu::objective(const precision_type & x, precision_type leakage_coefficient)
	  { return (x >= precision_type(0.0) ) ? x : leakage_coefficient * x; };
  	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type leaky_relu::derivative(const precision_type & x, precision_type leakage_coefficient )
	  { return (x >= precision_type(0.0) ) ? precision_type(1.0) : leakage_coefficient;  };

	// Exponential leaky rectified linear unit
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type exp_leaky_relu::objective(const precision_type & x, precision_type leakage_coefficient)
	  {
#if CUDA_ENABLED == 1
		return (x >= precision_type(0.0) ) ? x : leakage_coefficient * (exp(x) - precision_type(1.0));
#else
		return (x >= precision_type(0.0) ) ? x : leakage_coefficient * (std::exp(x) - precision_type(1.0));
#endif
	  };
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type exp_leaky_relu::derivative(const precision_type & x, precision_type leakage_coefficient)
	  {
#if CUDA_ENABLED == 1
		return (x >= precision_type(0.0) ) ? x : leakage_coefficient * (exp(x) - precision_type(1.0));
#else
		return (x >= precision_type(0.0) ) ? x : leakage_coefficient * (std::exp(x) - precision_type(1.0));
#endif
	  };

	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type softmax::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return exp(x);
#else
		return std::exp(x);
#endif
	  };
	template <class precision_type>
	  //to do
	  CUDA_CALLABLE_MEMBER precision_type softmax::derivative(const precision_type & x)
	  {return x;};

	template<class ACTIVATION_FUNCTION>
	  HOST activation_function<ACTIVATION_FUNCTION> get_activation_function(ACTIVATION_NAME name)
	  {
		try 
		{
		  switch(name)
		  { 
			case ACTIVATION_NAME::IDENTITY:
			  return activation_function<identity>();
			case ACTIVATION_NAME::SIGMOID:
			  return activation_function<sigmoid>();
			case ACTIVATION_NAME::SOFTPLUS:
			  return activation_function<softplus>();
			case ACTIVATION_NAME::TANH:
			  return activation_function<hyperbolic_tangent>();
			case ACTIVATION_NAME::RELU:
			  return activation_function<relu>();
			case ACTIVATION_NAME::LEAKY_RELU:
			  return activation_function<leaky_relu>();
			case ACTIVATION_NAME::EXP_LEAKY_RELU:
			  return activation_function<exp_leaky_relu>();
			case ACTIVATION_NAME::SOFTMAX:
			  return activation_function<softmax>();
			case ACTIVATION_NAME::INPUT:
			  throw std::runtime_error("Their is no activation for the input layer");
		  }
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
  }// END NAMESPACE ACTIVATION
}// END NAMESPACE ZINHART
