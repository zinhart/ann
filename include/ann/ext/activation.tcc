#include <stdexcept>
#include <iostream>
namespace zinhart
{
  namespace activation
  {
	// activation_function
	template <class precision_type>
	  HOST precision_type activation_function::operator()(ACTIVATION_NAME name, ACTIVATION_TYPE at, precision_type x)
	  {
		try
		{
		  if(name == ACTIVATION_NAME::IDENTITY) return ( *static_cast< activation_interface<identity>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::SIGMOID) return ( *static_cast< activation_interface<sigmoid>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::SOFTPLUS) return ( *static_cast< activation_interface<softplus>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::TANH) return ( *static_cast< activation_interface<hyperbolic_tangent>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::RELU) return ( *static_cast< activation_interface<relu>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::LEAKY_RELU) return ( *static_cast< activation_interface<leaky_relu>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::EXP_LEAKY_RELU) return ( *static_cast< activation_interface<exp_leaky_relu>* >(this) )(at, x);
		  else if(name == ACTIVATION_NAME::SOFTMAX) return ( *static_cast< activation_interface<softmax>* >(this) )(at, x);
		  else
			  throw std::runtime_error("Their is no activation for the layer specified");

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
	// activation_interface
	template <class ACTIVATION_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type activation_interface<ACTIVATION_FUNCTION>::operator()(ACTIVATION_TYPE at, precision_type x)
	  { return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x); }

	template <class ACTIVATION_FUNCTION>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type activation_interface<ACTIVATION_FUNCTION>::operator()(ACTIVATION_TYPE at, precision_type x, const precision_type coefficient)
	  { return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x, coefficient) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x, coefficient); }

    // input
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type input::objective(const precision_type & x)
	  {return x;}
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type input::derivative(const precision_type & x)
	  {return x;}

    // identity
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type identity::objective(const precision_type & x)
	  {return x;}
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type identity::derivative(const precision_type & x)
	  {return precision_type{1};}
	
	// sigmoid
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type sigmoid::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return precision_type{1.0} / ( precision_type{1.0} + exp(-x) );
#else
		return precision_type{1.0} / ( precision_type{1.0} + std::exp(-x) );
#endif
	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type sigmoid::derivative(const precision_type & x)
	  {
		return x * (precision_type{1.0} - x);
	  }

    // softplus
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type softplus::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return log(precision_type{1.0} + exp(x));
#else
  		return std::log(precision_type{1.0} + std::exp(x));
#endif
	  }
   	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type softplus::derivative(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return precision_type{1.0} / (precision_type{1.0} + exp(-x));
#else
		return precision_type{1.0} / (precision_type{1.0} + std::exp(-x));
#endif
	  }

	// hyperbolic tangent
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type hyperbolic_tangent::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return tanh(x);
#else
		return std::tanh(x);
#endif
	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type hyperbolic_tangent::derivative(const precision_type & x)
	  { return precision_type{1.0} - (x * x); }

	// rectified linear unit
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type relu::objective(const precision_type & x)
	  { return (x >= precision_type{0.0} ) ? x : precision_type{0.0}; }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type relu::derivative(const precision_type & x)
	  { return (x >= precision_type{0.0} ) ? precision_type{1.0} : precision_type{0.0};	}

	// leaky rectified linear unit
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type leaky_relu::objective(const precision_type & x, precision_type leakage_coefficient)
	  { return (x >= precision_type{0.0} ) ? x : leakage_coefficient * x; }
  	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type leaky_relu::derivative(const precision_type & x, precision_type leakage_coefficient )
	  { return (x >= precision_type{0.0} ) ? precision_type{1.0} : leakage_coefficient;  }

	// Exponential leaky rectified linear unit
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type exp_leaky_relu::objective(const precision_type & x, precision_type leakage_coefficient)
	  {
#if CUDA_ENABLED == 1
		return (x >= precision_type{0.0} ) ? x : leakage_coefficient * (exp(x) - precision_type{1.0})
#else
		return (x >= precision_type{0.0} ) ? x : leakage_coefficient * (std::exp(x) - precision_type{1.0});
#endif
	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type exp_leaky_relu::derivative(const precision_type & x, precision_type leakage_coefficient)
	  {
#if CUDA_ENABLED == 1
		return (x >= precision_type{0.0} ) ? x : leakage_coefficient * (exp(x) - precision_type{1.0});
#else
		return (x >= precision_type{0.0} ) ? x : leakage_coefficient * (std::exp(x) - precision_type{1.0});
#endif
	  }

	template <class precision_type>
	  CUDA_CALLABLE_MEMBER precision_type softmax::objective(const precision_type & x)
	  {
#if CUDA_ENABLED == 1
		return exp(x);
#else
		return std::exp(x);
#endif
	  }
	template <class precision_type>
	  //to do
	  CUDA_CALLABLE_MEMBER precision_type softmax::derivative(const precision_type & x)
	  {return x;}
  }// END NAMESPACE ACTIVATION
}// END NAMESPACE ZINHART
