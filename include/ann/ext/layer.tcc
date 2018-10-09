#include <concurrent_routines/serial/serial.hh>
#include <type_traits>
namespace zinhart
{
  namespace models
  {
	namespace layers
	{
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::input_layer input, zinhart::function_space::objective o)
		{ }
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::input_layer input, zinhart::function_space::derivative d)
		{ }

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::identity_layer identity, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( identity, *(start + i) );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::identity_layer identity, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = derivative( identity, *(start + i) );
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( sigmoid, *(start + i) );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) =	derivative( sigmoid, *(start + i) );
		}


	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softplus_layer softplus, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( softplus, *(start + i) );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softplus_layer softplus, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) =	derivative( softplus, *(start + i) );
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{

		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( hyperbolic_tangent, *(start + i) );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) =	derivative( hyperbolic_tangent, *(start + i) );
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::relu_layer relu, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( relu, *(start + i) );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::relu_layer relu, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) =	derivative( relu, *(start + i) );
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( leaky_relu, *(start + i), coefficient );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) =	derivative( leaky_relu, *(start + i), coefficient );
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) = objective( exp_leaky_relu, *(start + i), coefficient );
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(start + i) =	derivative( exp_leaky_relu, *(start + i), coefficient );
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softmax_layer softmax, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{
		/*  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(current_layer.softmax_layer, *i);
		*/
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softmax_layer softmax, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
	/*	  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(name, *i);*/
		}

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length)
		{
		/*  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(current_layer.batch_norm_layer, *i);*/
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length)
		{
	/*	  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(name, *i);*/
		}

	  template <class precision_type>
		template<class Callable, class ... Args>
		HOST void layer<precision_type>::activate(layer_info::generic_layer generic, zinhart::function_space::objective o, Callable && c, Args&& ...args)
		{
		}
	  template <class precision_type>
		template<class Callable, class ... Args>
		HOST void layer<precision_type>::activate(layer_info::generic_layer generic, zinhart::function_space::derivative d, Callable && c, Args&& ...args)
		{
		}

                                        
	  template <class precision_type>           
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::identity_layer identity, const precision_type & x)
		{ return x; }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::identity_layer identity, const precision_type & x )
		{ return precision_type{1}; }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::sigmoid_layer sigmoid, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return precision_type{1.0} / ( precision_type{1.0} + exp(-x) );
#else
  		  return precision_type{1.0} / ( precision_type{1.0} + std::exp(-x) );
#endif
		}
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::sigmoid_layer sigmoid, const precision_type & x)
		{ return x * (precision_type{1.0} - x); }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::softplus_layer softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return log(precision_type{1.0} + exp(x));
#else
  		  return std::log(precision_type{1.0} + std::exp(x));
#endif
		}
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::softplus_layer softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return precision_type{1.0} / (precision_type{1.0} + exp(-x));
#else
  		  return precision_type{1.0} / (precision_type{1.0} + std::exp(-x));
#endif
		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return tanh(x);
#else
  		  return std::tanh(x);
#endif
		}
	  template <class precision_type>
        CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x)
		{ return precision_type{1.0} - (x * x); }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::relu_layer relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? x : precision_type{0.0}; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::relu_layer relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : precision_type{0.0};}

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::leaky_relu_layer leaky_relu, const precision_type & x, const precision_type & coefficient)
		{ return (x >= precision_type{0.0} ) ? x : coefficient * x; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::leaky_relu_layer leaky_relu, const precision_type & x, const precision_type & coefficient )
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : coefficient;  }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x, const precision_type & coefficient)
		{
#if CUDA_ENABLED == 1
  		  return (x >= precision_type{0.0} ) ? x : coefficient * (exp(x) - precision_type{1.0})
#else
		  return (x >= precision_type{0.0} ) ? x : coefficient * (std::exp(x) - precision_type{1.0});
#endif
		}
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x, const precision_type & coefficient)
		{ 
#if CUDA_ENABLED == 1
  		  return (x >= precision_type{0.0} ) ? x : coefficient * (exp(x) - precision_type{1.0});
#else
  		  return (x >= precision_type{0.0} ) ? x : coefficient * (std::exp(x) - precision_type{1.0});
#endif
		}
	  template <class precision_type>
		template<class Callable, class ... Args>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::generic_layer generic, zinhart::function_space::objective o, Callable && c, Args&& ...args)
		{
#if CUDA_ENABLED == 1

#else

#endif
		}
	  template <class precision_type>
		template<class Callable, class ... Args>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::generic_layer generic, zinhart::function_space::derivative d, Callable && c, Args&& ...args)
		{
#if CUDA_ENABLED == 1

#else

#endif
		}

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
