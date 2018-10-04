#include <concurrent_routines/serial/serial.hh>
#include <type_traits>
namespace zinhart
{
  namespace models
  {
	namespace layers
	{

	  template <class precision_type>
		HOST layer<precision_type>::layer()
		{ init(0, 0, nullptr, nullptr, 1); }

	  template <class precision_type>
		HOST layer<precision_type>::layer(std::uint32_t start_index, std::uint32_t stop_index, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient)
		{ init(start_index, stop_index, total_activations, total_deltas, coefficient); }

	  template <class precision_type>
		HOST layer<precision_type>::layer(const layer & l)
		{ 
		  start_index = l.get_start_index();
		  stop_index = l.get_stop_index();
		  start_activations = l.get_start_activations();
		  stop_activations = l.get_stop_activations();
		  start_deltas = l.get_start_deltas();
		  stop_deltas = l.get_stop_deltas();
		  coefficient = l.get_coefficient();
		}
	  template <class precision_type>
		HOST layer<precision_type>::layer(layer && l)
		{ 
		  start_index = l.get_start_index();
		  stop_index = l.get_stop_index();
		  start_activations = l.get_start_activations();
		  stop_activations = l.get_stop_activations();
		  start_activations = l.get_start_deltas();
		  stop_activations = l.get_stop_deltas();
		  coefficient = l.get_coefficient();
		}
	  template <class precision_type>
		HOST void layer<precision_type>::init(std::uint32_t start_index, std::uint32_t stop_index, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient)
		{
		  this->start_index = start_index;
		  this->stop_index = stop_index;
		  this->start_activations = total_activations + this->start_index;
		  this->stop_activations = total_activations + this->stop_index;
		  this->start_deltas = total_deltas + this->start_index;
		  this->stop_deltas = total_deltas + this->stop_index;
		  this->coefficient = coefficient;
		}
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER  precision_type * layer<precision_type>::get_start_activations()const
		{ return start_activations; }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER  precision_type * layer<precision_type>::get_stop_activations()const
		{ return stop_activations; }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER  precision_type * layer<precision_type>::get_start_deltas()const
		{ return start_deltas; }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER  precision_type * layer<precision_type>::get_stop_deltas()const
		{ return stop_deltas; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_start_index()const
	    { return start_index;}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_stop_index()const
		{ return stop_index; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_total_nodes()const
		{ return stop_index - start_index; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void layer<precision_type>::set_coefficient(const precision_type & coefficient)
		{ this->coefficient = coefficient; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::get_coefficient()const
		{ return coefficient; }

	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::input_layer input, zinhart::function_space::objective o)
		{ }
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::identity_layer identity, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(identity, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(sigmoid, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softplus_layer softplus, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(softplus, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(hyperbolic_tangent, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::relu_layer relu, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(relu, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(leaky_relu, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::objective o)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(exp_leaky_relu, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softmax_layer softmax, zinhart::function_space::objective o)
		{
		/*  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(current_layer.softmax_layer, *i);*/
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::objective o)
		{
		/*  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = objective(current_layer.batch_norm_layer, *i);*/
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::input_layer input, zinhart::function_space::derivative d)
		{ }
		
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::identity_layer identity, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(identity, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(sigmoid, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softplus_layer softplus, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(softplus, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(hyperbolic_tangent, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::relu_layer relu, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(relu, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(leaky_relu, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::derivative d)
		{
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(exp_leaky_relu, *i);
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::softmax_layer softmax, zinhart::function_space::derivative d)
		{
	/*	  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(name, *i);*/
		}
	  template <class precision_type>
		HOST void layer<precision_type>::activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::derivative d)
		{
	/*	  for(precision_type * i = start_activations; i != stop_activations; ++i)
			*i = derivative(name, *i);*/
		}
                                        
	  template <class precision_type>           
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::identity_layer identity, const precision_type & x)
		{ return x; }
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
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::softplus_layer softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return log(precision_type{1.0} + exp(x));
#else
  		  return std::log(precision_type{1.0} + std::exp(x));
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
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::relu_layer relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? x : precision_type{0.0}; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::leaky_relu_layer leaky_relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? x : get_coefficient() * x; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return (x >= precision_type{0.0} ) ? x : get_coefficient() * (exp(x) - precision_type{1.0})
#else
		  return (x >= precision_type{0.0} ) ? x : get_coefficient() * (std::exp(x) - precision_type{1.0});
#endif
		}


	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::identity_layer identity, const precision_type & x )
		{ return precision_type{1}; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::sigmoid_layer sigmoid, const precision_type & x)
		{ return x * (precision_type{1.0} - x); }
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
        CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x)
		{ return precision_type{1.0} - (x * x); }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::relu_layer relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : precision_type{0.0};}
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::leaky_relu_layer leaky_relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : get_coefficient();  }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x)
		{ 
#if CUDA_ENABLED == 1
  		  return (x >= precision_type{0.0} ) ? x : get_coefficient() * (exp(x) - precision_type{1.0});
#else
  		  return (x >= precision_type{0.0} ) ? x : get_coefficient() * (std::exp(x) - precision_type{1.0});
#endif
		}

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
