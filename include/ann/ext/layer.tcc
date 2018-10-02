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
		HOST layer<precision_type>::layer(std::uint32_t start_index, std::uint32_t end_index, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient)
		{ init(start_index, end_index, total_activations, total_deltas, coefficient); }

	  template <class precision_type>
		HOST layer<precision_type>::layer(const layer & l)
		{ 
		  start_index = l.get_start_index();
		  end_index = l.get_end_index();
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
		  end_index = l.get_end_index();
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
		  this->end_index = end_index;
		  this->start_activations = total_activations + start_index;
		  this->stop_activations = total_activations + end_index;
		  this->start_deltas = total_deltas + start_index;
		  this->stop_deltas = total_deltas + end_index;
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
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_end_index()const
		{ return end_index; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_total_nodes()const
		{ return end_index - start_index; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void layer<precision_type>::set_coefficient(const precision_type & coefficient)
		{ this->coefficient = coefficient; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::get_coefficient()const
		{ return coefficient; }

	  template <class precision_type>
		template <class activation_name>			
		HOST void layer<precision_type>::activate(activation_name name, zinhart::function_space::OBJECTIVE obj)
		{

		  static_assert(
						std::is_same<activation_name, layer_name::identity>::value		      ||
						std::is_same<activation_name, layer_name::softplus>::value            ||
						std::is_same<activation_name, layer_name::hyperbolic_tangent>::value  ||
						std::is_same<activation_name, layer_name::relu>::value                || 
						std::is_same<activation_name, layer_name::leaky_relu>::value          ||
						std::is_same<activation_name, layer_name::exp_leaky_relu>::value      ||
						std::is_same<activation_name, layer_name::softmax>::value             ||  
						std::is_same<activation_name, layer_name::batch_normalization>::value,
						"activation name must be one of the following types \n layer_name::identity \n layer_name::softplus \n layer_name::hyperbolic_tangent \n layer_name::relu \n layer_name::leaky_relu \n layer_name::exp_leaky_relu \n layer_name::softmax \n layer_name::batch_normalization " 
					   ); 
		  for(precision_type * i = start_activations; i != stop_activations; ++i)
		  {
			*i = objective(name, *i);
		  }
		}
	  
	  template <class precision_type>
		template <class activation_name>			
		HOST void layer<precision_type>::activate(activation_name name, zinhart::function_space::DERIVATIVE obj)
		{

		  static_assert(
						std::is_same<activation_name, layer_name::identity>::value		      ||
						std::is_same<activation_name, layer_name::softplus>::value            ||
						std::is_same<activation_name, layer_name::hyperbolic_tangent>::value  ||
						std::is_same<activation_name, layer_name::relu>::value                || 
						std::is_same<activation_name, layer_name::leaky_relu>::value          ||
						std::is_same<activation_name, layer_name::exp_leaky_relu>::value      ||
						std::is_same<activation_name, layer_name::softmax>::value             ||  
						std::is_same<activation_name, layer_name::batch_normalization>::value,
						"activation name must be one of the following types \n layer_name::identity \n layer_name::softplus \n layer_name::hyperbolic_tangent \n layer_name::relu \n layer_name::leaky_relu \n layer_name::exp_leaky_relu \n layer_name::softmax \n layer_name::batch_normalization " 
					   );                       

		  for(precision_type * i = start_activations; i != stop_activations; ++i)
		  {
			*i = derivative(name, *i);
		  }
		}                                         
                                                
	  template <class precision_type>           
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::identity identity, const precision_type & x)
		{ return x; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::sigmoid sigmoid, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return precision_type{1.0} / ( precision_type{1.0} + exp(-x) );
#else
  		  return precision_type{1.0} / ( precision_type{1.0} + std::exp(-x) );
#endif
		}
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::softplus softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return log(precision_type{1.0} + exp(x));
#else
  		  return std::log(precision_type{1.0} + std::exp(x));
#endif
		}
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::hyperbolic_tangent hyperbolic_tangent, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return tanh(x);
#else
  		  return std::tanh(x);
#endif
		}


	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::relu relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? x : precision_type{0.0}; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::leaky_relu leaky_relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? x : get_coefficient() * x; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::objective(layer_name::exp_leaky_relu exp_leaky_relu, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return (x >= precision_type{0.0} ) ? x : get_coefficient() * (exp(x) - precision_type{1.0})
#else
		  return (x >= precision_type{0.0} ) ? x : get_coefficient() * (std::exp(x) - precision_type{1.0});
#endif
		}


	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::identity identity, const precision_type & x )
		{ return precision_type{1}; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::sigmoid sigmoid, const precision_type & x)
		{ return x * (precision_type{1.0} - x); }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::softplus softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return precision_type{1.0} / (precision_type{1.0} + exp(-x));
#else
  		  return precision_type{1.0} / (precision_type{1.0} + std::exp(-x));
#endif
		}
	  
	  template <class precision_type>
        CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::hyperbolic_tangent hyperbolic_tangent, const precision_type & x)
		{ return precision_type{1.0} - (x * x); }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::relu relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : precision_type{0.0};}
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::leaky_relu leaky_relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : get_coefficient();  }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type layer<precision_type>::derivative(layer_name::exp_leaky_relu exp_leaky_relu, const precision_type & x)
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
