#include <multi_core/serial/serial.hh>
#if CUDA_ENABLED == 1 
#else
#include <algorithm>
#include <numeric>
#include <mkl.h>
#endif
namespace zinhart
{
  namespace models
  {
	namespace layers
	{
	  template<class precision_type>
		HOST void layer<precision_type>::activate(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{ activate_impl(o, activations, length, bias); }	  

	  template<class precision_type>
		HOST void layer<precision_type>::activate(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{ activate_impl(o, d, deltas, error, activations, length); }

	  template<class precision_type>
		HOST void layer<precision_type>::activate(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{ activate_impl(h, d, deltas, activations, length); }

	  template<class precision_type>
		HOST void layer<precision_type>::set_size(std::uint32_t size)
		{ set_size_impl(size); }

	  template<class precision_type>
		HOST std::uint32_t layer<precision_type>::get_size()const
		{ return get_size_impl(); }

	  template<class precision_type>
		HOST void layer<precision_type>::set_bias(precision_type bias)
		{ set_bias_impl(bias); }

	  template<class precision_type>
		HOST precision_type layer<precision_type>::get_bias()const
		{ return get_bias_impl(); }

	  template<class precision_type>
		HOST std::string layer<precision_type>::name()const
		{ return name_impl(); }

	  template<class precision_type>
		HOST void input_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 
#else
#endif
		}
	  template<class precision_type>
		HOST void input_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
#endif
		}
	  template<class precision_type>
		HOST void input_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
#endif
		}

	  template<class precision_type>
		HOST void input_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t input_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void input_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type input_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }


	  template<class precision_type>
		HOST std::string input_layer<precision_type>::name_impl()const
		{ return "input"; }

	  template<class precision_type>
		HOST void identity_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::identity_layer(), o, activations, length, bias);
#endif
		}
	  template<class precision_type>
		HOST void identity_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::identity_layer(), o, d, deltas, error, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void identity_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::identity_layer(), h, d, deltas, activations, length);
#endif
		}

	  template<class precision_type>
		HOST void identity_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t identity_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void identity_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type identity_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST std::string identity_layer<precision_type>::name_impl()const
		{ return "identity"; }

	  template<class precision_type>
		HOST void sigmoid_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::sigmoid_layer(), o, activations, length, bias);
#endif
		}
	  template<class precision_type>
		HOST void sigmoid_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::sigmoid_layer(), o, d, deltas, error, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void sigmoid_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::sigmoid_layer(), h, d, deltas, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void sigmoid_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t sigmoid_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void sigmoid_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type sigmoid_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST std::string sigmoid_layer<precision_type>::name_impl()const
		{ return "sigmoid"; }

	  template<class precision_type>
		HOST void softplus_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::softplus_layer(), o, activations, length, bias);
#endif
		}
	  template<class precision_type>
		HOST void softplus_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::softplus_layer(), o, d, deltas, error, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void softplus_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::softplus_layer(), h, d, deltas, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void softplus_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t softplus_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void softplus_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type softplus_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }
	  
	  template<class precision_type>
		HOST std::string softplus_layer<precision_type>::name_impl()const
		{ return "softplus"; }

	  template<class precision_type>
		HOST void tanh_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::tanh_layer(), o, activations, length, bias);
#endif
		}
	  template<class precision_type>
		HOST void tanh_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::tanh_layer(), o, d, deltas, error, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void tanh_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::tanh_layer(), h, d, deltas, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void tanh_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t tanh_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void tanh_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type tanh_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST std::string tanh_layer<precision_type>::name_impl()const
		{ return "tanh"; }

	  template<class precision_type>
		HOST void relu_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::relu_layer(), o, activations, length, bias);
#endif
		}
	  template<class precision_type>
		HOST void relu_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::relu_layer(), o, d, deltas, error, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void relu_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::relu_layer(), h, d, deltas, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void relu_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t relu_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void relu_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type relu_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST std::string relu_layer<precision_type>::name_impl()const
		{ return "relu"; }

	  template<class precision_type>
		HOST void leaky_relu_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::leaky_relu_layer(), o, activations, length, coefficient, bias);
#endif
		}
	  template<class precision_type>
		HOST void leaky_relu_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::leaky_relu_layer(), o, d, deltas, error, activations, length, coefficient);
#endif
		}
	  template<class precision_type>
		HOST void leaky_relu_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::leaky_relu_layer(), h, d, deltas, activations, length, coefficient);
#endif
		}
	  template<class precision_type>
		HOST void leaky_relu_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t leaky_relu_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void leaky_relu_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type leaky_relu_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST std::string leaky_relu_layer<precision_type>::name_impl()const
		{ return "leaky_relu"; }

	  template<class precision_type>
		HOST void exp_leaky_relu_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::exp_leaky_relu_layer(), o, activations, length, coefficient, bias);
#endif
		}
	  template<class precision_type>
		HOST void exp_leaky_relu_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::exp_leaky_relu_layer(), o, d, deltas, error, activations, length, coefficient);
#endif
		}
	  template<class precision_type>
		HOST void exp_leaky_relu_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::exp_leaky_relu_layer(), h, d, deltas, activations, length, coefficient);
#endif
		}
	  template<class precision_type>
		HOST void exp_leaky_relu_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t exp_leaky_relu_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void exp_leaky_relu_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type exp_leaky_relu_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST std::string exp_leaky_relu_layer<precision_type>::name_impl()const
		{ return "exp_leaky_relu"; }

	  template<class precision_type>
		HOST softmax_layer<precision_type>::softmax_layer(std::uint32_t size)
		{ 
		  set_size_impl(size);
		  set_jacobian_size(get_size_impl());
#if CUDA_ENABLED == 1 

#else
		  jacobian = (precision_type *)mkl_malloc( get_jacobian_size() * sizeof(precision_type), 64);
#endif
	   	}
	  template<class precision_type>
		HOST softmax_layer<precision_type>::softmax_layer(const softmax_layer & sl)
		{
		  set_size_impl(sl.get_size_impl());
		  set_jacobian_size(get_size_impl());
#if CUDA_ENABLED == 1 

#else
		  jacobian = (precision_type *)mkl_malloc( get_jacobian_size() * sizeof(precision_type), 64);
#endif
		}
	  template<class precision_type>
		HOST softmax_layer<precision_type>::softmax_layer(softmax_layer && sl)
		{
		  set_size_impl(sl.get_size_impl());
		  set_jacobian_size(get_size_impl());
#if CUDA_ENABLED == 1 

#else
		  jacobian = sl.jacobian;//(precision_type *)mkl_malloc( this->get_size_impl() * sizeof(precision_type), 64);
		  sl.jacobian = nullptr;
#endif
		}
	  template<class precision_type>
		HOST softmax_layer<precision_type> & softmax_layer<precision_type>::operator = (const softmax_layer & sl)
		{
		  set_size_impl(sl.get_size_impl());
		  set_jacobian_size(get_size_impl());
#if CUDA_ENABLED == 1 

#else
		  jacobian = (precision_type *)mkl_malloc( get_jacobian_size() * sizeof(precision_type), 64);
#endif
		}
	  template<class precision_type>
		HOST softmax_layer<precision_type> & softmax_layer<precision_type>::operator = (softmax_layer && sl)
		{
		  set_size_impl(sl.get_size_impl());
		  set_jacobian_size(get_size_impl());
#if CUDA_ENABLED == 1 

#else
		  jacobian = sl.jacobian;//(precision_type *)mkl_malloc( this->get_size_impl() * sizeof(precision_type), 64);
		  sl.jacobian = nullptr;
#endif
		}

	  template<class precision_type>
		HOST softmax_layer<precision_type>::~softmax_layer()
		{
#if CUDA_ENABLED == 1 

#else
		  mkl_free(jacobian);
#endif
		}


	  template<class precision_type>
		HOST void softmax_layer<precision_type>::activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{  
#if CUDA_ENABLED == 1 

#else
		  a.activate(layer_info::softmax_layer(), o, activations, length, bias);
#endif
		}
	  template<class precision_type>
		HOST void softmax_layer<precision_type>::activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::softmax_layer(), o, d, deltas, jacobian, error, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void softmax_layer<precision_type>::activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
#if CUDA_ENABLED == 1
#else
		  a.activate(layer_info::softmax_layer(), h, d, deltas, jacobian, activations, length);
#endif
		}
	  template<class precision_type>
		HOST void softmax_layer<precision_type>::set_size_impl(std::uint32_t size)
		{ this->size = size; }

	  template<class precision_type>
		HOST std::uint32_t softmax_layer<precision_type>::get_size_impl()const
		{ return this->size; }

	  template<class precision_type>
		HOST void softmax_layer<precision_type>::set_bias_impl(precision_type bias)
		{ this->bias = bias; }

	  template<class precision_type>
		HOST precision_type softmax_layer<precision_type>::get_bias_impl()const
		{ return this->bias; }

	  template<class precision_type>
		HOST void softmax_layer<precision_type>::set_jacobian_size(std::uint32_t size)
		{ this->jacobian_size = size; }

	  template<class precision_type>
		HOST std::uint32_t softmax_layer<precision_type>::get_jacobian_size()const
		{ return this->jacobian_size; }

	  template<class precision_type>
		HOST std::string softmax_layer<precision_type>::name_impl()const
		{ return "softmax"; }

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::input_layer input, zinhart::function_space::objective o, const precision_type & bias)
		{ }
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::identity_layer identity, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( identity, *(activations + i) + bias );
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::identity_layer identity, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(identity, *(activations + i) ); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::identity_layer identity, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(identity, *(activations + i) );
		}
			
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( sigmoid, *(activations + i) + bias );
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::sigmoid_layer sigmoid, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(sigmoid, *(activations + i) ); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::sigmoid_layer sigmoid, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(sigmoid, *(activations + i) );
		}

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::softplus_layer softplus, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( softplus, *(activations + i) + bias );
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::softplus_layer softplus, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(softplus, *(activations + i) ); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::softplus_layer softplus, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(softplus, *(activations + i) );
		}


	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{

		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( hyperbolic_tangent, *(activations + i) + bias );
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(hyperbolic_tangent, *(activations + i) ); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::tanh_layer hyperbolic_tangent, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(hyperbolic_tangent, *(activations + i) );
		}

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::relu_layer relu, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( relu, *(activations + i) + bias );
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::relu_layer relu, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(relu, *(activations + i) ); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::relu_layer relu, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(relu, *(activations + i) );
		}

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & coefficient, const precision_type & bias)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( leaky_relu, *(activations + i) + bias, coefficient );
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(leaky_relu, *(activations + i), coefficient ); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::leaky_relu_layer leaky_relu, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(leaky_relu, *(activations + i), coefficient );
		}

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & coefficient, const precision_type & bias)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(activations + i) = objective( exp_leaky_relu, *(activations + i) + bias, coefficient );
		}

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) = *(error + i) * derivative(exp_leaky_relu, *(activations + i), coefficient); 
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
			*(deltas + i) *= derivative(exp_leaky_relu, *(activations + i), coefficient );
		}

	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::softmax_layer softmax, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias)
		{
		  std::uint32_t i{0};
		  precision_type sum{0.0};
		  precision_type max = *std::max_element(activations, activations + length);

		  for(i = 0; i < length; ++i)
		  {
			*(activations + i) = std::exp( ( *(activations + i) + bias) - max );
			// denominator
			sum += *(activations + i);
		  }
		  
		  // normalize
		  for(i = 0; i < length; ++i)
			*(activations + i) /= sum;
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::softmax_layer softmax, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, precision_type * jacobian, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
		  {
			for(std::uint32_t j = 0; j < length; ++j)
			{
			  jacobian[j] = (j == i) ? *(activations + i) * (precision_type{1.0} - *(activations + i)) : -*(activations + j) * *(activations + i);
			}
			// dot prod
  			precision_type sum{0};
			for(std::uint32_t j = 0; j < length; ++j)
			  sum += jacobian[j] * activations[j];

			// multiply by error
			*(deltas + i) = *(error + i) * sum;
		  }
		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::softmax_layer softmax, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, precision_type * jacobian, const precision_type * const activations, const std::uint32_t & length)
		{
		  for(std::uint32_t i = 0; i < length; ++i)
		  {
			for(std::uint32_t j = 0; j < length; ++j)
			  jacobian[j] = (j == i) ? *(activations + i) * (precision_type{1.0} - *(activations + i)) : -*(activations + j) * *(activations + i);

			// dot prod
		  	precision_type sum{0};
			for(std::uint32_t j = 0; j < length; ++j)
			  sum += jacobian[j] * activations[j];

			*(deltas + i) *=  sum;
		  } 
		}
/*
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, precision_type & scale, precision_type & shift, precision_type epsilon)
		{
		  std::uint32_t i;
		  precision_type batch_mean{0}, batch_variance{0};

		  for(i = 0; i < length; ++i)
			batch_mean += *(activations + i);
		  batch_mean /= length;
		  for(i = 0; i < length; ++i)
			batch_variance += ( *(activations + i) - batch_mean ) * ( *(activations + i) - batch_mean );
		  batch_variance /= length;

		  for(i = 0; i < length; ++i) 
  			*(activations + i) = ( *(activations + i) - batch_mean ) / std::sqrt(batch_variance + epsilon);
		  
		  *(activations + i) = scale * *(activations + i) + shift;

		}
	  template <class precision_type>
		HOST void activation<precision_type>::activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::derivative d, precision_type * activations, const std::uint32_t & length, precision_type & scale, precision_type & shift, precision_type epsilon)
		{
		  // to do
		}
*/
/*
	  template <class precision_type>
		template<class Callable, class ... Args>
		HOST void activation<precision_type>::activate(layer_info::generic_layer generic, zinhart::function_space::objective o, Callable && c, Args&& ...args)
		{
		}
	  template <class precision_type>
		template<class Callable, class ... Args>
		HOST void activation<precision_type>::activate(layer_info::generic_layer generic, zinhart::function_space::derivative d, Callable && c, Args&& ...args)
		{
		}
*/
                                        
	  template <class precision_type>           
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::identity_layer identity, const precision_type & x)
		{ return x; }
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::identity_layer identity, const precision_type & x )
		{ return precision_type{1}; }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::sigmoid_layer sigmoid, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return precision_type{1.0} / ( precision_type{1.0} + exp(-x) );
#else
  		  return precision_type{1.0} / ( precision_type{1.0} + std::exp(-x) );
#endif
		}
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::sigmoid_layer sigmoid, const precision_type & x)
		{ return x * (precision_type{1.0} - x); }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::softplus_layer softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return log(precision_type{1.0} + exp(x));
#else
  		  return std::log(precision_type{1.0} + std::exp(x));
#endif
		}
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::softplus_layer softplus, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return precision_type{1.0} / (precision_type{1.0} + exp(-x));
#else
  		  return precision_type{1.0} / ( precision_type{1.0} + std::exp(-x) );
#endif
		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x)
		{
#if CUDA_ENABLED == 1
  		  return tanh(x);
#else
  		  return std::tanh(x);
#endif
		}
	  template <class precision_type>
        CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x)
		{ 

#if CUDA_ENABLED == 1
  		  return precision_type{1.0} - precision_type{ x * x };
#else
		  return precision_type{1.0} - precision_type{ x * x };
#endif
		}

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::relu_layer relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? x : precision_type{0.0}; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::relu_layer relu, const precision_type & x)
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : precision_type{0.0};}

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::leaky_relu_layer leaky_relu, const precision_type & x, const precision_type & coefficient)
		{ return (x >= precision_type{0.0} ) ? x : coefficient * x; }
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::leaky_relu_layer leaky_relu, const precision_type & x, const precision_type & coefficient )
		{ return (x >= precision_type{0.0} ) ? precision_type{1.0} : coefficient;  }

	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x, const precision_type & coefficient)
		{
#if CUDA_ENABLED == 1
  		  return (x > precision_type{0.0} ) ? x : coefficient * (exp(x) - precision_type{1.0})
#else
		  return (x > precision_type{0.0} ) ? x : coefficient * (std::exp(x) - precision_type{1.0});
#endif
		}
	  template <class precision_type>
  		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x, const precision_type & coefficient)
		{ 
#if CUDA_ENABLED == 1
  		  return (x > precision_type{0.0} ) ? precision_type{1.0} : x + coefficient;
#else
  		  return (x > precision_type{0.0} ) ? precision_type{1.0} : x + coefficient;
#endif
		}
	  template <class precision_type>
		template<class Callable, class ... Args>
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::objective(layer_info::generic_layer generic, zinhart::function_space::objective o, Callable && c, Args&& ...args)
		{
#if CUDA_ENABLED == 1

#else

#endif
		}
	  template <class precision_type>
		template<class Callable, class ... Args>
		CUDA_CALLABLE_MEMBER precision_type activation<precision_type>::derivative(layer_info::generic_layer generic, zinhart::function_space::derivative d, Callable && c, Args&& ...args)
		{
#if CUDA_ENABLED == 1

#else

#endif
		}

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
