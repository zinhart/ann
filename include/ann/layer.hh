#ifndef LAYER_HH
#define LAYER_HH
#include <ann/function_space.hh>
#include <cassert>
//#include <utility> // for std::forward
namespace zinhart
{
  namespace models
  {
	namespace layers
	{
	  namespace layer_info
	  {
		enum input_layer : std::uint32_t;
		enum identity_layer : std::uint32_t;
		enum sigmoid_layer : std::uint32_t;
		enum softplus_layer : std::uint32_t;
		enum tanh_layer : std::uint32_t;
		enum relu_layer : std::uint32_t;
		enum leaky_relu_layer : std::uint32_t;
		enum exp_leaky_relu_layer : std::uint32_t;
		enum softmax_layer : std::uint32_t;
		enum batch_normalization_layer : std::uint32_t;
		// grouped into one type for conveniece
		union layer_type
		{
		  input_layer input;
		  identity_layer identity;
		  sigmoid_layer sigmoid;
		  softplus_layer softplus;
		  tanh_layer hyperbolic_tangent;
		  relu_layer relu;
		  leaky_relu_layer leaky_relu;
		  exp_leaky_relu_layer exp_leaky_relu;
		  softmax_layer softmax;
		  batch_normalization_layer batch_normalization;
		};
	  }
	  // thread safe layer class
	  template <class precision_type>
		class layer
		{
		  private:
			std::uint32_t start_index;
			std::uint32_t stop_index;
			precision_type * start_activations;
			precision_type * stop_activations;
			precision_type * start_deltas;
			precision_type * stop_deltas;
			precision_type coefficient;
			CUDA_CALLABLE_MEMBER  precision_type * get_start_activations()const;
			CUDA_CALLABLE_MEMBER  precision_type * get_stop_activations()const;
			CUDA_CALLABLE_MEMBER  precision_type * get_start_deltas()const;
			CUDA_CALLABLE_MEMBER  precision_type * get_stop_deltas()const;
		  public:
			enum class activation : std::uint32_t {input = std::uint32_t{0}, identity, sigmoid, softplus, tanh, relu, leaky_relu, exp_leaky_relu, softmax, batch_norm};
			HOST layer & operator = (const layer&) = delete;
			HOST layer & operator = (layer&&) = delete;
			HOST layer();
			HOST layer(const layer&);
			HOST layer(layer&&);
			HOST layer(std::uint32_t start, std::uint32_t stop, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient = 1);
			HOST void init(std::uint32_t start, std::uint32_t stop, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient = 1);
			CUDA_CALLABLE_MEMBER std::uint32_t get_start_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_stop_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_total_nodes()const;
			CUDA_CALLABLE_MEMBER void set_coefficient(const precision_type & coefficient);
			CUDA_CALLABLE_MEMBER precision_type get_coefficient()const;
  			
			HOST void activate(layer_info::input_layer input, zinhart::function_space::objective o);
			HOST void activate(layer_info::identity_layer identity, zinhart::function_space::objective o);
			HOST void activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::objective o);
			HOST void activate(layer_info::softplus_layer softplus, zinhart::function_space::objective o);
			HOST void activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::objective o);
			HOST void activate(layer_info::relu_layer relu, zinhart::function_space::objective o);
			HOST void activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::objective o);
			HOST void activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::objective o);
			HOST void activate(layer_info::softmax_layer softmax, zinhart::function_space::objective o);
			HOST void activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::objective o);

			HOST void activate(layer_info::input_layer input, zinhart::function_space::derivative d);
			HOST void activate(layer_info::identity_layer identity, zinhart::function_space::derivative d);
			HOST void activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::derivative d);
			HOST void activate(layer_info::softplus_layer softplus, zinhart::function_space::derivative d);
			HOST void activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::derivative d);
			HOST void activate(layer_info::relu_layer relu, zinhart::function_space::derivative d);
			HOST void activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::derivative d);
			HOST void activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::derivative d);
			HOST void activate(layer_info::softmax_layer softmax, zinhart::function_space::derivative d);
			HOST void activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::derivative d);

			
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::identity_layer identity, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::sigmoid_layer sigmoid, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::softplus_layer sigmoid, const precision_type & x);
	        CUDA_CALLABLE_MEMBER precision_type objective(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::relu_layer relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::leaky_relu_layer leaky_relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x);


  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::identity_layer identity, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::sigmoid_layer sigmoid, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::softplus_layer softplus, const precision_type & x);
	        CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::relu_layer relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::leaky_relu_layer leaky_relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x);
		};

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
#include <ann/ext/layer.tcc>
#endif
