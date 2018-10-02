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
	  namespace layer_name
	  {
		
		enum input : std::uint32_t;
		enum identity : std::uint32_t;
		enum sigmoid : std::uint32_t;
		enum softplus : std::uint32_t;
		enum hyperbolic_tangent : std::uint32_t;
		enum relu : std::uint32_t;
		enum leaky_relu : std::uint32_t;
		enum exp_leaky_relu : std::uint32_t;
		enum softmax : std::uint32_t;
		enum batch_normalization : std::uint32_t;
	/*	enum input;
		enum identity;
		enum sigmoid;
		enum softplus;
		enum hyperbolic_tangent;
		enum relu;
		enum leaky_relu;
		enum exp_leaky_relu;
		enum softmax;
		enum batch_normalization;*/
	  }

	  template <class precision_type>
		class layer
		{
		  private:
			std::uint32_t start_index;
			std::uint32_t end_index;
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
			HOST layer & operator = (const layer&) = delete;
			HOST layer & operator = (layer&&) = delete;
			HOST layer();
			HOST layer(const layer&);
			HOST layer(layer&&);
			HOST layer(std::uint32_t start, std::uint32_t stop, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient = 1);
			HOST void init(std::uint32_t start, std::uint32_t stop, precision_type * total_activations, precision_type * total_deltas, const precision_type & coefficient = 1);
			CUDA_CALLABLE_MEMBER std::uint32_t get_start_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_end_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_total_nodes()const;
			CUDA_CALLABLE_MEMBER void set_coefficient(const precision_type & coefficient);
			CUDA_CALLABLE_MEMBER precision_type get_coefficient()const;
            template <class activation_name>			
  			  HOST void activate(activation_name name, zinhart::function_space::OBJECTIVE obj);
//			HOST void activate(layer_name::softmax, zinhart::function_space::OBJECTIVE obj);
//			HOST void activate(layer_name::batch_normalization, zinhart::function_space::OBJECTIVE obj);
			template <class activation_name>			
			  HOST void activate(activation_name name, zinhart::function_space::DERIVATIVE der);

//			HOST void activate(layer_name::softmax, zinhart::function_space::DERIVATIVE obj);
//			HOST void activate(layer_name::batch_normalization, zinhart::function_space::DERIVATIVE obj);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_name::identity identity, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_name::sigmoid sigmoid, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_name::softplus sigmoid, const precision_type & x);
	        CUDA_CALLABLE_MEMBER precision_type objective(layer_name::hyperbolic_tangent hyperbolic_tangent, const precision_type & x);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_name::relu relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_name::leaky_relu leaky_relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_name::exp_leaky_relu exp_leaky_relu, const precision_type & x);
  			//CUDA_CALLABLE_MEMBER precision_type objective(layer_name::softmax name, const precision_type & x);
  			//CUDA_CALLABLE_MEMBER precision_type objective(layer_name::batch_normalization name, const precision_type & x);


  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::identity identity, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::sigmoid sigmoid, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::softplus softplus, const precision_type & x);
	        CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::hyperbolic_tangent hyperbolic_tangent, const precision_type & x);


  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::relu relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::leaky_relu leaky_relu, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::exp_leaky_relu exp_leaky_relu, const precision_type & x);
  			//CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::softmax name, const precision_type & x);
  			//CUDA_CALLABLE_MEMBER precision_type derivative(layer_name::batch_normalization name, const precision_type & x);
		};

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
#include <ann/ext/layer.tcc>
#endif
