#ifndef LAYER_HH
#define LAYER_HH
#include <ann/activation.hh>
namespace zinhart
{
  namespace models
  {
	namespace layers
	{
	  enum class layer_name : std::uint32_t {INPUT = std::uint32_t{0}, IDENTITY, SIGMOID, SOFTPLUS, TANH, RELU, LEAKY_RELU, EXP_LEAKY_RELU, SOFTMAX, BATCH_NORMALIZATION};
	  	  template <class precision_type>
		class layer
		{
		  private:
			std::uint32_t start_index;
			std::uint32_t end_index;
			precision_type * start_activations;
			precision_type * end_activations;
			precision_type * start_deltas;
			precision_type * end_deltas;
			layer_name name;
#if CUDA_ENABLED == 1
#else

			// layer name will be used to as hashes to return the proper activation object
			zinhart::activation::activation_test activation_map[9] = {zinhart::activation::input(), zinhart::activation::identity(), 
						  zinhart::activation::sigmoid(), 
						  zinhart::activation::softplus(), zinhart::activation::hyperbolic_tangent(), zinhart::activation::relu(), 
						  zinhart::activation::leaky_relu(), zinhart::activation::exp_leaky_relu(), 
						  zinhart::activation::softmax()/*, zinhart::activation::batch_normalization()*/
						 };
#endif
			// activation_function f; //to do later
		  public:
			HOST layer() = delete;
			HOST layer(const layer&) = default;
			HOST layer(layer&&) = default;
			HOST layer & operator = (const layer&) = default;
			HOST layer & operator = (layer&&) = default;
			HOST layer(layer_name name);
			HOST layer(layer_name name, std::uint32_t start, std::uint32_t stop, precision_type * total_activations, precision_type * total_deltas);
			HOST void init(layer_name name, std::uint32_t start, std::uint32_t stop, precision_type * total_activations, precision_type * total_deltas);
			template <class ... Args>
  			  HOST void activate(Args && ... args);
			CUDA_CALLABLE_MEMBER std::uint32_t get_start_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_end_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_total_nodes()const;
		};

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
#include <ann/ext/layer.tcc>
#endif
