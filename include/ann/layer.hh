#ifndef LAYER_HH
#define LAYER_HH
namespace zinhart
{
  namespace models
  {
	namespace layers
	{
	  enum class layer_name : std::uint32_t {INPUT = std::uint32_t{0}, IDENTITY, SIGMOID, SOFTPLUS, TANH, RELU, LEAKY_RELU, EXP_LEAKY_RELU, SOFTMAX};
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
			CUDA_CALLABLE_MEMBER std::uint32_t get_start_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_end_index()const;
			CUDA_CALLABLE_MEMBER std::uint32_t get_total_nodes()const;
		};
	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
#include <ann/ext/layer.tcc>
#endif
