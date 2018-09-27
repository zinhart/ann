#include <concurrent_routines/serial/serial.hh>
namespace zinhart
{
  namespace models
  {
	namespace layers
	{

	  template <class precision_type>
  		HOST layer<precision_type>::layer()
		{

		}
	  template <class precision_type>
  		HOST layer<precision_type>::layer(LAYER_NAME name)
		{ init(name, 0, 0, nullptr, nullptr); }

	  template <class precision_type>
		HOST layer<precision_type>::layer(LAYER_NAME name, std::uint32_t start_index, std::uint32_t end_index, precision_type * total_activations, precision_type * total_deltas)
		{ init(name, start_index, end_index, total_activations, total_deltas); }

	  template <class precision_type>
		HOST void layer<precision_type>::init(LAYER_NAME name, std::uint32_t start_index, std::uint32_t stop_index, precision_type * total_activations, precision_type * total_deltas)
		{
		  this->start_index = start_index;
		  this->end_index = end_index;
		  this->start_activations = total_activations + start_index;
		  this->end_activations = total_activations + end_index;
		  this->start_deltas = total_deltas + start_index;
		  this->end_deltas = total_deltas + end_index;
		  activation_map = new zinhart::activation::activation_test[9]{
			              zinhart::activation::input(), zinhart::activation::identity(), zinhart::activation::sigmoid(), 
						  zinhart::activation::softplus(), zinhart::activation::hyperbolic_tangent(), zinhart::activation::relu(), 
						  zinhart::activation::leaky_relu(), zinhart::activation::exp_leaky_relu(), 
						  zinhart::activation::softmax()/*, zinhart::activation::batch_normalization()*/
						 };
		  activation_map1 = new zinhart::activation::activation_test * [9];
		  activation_map1[0] = new zinhart::activation::input();
		  activation_map1[1] = new zinhart::activation::identity();
		  activation_map1[2] = new zinhart::activation::sigmoid();
		  activation_map1[3] = new zinhart::activation::softplus();
		  activation_map1[4] = new zinhart::activation::hyperbolic_tangent();
		  activation_map1[5] = new zinhart::activation::relu();
		  activation_map1[6] = new zinhart::activation::leaky_relu();
		  activation_map1[7] = new zinhart::activation::exp_leaky_relu();
		  activation_map1[8] = new zinhart::activation::softmax();
		}

	  template <class precision_type>
  		template <class ... Args>
		HOST void layer<precision_type>::objective(LAYER_NAME Name, std::uint32_t thread_id, std::uint32_t n_threads, Args && ... args)
		{
		  std::uint32_t start{0}, stop{0}, i{0};
		  zinhart::serial::map(thread_id, n_threads, get_total_nodes(), start, stop);
		  for(i = start; i != stop; ++i)
		  {
			// objective
			*(activation_map[std::uint32_t{Name}]).objective(*(start_activations + i), args...);

		//	auto x = static_cast<decltype()>();

			// get the specific activation from the hash_table using layer_name which is determined when a layer object is created
			// have to downcase to typeof(layer_name::Name)
		//	*activation_map[std::uint32_t{Name}].objective(*(start_activations + i), args...);
		  }

		}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_start_index()const
	    { return start_index;}

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_end_index()const
		{ return end_index; }

	  template <class precision_type>
		CUDA_CALLABLE_MEMBER std::uint32_t layer<precision_type>::get_total_nodes()const
		{ return end_index - start_index; }

	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
