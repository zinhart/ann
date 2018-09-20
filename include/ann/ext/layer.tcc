namespace zinhart
{
  namespace models
  {
	namespace layers
	{
  //	  static const layer_name activation_map[] = {layer_name::INPUT, layer_name::IDENTITY, layer_name::SIGMOID, layer_name::SOFTPLUS, layer_name::TANH, layer_name::RELU, layer_name::LEAKY_RELU, layer_name::EXP_LEAKY_RELU, layer_name::SOFTMAX, layer_name::BATCH_NORMALIZATION};
	//template <class precision_type>
  	  //const layer_name * layer<precision_type>::activation_table = activation_map;
	  template <class precision_type>
  		HOST layer<precision_type>::layer(layer_name name)
		{ init(name, 0, 0, nullptr, nullptr); }

	  template <class precision_type>
		HOST layer<precision_type>::layer(layer_name name, std::uint32_t start_index, std::uint32_t end_index, precision_type * total_activations, precision_type * total_deltas)
		{ init(name, start_index, end_index, total_activations, total_deltas); }

	  template <class precision_type>
		HOST void layer<precision_type>::init(layer_name name, std::uint32_t start_index, std::uint32_t stop_index, precision_type * total_activations, precision_type * total_deltas)
		{
		  this->name = name;
		  this->start_index = start_index;
		  this->end_index = end_index;
		  this->start_activations = total_activations + start_index;
		  this->end_activations = total_activations + end_index;
		  this->start_deltas = total_deltas + start_index;
		  this->end_deltas = total_deltas + end_index;
		}

	  template <class ... Args>
		HOST void activate(Args && ... args)
		{
		  // get the specific activation from the hash_table using layer_name which is determined when a layer object is created
		  
		  
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
