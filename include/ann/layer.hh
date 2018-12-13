#ifndef LAYER_HH
#define LAYER_HH
#include <ann/function_space.hh>
#include <cassert>
#include <utility>
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
		enum generic_layer : std::uint32_t;
		enum hidden_layer : std::uint32_t;
		enum output_layer : std::uint32_t;
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
		  generic_layer universal_layer;
		};
	  }

	  template <class precision_type>
		class activation
		{
		  public:
			HOST activation() = default;
			HOST activation(const activation&) = default;
			HOST activation(activation&&) = default;
			HOST activation & operator = (const activation&) = default;
			HOST activation & operator = (activation&&) = default;
			HOST void activate(layer_info::input_layer input, zinhart::function_space::objective o, const precision_type & bias = 1);
			HOST void activate(layer_info::input_layer input, zinhart::function_space::derivative d);

			HOST void activate(layer_info::identity_layer identity, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1);
			HOST void activate(layer_info::identity_layer identity, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::identity_layer identity, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length);

			HOST void activate(layer_info::sigmoid_layer sigmoid, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1);
			HOST void activate(layer_info::sigmoid_layer sigmoid, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::sigmoid_layer sigmoid, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length);

			HOST void activate(layer_info::softplus_layer softplus, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1);
			HOST void activate(layer_info::softplus_layer softplus, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::softplus_layer softplus, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length);

			HOST void activate(layer_info::tanh_layer hyperbolic_tangent, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1);
			HOST void activate(layer_info::tanh_layer hyperbolic_tangent, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::tanh_layer hyperbolic_tangent, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length);

			HOST void activate(layer_info::relu_layer relu, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1);
			HOST void activate(layer_info::relu_layer relu, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::relu_layer relu, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length);

			HOST void activate(layer_info::leaky_relu_layer leaky_relu, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & coefficient = 0.1, const precision_type & bias = 1);
			HOST void activate(layer_info::leaky_relu_layer leaky_relu, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient = 0.1);
			HOST void activate(layer_info::leaky_relu_layer leaky_relu, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient = 0.1);

			HOST void activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & coefficient = 0.1, const precision_type & bias = 1);
			HOST void activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient = 0.1);
			HOST void activate(layer_info::exp_leaky_relu_layer exp_leaky_relu, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length, const precision_type & coefficient = 0.1);

			HOST void activate(layer_info::softmax_layer softmax, zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1);
			HOST void activate(layer_info::softmax_layer softmax, layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, precision_type * jacobian, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::softmax_layer softmax, layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, precision_type * jacobian, const precision_type * const activations, const std::uint32_t & length);

			HOST void activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::objective o, precision_type * start, const std::uint32_t & length,                                precision_type & scale = 1, precision_type & shift = 0, precision_type epsilon = 1.e-6);
			HOST void activate(layer_info::batch_normalization_layer batch_norm, zinhart::function_space::derivative d, precision_type * start, const std::uint32_t & length,
				               precision_type & scale, precision_type & shift, precision_type epsilon = 1.e-6);
/*
			template<class Callable, class ... Args>
			  HOST void activate(layer_info::generic_layer generic_layer, zinhart::function_space::objective o, Callable && c, Args&& ...args);
			template<class Callable, class ... Args>
			  HOST void activate(layer_info::generic_layer generic_layer, zinhart::function_space::derivative d, Callable && c, Args&& ...args);
*/
			// vectorized activation functions and their first order derivatives
  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::identity_layer identity, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::identity_layer identity, const precision_type & x);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::sigmoid_layer sigmoid, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::sigmoid_layer sigmoid, const precision_type & x);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::softplus_layer softplus, const precision_type & x);
  			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::softplus_layer softplus, const precision_type & x);

	        CUDA_CALLABLE_MEMBER precision_type objective(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x);
	        CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::tanh_layer hyperbolic_tangent, const precision_type & x);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::relu_layer relu, const precision_type & x);
			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::relu_layer relu, const precision_type & x);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::leaky_relu_layer leaky_relu, const precision_type & x, const precision_type & coefficient = 0.1);
 			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::leaky_relu_layer leaky_relu, const precision_type & x, const precision_type & coefficient = 0.1);

  			CUDA_CALLABLE_MEMBER precision_type objective(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x, const precision_type & coefficient = 0.1);
			CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::exp_leaky_relu_layer exp_leaky_relu, const precision_type & x, const precision_type & coefficient = 0.1);

			template<class Callable, class ... Args>
			  CUDA_CALLABLE_MEMBER precision_type objective(layer_info::generic_layer generic, zinhart::function_space::objective o, Callable && c, Args&& ...args);
			template<class Callable, class ... Args>
			  CUDA_CALLABLE_MEMBER precision_type derivative(layer_info::generic_layer generic, zinhart::function_space::derivative d, Callable && c, Args&& ...args);
		};

	  template<class precision_type>
		class layer
		{
		  public:
			HOST virtual ~layer() = default;
			HOST void activate(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0);	
			HOST void activate(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length);
			HOST void activate(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length);
			HOST void set_size(std::uint32_t size);
			HOST std::uint32_t get_size()const;
			HOST void set_bias(precision_type bias);
			HOST precision_type get_bias()const;
			HOST std::string name()const;
		  protected:
			std::uint32_t size;
			precision_type bias;
			activation<precision_type> a;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) = 0;	
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length)= 0;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length)= 0;
			HOST virtual void set_size_impl(std::uint32_t size) = 0;
			HOST virtual std::uint32_t get_size_impl()const = 0;
			HOST virtual void set_bias_impl(precision_type bias) = 0;
			HOST virtual precision_type get_bias_impl()const = 0;
			HOST virtual std::string name_impl()const = 0;

		};

	  template<class precision_type>
  		class input_layer : public layer<precision_type>
		{
		  public:
			HOST input_layer() = default;
			HOST input_layer(const input_layer&) = default;
			HOST input_layer(input_layer&&) = default;
			HOST input_layer & operator = (const input_layer&) = default;
			HOST input_layer & operator = (input_layer&&) = default;
			HOST ~input_layer() = default; 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class identity_layer : public layer<precision_type>
		{
		  public:
			HOST identity_layer() = default;
			HOST identity_layer(const identity_layer&) = default;
			HOST identity_layer(identity_layer&&) = default;
			HOST identity_layer & operator = (const identity_layer&) = default;
			HOST identity_layer & operator = (identity_layer&&) = default;
			HOST ~identity_layer() = default; 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class sigmoid_layer : public layer<precision_type>
		{
		  public:
			HOST sigmoid_layer() = default;
			HOST sigmoid_layer(const sigmoid_layer&) = default;
			HOST sigmoid_layer(sigmoid_layer&&) = default;
			HOST sigmoid_layer & operator = (const sigmoid_layer&) = default;
			HOST sigmoid_layer & operator = (sigmoid_layer&&) = default;
			HOST ~sigmoid_layer() = default; 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
  		};


	  template<class precision_type>
  		class softplus_layer : public layer<precision_type>
		{
		  public:
			HOST softplus_layer() = default;
			HOST softplus_layer(const softplus_layer&) = default;
			HOST softplus_layer(softplus_layer&&) = default;
			HOST softplus_layer & operator = (const softplus_layer&) = default;
			HOST softplus_layer & operator = (softplus_layer&&) = default;
			HOST ~softplus_layer() = default; 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class tanh_layer : public layer<precision_type>
		{
		  public:
			HOST tanh_layer() = default;
			HOST tanh_layer(const tanh_layer&) = default;
			HOST tanh_layer(tanh_layer&&) = default;
			HOST tanh_layer & operator = (const tanh_layer&) = default;
			HOST tanh_layer & operator = (tanh_layer&&) = default;
			HOST ~tanh_layer() = default; 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class relu_layer : public layer<precision_type>
		{
		  public:
			HOST relu_layer() = default;
			HOST relu_layer(const relu_layer&) = default;
			HOST relu_layer(relu_layer&&) = default;
			HOST relu_layer & operator = (const relu_layer&) = default;
			HOST relu_layer & operator = (relu_layer&&) = default;
			HOST ~relu_layer() = default; 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class leaky_relu_layer : public layer<precision_type>
		{
		  public:
			HOST leaky_relu_layer() = default;
			HOST leaky_relu_layer(const leaky_relu_layer&) = default;
			HOST leaky_relu_layer(leaky_relu_layer&&) = default;
			HOST leaky_relu_layer & operator = (const leaky_relu_layer&) = default;
			HOST leaky_relu_layer & operator = (leaky_relu_layer&&) = default;
			HOST ~leaky_relu_layer() = default; 
			precision_type coefficient{0.1};
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class exp_leaky_relu_layer : public layer<precision_type>
		{
		  public:
			HOST exp_leaky_relu_layer() = default;
			HOST exp_leaky_relu_layer(const exp_leaky_relu_layer&) = default;
			HOST exp_leaky_relu_layer(exp_leaky_relu_layer&&) = default;
			HOST exp_leaky_relu_layer & operator = (const exp_leaky_relu_layer&) = default;
			HOST exp_leaky_relu_layer & operator = (exp_leaky_relu_layer&&) = default;
			HOST ~exp_leaky_relu_layer() = default; 
			precision_type coefficient{0.1};
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};

	  template<class precision_type>
  		class softmax_layer : public layer<precision_type>
		{
		  public:
			// force construction with a size so that the jacobian matrix can be properly initialized
			HOST softmax_layer(std::uint32_t size);
			HOST softmax_layer(const softmax_layer&);
			HOST softmax_layer(softmax_layer&&);
			HOST softmax_layer & operator = (const softmax_layer&);
			HOST softmax_layer & operator = (softmax_layer&&);
			HOST ~softmax_layer(); 
		  private:
			using layer<precision_type>::a;
			using layer<precision_type>::size;
			using layer<precision_type>::bias;
			HOST std::uint32_t jacobian_size;
			HOST void set_jacobian_size(std::uint32_t size);
			std::uint32_t get_jacobian_size()const;
			precision_type * jacobian{nullptr};
			HOST virtual void activate_impl(zinhart::function_space::objective o, precision_type * activations, const std::uint32_t & length, const precision_type & bias = 1.0) override;
			HOST virtual void activate_impl(layer_info::output_layer o, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const error, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void activate_impl(layer_info::hidden_layer h, zinhart::function_space::derivative d, precision_type * deltas, const precision_type * const activations, const std::uint32_t & length) override;
			HOST virtual void set_size_impl(std::uint32_t size) override;
			HOST virtual std::uint32_t get_size_impl()const override;
			HOST virtual void set_bias_impl(precision_type bias)override;
			HOST virtual precision_type get_bias_impl()const override;
			HOST virtual std::string name_impl()const override;
		};
	  template <class precision_type>
		std::shared_ptr<layer<precision_type>> make_layer(std::string name, std::uint32_t size, precision_type hint = 0.0)
		{
		  if(name == "input")
		  {
			std::shared_ptr<layer<precision_type>> input{std::make_shared<input_layer<precision_type>>()};
			input->set_size(size);
			return input;
		  }
		  else if(name == "identity")
		  {
			std::shared_ptr<layer<precision_type>> identity{std::make_shared<identity_layer<precision_type>>()};
			identity->set_size(size);
			return identity;
		  }
		  
		  else if(name == "sigmoid")
		  {
			std::shared_ptr<layer<precision_type>> sigmoid{std::make_shared<sigmoid_layer<precision_type>>()};
			sigmoid->set_size(size);
			return sigmoid;
		  }
		  else if(name == "softplus")
		  {
			std::shared_ptr<layer<precision_type>> softplus{std::make_shared<softplus_layer<precision_type>>()};
			softplus->set_size(size);
			return softplus;
		  }
		  else if(name == "tanh")
		  {
			std::shared_ptr<layer<precision_type>> tanh{std::make_shared<tanh_layer<precision_type>>()};
			tanh->set_size(size);
			return tanh;
		  }
		  else if(name == "relu")
		  {
			std::shared_ptr<layer<precision_type>> relu{std::make_shared<relu_layer<precision_type>>()};
			relu->set_size(size);
			return relu;
		  }
		  else if(name == "leaky_relu")
		  {
			std::shared_ptr<layer<precision_type>> leaky_relu{std::make_shared<leaky_relu_layer<precision_type>>()};
			leaky_relu->set_size(size);
			return leaky_relu;
		  }
		  else if(name == "exp_leaky_relu")
		  {
			std::shared_ptr<layer<precision_type>> exp_leaky_relu{std::make_shared<exp_leaky_relu_layer<precision_type>>()};
			exp_leaky_relu->set_size(size);
			return exp_leaky_relu;
		  }
		  else if(name == "softmax")
		  {
			std::shared_ptr<layer<precision_type>> softmax{std::make_shared<softmax_layer<precision_type>>(size)};
			return softmax;
		  }
		  
		}
	}//END NAMESPACE LAYERS
  }//END NAMESPACE MODELS
}//END NAMESPACE ZINHART
#include <ann/ext/layer.tcc>
#endif
