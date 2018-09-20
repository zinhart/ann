#ifndef ACTIVATION_HH
#define ACTIVATION_HH
#include "concurrent_routines/concurrent_routines_error.hh"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <iostream>
#if CUDA_ENABLED == true
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  namespace activation
  {
	enum class ACTIVATION_NAME : std::uint32_t {INPUT = std::uint32_t{0}, IDENTITY, SIGMOID, SOFTPLUS, TANH, RELU, LEAKY_RELU, EXP_LEAKY_RELU, SOFTMAX};
	enum class ACTIVATION_TYPE : std::uint32_t {OBJECTIVE = std::uint32_t{0}, DERIVATIVE}; 
	std::uint32_t total_activation_types();
	using Neurons = std::uint32_t;
	using LAYER_INFO = std::pair<ACTIVATION_NAME, Neurons>;
	template <class ACTIVATION_FUNCTION>
	  class activation_interface;
	class identity;
	class sigmoid;
	class softplus;
	class hyperbolic_tangent;
	class relu;
	class leaky_relu;
	class exp_leaky_relu;
	class softmax;
	class activation_function
	{
	  public:
		CUDA_CALLABLE_MEMBER activation_function() = default;
		CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
		CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
		CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
		CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
		CUDA_CALLABLE_MEMBER ~activation_function() = default;
		template <class precision_type>
		  HOST precision_type operator ()(ACTIVATION_NAME name, ACTIVATION_TYPE at, precision_type x);
	};

	// activation_interface will be the common interface!
	template <class ACTIVATION_FUNCTION>
	  class activation_interface : public activation_function
	  {
		public:
		  CUDA_CALLABLE_MEMBER activation_interface() = default;
		  CUDA_CALLABLE_MEMBER activation_interface(const activation_interface&) = default;
		  CUDA_CALLABLE_MEMBER activation_interface(activation_interface&&) = default;
		  CUDA_CALLABLE_MEMBER activation_interface & operator = (const activation_interface&) = default;
		  CUDA_CALLABLE_MEMBER activation_interface & operator = (activation_interface&&) = default;
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator ()( ACTIVATION_TYPE at, precision_type x);
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator ()( ACTIVATION_TYPE at, precision_type x, const precision_type coefficient);
	  };
	// a placeholder class used simulate all the different activations as 1 type
	class activation_test
	{
	  public:
		CUDA_CALLABLE_MEMBER activation_test() = default;
		CUDA_CALLABLE_MEMBER activation_test(const activation_test&) = default;
		CUDA_CALLABLE_MEMBER activation_test(activation_test&&) = default;
		CUDA_CALLABLE_MEMBER activation_test & operator = (const activation_test&) = default;
		CUDA_CALLABLE_MEMBER activation_test & operator = (activation_test&&) = default;
		CUDA_CALLABLE_MEMBER ~activation_test() = default;
	};

	class input : public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER input() = default;
		 CUDA_CALLABLE_MEMBER input(const input&) = default;
		 CUDA_CALLABLE_MEMBER input(input&&) = default;
		 CUDA_CALLABLE_MEMBER input & operator = (const input&) = default;
		 CUDA_CALLABLE_MEMBER input & operator = (input&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);
	};
	class identity : public activation_interface<identity>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER identity() = default;
		 CUDA_CALLABLE_MEMBER identity(const identity&) = default;
		 CUDA_CALLABLE_MEMBER identity(identity&&) = default;
		 CUDA_CALLABLE_MEMBER identity & operator = (const identity&) = default;
		 CUDA_CALLABLE_MEMBER identity & operator = (identity&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);
	};

	class sigmoid : public activation_interface<sigmoid>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER sigmoid() = default;
		 CUDA_CALLABLE_MEMBER sigmoid(const sigmoid&) = default;
		 CUDA_CALLABLE_MEMBER sigmoid(sigmoid&&) = default;
		 CUDA_CALLABLE_MEMBER sigmoid & operator = (const sigmoid&) = default;
		 CUDA_CALLABLE_MEMBER sigmoid & operator = (sigmoid&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);
	};

	class softplus : public activation_interface<softplus>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER softplus() = default;
		 CUDA_CALLABLE_MEMBER softplus(const softplus&) = default;
		 CUDA_CALLABLE_MEMBER softplus(softplus&&) = default;
		 CUDA_CALLABLE_MEMBER softplus & operator = (const softplus&) = default;
		 CUDA_CALLABLE_MEMBER softplus & operator = (softplus&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);

	};

	class hyperbolic_tangent : public activation_interface<hyperbolic_tangent>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER hyperbolic_tangent() = default;
		 CUDA_CALLABLE_MEMBER hyperbolic_tangent(const hyperbolic_tangent&) = default;
		 CUDA_CALLABLE_MEMBER hyperbolic_tangent(hyperbolic_tangent&&) = default;
		 CUDA_CALLABLE_MEMBER hyperbolic_tangent & operator = (const hyperbolic_tangent&) = default;
		 CUDA_CALLABLE_MEMBER hyperbolic_tangent & operator = (hyperbolic_tangent&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);
	};

	class relu : public activation_interface<relu>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER relu() = default;
		 CUDA_CALLABLE_MEMBER relu(const relu&) = default;
		 CUDA_CALLABLE_MEMBER relu(relu&&) = default;
		 CUDA_CALLABLE_MEMBER relu & operator = (const relu&) = default;
		 CUDA_CALLABLE_MEMBER relu & operator = (relu&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);
	};



	class leaky_relu : public activation_interface<leaky_relu>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER leaky_relu() = default;
		 CUDA_CALLABLE_MEMBER leaky_relu(const leaky_relu&) = default;
		 CUDA_CALLABLE_MEMBER leaky_relu(leaky_relu&&) = default;
		 CUDA_CALLABLE_MEMBER leaky_relu & operator = (const leaky_relu&) = default;
		 CUDA_CALLABLE_MEMBER leaky_relu & operator = (leaky_relu&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x, precision_type leakage_coefficient = 0.1);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x, precision_type leakage_coefficient = 0.1);
	};



	class exp_leaky_relu : public activation_interface<exp_leaky_relu>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER exp_leaky_relu() = default;
		 CUDA_CALLABLE_MEMBER exp_leaky_relu(const exp_leaky_relu&) = default;
		 CUDA_CALLABLE_MEMBER exp_leaky_relu(exp_leaky_relu&&) = default;
		 CUDA_CALLABLE_MEMBER exp_leaky_relu & operator = (const exp_leaky_relu&) = default;
		 CUDA_CALLABLE_MEMBER exp_leaky_relu & operator = (exp_leaky_relu&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x, precision_type leakage_coefficient = 0.1);
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x, precision_type leakage_coefficient = 0.1);
	};

	class softmax : public activation_interface<softmax>, public activation_test
	{
	  public:
		 CUDA_CALLABLE_MEMBER softmax() = default;
		 CUDA_CALLABLE_MEMBER softmax(const softmax&) = default;
		 CUDA_CALLABLE_MEMBER softmax(softmax&&) = default;
		 CUDA_CALLABLE_MEMBER softmax & operator = (const softmax&) = default;
		 CUDA_CALLABLE_MEMBER softmax & operator = (softmax&&) = default;
		 template <class precision_type>
		   CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & x);
		 template <class precision_type>
		   //to do
		   CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & x);
	};

	template<class ACTIVATION_FUNCTION>
	  HOST activation_interface<ACTIVATION_FUNCTION> get_activation_interface(ACTIVATION_NAME name);
	HOST std::string get_activation_name(ACTIVATION_NAME name);
#if CUDA_ENABLED == 1
	//wrppers for host functions to use to call kernels here, the wrappers will calculate the block_parameters and the threads per block
	template <class precision_type>
	  HOST std::int32_t call_activation_function(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, precision_type * device_Wx_plus_b, const std::uint32_t & current_layer_size, const std::uint32_t & device_id = 0);
	template <class precision_type>
	  HOST std::int32_t call_activation_function(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, precision_type * device_Wx_plus_b, const std::uint32_t & current_layer_size, const cudaStream_t & stream, const std::uint32_t & device_id = 0);

	std::int32_t call_activation_function(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, double coefficient, std::uint32_t layer_size);
	#endif
  }// END NAMESPACE ACTIVATION
}// END NAMESPACE ZINHART
#include <ann/ext/activation.tcc>
#endif
