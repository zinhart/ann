#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "typedefs.cuh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include <cstdint>
#include <utility>
#if CUDA_ENABLED == true
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  enum class ACTIVATION_NAME : std::uint32_t {INPUT = std::uint32_t(1), IDENTITY, SIGMOID, SOFTPLUS, TANH, RELU, LEAKY_RELU, EXP_LEAKY_RELU, SOFTMAX};
  enum class ACTIVATION_TYPE : std::uint32_t {OBJECTIVE = std::uint32_t(0), DERIVATIVE}; 
  using Neurons = std::uint32_t;
  using LAYER_INFO = std::pair<ACTIVATION_NAME, Neurons>;

  // activation will be the common interface!
  template <class ACTIVATION_FUNCTION>
	class activation
	{
	  public:
		CUDA_CALLABLE_MEMBER activation() = default;
		CUDA_CALLABLE_MEMBER activation(const activation&) = default;
		CUDA_CALLABLE_MEMBER activation(activation&&) = default;
		CUDA_CALLABLE_MEMBER activation & operator = (const activation&) = default;
		CUDA_CALLABLE_MEMBER activation & operator = (activation&&) = default;
		template <class Precision_Type>
		  CUDA_CALLABLE_MEMBER Precision_Type operator ()(Precision_Type x, ACTIVATION_TYPE at)
		  { return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x); };
		template <class Precision_Type>
		  CUDA_CALLABLE_MEMBER Precision_Type operator ()(Precision_Type x, ACTIVATION_TYPE at, const Precision_Type coefficient)
		  { return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x, coefficient) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x, coefficient); };
	};

  class identity : public activation<identity>
  {
	public:
	   CUDA_CALLABLE_MEMBER identity() = default;
	   CUDA_CALLABLE_MEMBER identity(const identity&) = default;
	   CUDA_CALLABLE_MEMBER identity(identity&&) = default;
	   CUDA_CALLABLE_MEMBER identity & operator = (const identity&) = default;
	   CUDA_CALLABLE_MEMBER identity & operator = (identity&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x)
	     {return x;};
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x)
	     {return Precision_Type{1};};
  };

  class sigmoid : public activation<sigmoid>
  {
	public:
	   CUDA_CALLABLE_MEMBER sigmoid() = default;
	   CUDA_CALLABLE_MEMBER sigmoid(const sigmoid&) = default;
	   CUDA_CALLABLE_MEMBER sigmoid(sigmoid&&) = default;
	   CUDA_CALLABLE_MEMBER sigmoid & operator = (const sigmoid&) = default;
	   CUDA_CALLABLE_MEMBER sigmoid & operator = (sigmoid&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type  x)
		 {
#if CUDA_ENABLED == 1
			return Precision_Type{1.0} / ( Precision_Type{1.0} + exp(-x) );
#else
			return Precision_Type{1.0} / ( Precision_Type{1.0} + std::exp(-x) );
#endif
		 };
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x)
		 {
		   return x * (Precision_Type(1.0) - x);
		 };
  };
  class softmax : public activation<softmax>
  {
	public:
	   CUDA_CALLABLE_MEMBER softmax() = default;
	   CUDA_CALLABLE_MEMBER softmax(const softmax&) = default;
	   CUDA_CALLABLE_MEMBER softmax(softmax&&) = default;
	   CUDA_CALLABLE_MEMBER softmax & operator = (const softmax&) = default;
	   CUDA_CALLABLE_MEMBER softmax & operator = (softmax&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x)
		 {
#if CUDA_ENABLED == 1
			return exp(x);
#else
			return std::exp(x);
#endif
		 };
	   template <class Precision_Type>
		 //to do
	     CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x)
	     {return x;};
  };


  class softplus : public activation<softplus>
  {
	public:
	   CUDA_CALLABLE_MEMBER softplus() = default;
	   CUDA_CALLABLE_MEMBER softplus(const softplus&) = default;
	   CUDA_CALLABLE_MEMBER softplus(softplus&&) = default;
	   CUDA_CALLABLE_MEMBER softplus & operator = (const softplus&) = default;
	   CUDA_CALLABLE_MEMBER softplus & operator = (softplus&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x)
		 {
#if CUDA_ENABLED == 1
		    if(Precision_Type(1.0) + exp(x) <=0)
			  printf("Gotcha\n");
			return log(Precision_Type(1.0) + exp(x));
#else
			return std::log(Precision_Type(1.0) + std::exp(x));
#endif
		 };
	template <class Precision_Type>
	   CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x)
	   {
#if CUDA_ENABLED == 1
		 return Precision_Type(1.0) / (Precision_Type(1.0) + exp(-x));
#else
		 return Precision_Type(1.0) / (Precision_Type(1.0) + std::exp(-x));
#endif
	   };
  };

  class hyperbolic_tangent : public activation<hyperbolic_tangent>
  {
	public:
	   CUDA_CALLABLE_MEMBER hyperbolic_tangent() = default;
	   CUDA_CALLABLE_MEMBER hyperbolic_tangent(const hyperbolic_tangent&) = default;
	   CUDA_CALLABLE_MEMBER hyperbolic_tangent(hyperbolic_tangent&&) = default;
	   CUDA_CALLABLE_MEMBER hyperbolic_tangent & operator = (const hyperbolic_tangent&) = default;
	   CUDA_CALLABLE_MEMBER hyperbolic_tangent & operator = (hyperbolic_tangent&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x)
		 {
#if CUDA_ENABLED == 1
			return tanh(-x);
#else
			return std::tanh(-x);
#endif
		 };
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x)
		 {
		   return Precision_Type(1.0) - (x * x);
		 };
  };

  class relu : public activation<relu>
  {
	public:
	   CUDA_CALLABLE_MEMBER relu() = default;
	   CUDA_CALLABLE_MEMBER relu(const relu&) = default;
	   CUDA_CALLABLE_MEMBER relu(relu&&) = default;
	   CUDA_CALLABLE_MEMBER relu & operator = (const relu&) = default;
	   CUDA_CALLABLE_MEMBER relu & operator = (relu&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x)
		 {
		   return (x >= Precision_Type{0.0} ) ? x : Precision_Type{0.0};
		 };
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x)
		 {
		   return (x >= Precision_Type{0.0} ) ? Precision_Type{1.0} : Precision_Type{0.0};
		 };
  };



  class leaky_relu : public activation<leaky_relu>
  {
	public:
	   CUDA_CALLABLE_MEMBER leaky_relu() = default;
	   CUDA_CALLABLE_MEMBER leaky_relu(const leaky_relu&) = default;
	   CUDA_CALLABLE_MEMBER leaky_relu(leaky_relu&&) = default;
	   CUDA_CALLABLE_MEMBER leaky_relu & operator = (const leaky_relu&) = default;
	   CUDA_CALLABLE_MEMBER leaky_relu & operator = (leaky_relu&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x, Precision_Type leakage_coefficient = 0.1)
		 {
		   return (x >= Precision_Type(0.0) ) ? x : leakage_coefficient * x;
		 };
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x, Precision_Type leakage_coefficient = 0.1)
		 {
		   return (x >= Precision_Type(0.0) ) ? Precision_Type(1.0) : leakage_coefficient;
		 };
  };



  class exp_leaky_relu : public activation<exp_leaky_relu>
  {
	public:
	   CUDA_CALLABLE_MEMBER exp_leaky_relu() = default;
	   CUDA_CALLABLE_MEMBER exp_leaky_relu(const exp_leaky_relu&) = default;
	   CUDA_CALLABLE_MEMBER exp_leaky_relu(exp_leaky_relu&&) = default;
	   CUDA_CALLABLE_MEMBER exp_leaky_relu & operator = (const exp_leaky_relu&) = default;
	   CUDA_CALLABLE_MEMBER exp_leaky_relu & operator = (exp_leaky_relu&&) = default;
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type objective(const Precision_Type & x, Precision_Type leakage_coefficient = 0.1)
		 {
#if CUDA_ENABLED == 1
		   return (x >= Precision_Type(0.0) ) ? x : leakage_coefficient * (exp(x) - Precision_Type(1.0));
#else
		   return (x >= Precision_Type(0.0) ) ? x : leakage_coefficient * (std::exp(x) - Precision_Type(1.0));
#endif
		 };
	   template <class Precision_Type>
		 CUDA_CALLABLE_MEMBER Precision_Type derivative(const Precision_Type & x, Precision_Type leakage_coefficient = 0.1)
		 {

#if CUDA_ENABLED == 1
		   return (x >= Precision_Type(0.0) ) ? x : leakage_coefficient * (exp(x) - Precision_Type(1.0));
#else
		   return (x >= Precision_Type(0.0) ) ? x : leakage_coefficient * (std::exp(x) - Precision_Type(1.0));
#endif
		 };
  };

#if CUDA_ENABLED == 1
  //wrppers for host functions to use to call kernels here, the wrappers will calculate the block_parameters and the threads per block
  template <class Precision_Type>
	HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, Precision_Type * device_Wx_plus_b, const std::uint32_t & current_layer_size, const std::uint32_t & device_id = 0);
  template <class Precision_Type>
	HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, Precision_Type * device_Wx_plus_b, const std::uint32_t & current_layer_size, const cudaStream_t & stream, const std::uint32_t & device_id = 0);

  std::int32_t call_activation(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, double coefficient, std::uint32_t layer_size);
  #endif
}
#endif
