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
  enum class ACTIVATION_NAME : std::uint8_t {INPUT = std::uint8_t(0), IDENTITY, SIGMOID, SOFTMAX, SOFTPLUS, TANH, RELU, LEAKY_RELU, EXP_LEAKY_RELU};
  enum class ACTIVATION_TYPE : std::uint8_t {OBJECTIVE = std::uint8_t(0), DERIVATIVE}; 
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
		CUDA_CALLABLE_MEMBER double operator ()(double x, ACTIVATION_TYPE at)
		{ return (at == ACTIVATION_TYPE::OBJECTIVE) ? static_cast<ACTIVATION_FUNCTION*>(this)->objective(x) : static_cast<ACTIVATION_FUNCTION*>(this)->derivative(x); };
	};

  class identity : public activation<identity>
  {
	public:
	   CUDA_CALLABLE_MEMBER identity() = default;
	   CUDA_CALLABLE_MEMBER identity(const identity&) = default;
	   CUDA_CALLABLE_MEMBER identity(identity&&) = default;
	   CUDA_CALLABLE_MEMBER identity & operator = (const identity&) = default;
	   CUDA_CALLABLE_MEMBER identity & operator = (identity&&) = default;
	   CUDA_CALLABLE_MEMBER double objective(const double & x)
	   {return x;};
	   CUDA_CALLABLE_MEMBER double derivative(const double & x)
	   {return 1;};
  };

  class sigmoid : public activation<sigmoid>
  {
	public:
	   CUDA_CALLABLE_MEMBER sigmoid() = default;
	   CUDA_CALLABLE_MEMBER sigmoid(const sigmoid&) = default;
	   CUDA_CALLABLE_MEMBER sigmoid(sigmoid&&) = default;
	   CUDA_CALLABLE_MEMBER sigmoid & operator = (const sigmoid&) = default;
	   CUDA_CALLABLE_MEMBER sigmoid & operator = (sigmoid&&) = default;
	   CUDA_CALLABLE_MEMBER double objective(const double & x)
	   {
#if CUDA_ENABLED == 1
		  return double(1.0) / ( double(1.0) + exp(-x) );
#else
		  return double(1.0) / ( double(1.0) + std::exp(-x) );
#endif
	   };
	   CUDA_CALLABLE_MEMBER double derivative(const double & x)
	   {
		 return x * (double(1.0) - x);
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
	   CUDA_CALLABLE_MEMBER double objective(const double & x)
	   {
#if CUDA_ENABLED == 1
		  return exp(x);
#else
		  return std::exp(x);
#endif
	   };
	   //to do
	   CUDA_CALLABLE_MEMBER double derivative(const double & x)
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
	   CUDA_CALLABLE_MEMBER double objective(const double & x)
	   {
#if CUDA_ENABLED == 1
		  return log(double(1.0) + exp(x));
#else
		  return std::log(double(1.0) + std::exp(x));
#endif
	   };
	   CUDA_CALLABLE_MEMBER double derivative(const double & x)
	   {
#if CUDA_ENABLED == 1
		 return double(1.0) / (double(1.0) + exp(-x));
#else
		 return double(1.0) / (double(1.0) + std::exp(-x));
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
	   CUDA_CALLABLE_MEMBER double objective(const double & x)
	   {
#if CUDA_ENABLED == 1
		  return tanh(-x);
#else
		  return std::tanh(-x);
#endif
	   };
	   CUDA_CALLABLE_MEMBER double derivative(const double & x)
	   {
		 return double(1.0) - (x * x);
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
	   CUDA_CALLABLE_MEMBER double objective(const double & x)
	   {
		 return (x >= double(0.0) ) ? x : double(0.0);
	   };
	   CUDA_CALLABLE_MEMBER double derivative(const double & x)
	   {
		 return (x >= double(0.0) ) ? double(1.0) : double(0.0);
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
	   CUDA_CALLABLE_MEMBER double objective(const double & x, double leakage_coefficient = 0.1)
	   {
		 return (x >= double(0.0) ) ? x : leakage_coefficient * x;
	   };
	   CUDA_CALLABLE_MEMBER double derivative(const double & x, double leakage_coefficient = 0.1)
	   {
   		 return (x >= double(0.0) ) ? double(1.0) : leakage_coefficient;
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
	   CUDA_CALLABLE_MEMBER double objective(const double & x, double leakage_coefficient = 0.1)
	   {
#if CUDA_ENABLED == 1
		 return (x >= double(0.0) ) ? x : leakage_coefficient * (exp(x) - double(1.0));
#else
		 return (x >= double(0.0) ) ? x : leakage_coefficient * (std::exp(x) - double(1.0));
#endif
	   };
	   CUDA_CALLABLE_MEMBER double derivative(const double & x, double leakage_coefficient = 0.1)
	   {

#if CUDA_ENABLED == 1
		 return (x >= double(0.0) ) ? x : leakage_coefficient * (exp(x) - double(1.0));
#else
		 return (x >= double(0.0) ) ? x : leakage_coefficient * (std::exp(x) - double(1.0));
#endif
	   };
  };



  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_identity(ACTIVATION_TYPE activation_type, Numeric_Type & input)	
	{
#if CUDA_ENABLED == 1	
	  printf("CUDA_ENABLED activation_identity\n");
#else	
	  printf("CUDA_DISABLED activation_identity\n");
#endif
	  switch(activation_type)
	  {
		case ACTIVATION_TYPE::OBJECTIVE:
		  return input;
		case ACTIVATION_TYPE::DERIVATIVE:
		  return Numeric_Type(1);
#if CUDA_ENABLED == 1	
		default:
		  printf("default case called in activation_identity\n");
		  return 0;
#else	
		default:
		  std::cerr<<"default case called in activation_identity\n";
		  return 0;
#endif
	  }
	}

  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_softmax(ACTIVATION_TYPE activation_type, Numeric_Type & input_i, Numeric_Type & input_j, Numeric_Type & kronecker_delta)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_softmax\n");
#else
	  printf("CUDA_DISABLED activation_softmax\n");
#endif
	  switch(activation_type)
	  {
#if CUDA_ENABLED == 1
		case ACTIVATION_TYPE::OBJECTIVE:
		  return exp(input_i);
		default:
		  printf("default case called in activation_softmax\n");
		  return 0;
#else
		case ACTIVATION_TYPE::OBJECTIVE:
		  return std::exp(input_i);
		default:
		  std::cerr<<"default case called in activation_identity\n";
		  return 0;
#endif
  		case ACTIVATION_TYPE::DERIVATIVE:
  		  return input_i * (kronecker_delta - input_j);
	}
  }

  template<class Numeric_Type>
   	CUDA_CALLABLE_MEMBER Numeric_Type activation_sigmoid(ACTIVATION_TYPE activation_type, Numeric_Type & input)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_sigmoid\n");
#else
	  printf("CUDA_DISABLED activation_sigmoid\n");
#endif
	  switch(activation_type)
	  {
#if CUDA_ENABLED == 1
		case ACTIVATION_TYPE::OBJECTIVE:
		  return Numeric_Type(1.0) / (Numeric_Type(1.0) + exp(-input));
		case ACTIVATION_TYPE::DERIVATIVE:
		  return input * (Numeric_Type(1.0) - input);
		default:
		  printf("default case called in activation_sigmoid\n");
		  return 0;
#else
		case ACTIVATION_TYPE::OBJECTIVE:
		  return Numeric_Type(1.0) / (Numeric_Type(1.0) + std::exp(-input));
		case ACTIVATION_TYPE::DERIVATIVE:
		  return input * (Numeric_Type(1.0) - input);
		default:
		  std::cerr<<"default case called in activation_sigmoid\n";
		  return 0;
#endif
	  }
  }

  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_softplus(ACTIVATION_TYPE activation_type, Numeric_Type & input)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_softplus\n");
#else
	  printf("CUDA_DISABLED activation_softplus\n");

#endif
	  switch(activation_type)
	  {
#if CUDA_ENABLED == 1
		case ACTIVATION_TYPE::OBJECTIVE:
		  return log(Numeric_Type(1.0) + exp(input));
		case ACTIVATION_TYPE::DERIVATIVE:
		  return Numeric_Type(1.0) / (Numeric_Type(1.0) + exp(-input));
		default:
		  printf("default case called in activation_softplus\n");
		  return 0;
#else
		case ACTIVATION_TYPE::OBJECTIVE:
		  return log(Numeric_Type(1.0) + std::exp(input));
		case ACTIVATION_TYPE::DERIVATIVE:
		  return Numeric_Type(1.0) / (Numeric_Type(1.0) + std::exp(-input));
		default:
		  std::cerr<<"default case called in activation_softplus\n";
		  return 0;
#endif
	}
  }

  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_tanh(ACTIVATION_TYPE activation_type, Numeric_Type & input)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_tanh\n");
#else
	  printf("CUDA_DISABLED activation_tanh\n");
#endif
	  switch(activation_type)
	  {
#if CUDA_ENABLED == 1
		case ACTIVATION_TYPE::OBJECTIVE:
		  return tanh(-input);
		case ACTIVATION_TYPE::DERIVATIVE:
		  return Numeric_Type(1.0) - (input * input);
		default:
		  printf("default case called in activation_tanh\n");
		  return 0;
#else
		case ACTIVATION_TYPE::OBJECTIVE:
		  return std::tanh(-input);
		case ACTIVATION_TYPE::DERIVATIVE:
		  return Numeric_Type(1.0) - (input * input);
		default:
		  std::cerr<<"default case called in activation_tanh\n";
		  return 0;
#endif
	  }
  }

  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_relu(ACTIVATION_TYPE activation_type, Numeric_Type & input)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_relu\n");
#else
	  printf("CUDA_DISABLED activation_relu\n");
#endif
	  switch(activation_type)
	  {
		case ACTIVATION_TYPE::OBJECTIVE:
		  return (input >= Numeric_Type(0.0) ) ? input : Numeric_Type(0.0);
		case ACTIVATION_TYPE::DERIVATIVE:
		  return (input >= Numeric_Type(0.0) ) ? Numeric_Type(1.0) : Numeric_Type(0.0);
#if CUDA_ENABLED == 1
		default:
		  printf("default case called in activation_relu\n");
		  return 0;
#else
		default:
		  std::cerr<<"default case called in activation_relu\n";
		  return 0;
#endif
	  }
	}

  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_leaky_relu(ACTIVATION_TYPE activation_type, Numeric_Type & input, Numeric_Type & leakage_coefficient)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_leaky_relu\n");
#else
	  printf("CUDA_DISABLED activation_leaky_relu\n");
#endif
	  switch(activation_type)
	  {
		case ACTIVATION_TYPE::OBJECTIVE:
		  return (input >= Numeric_Type(0.0) ) ? input : leakage_coefficient * input;
		case ACTIVATION_TYPE::DERIVATIVE:
		  return (input >= Numeric_Type(0.0) ) ? Numeric_Type(1.0) : leakage_coefficient;
#if CUDA_ENABLED == 1
		default:
		  printf("default case called in activation_leaky_relu\n");
		  return 0;
#else
		default:
		  std::cerr<<"default case called in activation_leaky_relu\n";
		  return 0;
#endif
	  }
	}

  template<class Numeric_Type>
	CUDA_CALLABLE_MEMBER Numeric_Type activation_exponential_leaky_relu(ACTIVATION_TYPE activation_type, Numeric_Type & input, Numeric_Type & leakage_coefficient)
	{
#if CUDA_ENABLED == 1
	  printf("CUDA_ENABLED activation_exponential_leaky_relu\n");
#else
	  printf("CUDA_DISABLED activation_exponential_leaky_relu\n");
#endif

#if CUDA_ENABLED == 1
	  switch(activation_type)
	  {
		case ACTIVATION_TYPE::OBJECTIVE:
		  return (input >= Numeric_Type(0.0) ) ? input : leakage_coefficient * (exp(input) - Numeric_Type(1.0));
		case ACTIVATION_TYPE::DERIVATIVE:
		  return (input >= Numeric_Type(0.0) ) ? Numeric_Type(1.0) : leakage_coefficient * (exp(input) - Numeric_Type(1.0)) + leakage_coefficient;
		default:
		  printf("default case called in activation_leaky_relu\n");
		  return 0;
	  }
#else
	  printf("CUDA_DISABLED activation_leaky_relu\n");
	  switch(activation_type)
	  {
		case ACTIVATION_TYPE::OBJECTIVE:
		  return (input >= Numeric_Type(0.0) ) ? input : leakage_coefficient * (std::exp(input) - Numeric_Type(1.0));
		case ACTIVATION_TYPE::DERIVATIVE:
		  return (input >= Numeric_Type(0.0) ) ? Numeric_Type(1.0) : leakage_coefficient * (std::exp(input) - Numeric_Type(1.0)) + leakage_coefficient;
		default:
		  std::cerr<<"default case called in activation_leaky_relu\n";
		  return 0;
	  }
#endif
  } 


#if CUDA_ENABLED == 1
  //wrappers for host functions to use to call kernels here, the wrappers will calculate the block_parameters and the threads per block
  std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, std::uint32_t current_layer_size);

  std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, std::uint32_t current_layer_size, const cudaStream_t & stream);

  std::int32_t call_activation(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, double coefficient, std::uint32_t layer_size);
  //activation function kernels here
  __global__ void activation_kernel(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, std::uint32_t size); //everything that's not leaky relu, elu, or softmax
  __global__ void activation_kernel_coeff(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, double coefficient, std::uint32_t size);//leaky relu or elu
  __global__ void activation_kernel_softmax(ACTIVATION_TYPE activation_type,double * Wx_plus_b, std::uint32_t size);//not sure atm what other parameters are necesary 12/28/17
  //end activation function kernels
#endif
}
#endif
