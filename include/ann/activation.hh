#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "typedefs.cuh"
#include <cstdint>
#include <utility>
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  enum class ACTIVATION_NAME : std::uint8_t/*ACTIVATION_TYPE*/ {INPUT = 0, IDENTITY, SOFTMAX, SIGMOID, SOFTPLUS, TANH, RELU, LEAKY_RELU, EXP_LEAKY_RELU};
  enum class ACTIVATION_TYPE : std::uint8_t/*ACTIVATION_TYPE*/ {OBJECTIVE = 0, DERIVATIVE}; 
  using Neurons = std::uint32_t;
  using LAYER_INFO = std::pair<ACTIVATION_NAME, Neurons>;
  template<ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type>
   	class activation_function;
  template<>	
	class activation_function<ACTIVATION_NAME::INPUT,ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x){return x;};
   };
  template<>	
	class activation_function<ACTIVATION_NAME::INPUT,ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x){return x;};
   };
  template<>	
	class activation_function<ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x){return x;};
   };
  template<>	
	class activation_function<ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x){return 1;};
   };
 template<>	
	class activation_function<ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
#if CUDA_ENABLED == 1
		  return double(1.0) / ( double(1.0) + exp(-x) );
#else
		  return double(1.0) / ( double(1.0) + std::exp(-x) );
#endif
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   //to do
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
		return x * (double(1.0) - x);

	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::SOFTMAX, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
#if CUDA_ENABLED == 1
		  return exp(x);
#else
		  return std::exp(x);
#endif
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::SOFTMAX, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   //to do
	   CUDA_CALLABLE_MEMBER double operator ()(double x){return 1;};
   };
  template<>	
	class activation_function<ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
#if CUDA_ENABLED == 1
		  return log(double(1.0) + exp(x));
#else
		  return std::log(double(1.0) + std::exp(x));
#endif
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
#if CUDA_ENABLED == 1
		 return double(1.0) / (double(1.0) + exp(-x));
#else
		 return double(1.0) / (double(1.0) + std::exp(-x));
#endif
	  };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::TANH, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
#if CUDA_ENABLED == 1
		  return tanh(-x);
#else
		  return std::tanh(-x);
#endif
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::TANH, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
		 return double(1.0) - (x * x);
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::RELU, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
		  return (x >= double(0.0) ) ? x : double(0.0);
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::RELU, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x)
	   {
   		 return (x >= double(0.0) ) ? double(1.0) : double(0.0);
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::LEAKY_RELU, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x, double leakage_coefficient = 0.1)
	   {
		 return (x >= double(0.0) ) ? x : leakage_coefficient * x;
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::LEAKY_RELU, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x, double leakage_coefficient = 0.1)
	   {
   		 return (x >= double(0.0) ) ? double(1.0) : leakage_coefficient;
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::EXP_LEAKY_RELU, ACTIVATION_TYPE::OBJECTIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x, double leakage_coefficient = 0.1)
	   {
#if CUDA_ENABLED == 1
		 return (x >= double(0.0) ) ? x : leakage_coefficient * (exp(x) - double(1.0));
#else
		 return (x >= double(0.0) ) ? x : leakage_coefficient * (std::exp(x) - double(1.0));
#endif
	   };
   };
  template<>	
	class activation_function<ACTIVATION_NAME::EXP_LEAKY_RELU, ACTIVATION_TYPE::DERIVATIVE>
   {
	 public:
	   CUDA_CALLABLE_MEMBER activation_function() = default;
	   CUDA_CALLABLE_MEMBER activation_function(const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function(activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (const activation_function&) = default;
	   CUDA_CALLABLE_MEMBER activation_function & operator = (activation_function&&) = default;
	   CUDA_CALLABLE_MEMBER double operator ()(double x, double leakage_coefficient = 0.1)
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


  //wrappers for host functions to use to call kernels here, the wrappers will calculate the block_parameters and the threads per block
  std::int32_t call_activation(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, std::uint32_t layer_size);
  std::int32_t call_activation(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, double coefficient, std::uint32_t layer_size);
#if CUDA_ENABLED == 1
  //activation function kernels here
  __global__ void activation_kernel(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, std::uint32_t size); //everything that's not leaky relu, elu, or softmax
  __global__ void activation_kernel_coeff(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * Wx_plus_b, double coefficient, std::uint32_t size);//leaky relu or elu
  __global__ void activation_kernel_softmax(ACTIVATION_TYPE activation_type,double * Wx_plus_b, std::uint32_t size);//not sure atm what other parameters are necesary 12/28/17
  //end activation function kernels
#endif
  /*class Layer
  {
	public:
	  Layer() = default;
	  Layer(const Layer &) = default;
	  Layer(Layer &&) = default;
	  Layer & operator =(const Layer &) = default;
	  Layer & operator =(Layer &&) = default;
	  ~Layer() = default;
	  CUDA_CALLABLE_MEMBER pt operator()(double & input, LAYER_NAME ln, ACTIVATION f)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED Layer operator()\n");
#else
		printf("CUDA_DISABLED Layer operator()\n");
#endif
		if(f == ACTIVATION::OBJECTIVE)
		{
		  switch(ln)
		  {
			case LAYER_NAME::IDENTITY :
			  return input;
			case LAYER_NAME::SOFTMAX :
#if CUDA_ENABLED == 1
			  return exp(input);
#else
		      return std::exp(input);
#endif
		   case LAYER_NAME::TANH :
#if CUDA_ENABLED == 1
			  return tanh(-input);
#else
			  std::tanh(-input);
#endif
			case LAYER_NAME::RELU :
			 return (input >= 0.0) ? input : 0.0;
			default:
			 return 0.0;
		  }
		}
		else if(f == ACTIVATION::DERIVATIVE)
		{
		  switch(ln)
		  {
			case LAYER_NAME::IDENTITY :
			  return 1.0;
			case LAYER_NAME::SOFTMAX :
			  return input * (1.0 - input);
			case LAYER_NAME::TANH :
			 return 1.0 - (input * input);
			case LAYER_NAME::RELU :
			 return (input >= 0.0) ? 1.0 : 0.0;
			default:
			  return 0.0;
		  }
		}
		else
		  return 0.0;
	  }
	  CUDA_CALLABLE_MEMBER pt operator()(double & input, double & coefficient, LAYER_NAME ln, ACTIVATION f)
	  {

#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED Layer operator() (coeff)\n");
#else
		printf("CUDA_DISABLED Layer operator() (coeff)\n");
#endif
		if(f == ACTIVATION::OBJECTIVE)
		{
		  switch(ln)
		  {
			case LAYER_NAME::LEAKY_RELU :
			  return (input >= 0.0) ? input : coefficient * input;
		   default :
			return 0.0;
		  }
		}
		else if(f == ACTIVATION::DERIVATIVE)
		{
		  switch(ln)
		  {
			case LAYER_NAME::LEAKY_RELU :
			  return (input >= 0.0) ? 1 : coefficient;
		   default :
			return 0.0;
		  }
		}
		else
		  return 0.0;
	  }
  };*/
/*  pt call_activation(Layer & L, double & input, LAYER_NAME ln, ACTIVATION f);
  pt call_activation(Layer & L, double & input, double & coefficient, LAYER_NAME ln, ACTIVATION f);*/

}
#endif
