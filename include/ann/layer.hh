#ifndef LAYER_H
#define LAYER_H
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
  using LAYER_TYPE = std::uint8_t;
  enum class LAYER_NAME : LAYER_TYPE {INPUT = 0, IDENTITY, SOFTMAX, SIGMOID, TANH, RELU, LEAKY_RELU};
  enum class ACTIVATION : LAYER_TYPE {OBJECTIVE = 0, DERIVATIVE}; 
  using Neurons = std::uint32_t;
  using LAYER_INFO = std::pair<LAYER_NAME, Neurons>;
  
  class Layer
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
  };
  pt call_activation(Layer & L, double & input, LAYER_NAME ln, ACTIVATION f);
  pt call_activation(Layer & L, double & input, double & coefficient, LAYER_NAME ln, ACTIVATION f);

#if CUDA_ENABLED == 1
  //activation function kernels here
  __global__ void activation_kernel(LAYER_NAME ln, double * Wx_plus_b, std::uint32_t size); 
  __global__ void activation_kernel_coeff(LAYER_NAME ln, double * Wx_plus_b, std::uint32_t size); 

#endif

  

}
#endif
