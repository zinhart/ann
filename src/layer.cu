#include "ann/layer.hh"
namespace zinhart
{
  pt call_activation(Layer & L, double & input, LAYER_NAME ln, ACTIVATION f)
  { return L(input, ln, f); } 
  pt call_activation(Layer & L, double & input, double & coefficient, LAYER_NAME ln, ACTIVATION f)
  { return L(input, coefficient, ln, f); }

#if CUDA_ENABLED == 1
  //activation function kernels here
  __global__ void activation_kernel(LAYER_NAME ln, double * Wx_plus_b, std::uint32_t size)
  {
	Layer L;
	std::uint32_t i = threadIdx.x;
	if(i > size)
	  return;
	  L(Wx_plus_b[i], ln, ACTIVATION::OBJECTIVE);
  }	
#endif
}
