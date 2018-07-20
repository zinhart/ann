#include "ann/activation.hh"
#include <cassert>
namespace zinhart
{
  namespace activation
  {
  	std::uint32_t total_activation_types()
	{
	  return 9;
	}
	HOST std::string get_activation_name(ACTIVATION_NAME name)
	{
	  assert(std::uint32_t(name) >= 0u && std::uint32_t(name) < total_activation_types());
	  std::string ret;
	  switch(name)
	  { 
		case ACTIVATION_NAME::INPUT:
		  ret = "INPUT";
		  break;
		case ACTIVATION_NAME::IDENTITY:
		  ret = "IDENTITY";
		  break;
		case ACTIVATION_NAME::SIGMOID:
		  ret = "SIGMOID";
		  break;
		case ACTIVATION_NAME::SOFTPLUS:
		  ret = "SOFTPLUS";
		  break;
		case ACTIVATION_NAME::TANH:
		  ret = "TANH";
		  break;
		case ACTIVATION_NAME::RELU:
		  ret = "RELU";
		  break;
		case ACTIVATION_NAME::LEAKY_RELU:
		  ret = "LEAKY_RELU";
		  break;
		case ACTIVATION_NAME::EXP_LEAKY_RELU:
		  ret = "EXP_LEAKY_RELU";
		  break;
		case ACTIVATION_NAME::SOFTMAX:
		  ret = "SOFTMAX";
		  break;
	  }
	  return ret;
	}
  }
}
