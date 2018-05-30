#include "ann/activation.hh"
#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
namespace zinhart
{
  // wrappers for host functions to use to call kernels here, the wrappers will calculate the block_parameters and the threads per block
  std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, std::uint32_t current_layer_size)
  {
	cudaError_t error_id;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	dim3 block_launch;
	std::int32_t warp_size = properties.warpSize;
	std::int32_t threads_per_block = (current_layer_size + warp_size -1) / warp_size * warp_size;
	if(threads_per_block > 4 * warp_size)
	  threads_per_block = 4 * warp_size;
	block_launch.x = (current_layer_size + threads_per_block - 1) / threads_per_block;// number of blocks
	block_launch.y = 1;
	block_launch.z = 1;
	//std::cout<<"current_layer_size: "<<current_layer_size<<" threads_per_block: "<<threads_per_block<<" warp_size: "<<warp_size <<" block_launch.x: " <<block_launch.x<< " block_launch.y: " <<block_launch.y<< " block_launch.z: " <<block_launch.z<<"\n";
	//call kernel
	activation_kernel<<<block_launch, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, current_layer_size);
	cudaDeviceSynchronize();
  	error_id = cudaGetLastError();
	if(error_id != cudaSuccess)
	{
	  std::cerr<<"activation_kernel failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
	  return 1;
	}
	return 0;
  }

  // this method does not synchronize interally
  std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, std::uint32_t current_layer_size, const cudaStream_t & stream)
  {
	cudaError_t error_id;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	dim3 block_launch;
	std::int32_t warp_size = properties.warpSize;
	std::int32_t threads_per_block = (current_layer_size + warp_size -1) / warp_size * warp_size;
	if(threads_per_block > 4 * warp_size)
	  threads_per_block = 4 * warp_size;
	block_launch.x = (current_layer_size + threads_per_block - 1) / threads_per_block;// number of blocks
	block_launch.y = 1;
	block_launch.z = 1;
	//std::cout<<"current_layer_size: "<<current_layer_size<<" threads_per_block: "<<threads_per_block<<" warp_size: "<<warp_size <<" block_launch.x: " <<block_launch.x<< " block_launch.y: " <<block_launch.y<< " block_launch.z: " <<block_launch.z<<"\n";
	//call kernel
	activation_kernel<<<block_launch, threads_per_block,0, stream>>>(activation_name, activation_type, device_Wx_plus_b, current_layer_size);
	//cudaDeviceSynchronize();
  	error_id = cudaGetLastError();
	if(error_id != cudaSuccess)
	{
	  std::cerr<<"activation_kernel failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
	  return 1;
	}
	return 0;
  }

 /* 
	std::int32_t call_activation(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, double coefficient, std::uint32_t layer_size)
  {
	cudaError_t error_id;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	std::int32_t warp_size = properties.warpSize;
	std::int32_t threads_per_block = (layer_size + warp_size - 1) / (warp_size * warp_size);
    if(threads_per_block > 4 * warp_size)
	  threads_per_block = 4 * warp_size;	  
	dim3 block_launch;
	block_launch.x = (layer_size + threads_per_block - 1) / threads_per_block;
	block_launch.y = 1; 
	block_launch.z = 1;
	//call kernel
	activation_kernel_coeff<<<block_launch, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, coefficient, layer_size);
	cudaDeviceSynchronize();
	error_id = cudaGetLastError();
	if(error_id != cudaSuccess)
	{
	  std::cerr<<"activation_kernel_coeff failed to launch with error: "<<cudaGetErrorString(error_id);
	  return 1;
	}
	//copy memory from host to device
	return 0;
  }*/


  //activation function kernels here
	__global__ void activation_kernel(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, std::uint32_t layer_size) //everything that's not leaky relu, elu, or softmax
  {

	const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("thread_id: %d\n", thread_id);
	if(thread_id >= layer_size)
	  return;
	if(activation_name == ACTIVATION_NAME::IDENTITY)
	{
	  activation<identity> f;
	  device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);
	}
	else if(activation_name == ACTIVATION_NAME::SIGMOID)
	{
	  activation<sigmoid> f;
	  device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);
	}
	else if(activation_name == ACTIVATION_NAME::SOFTPLUS)
	{
	  activation<softplus> f;
	  device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);
	}
	else if(activation_name == ACTIVATION_NAME::TANH)
	{
	  activation<hyperbolic_tangent> f;
	  device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);
	}
	else if(activation_name == ACTIVATION_NAME::RELU)
	{
	  activation<relu> f;
	  device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);
	}
	else
	  return;
  }
  __global__ void activation_kernel_coeff(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, double coefficient, std::uint32_t layer_size)//leaky relu or elu
  {
	std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("thread_id: %d\n", thread_id);
	if(thread_id > layer_size)
	  return;
	switch(activation_name)
	{
	  case ACTIVATION_NAME::LEAKY_RELU:
		device_Wx_plus_b[thread_id] = activation_leaky_relu(activation_type, device_Wx_plus_b[thread_id], coefficient);
		break;
	  case ACTIVATION_NAME::EXP_LEAKY_RELU:
		device_Wx_plus_b[thread_id] = activation_exponential_leaky_relu(activation_type, device_Wx_plus_b[thread_id], coefficient);
		break;
	  default:
		return;
	}
  }
  __global__ void activation_kernel_softmax(ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, std::uint32_t layer_size)
  {
	//to do
	std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id > layer_size)
  	  return;
	return;
  }
}
