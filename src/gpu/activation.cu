#include "ann/activation.hh"
#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
namespace zinhart
{
  template HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, float * device_Wx_plus_b, const std::uint32_t & current_layer_size, const std::uint32_t & device_id);
  template HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, const std::uint32_t & current_layer_size, const std::uint32_t & device_id);
  template HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, float * device_Wx_plus_b, const std::uint32_t & current_layer_size, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, double * device_Wx_plus_b, const std::uint32_t & current_layer_size, const cudaStream_t & stream, const std::uint32_t & device_id);


  //activation function kernels here
   template <class Precision_Type, template <class> class activation_function, class ACT>
 	 __global__ void activation_kernel(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, Precision_Type * device_Wx_plus_b, activation_function<ACT> f,const std::uint32_t layer_size) //everything that's not leaky relu, elu, or softmax
	 {
	   //  in case later I decide to use shared memory
	   //  extern __shared__ std::uint8_t device_Wx_plus_b_shared[];
	   //  Precision_Type * tile = reinterpret_cast<Precision_Type*>(tile);
	     
	 /*  check out a grid stride loop here in the future
	  *  for (std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < layer_size; thread_id += blockDim.x * gridDim.x)
  		 device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);*/
 	   const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	   if(thread_id >= layer_size)
		 return;
	   device_Wx_plus_b[thread_id] =  f(device_Wx_plus_b[thread_id], activation_type);
	 }
   //everything that's not leaky relu, elu, or softmax
   template <class Precision_Type>
 	 __global__ void activation_kernel(ACTIVATION_NAME activation_name, ACTIVATION_TYPE activation_type, Precision_Type * device_Wx_plus_b, const std::uint32_t layer_size, const std::uint32_t shared_memory_length)   
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
	  /*
	  case ACTIVATION_NAME::LEAKY_RELU:
		device_Wx_plus_b[thread_id] = activation_leaky_relu(activation_type, device_Wx_plus_b[thread_id], coefficient);
		break;
	  case ACTIVATION_NAME::EXP_LEAKY_RELU:
		device_Wx_plus_b[thread_id] = activation_exponential_leaky_relu(activation_type, device_Wx_plus_b[thread_id], coefficient);
		break;
	  default:
		return;*/
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

  // wrappers for host functions to use to call kernels here, the wrappers will calculate the block_parameters and the threads per block
  template <class Precision_Type>
	HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, Precision_Type * device_Wx_plus_b, const std::uint32_t & current_layer_size, const std::uint32_t & device_id)
	{
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  grid_space::get_launch_params(num_blocks, threads_per_block, current_layer_size, device_id);
	  if(activation_name == ACTIVATION_NAME::IDENTITY)
	  {
	    activation<identity> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::SIGMOID)
	  {
	    activation<sigmoid> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::SOFTMAX)
	  {
		activation<softmax> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::SOFTPLUS)
	  {
	    activation<softplus> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::TANH)
	  {
	    activation<hyperbolic_tangent> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::RELU)
	  {
	    activation<relu> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::LEAKY_RELU)
	  {
	    activation<leaky_relu> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::EXP_LEAKY_RELU)
	  {
	    activation<exp_leaky_relu> f;
		activation_kernel<<<num_blocks, threads_per_block>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else
	  {
		// probably the input layer was passed
		return 1;
	  }
	  
	  //std::cout<<"current_layer_size: "<<current_layer_size<<" threads_per_block: "<<threads_per_block<<" warp_size: "<<warp_size <<" block_launch.x: " <<block_launch.x<< " block_launch.y: " <<block_launch.y<< " block_launch.z: " <<block_launch.z<<"\n";
	  //call kernel
	  cudaDeviceSynchronize();
	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()),__FILE__,__LINE__);
	}

  // this method does not synchronize interally
   template <class Precision_Type>
	HOST std::int32_t call_activation(const ACTIVATION_NAME activation_name, const ACTIVATION_TYPE activation_type, Precision_Type * device_Wx_plus_b, const std::uint32_t & current_layer_size, const cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  std::uint32_t shared_memory_bytes{0};
	  grid_space::get_launch_params(num_blocks, threads_per_block, current_layer_size, shared_memory_bytes, device_id, Precision_Type{});
	  const std::uint32_t shared_memory_length =  shared_memory_bytes / sizeof(Precision_Type);
	  if(activation_name == ACTIVATION_NAME::IDENTITY)
	  {
	    activation<identity> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::SIGMOID)
	  {
	    activation<sigmoid> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::SOFTMAX)
	  {
		activation<softmax> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::SOFTPLUS)
	  {
	    activation<softplus> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::TANH)
	  {
	    activation<hyperbolic_tangent> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::RELU)
	  {
	    activation<relu> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::LEAKY_RELU)
	  {
	    activation<leaky_relu> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else if(activation_name == ACTIVATION_NAME::EXP_LEAKY_RELU)
	  {
	    activation<exp_leaky_relu> f;
		activation_kernel<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(activation_name, activation_type, device_Wx_plus_b, f,current_layer_size);
	  }
	  else
	  {
		// probably the input layer was passed
		return 1;
	  }
	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()),__FILE__,__LINE__);
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


  
}
