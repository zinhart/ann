#ifndef TYPEDEFS_CUH
#define TYPEDEFS_CUH

#if CUDA_ENABLED == true//this is defined in top level cmake lists file
  #define CUDA_CALLABLE_MEMBER __host__ __device__
  #define HOST __host__
  #define DEVICE __device__
  #define CONSTANT __constant__
  #define SHARED __shared__
  #include <cuda.h>
  #include <builtin_types.h>
  #include <cuda_runtime_api.h>
  #include <cublas_v2.h>
  #include <thrust/host_vector.h>
  #include <thrust/device_vector.h>
#else
  #define CUDA_CALLABLE_MEMBER
  #define HOST
  #define DEVICE
  #define CONSTANT
  #define SHARED
  #include "mkl.h"
#endif

namespace zinhart
{
#if CUDA_ENABLED == true
  using uint = unsigned int;
  using precision_type = double;
  using pt = precision_type;
  template <typename T>
	struct kernel_vector
	{
	  kernel_vector(){};
	  kernel_vector(thrust::device_vector<T> & d_vec)
	  {
		k_elements = thrust::raw_pointer_cast( &d_vec[0] );
		k_size  = ( uint ) d_vec.size();
	  }
	  T*  k_elements;
	  uint1 k_size;
	  uint1 CUDA_CALLABLE_MEMBER size()
	  {return k_size;}
	};
  template <typename T>
	kernel_vector<T> to_kernel_vector(thrust::device_vector<T>& d_vec)
	{
	  kernel_vector<T> k_vector;
	  k_vector.k_elements = thrust::raw_pointer_cast(&d_vec[0]);
	  k_vector.k_size  = (uint) d_vec.size();
	  return k_vector;
	};
  template <typename T>
	struct kernel_matrix
	{
	  kernel_matrix(){};
	  kernel_matrix(thrust::device_vector<T> & d_vec, uint width, uint height)
	  {	
		k_elements = thrust::raw_pointer_cast( &d_vec[0] );
		k_width = width;
		k_height = height;
	  }
	  CUDA_CALLABLE_MEMBER uint2  dim()
	  {return uint2(k_width,k_height);}
	  uint1 k_width;
	  uint1 k_height;
	  T * k_elements; 
	};
  template <typename T>
	kernel_matrix<T> to_kernel_matrix(thrust::device_vector<T> & d_vec, uint width, uint height)
	{
	  kernel_matrix<T> k_matrix;
	  k_matrix.k_elements = thrust::raw_pointer_cast( &d_vec[0] );
	  k_matrix.k_width = width;
	  k_matrix.k_height = height;
	  return k_matrix;
	}
#endif
}
#endif
