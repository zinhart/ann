#include "ann/activation.hh"
#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include<cublas_v2.h>
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <memory>
#include <vector>
using namespace zinhart;

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> pos_real(0, std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> reals(std::numeric_limits<double>::min(), std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> neg_real(std::numeric_limits<double>::min(), -1 );


TEST(activation_test_sync, call_activation_identity_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<identity> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i], ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size));
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i], activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}



/*
 * ACTIVATION IDENTITY DERIVATIVE
 * */
TEST(activation_test_sync, call_activation_identity_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<identity> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i], activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}


/*
 * ACTIVATION IDENTITY SIGMOID
 * */
TEST(activation_test_sync, call_activation_sigmoid_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<sigmoid> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

/*
 * ACTIVATION SIGMOID DERIVATIVE
 * */
TEST(activation_test_sync, call_activation_sigmoid_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<sigmoid> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size) );
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i], activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}


//to do
//TEST(activation_test, call_activation_softmax)
//{
//  std::random_device rd;
//  std::mt19937 mt(rd());
//  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
//  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
//  std::shared_ptr<double> activation;
//  std::shared_ptr<double> activation_vector_copy;
//  double * device_activation_vector;
//  cudaError_t error_id;
//  std::uint16_t activation_vector_size = Z_plus(mt);
//  activation = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
//  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );//will store the results of call activation
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
//  error_id = cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) );
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";/**/
//  for(std::int16_t i = 0; i < activation_vector_size; ++i)
//  {	
//	activation.get()[i] = real(mt);
//	activation_vector_copy.get()[i] = activation.get()[i];
//  }
//  error_id = cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  ASSERT_EQ(call_activation(ACTIVATION_NAME::SOFTMAX_OUTPUT, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size), 0);
//  
//  error_id = cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//
//  for(std::int16_t i = 0; i < activation_vector_size; ++i)
//  {
//	ASSERT_EQ(activation.get()[i],activation_vector_copy.get()[i]);
//  }
//  error_id = cudaFree(device_activation_vector);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//}

TEST(activation_test_sync, call_activation_softplus_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<softplus> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size));
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_sync, call_activation_softplus_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<softplus> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  //randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}
TEST(activation_test_sync, call_activation_tanh_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<hyperbolic_tangent> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::TANH, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_sync, call_activation_tanh_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  //the activation function itself
  activation<hyperbolic_tangent> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::TANH, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}
TEST(activation_test_sync, call_activation_relu_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<relu> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::RELU, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_sync, call_activation_relu_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<relu> act;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  // will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_vector_size], std::default_delete<double[]>() );
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__)); 
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	// values post-activation
	activation_vector.get()[i] = act(activation_vector.get()[i],ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  //call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(call_activation(ACTIVATION_NAME::RELU, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size), 0);
  // copy activations from device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy.get(), device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}




TEST(activation_test_async, call_activation_identity_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<identity> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_async, call_activation_identity_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<identity> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}






TEST(activation_test_async, call_activation_sigmoid_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<sigmoid> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}


TEST(activation_test_async, call_activation_sigmoid_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<sigmoid> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}


TEST(activation_test_async, call_activation_softplus_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<softplus> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_async, call_activation_softplus_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<softplus> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}




TEST(activation_test_async, call_activation_tanh_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<hyperbolic_tangent> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::TANH, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_async, call_activation_tanh_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<hyperbolic_tangent> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::TANH, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}














TEST(activation_test_async, call_activation_relu_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<relu> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::OBJECTIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::RELU, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}

TEST(activation_test_async, call_activation_relu_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  // the activation function itself
  activation<relu> act;
  double * activation_vector;
  double * activation_vector_copy;
  double * device_activation_vector;

  // cuda stream
  cudaStream_t stream;

  // create stream
  cudaStreamCreate(&stream);

  // arbitrary layer size  based on ushort max
  std::uint16_t activation_vector_size = Z_plus(mt);
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  // will store the results of call activation
  ASSERT_EQ(0,zinhart::check_cuda_api(cudaHostAlloc((void**)&activation_vector_copy, sizeof(double) * activation_vector_size, cudaHostAllocDefault),__FILE__, __LINE__));
  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation vector size: "<<activation_vector_size<<" total bytes: "<<std::uint32_t(activation_vector_size) * sizeof(double)<<"\n";
  // allocate device activation vector
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double) ),__FILE__,__LINE__));
  // randomly initialize activations
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {	
	// values pre-activation
	activation_vector[i] = real(mt);
	activation_vector_copy[i] = activation_vector[i];
	// values post-activation
	activation_vector[i] = act(activation_vector[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  // copy pre-activation values to device
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_vector_size * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(device_activation_vector, activation_vector_copy, activation_vector_size * sizeof(double), cudaMemcpyHostToDevice, stream),__FILE__,__LINE__));
  // call activation function make sure the kernel wrapper returns 0
  ASSERT_EQ(0, call_activation(ACTIVATION_NAME::RELU, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_vector_size, stream));
  // copy activations from device to host
  //ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy( activation_vector_copy, device_activation_vector, std::uint32_t(activation_vector_size) * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(activation_vector_copy, device_activation_vector, activation_vector_size * sizeof(double), cudaMemcpyDeviceToHost, stream),__FILE__,__LINE__));

  // synchronize stream
  cudaStreamSynchronize(stream);

  // validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_vector_size; ++i)
  {
	ASSERT_EQ(activation_vector[i], activation_vector_copy[i]);
  }

  // release pinned host  memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(activation_vector_copy),__FILE__,__LINE__));

  // release device memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(device_activation_vector),__FILE__,__LINE__));

  // destroy cuda stream
  cudaStreamDestroy(stream);
  // reset device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__,__LINE__));
}
