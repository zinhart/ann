#include "ann/activation.hh"
#include <cublas_v2.h>
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <memory>
using namespace zinhart;

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> pos_real(0, std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> reals(std::numeric_limits<double>::min(), std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> neg_real(std::numeric_limits<double>::min(), -1 );
/*
 * CUDA WRAPPERS
 * */
/*
 * ACTIVATION OBJECTIVE
 * */
TEST(activation_test, call_activation_identity_objective)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  //the activation function itself
  activation<zinhart::identity> act_identity;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  //arbitrary layer size limited to ushort max
  std::uint16_t activation_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
  //will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
  //std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
  error_id = cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_size) * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  //randomly initialize activations
  for(std::int16_t i = 0; i < activation_size; ++i)
  {
	//values pre activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	//values post activation
	activation_vector.get()[i] = act_identity(activation_vector.get()[i],ACTIVATION_TYPE::OBJECTIVE); 
  }
  //copy activations from host to device
  error_id = cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  //call activation function
  ASSERT_EQ(call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::OBJECTIVE, device_activation_vector, activation_size), 0);
  //copy activations from device to host
  error_id = cudaMemcpy(activation_vector_copy.get(), device_activation_vector, activation_size * sizeof(double), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  //validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  //release device memory
  error_id = cudaFree(device_activation_vector);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}
TEST(activation_test, call_activation_identity_derivative)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
  //the activation function itself
  activation<identity> act_identity;
  std::shared_ptr<double> activation_vector;
  std::shared_ptr<double> activation_vector_copy;
  double * device_activation_vector;
  cudaError_t error_id;
  //arbitrary layer size limited to ushort max
  std::uint16_t activation_size = Z_plus(mt);
  activation_vector = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
  //will store the results of call activation
  activation_vector_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
  error_id = cudaMalloc( (void **) &device_activation_vector, std::uint32_t(activation_size) * sizeof(double) );
  if(error_id != cudaSuccess)
	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  //randomly initialize activations
  for(std::int16_t i = 0; i < activation_size; ++i)
  {	
	//values pre activation
	activation_vector.get()[i] = real(mt);
	activation_vector_copy.get()[i] = activation_vector.get()[i];
	//values post activation
	activation_vector.get()[i] = act_identity(activation_vector.get()[i], ACTIVATION_TYPE::DERIVATIVE);
  }
  //copy activations from host to device
  error_id = cudaMemcpy(device_activation_vector, activation_vector_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  //call activation function
  ASSERT_EQ(call_activation(ACTIVATION_NAME::IDENTITY, ACTIVATION_TYPE::DERIVATIVE, device_activation_vector, activation_size), 0);
  //copy activations from device to host
  error_id = cudaMemcpy(activation_vector_copy.get(), device_activation_vector, activation_size * sizeof(double), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  //validate each value in activation copy is the same since this is the identity function
  for(std::int16_t i = 0; i < activation_size; ++i)
  {
	ASSERT_EQ(activation_vector.get()[i],activation_vector_copy.get()[i]);
  }
  //release device memory
  error_id = cudaFree(device_activation_vector);
  if(error_id != cudaSuccess)
	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}

//TEST(activation_test, call_activation_sigmoid_object)
//{
//  std::random_device rd;
//  std::mt19937 mt(rd());
//  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
//  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
//  std::shared_ptr<double> activation;
//  std::shared_ptr<double> activation_copy;
//  double * device_activations;
//  cudaError_t error_id;
//  std::uint16_t activation_size = Z_plus(mt);
//  activation = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
//  activation_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );//will store the results of call activation
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
//  error_id = cudaMalloc( (void **) &device_activations, std::uint32_t(activation_size) * sizeof(double) );
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {	
//	activation.get()[i] = real(mt);
//	activation_copy.get()[i] = activation.get()[i];
//  }
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  ASSERT_EQ(call_activation(ACTIVATION_NAME::SIGMOID, ACTIVATION_TYPE::OBJECTIVE, device_activations, activation_size), 0);
//  
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {
//	ASSERT_EQ(activation.get()[i],activation_copy.get()[i]);
//  }
//  error_id = cudaFree(device_activations);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//}
//to do
//TEST(activation_test, call_activation_softmax)
//{
//  std::random_device rd;
//  std::mt19937 mt(rd());
//  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
//  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
//  std::shared_ptr<double> activation;
//  std::shared_ptr<double> activation_copy;
//  double * device_activations;
//  cudaError_t error_id;
//  std::uint16_t activation_size = Z_plus(mt);
//  activation = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
//  activation_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );//will store the results of call activation
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
//  error_id = cudaMalloc( (void **) &device_activations, std::uint32_t(activation_size) * sizeof(double) );
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";/**/
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {	
//	activation.get()[i] = real(mt);
//	activation_copy.get()[i] = activation.get()[i];
//  }
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  ASSERT_EQ(call_activation(ACTIVATION_NAME::SOFTMAX, ACTIVATION_TYPE::OBJECTIVE, device_activations, activation_size), 0);
//  
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {
//	ASSERT_EQ(activation.get()[i],activation_copy.get()[i]);
//  }
//  error_id = cudaFree(device_activations);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//}
//
//TEST(activation_test, call_activation_softplus_objective)
//{
//  std::random_device rd;
//  std::mt19937 mt(rd());
//  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
//  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
//  std::shared_ptr<double> activation;
//  std::shared_ptr<double> activation_copy;
//  double * device_activations;
//  cudaError_t error_id;
//  std::uint16_t activation_size = Z_plus(mt);
//  activation = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
//  activation_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );//will store the results of call activation
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
//  error_id = cudaMalloc( (void **) &device_activations, std::uint32_t(activation_size) * sizeof(double) );
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";/**/
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {	
//	activation.get()[i] = real(mt);
//	activation_copy.get()[i] = activation.get()[i];
//  }
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  ASSERT_EQ(call_activation(ACTIVATION_NAME::SOFTPLUS, ACTIVATION_TYPE::OBJECTIVE, device_activations, activation_size), 0);
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {
//	ASSERT_EQ(activation.get()[i],activation_copy.get()[i]);
//  }
//  error_id = cudaFree(device_activations);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//}
//
//TEST(activation_test, call_activation_tanh_objective)
//{
//  std::random_device rd;
//  std::mt19937 mt(rd());
//  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
//  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
//  std::shared_ptr<double> activation;
//  std::shared_ptr<double> activation_copy;
//  double * device_activations;
//  cudaError_t error_id;
//  std::uint16_t activation_size = Z_plus(mt);
//  activation = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
//  activation_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );//will store the results of call activation
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
//  error_id = cudaMalloc( (void **) &device_activations, std::uint32_t(activation_size) * sizeof(double) );
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";/**/
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {	
//	activation.get()[i] = real(mt);
//	activation_copy.get()[i] = activation.get()[i];
//  }
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  ASSERT_EQ(call_activation(ACTIVATION_NAME::TANH, ACTIVATION_TYPE::OBJECTIVE, device_activations, activation_size), 0);
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {
//	ASSERT_EQ(activation.get()[i],activation_copy.get()[i]);
//  }
//  error_id = cudaFree(device_activations);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//}
//
//TEST(activation_test, call_activation_relu_objective)
//{
//  std::random_device rd;
//  std::mt19937 mt(rd());
//  std::uniform_int_distribution<std::uint16_t> Z_plus(1, std::numeric_limits<std::uint16_t>::max() / 2  );
//  std::uniform_real_distribution<float> real(std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
//  std::shared_ptr<double> activation;
//  std::shared_ptr<double> activation_copy;
//  double * device_activations;
//  cudaError_t error_id;
//  std::uint16_t activation_size = Z_plus(mt);
//  activation = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );
//  activation_copy = std::shared_ptr<double> ( new double[activation_size], std::default_delete<double[]>() );//will store the results of call activation
//  std::cout<<"max value of ushort: "<<std::numeric_limits<std::uint16_t>::max() <<" activation size: "<<activation_size<<" total bytes: "<<std::uint32_t(activation_size) * sizeof(double)<<"\n";
//  error_id = cudaMalloc( (void **) &device_activations, std::uint32_t(activation_size) * sizeof(double) );
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation allocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";/**/
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {	
//	activation.get()[i] = real(mt);
//	activation_copy.get()[i] = activation.get()[i];
//  }
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//  ASSERT_EQ(call_activation(ACTIVATION_NAME::RELU, ACTIVATION_TYPE::OBJECTIVE, device_activations, activation_size), 0);
//  
//  error_id = cudaMemcpy(device_activations, activation_copy.get(), activation_size * sizeof(double), cudaMemcpyHostToDevice);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//
//  for(std::int16_t i = 0; i < activation_size; ++i)
//  {
//	ASSERT_EQ(activation.get()[i],activation_copy.get()[i]);
//  }
//  error_id = cudaFree(device_activations);
//  if(error_id != cudaSuccess)
//	std::cerr<<"device activation deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
//}
