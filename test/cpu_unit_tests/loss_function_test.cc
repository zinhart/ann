#include <gtest/gtest.h>
#include "ann/loss_function.hh"
#include <random>
#include <limits>

using namespace zinhart::function_space;
using namespace zinhart::function_space::error_metrics;
/*
TEST(loss_function_test, cross_entropy_multi_class_objective)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto ce = [](double kth_output, double kth_target, double epsilon = 1.e-30){return kth_target * log(kth_output + epsilon);};
  zinhart::error_metrics::loss_function loss;
  loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, LOSS_FUNCTION_TYPE::OBJECTIVE, parallel_error, outputs, targets, n_elements, results);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, ce);
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  EXPECT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}

TEST(loss_function_test, cross_entropy_multi_class_derivative)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto ce_derivative = [](double kth_output, double kth_target, double epsilon = 1.e-30){return kth_output - kth_target;};
  zinhart::error_metrics::loss_function loss;
  loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, LOSS_FUNCTION_TYPE::DERIVATIVE, parallel_error, outputs, targets, n_elements, results);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, ce_derivative);
  
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  EXPECT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}

TEST(loss_function_test, mean_square_error_objective)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto mse = [](double kth_output, double kth_target){return 0.5 * (kth_output - kth_target) * (kth_output - kth_target);};
  zinhart::error_metrics::loss_function loss;
  loss(LOSS_FUNCTION_NAME::MSE, LOSS_FUNCTION_TYPE::OBJECTIVE, parallel_error, outputs, targets, n_elements, results);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, mse);
  
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  ASSERT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}
TEST(loss_function_test, mean_square_error_derivative)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto mse_derivative = [](double kth_output, double kth_target){return double{2.0} * (kth_output - kth_target);};
  zinhart::error_metrics::loss_function loss;
  loss(LOSS_FUNCTION_NAME::MSE, LOSS_FUNCTION_TYPE::DERIVATIVE, parallel_error, outputs, targets, n_elements, results);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, mse_derivative);
  
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  ASSERT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}
TEST(loss_function_test_2, cross_entropy_multi_class_objective)
{

  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto ce = [](double kth_output, double kth_target, double epsilon = 1.e-30){return kth_target * log(kth_output + epsilon);};
  zinhart::error_metrics::loss_function loss;
  parallel_error = loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, LOSS_FUNCTION_TYPE::OBJECTIVE, outputs, targets, n_elements);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, ce);
  
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  EXPECT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}


TEST(loss_function_test_2, cross_entropy_multi_class_derivative)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto ce_derivative = [](double kth_output, double kth_target, double epsilon = 1.e-30){return kth_output - kth_target;};
  zinhart::error_metrics::loss_function loss;
  parallel_error = loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, LOSS_FUNCTION_TYPE::DERIVATIVE, outputs, targets, n_elements);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, ce_derivative);
  
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  EXPECT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}
*/
TEST(loss_function_test_2, mean_square_error_objective)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0}, batch_size{uint_dist(mt)};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto mse = [batch_size](double kth_output, double kth_target){return double{1} / double{batch_size}  * (kth_output - kth_target) * (kth_output - kth_target);};
  loss_function loss;
  parallel_error = loss(LOSS_FUNCTION_NAME::MSE, OBJECTIVE(), outputs, targets, n_elements, batch_size);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, mse);
  
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  ASSERT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}
/*
TEST(loss_function_test_2, mean_square_error_derivative)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
  }	
  auto mse_derivative = [batch_size](double kth_output, double kth_target){return double{2.0} / double{batch_size} * (kth_output - kth_target);};
  zinhart::error_metrics::loss_function loss;
  parallel_error = loss(LOSS_FUNCTION_NAME::MSE, LOSS_FUNCTION_TYPE::DERIVATIVE, outputs, targets, n_elements, batch_size);
  serial_error = zinhart::serial::neumaier_sum(outputs_test, targets_test, n_elements, mse_derivative);
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  ASSERT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
}*/
