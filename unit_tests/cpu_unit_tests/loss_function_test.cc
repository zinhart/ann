#include <gtest/gtest.h>
#include <ann/loss_function.hh>
#include <random>
#include <limits>

//using namespace zinhart::function_space;
//using namespace zinhart::function_space::error_metrics;
using namespace zinhart::loss_functions;
using namespace zinhart::function_space;

TEST(loss_function_test, cross_entropy_multi_class)
{
  // random number generators
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());

  // loop counters & testing variables
  const std::uint32_t alignment{64};
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
  double sum{0}, sum_test{0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  double * results{nullptr}, * results_test{nullptr};

  // a loss function
  loss_function<double> * ce_multi_class = new cross_entropy_multi_class<double>();
  // an error function
  error_function<double> e;

  // allocate 
  outputs = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  targets = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  outputs_test = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  targets_test = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  results = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  results_test = (double*) mkl_malloc(n_elements * sizeof(double), alignment);


  // initialize vectors
  for(i = 0; i < n_elements; ++i)
  {
	outputs[i] = real_dist(mt);
	outputs_test[i] = outputs[i];
	targets[i] = real_dist(mt);
	targets_test[i] = targets[i];
	results[i] = 0.0;
	results_test[i] = 0.0;
  }	

  // calculate error
  sum = ce_multi_class->error(objective(), outputs, targets, n_elements);

  // perform error test
  for(i = 0; i < n_elements; ++i)
	sum_test += e.objective(loss_attributes::cross_entropy_multi_class(), *(outputs_test + i), *(targets_test + i));
  
  // validate loss objective
  ASSERT_DOUBLE_EQ(sum, sum_test);

  // calculate error derivative
  ce_multi_class->error(derivative(), outputs, targets, results, n_elements);

  // perform error derivative test
  for(i = 0; i < n_elements; ++i)
	*(results_test + i) = e.derivative(loss_attributes::cross_entropy_multi_class(), *(outputs_test + i), *(targets_test + i));

  // cleanup
  delete ce_multi_class;
  mkl_free(outputs);
  mkl_free(targets);
  mkl_free(results);
  mkl_free(outputs_test);
  mkl_free(targets_test);
  mkl_free(results_test);
}


TEST(loss_function_test, mean_squared_error)
{
  // random number generators
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());

  // loop counters & testing variables
  const std::uint32_t alignment{64};
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
  double sum{0}, sum_test{0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  double * results{nullptr}, * results_test{nullptr};

  // a loss function
  loss_function<double> * mse = new mean_squared_error<double>();
  // an error function
  error_function<double> e;

  // allocate 
  outputs = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  targets = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  outputs_test = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  targets_test = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  results = (double*) mkl_malloc(n_elements * sizeof(double), alignment);
  results_test = (double*) mkl_malloc(n_elements * sizeof(double), alignment);


  // initialize vectors
  for(i = 0; i < n_elements; ++i)
  {
	outputs[i] = real_dist(mt);
	outputs_test[i] = outputs[i];
	targets[i] = real_dist(mt);
	targets_test[i] = targets[i];
	results[i] = 0.0;
	results_test[i] = 0.0;
  }	

  // calculate error
  sum = mse->error(objective(), outputs, targets, n_elements);

  // perform error test
  for(i = 0; i < n_elements; ++i)
	sum_test += e.objective(loss_attributes::mean_squared_error(), *(outputs_test + i), *(targets_test + i));
  
  // validate loss objective
  ASSERT_DOUBLE_EQ(sum, sum_test);

  // calculate error derivative
  mse->error(derivative(), outputs, targets, results, n_elements);

  // perform error derivative test
  for(i = 0; i < n_elements; ++i)
	*(results_test + i) = e.derivative(loss_attributes::mean_squared_error(), *(outputs_test + i), *(targets_test + i));

  // cleanup
  delete mse;
  mkl_free(outputs);
  mkl_free(targets);
  mkl_free(results);
  mkl_free(outputs_test);
  mkl_free(targets_test);
  mkl_free(results_test);
}
/*
TEST(loss_function_test, cross_entropy_multi_class_objective)
{

  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
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
  loss_function loss;
  parallel_error = loss(LOSS_FUNCTION_NAME::CROSS_ENTROPY_MULTI_CLASS, objective(), outputs, targets, n_elements);
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
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
  double kth_target{0.0}, kth_output{0.0}, serial_error{0.0}, parallel_error{0.0};
  double * outputs{nullptr}, * targets{nullptr}, * error{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr}, * error_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  error = new double[n_elements];
  error_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
	error[i] = 0.0;
	error_test[i] = 0.0;
  }	
  auto ce_derivative = [](double kth_output, double kth_target){return kth_output - kth_target;};
  loss_function loss;
  for(i = 0; i < n_elements; ++i)
	*(error_test + i ) = ce_derivative( *(outputs + i), *(targets + i) );
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  EXPECT_DOUBLE_EQ( serial_error, parallel_error );
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
  delete [] error;
  delete [] error_test;
}

TEST(loss_function_test, mean_square_error_objective)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, batch_size{uint_dist(mt)};
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
  parallel_error = loss(LOSS_FUNCTION_NAME::MSE, objective(), outputs, targets, n_elements, batch_size);
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
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
  double kth_target{0.0}, kth_output{0.0}, parallel_error{0.0}, batch_size{uint_dist(mt)};
  double * outputs{nullptr}, * targets{nullptr}, * error{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr}, * error_test{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  outputs = new double[n_elements];
  targets = new double[n_elements];
  outputs_test = new double[n_elements];
  targets_test = new double[n_elements];
  error = new double[n_elements];
  error_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_output = real_dist(mt);
	kth_target = real_dist(mt);
	outputs[i] = kth_output;
	outputs_test[i] = kth_output;
	targets[i] = kth_target;
	targets_test[i] = kth_target;
	error[i] = 0.0;
	error_test[i] = 0.0;
  }	
  auto mse_derivative = [batch_size](double kth_output, double kth_target){return double{2.0} / double{batch_size} * (kth_output - kth_target);};
  loss_function loss;
  loss(LOSS_FUNCTION_NAME::MSE, derivative(), outputs, targets, error, n_elements, batch_size);
  for(i = 0; i < n_elements; ++i)
	*(error_test + i ) = mse_derivative( *(outputs + i), *(targets + i) );
  
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 

  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_DOUBLE_EQ(error[i], error_test[i]);
  }
  
  delete [] outputs;
  delete [] outputs_test;
  delete [] targets;
  delete [] targets_test;
  delete [] error;
  delete [] error_test;
}*/
