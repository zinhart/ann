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
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());

  // loop counters & testing variables
  const std::uint32_t alignment{64};
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
  double sum{0}, sum_test{0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  double * results{nullptr}, * results_test{nullptr};

  // a loss function
  loss_function<double> * loss = new cross_entropy_multi_class<double>();
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
  sum = loss->error(objective(), outputs, targets, n_elements);

  // perform error test
  for(i = 0; i < n_elements; ++i)
	sum_test += -e.objective(loss_attributes::cross_entropy_multi_class(), *(outputs_test + i), *(targets_test + i));
  
  // validate loss objective
  EXPECT_DOUBLE_EQ(sum, sum_test);

  // calculate error derivative
  loss->error(derivative(), outputs, targets, results, n_elements);

  // perform error derivative test
  for(i = 0; i < n_elements; ++i)
	*(results_test + i) = e.derivative(loss_attributes::cross_entropy_multi_class(), *(outputs_test + i), *(targets_test + i));

  // validate loss derivative
  for(i = 0; i < n_elements; ++i)
    EXPECT_DOUBLE_EQ(*(results + i), *(results_test + i));

  // cleanup
  delete loss;
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
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), /*std::numeric_limits<std::uint16_t>::max()*/ 30);
  std::mt19937 mt(rd());

  // loop counters & testing variables
  const std::uint32_t alignment{64};
  std::uint32_t n_elements{uint_dist(mt)}, i{0};
  double sum{0}, sum_test{0};
  double * outputs{nullptr}, * targets{nullptr}, * outputs_test{nullptr}, * targets_test{nullptr};
  double * results{nullptr}, * results_test{nullptr};

  // a loss function
  loss_function<double> * loss = new mean_squared_error<double>();
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
  sum = loss->error(objective(), outputs, targets, n_elements);

  // perform error test
  for(i = 0; i < n_elements; ++i)
	sum_test += e.objective(loss_attributes::mean_squared_error(), *(outputs_test + i), *(targets_test + i));
  sum_test /= n_elements;
  
  // validate loss objective
  EXPECT_DOUBLE_EQ(sum, sum_test);

  // calculate error derivative
  loss->error(derivative(), outputs, targets, results, n_elements);

  // perform error derivative test
  for(i = 0; i < n_elements; ++i)
	*(results_test + i) = e.derivative(loss_attributes::mean_squared_error(), *(outputs_test + i), *(targets_test + i), n_elements);

  // validate loss derivative
  for(i = 0; i < n_elements; ++i)
    EXPECT_DOUBLE_EQ(*(results + i), *(results_test + i));

  // cleanup
  delete loss;
  mkl_free(outputs);
  mkl_free(targets);
  mkl_free(results);
  mkl_free(outputs_test);
  mkl_free(targets_test);
  mkl_free(results_test);
}
