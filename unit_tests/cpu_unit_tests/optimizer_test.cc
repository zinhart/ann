#include <ann/optimizer.hh>
#include <gtest/gtest.h>
#include <random>
#include <limits>

using namespace zinhart::optimizers;

TEST(optimizer, sgd_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max() );
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<sgd<double>>(learning_rate);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, momentum_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, momentum_term{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<momentum<double>>(length, learning_rate, momentum_term);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<momentum<double>>(length, learning_rate, momentum_term);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, nesterov_momentum_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, momentum_term{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<nesterov_momentum<double>>(length, learning_rate, momentum_term);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<nesterov_momentum<double>>(length, learning_rate, momentum_term);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, adagrad_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<adagrad<double>>(length, learning_rate, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<adagrad<double>>(length, learning_rate, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, conjugate_gradient_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<conjugate_gradient<double>>(length, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<conjugate_gradient<double>>(length, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, adadelta_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double gamma{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<adadelta<double>>(length, gamma, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<adadelta<double>>(length, gamma, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, rms_prop_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<rms_prop<double>>(length, learning_rate, beta, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<rms_prop<double>>(length, learning_rate, beta, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, rprop_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate_pos{real_dist(mt)}, learning_rate_neg{real_dist(mt)}, delta_max{real_dist(mt)}, delta_min{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<rprop<double>>(length, learning_rate_pos, learning_rate_neg, delta_max, delta_min);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<rprop<double>>(length, learning_rate_pos, learning_rate_neg, delta_max, delta_min);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}

TEST(optimizer, adamax_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, beta_1_t{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<adamax<double>>(length, learning_rate, beta_1, beta_2, beta_1_t, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<adamax<double>>(length, learning_rate, beta_1, beta_2, beta_1_t, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // validation of moment update
  op->update_bias_correction();
  EXPECT_DOUBLE_EQ(op->get_bias_corrected_first_moment(), beta_1_t * beta_1 );

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}


TEST(optimizer, amsgrad_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<amsgrad<double>>(length, learning_rate, beta_1, beta_2, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<amsgrad<double>>(length, learning_rate, beta_1, beta_2, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}


TEST(optimizer, adam_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, beta_1_t{real_dist(mt)}, beta_2_t{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<adam<double>>(length, learning_rate, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<adam<double>>(length, learning_rate, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // validation of moment update
  op->update_bias_correction();
  EXPECT_DOUBLE_EQ(op->get_bias_corrected_first_moment(), beta_1_t * beta_1 );
  EXPECT_DOUBLE_EQ(op->get_bias_corrected_second_moment(), beta_2_t * beta_2 );

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}


TEST(optimizer, nadam_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::thread_pool::get_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, gamma{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, beta_1_t{real_dist(mt)}, beta_2_t{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool::pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;

  // the optimizer
  std::shared_ptr<optimizer<double>> op = std::make_shared<nadam<double>>(length, learning_rate, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);
  std::shared_ptr<optimizer<double>> op_test = std::make_shared<nadam<double>>(length, learning_rate, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);

  // allocate
  weights = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient = (double*) mkl_malloc(length * sizeof(double), alignment);
  weights_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  gradient_test = (double*) mkl_malloc(length * sizeof(double), alignment);
  
  // initialize
  for(i = 0; i < length; ++i)
  {
	weights[i] = real_dist(mt);
	weights_test[i] = weights[i];
	gradient[i] = real_dist(mt);
	gradient_test[i] = gradient[i];
  }
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
  {
	// add tasks to pool and store in results
	results.push_back(pool.add_task(optimize_m, op, weights, gradient, length, n_threads, thread_id));
  }
  // serial version in test
  optimize_m(op_test, weights_test, gradient_test, length);

  // synchronize
  for(thread_id = 0; thread_id < n_threads; ++thread_id) 
	results[thread_id].get();

  // validate
  for(i = 0; i < length; ++i)
  {
	EXPECT_DOUBLE_EQ(weights[i], weights_test[i])<<"i: "<<i;
	EXPECT_DOUBLE_EQ(gradient[i], gradient_test[i])<<"i: "<<i;
  }	

  // validation of moment update
  op->update_bias_correction();
  EXPECT_DOUBLE_EQ(op->get_bias_corrected_first_moment(), beta_1_t * beta_1 );
  EXPECT_DOUBLE_EQ(op->get_bias_corrected_second_moment(), beta_2_t * beta_2 );

  // cleanup
  mkl_free(weights);
  mkl_free(gradient);
  mkl_free(weights_test);
  mkl_free(gradient_test);
}
