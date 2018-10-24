#include <ann/optimizer.hh>
#include <gtest/gtest.h>
#include <random>
#include <limits>

using namespace zinhart::optimizers;

TEST(optimizer, sgd_thread_safety)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max() );
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, momentum_term{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, momentum_term{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double gamma{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate_pos{real_dist(mt)}, learning_rate_neg{real_dist(mt)}, delta_max{real_dist(mt)}, delta_min{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, beta_1_t{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
  std::uniform_int_distribution<std::uint32_t> thread_dist(zinhart::multi_core::default_thread_pool::get_default_thread_pool().size(), 20);
  std::uniform_int_distribution<std::uint32_t> uint_dist(20, std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t length{uint_dist(mt)}, n_threads{thread_dist(mt)}, i{0}, thread_id{0};
  double * weights{nullptr}, * gradient{nullptr}, * weights_test{nullptr}, * gradient_test{nullptr};
  double learning_rate{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, epsilon{real_dist(mt)};
  const std::uint32_t alignment = 64;

  // the thread pool & futures
  zinhart::multi_core::thread_pool pool(n_threads);
  std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;

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
/*
TEST(optimizer, adamax)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_mean{nullptr},* prior_variance{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_mean_test{nullptr}, * prior_variance_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_mean{0.0}, kth_prior_variance{0.0}, beta_1_t{real_dist(mt)}, eta{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, epsilon{real_dist(mt)}; 
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_mean = new double[n_elements];
  prior_variance = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_mean_test = new double[n_elements];
  prior_variance_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_mean = real_dist(mt);
	kth_prior_variance = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_mean[i] = kth_prior_mean;
	prior_variance[i] = kth_prior_variance;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_mean_test[i] = kth_prior_mean;
	prior_variance_test[i] = kth_prior_variance;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::ADAMAX(), theta, prior_mean, prior_variance, gradient, beta_1_t, n_elements, results, eta, beta_1, beta_2, epsilon);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_mean_test[i] = beta_1 * prior_mean_test[i] + (double{1.0} - beta_1) * gradient_test[i]; 
	prior_variance_test[i] = (beta_2 * prior_variance_test[i] > fabs(gradient_test[i])) ? beta_2 * prior_variance_test[i] : fabs(gradient_test[i]);
	theta_test[i] -= (eta / (double{1.0} - beta_1_t) ) * (prior_mean_test[i] / (prior_variance_test[i] + epsilon)); 
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_mean[i], prior_mean_test[i]);
	ASSERT_DOUBLE_EQ(prior_variance[i], prior_variance_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_mean;
  delete [] prior_variance;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_mean_test;
  delete [] prior_variance_test;
}

TEST(optimizer, adamax_moment_update)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double beta_1_t{dist(mt)}, beta_1_t_copy{beta_1_t}, beta_1{dist(mt)};
  beta_1_t_copy *= beta_1;
  zinhart::optimizers::adamax::moment_update<double>(beta_1_t, beta_1);
  ASSERT_EQ(beta_1_t_copy, beta_1_t);
}

TEST(optimizer, amsgrad)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_mean{nullptr}, * prior_variance{nullptr},* prior_bias_corrected_variance{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_mean_test{nullptr}, * prior_variance_test{nullptr}, * prior_bias_corrected_variance_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_mean{0.0}, kth_prior_variance{0.0}, kth_prior_bias_corrected_variance{0.0}, 
		 eta{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, epsilon{real_dist(mt)}; 
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_mean = new double[n_elements];
  prior_variance = new double[n_elements];
  prior_bias_corrected_variance = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_mean_test = new double[n_elements];
  prior_variance_test = new double[n_elements];
  prior_bias_corrected_variance_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_mean = real_dist(mt);
	kth_prior_variance = real_dist(mt);
	kth_prior_bias_corrected_variance = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_mean[i] = kth_prior_mean;
	prior_variance[i] = kth_prior_variance;
	prior_bias_corrected_variance[i] = kth_prior_bias_corrected_variance;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_mean_test[i] = kth_prior_mean;
	prior_variance_test[i] = kth_prior_variance;
	prior_bias_corrected_variance_test[i] = kth_prior_bias_corrected_variance;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::AMSGRAD(), theta, prior_mean, prior_variance, prior_bias_corrected_variance, gradient, n_elements, results, eta, beta_1, beta_2, epsilon);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_mean_test[i] = beta_1 * prior_mean_test[i] + (double{1} - beta_1) * gradient_test[i];
	prior_variance_test[i] = beta_2 * prior_variance_test[i] + (double{1} - beta_2) * gradient_test[i] * gradient_test[i];
	// max(prior_variance > prior_bias_corrected_variance)
	prior_bias_corrected_variance_test[i] = (prior_variance_test[i] > prior_bias_corrected_variance_test[i]) ? prior_variance_test[i] : prior_bias_corrected_variance_test[i];
	theta_test[i] -= eta * ( prior_mean_test[i] ) / ( sqrt( prior_bias_corrected_variance_test[i]) + epsilon ) ;
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_mean[i], prior_mean_test[i]);
	ASSERT_DOUBLE_EQ(prior_variance[i], prior_variance_test[i]);
	ASSERT_DOUBLE_EQ(prior_bias_corrected_variance[i], prior_bias_corrected_variance_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_mean;
  delete [] prior_variance;
  delete [] prior_bias_corrected_variance;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_mean_test;
  delete [] prior_variance_test;
  delete [] prior_bias_corrected_variance_test;
}

TEST(optimizer, adam)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_mean{nullptr},* prior_variance{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_mean_test{nullptr}, * prior_variance_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_mean{0.0}, kth_prior_variance{0.0}, beta_1_t{real_dist(mt)}, beta_2_t{real_dist(mt)},
		 eta{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, epsilon{real_dist(mt)}; 
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_mean = new double[n_elements];
  prior_variance = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_mean_test = new double[n_elements];
  prior_variance_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_mean = real_dist(mt);
	kth_prior_variance = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_mean[i] = kth_prior_mean;
	prior_variance[i] = kth_prior_variance;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_mean_test[i] = kth_prior_mean;
	prior_variance_test[i] = kth_prior_variance;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::ADAM(), theta, prior_mean, prior_variance, gradient, beta_1_t, beta_2_t, n_elements, results, eta, beta_1, beta_2, epsilon);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_mean_test[i] = beta_1 * prior_mean_test[i] + (double{1} - beta_1) * gradient_test[i];
	prior_variance_test[i] = beta_2 * prior_variance_test[i] + (double{1} - beta_2) * gradient_test[i] * gradient_test[i];
	double bias_corrected_mean{ prior_mean_test[i] / (double{1} - beta_1_t) };
	double bias_corrected_variace{ prior_variance_test[i] / (double{1} - beta_2_t) };
	theta_test[i] -= eta * ( bias_corrected_mean ) / (sqrt( bias_corrected_variace ) + epsilon ) ;
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_mean[i], prior_mean_test[i]);
	ASSERT_DOUBLE_EQ(prior_variance[i], prior_variance_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_mean;
  delete [] prior_variance;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_mean_test;
  delete [] prior_variance_test;
}
TEST(optimizer, adam_moment_update)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double beta_1_t{dist(mt)}, beta_1_t_copy{beta_1_t}, beta_2_t{dist(mt)}, beta_2_t_copy{beta_2_t}, beta_1{dist(mt)}, beta_2{dist(mt)};
  beta_1_t_copy *= beta_1;
  beta_2_t_copy *= beta_2;

  zinhart::optimizers::adam::moment_update<double>(beta_1_t, beta_2_t, beta_1, beta_2);
  
  ASSERT_EQ(beta_1_t_copy, beta_1_t);
  ASSERT_EQ(beta_2_t_copy, beta_2_t);
}


TEST(optimizer, nadam)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_mean{nullptr},* prior_variance{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_mean_test{nullptr}, * prior_variance_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_mean{0.0}, kth_prior_variance{0.0}, beta_1_t{real_dist(mt)}, beta_2_t{real_dist(mt)},
		 eta{real_dist(mt)}, gamma{real_dist(mt)}, beta_1{real_dist(mt)}, beta_2{real_dist(mt)}, epsilon{real_dist(mt)}; 
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_mean = new double[n_elements];
  prior_variance = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_mean_test = new double[n_elements];
  prior_variance_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_mean = real_dist(mt);
	kth_prior_variance = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_mean[i] = kth_prior_mean;
	prior_variance[i] = kth_prior_variance;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_mean_test[i] = kth_prior_mean;
	prior_variance_test[i] = kth_prior_variance;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::NADAM(), theta, prior_mean, prior_variance, gradient, beta_1_t, beta_2_t, n_elements, results, eta, gamma, beta_1, beta_2, epsilon);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_mean_test[i] = beta_1 * prior_mean_test[i] + (double{1} - beta_1) * gradient_test[i];
	prior_variance_test[i] = beta_2 * prior_variance_test[i] + (double{1} - beta_2) * gradient_test[i] * gradient_test[i];
	double prior_bias_corrected_mean{prior_mean_test[i] / (double{1} - beta_1_t )};
	double prior_bias_corrected_variance{prior_variance_test[i] / (double{1} - beta_2_t)};
	theta_test[i] -= eta / ( sqrt(prior_bias_corrected_variance) + epsilon ) * (beta_1 * prior_bias_corrected_mean + (double{1} - beta_1) / (double{1} - beta_1_t) * gradient_test[i] );
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_mean[i], prior_mean_test[i]);
	ASSERT_DOUBLE_EQ(prior_variance[i], prior_variance_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_mean;
  delete [] prior_variance;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_mean_test;
  delete [] prior_variance_test;
}
TEST(optimizer, nadam_moment_update)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double beta_1_t {dist(mt)}, beta_1_t_copy{beta_1_t}, beta_2_t{dist(mt)}, beta_2_t_copy{beta_2_t}, beta_1{dist(mt)}, beta_2{dist(mt)};
  beta_1_t_copy *= beta_1;
  beta_2_t_copy *= beta_2;
  
  zinhart::optimizers::nadam::moment_update<double>(beta_1_t, beta_2_t, beta_1, beta_2);

  ASSERT_EQ(beta_1_t_copy, beta_1_t);
  ASSERT_EQ(beta_2_t_copy, beta_2_t);
}
*/
