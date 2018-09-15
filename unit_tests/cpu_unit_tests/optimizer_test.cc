#include <ann/optimizer.hh>
#include <gmock/gmock.h>
#include <random>
#include <limits>

TEST(optimizer, sgd)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * theta_test{nullptr}, * gradient_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, eta{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::SGD(), theta, gradient, n_elements, results, eta);
  for(i = 0; i < n_elements; ++i)
  {
	theta_test[i] -= eta * gradient_test[i]; 
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] theta_test;
  delete [] gradient_test;
}

TEST(optimizer, momentum)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_velocity{nullptr}, * theta_test{nullptr}, * gradient_test{nullptr}, *prior_velocity_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_velocity{0.0}, gamma{real_dist(mt)}, eta{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_velocity = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_velocity_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_velocity = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_velocity[i] = kth_velocity;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_velocity_test[i] = kth_velocity;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::MOMENTUM(), theta, n_elements, prior_velocity, gradient, gamma, eta, results);
  for(i = 0; i < n_elements; ++i)
  {   		   
	double current_velocity{ gamma * prior_velocity_test[i] + eta * gradient_test[i] };
	theta_test[i] -= current_velocity;
    prior_velocity_test[i] = current_velocity;	
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_velocity[i], prior_velocity_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_velocity;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_velocity_test;
}

TEST(optimizer, nesterov_momentum)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_velocity{nullptr}, * theta_test{nullptr}, * gradient_test{nullptr}, *prior_velocity_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_velocity{0.0}, gamma{real_dist(mt)}, eta{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_velocity = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_velocity_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_velocity = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_velocity[i] = kth_velocity;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_velocity_test[i] = kth_velocity;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::NESTEROV_MOMENTUM(), theta, n_elements, prior_velocity, gradient, gamma, eta, results);
  for(i = 0; i < n_elements; ++i)
  {   		   
	double current_velocity { gamma * prior_velocity_test[i] + eta * gradient_test[i] * (  theta_test[i] - gamma * prior_velocity_test[i]) };
	theta_test[i] -= current_velocity;
    prior_velocity_test[i] = current_velocity;	
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_velocity[i], prior_velocity_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_velocity;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_velocity_test;
}
TEST(optimizer, adagrad)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * theta_test{nullptr}, * gradient_test{nullptr}, * prior_gradient_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_gradient{0.0}, eta{real_dist(mt)}, epsilon{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_gradient = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_gradient_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_gradient = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_gradient[i] = kth_prior_gradient;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_gradient_test[i] = kth_prior_gradient;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::ADAGRAD(), theta, n_elements, prior_gradient, gradient, eta, epsilon, results);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_gradient_test[i] += gradient_test[i] * gradient_test[i];
	theta_test[i] -= eta * gradient_test[i] / sqrt(prior_gradient_test[i] + epsilon);
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_gradient[i], prior_gradient_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_gradient;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_gradient_test;
}

TEST(optimizer, conjugate_gradient)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_hessian{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_gradient_test{nullptr}, * prior_hessian_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_gradient{0.0}, kth_prior_hessian{0.0}, eta{real_dist(mt)}, epsilon{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_gradient = new double[n_elements];
  prior_hessian = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_gradient_test = new double[n_elements];
  prior_hessian_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_gradient = real_dist(mt);
	kth_prior_hessian = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_gradient[i] = kth_prior_gradient;
	prior_hessian[i] = kth_prior_hessian;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_gradient_test[i] = kth_prior_gradient;
	prior_hessian_test[i] = kth_prior_hessian;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::CONJUGATE_GRADIENT(), theta, prior_gradient, prior_hessian, gradient, epsilon, n_elements, results);
  for(i = 0; i < n_elements; ++i)
  {   		   

	double gamma { ( gradient_test[i] - prior_gradient_test[i] ) * gradient_test[i] / ( prior_gradient_test[i] * prior_gradient_test[i] + epsilon ) };
	double step { gradient_test[i] + ( gamma * prior_hessian_test[i] ) };
	theta_test[i] += step;
	prior_gradient_test[i] = gradient_test[i];
	prior_hessian_test[i] = step;
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_gradient[i], prior_gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_hessian[i], prior_hessian_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_gradient;
  delete [] prior_hessian;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_gradient_test;
  delete [] prior_hessian_test;
}

TEST(optimizer, adadelta)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_delta{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_gradient_test{nullptr}, * prior_delta_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_gradient{0.0}, kth_prior_delta{0.0}, gamma{real_dist(mt)}, epsilon{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_gradient = new double[n_elements];
  prior_delta = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_gradient_test = new double[n_elements];
  prior_delta_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_gradient = real_dist(mt);
	kth_prior_delta = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_gradient[i] = kth_prior_gradient;
	prior_delta[i] = kth_prior_delta;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_gradient_test[i] = kth_prior_gradient;
	prior_delta_test[i] = kth_prior_delta;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::ADADELTA(), theta, prior_gradient, prior_delta, gradient, n_elements, results, gamma, epsilon);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_gradient_test[i] = gamma * prior_gradient_test[i] + (double{1.0} - gamma) * gradient_test[i] * gradient_test[i];
	double delta { -(sqrt(prior_delta_test[i] * prior_delta_test[i] + epsilon) / sqrt(prior_gradient_test[i] * prior_gradient_test[i] + epsilon)) * gradient_test[i] };
	theta_test[i] += delta;
	prior_delta_test[i] = gamma * prior_delta_test[i] + (double{1.0} - gamma) * delta * delta;
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_gradient[i], prior_gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_delta[i], prior_delta_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_gradient;
  delete [] prior_delta;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_gradient_test;
  delete [] prior_delta_test;
}

TEST(optimizer, rms_prop)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * theta_test{nullptr}, * gradient_test{nullptr}, * prior_gradient_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_gradient{0.0}, eta{real_dist(mt)}, gamma{real_dist(mt)}, epsilon{real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_gradient = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_gradient_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_gradient = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_gradient[i] = kth_prior_gradient;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_gradient_test[i] = kth_prior_gradient;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::RMS_PROP(), theta, prior_gradient, gradient, n_elements, results, eta, gamma, epsilon);
  for(i = 0; i < n_elements; ++i)
  {   		   
	prior_gradient_test[i] = gamma * prior_gradient_test[i] * prior_gradient_test[i] + (double{1} - gamma) * gradient_test[i] * gradient_test[i];
	theta_test[i] -= eta * gradient_test[i] / sqrt(prior_gradient_test[i] * prior_gradient_test[i] + epsilon);
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_gradient[i], prior_gradient_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_gradient;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_gradient_test;
}

TEST(optimizer, rprop)
{
  std::random_device rd;
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uniform_int_distribution<std::uint32_t> uint_dist(zinhart::parallel::default_thread_pool::get_default_thread_pool().size(), std::numeric_limits<std::uint16_t>::max());
  std::mt19937 mt(rd());
  std::uint32_t n_elements{uint_dist(mt)}, i{0}, j{0};
  double * theta{nullptr}, * gradient{nullptr}, * prior_gradient{nullptr}, * prior_delta{nullptr}; 
  double * theta_test{nullptr}, * gradient_test{nullptr}, * prior_gradient_test{nullptr}, * prior_delta_test{nullptr};
  double kth_theta{0.0}, kth_gradient{0.0}, kth_prior_gradient{0.0}, kth_prior_delta{0.0}, eta_pos{real_dist(mt)}, eta_neg{real_dist(mt)}, delta_max{real_dist(mt)}, delta_min{delta_max - real_dist(mt)};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  theta = new double[n_elements];
  gradient = new double[n_elements];
  prior_gradient = new double[n_elements];
  prior_delta = new double[n_elements];
  theta_test = new double[n_elements];
  gradient_test = new double[n_elements];
  prior_gradient_test = new double[n_elements];
  prior_delta_test = new double[n_elements];
  for(i = 0; i < n_elements; ++i)
  {
	kth_theta = real_dist(mt);
	kth_gradient = real_dist(mt);
	kth_prior_gradient = real_dist(mt);
	kth_prior_delta = real_dist(mt);
	theta[i] = kth_theta;
	gradient[i] = kth_gradient;
	prior_gradient[i] = kth_prior_gradient;
	prior_delta[i] = kth_prior_delta;
	theta_test[i] = kth_theta;
	gradient_test[i] = kth_gradient;
	prior_gradient_test[i] = kth_prior_gradient;
	prior_delta_test[i] = kth_prior_delta;
  }
  zinhart::optimizers::optimizer op;
  op(zinhart::optimizers::RPROP(), theta, prior_gradient, prior_delta, gradient, n_elements, results, eta_pos, eta_neg, delta_max, delta_min);
  for(i = 0; i < n_elements; ++i)
  {   		   
	if(gradient_test[i] * prior_gradient_test[i] > 0) // if the sign of the gradient has stayed positive
	{
	  prior_delta_test[i] = ( prior_delta_test[i] * eta_pos < delta_max) ? prior_delta_test[i] * eta_pos : delta_max;
	  theta_test[i] = theta_test[i] - gradient_test[i] * prior_delta_test[i];
	  prior_gradient_test[i] = gradient_test[i]; 
	}
	else if(gradient_test[i] * prior_gradient_test[i] < 0)// if the sign of the gradient has stayed negative
	{
	  prior_delta_test[i] = ( prior_delta_test[i] * eta_neg > delta_min) ? prior_delta_test[i] * eta_neg : delta_min;
	  prior_gradient_test[i] = 0;
	} 
	else// if either the prior or current gradient is 0, because of a negative gradient
	{
	  theta_test[i] = theta_test[i] - gradient_test[i] * prior_delta_test[i];
	  prior_gradient_test[i] = gradient_test[i]; 
	}
  }
  for(i = 0; i < results.size(); ++i)
	results[i].get();
  results.clear(); 
  for(i = 0; i < n_elements; ++i)
  {
    EXPECT_DOUBLE_EQ(theta[i], theta_test[i]);
	ASSERT_DOUBLE_EQ(gradient[i], gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_gradient[i], prior_gradient_test[i]);
	ASSERT_DOUBLE_EQ(prior_delta[i], prior_delta_test[i]);
  }
  delete [] theta;
  delete [] gradient;
  delete [] prior_gradient;
  delete [] prior_delta;
  delete [] theta_test;
  delete [] gradient_test;
  delete [] prior_gradient_test;
  delete [] prior_delta_test;
}

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
