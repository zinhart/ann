#include "ann/optimizer.hh"
#include "gtest/gtest.h"
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
  op(zinhart::optimizers::OPTIMIZER_NAME::SGD, theta, gradient, n_elements, results, eta);
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
  op(zinhart::optimizers::OPTIMIZER_NAME::MOMENTUM, theta, n_elements, prior_velocity, gradient, gamma, eta, results);
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
  op(zinhart::optimizers::OPTIMIZER_NAME::NESTEROV_MOMENTUM, theta, n_elements, prior_velocity, gradient, gamma, eta, results);
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
  op(zinhart::optimizers::OPTIMIZER_NAME::ADAGRAD, theta, n_elements, prior_gradient, gradient, eta, epsilon, results);
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
  op(zinhart::optimizers::OPTIMIZER_NAME::CONJUGATE_GRADIENT, theta, prior_gradient, prior_hessian, gradient, epsilon, n_elements, results);
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

/*
TEST(optimizer, adadelta)
{
  std::random_device rd;

  std::uniform_real_distribution<float> dist(0, std::numeric_limits<float>::max());

  std::mt19937 mt(rd());
  double theta{dist(rd)}, prior_gradient{dist(rd)}, prior_gradient_copy{prior_gradient}, prior_delta{dist(rd)}, 
							current_gradient{dist(rd)}, theta_copy{theta}, gamma{0.9}, epsilon{dist(rd)};
  prior_gradient_copy = gamma * prior_gradient_copy + (1 - gamma) * current_gradient * current_gradient;
  double delta = -(sqrt(prior_delta + epsilon) / sqrt(prior_gradient_copy + epsilon)) * current_gradient;
  theta_copy += delta;
  zinhart::optimizers::optimizer<zinhart::optimizers::adadelta> op;
  op(zinhart::optimizers::OPTIMIZER_NAME::ADADELTA, theta, prior_gradient, prior_delta, current_gradient, gamma, epsilon);
  ASSERT_EQ(theta, theta_copy);
}
TEST(optimizer, rms_prop)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta{dist(rd)}, prior_gradient{dist(rd)}, prior_gradient_copy{prior_gradient}, current_gradient{dist(rd)}, theta_copy{theta}, eta{dist(rd)}, beta{0.9},  epsilon{dist(rd)};
  prior_gradient_copy = beta * prior_gradient_copy + (1 - beta) * current_gradient * current_gradient;
  theta_copy -= eta * current_gradient / sqrt(prior_gradient_copy + epsilon);
  zinhart::optimizers::optimizer<zinhart::optimizers::rms_prop> op;
  op(zinhart::optimizers::OPTIMIZER_NAME::RMS_PROP, theta, prior_gradient, current_gradient, eta, beta, epsilon);
  ASSERT_EQ(theta, theta_copy);
}

TEST(optimizer, adamax)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta {dist(mt)}, prior_mean{dist(mt)}, prior_mean_copy{prior_mean}, prior_variance{dist(mt)}, prior_variance_copy{prior_variance}, current_gradient{dist(mt)}, 
	  						  			  theta_copy{theta},  beta_1_t{0.9}, eta{0.003}, beta_1{0.9}, beta_2{0.999}, epsilon{1e-8};
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1.0) - beta_1) * current_gradient; 
  prior_variance_copy = (beta_2 * prior_variance_copy > fabs(current_gradient)) ? beta_2 * prior_variance : fabs(current_gradient);
  theta_copy -= (eta / (double(1.0) - beta_1_t) ) * (prior_mean_copy / (prior_variance_copy + epsilon));

  zinhart::optimizers::optimizer<zinhart::optimizers::adamax> op;
  op(zinhart::optimizers::OPTIMIZER_NAME::ADAMAX, theta, prior_mean, prior_variance, current_gradient, beta_1_t, eta, beta_1, beta_2, epsilon);

  ASSERT_EQ(theta, theta_copy);

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
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta{dist(mt)}, prior_mean{dist(mt)}, prior_mean_copy{prior_mean}, 
		 prior_variance{dist(mt)}, prior_variance_copy{prior_variance}, 
		 prior_bias_corrected_variance{dist(mt)}, prior_bias_corrected_variance_copy{prior_variance_copy},
		 current_gradient{dist(mt)}, theta_copy{theta}, eta{0.0001}, beta_1{0.9},
		 beta_2{0.999},  epsilon{dist(mt)};
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1) - beta_1) * current_gradient;
  prior_variance_copy = beta_2 * prior_mean_copy + (double(1) - beta_2) * current_gradient * current_gradient;
  prior_bias_corrected_variance_copy = (prior_variance_copy > prior_bias_corrected_variance_copy) ? prior_variance : prior_bias_corrected_variance_copy; 
  theta_copy -= eta * prior_mean_copy / ( sqrt(prior_bias_corrected_variance_copy) + epsilon  );

  zinhart::optimizers::optimizer<zinhart::optimizers::amsgrad> op;
  op(zinhart::optimizers::OPTIMIZER_NAME::AMSGRAD, theta, prior_mean, prior_variance, prior_bias_corrected_variance, current_gradient, eta, beta_1, beta_2, epsilon);

  ASSERT_EQ(theta, theta_copy);
}

TEST(optimizer, adam)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta{dist(mt)}, prior_mean{dist(mt)}, prior_mean_copy{prior_mean}, prior_variance{dist(mt)}, prior_variance_copy{prior_variance}, current_gradient{dist(mt)}, 
									  theta_copy{theta}, beta_1_t{0.999}, beta_2_t{0.999}, eta{0.0001}, beta_1{0.9},
									  beta_2{0.999},  epsilon{dist(mt)};
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1) - beta_1) * current_gradient;
  prior_variance_copy = beta_2 * prior_mean_copy + (double(1) - beta_2) * current_gradient * current_gradient;
  theta_copy -= eta * ( prior_mean_copy / (double(1) - beta_1_t) ) / (sqrt( prior_variance_copy / (double(1) - beta_2_t) ) + epsilon );

  zinhart::optimizers::optimizer<zinhart::optimizers::adam> op;
  op(zinhart::optimizers::OPTIMIZER_NAME::ADAM, theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon);

  ASSERT_EQ(theta, theta_copy);
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
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta{dist(mt)}, prior_mean{dist(mt)}, prior_mean_copy{prior_mean}, prior_variance{dist(mt)}, prior_variance_copy{prior_variance}, current_gradient{dist(mt)}, 
										theta_copy{theta}, eta{0.001}, gamma{0.9}, beta_1{0.9},
									   	beta_2{0.999}, beta_1_t{0.9}, beta_2_t{ 0.999 }, epsilon{1e-8};
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1) - beta_1) * current_gradient;
  prior_variance_copy = beta_2 * prior_variance_copy + (double(1) - beta_2) * current_gradient * current_gradient;
  double prior_mean_hat {prior_mean_copy / (double(1) - beta_1_t )};
  double prior_variance_hat {prior_variance_copy / (double(1) - beta_2_t)}; 
  theta -= eta / ( sqrt(prior_variance_hat) + epsilon ) * (beta_1 * prior_mean_hat + (double(1) - beta_1) / (double(1) - beta_1_t) * current_gradient  );
  zinhart::optimizers::optimizer<zinhart::optimizers::nadam> op;
  op(zinhart::optimizers::OPTIMIZER_NAME::NADAM, theta, prior_mean, prior_variance, current_gradient, eta, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon ); 
  ASSERT_EQ(theta, theta_copy);
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
}*/
