#include "ann/optimizer.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
using namespace zinhart;
TEST(optimizer, sgd)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());	
  double theta = dist(mt), gradient = dist(mt), eta = dist(mt), theta_copy = theta;
  optimizer op;
  call_sgd(op, theta, gradient, eta);
  ASSERT_EQ(theta_copy - eta * gradient, theta);
}

TEST(optimizer, momentum)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(mt), theta_copy = theta, current_gradient = dist(mt), 
		 prior_velocity = dist(mt), gamma = dist(mt), eta = dist(mt);
  double velocity = gamma * prior_velocity + eta * current_gradient;
  theta_copy -= velocity;
  optimizer op;
  call_momentum(op, theta, prior_velocity, current_gradient, gamma, eta);
  ASSERT_EQ(theta, theta_copy);
}

TEST(optimizer, nesterov_momentum)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(mt), prior_velocity = dist(mt), current_gradient = dist(mt), theta_copy = theta, gamma = dist(mt), eta = dist(mt);
  double velocity = gamma * prior_velocity + eta * current_gradient * ( theta - gamma * prior_velocity);
  theta_copy -= velocity;
  optimizer op;
  call_nesterov_momentum(op, theta, prior_velocity, current_gradient,  gamma, eta);
  ASSERT_EQ(theta, theta_copy);
}
TEST(optimizer, conjugate_gradient)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(mt), prior_gradient = dist(mt), hessian = dist(mt), current_gradient = dist(mt), theta_copy = theta, epsilon = 1.e-30;
  double gamma = ( current_gradient - prior_gradient ) * current_gradient / ( prior_gradient * prior_gradient + epsilon );
  double step = current_gradient + ( gamma * hessian );
  theta_copy += step;
  optimizer op;
  call_conjugate_gradient(op, theta, prior_gradient, hessian, current_gradient, epsilon);
  ASSERT_EQ(theta, theta_copy);
}
TEST(optimizer, adagrad)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(rd), prior_gradient = dist(rd), prior_gradient_copy = prior_gradient, current_gradient = dist(rd), theta_copy = theta, eta = dist(rd), epsilon = dist(rd);
  prior_gradient_copy += current_gradient * current_gradient;
  theta_copy -= eta * current_gradient / sqrt(prior_gradient_copy + epsilon);
  optimizer op;
  call_adagrad(op, theta, prior_gradient, current_gradient, eta, epsilon);
  ASSERT_EQ(theta, theta_copy);
}
TEST(optimizer, adadelta)
{
  std::random_device rd;

  std::uniform_real_distribution<float> dist(0, std::numeric_limits<float>::max());

  std::mt19937 mt(rd());
  double theta = dist(rd), prior_gradient = dist(rd), prior_gradient_copy = prior_gradient, prior_delta = dist(rd), 
							current_gradient = dist(rd), theta_copy = theta, gamma = 0.9, epsilon = dist(rd);
  prior_gradient_copy = gamma * prior_gradient_copy + (1 - gamma) * current_gradient * current_gradient;
  double delta = -(sqrt(prior_delta + epsilon) / sqrt(prior_gradient_copy + epsilon)) * current_gradient;
  theta_copy += delta;
  optimizer op;
  call_adadelta(op,	theta, prior_gradient, prior_delta, current_gradient, gamma, epsilon);
  ASSERT_EQ(theta, theta_copy);

}
TEST(optimizer, rms_prop)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(rd), prior_gradient = dist(rd), prior_gradient_copy = prior_gradient, current_gradient = dist(rd), theta_copy = theta, eta = dist(rd), beta = 0.9,  epsilon = dist(rd);
  prior_gradient_copy = beta * prior_gradient_copy + (1 - beta) * current_gradient * current_gradient;
  theta_copy -= eta * current_gradient / sqrt(prior_gradient_copy + epsilon);
  optimizer op;
  call_rms_prop(op, theta, prior_gradient, current_gradient, eta, beta, epsilon );
  ASSERT_EQ(theta, theta_copy);

}
TEST(optimizer, adam)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(mt), prior_mean = dist(mt), prior_mean_copy = prior_mean, prior_variance = dist(mt), prior_variance_copy = prior_variance,  current_gradient = dist(mt), 
									  theta_copy = theta, beta_1_t = 0.999, beta_2_t = 0.999, eta = 0.0001 , beta_1 = 0.9,
									  beta_2 = 0.999,  epsilon = dist(mt);
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1) - beta_1_t) * current_gradient;
  prior_variance_copy = beta_2 * prior_mean_copy + (double(1) - beta_2_t) * current_gradient * current_gradient;
  theta_copy -= eta * ( prior_mean_copy / (double(1) - beta_1_t) ) / (sqrt( prior_variance_copy / (double(1) - beta_2_t) ) + epsilon );
  optimizer op;
  call_adam(op, theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon);
  ASSERT_EQ(theta, theta_copy);
}
TEST(optimizer, adam_moment_update)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double beta_1_t = dist(mt), beta_1_t_copy = beta_1_t, beta_2_t = dist(mt), beta_2_t_copy = beta_2_t, beta_1 = dist(mt), beta_2 = dist(mt);
  beta_1_t_copy *= beta_1;
  beta_2_t_copy *= beta_2;
  optimizer op;
  call_adam_moment_update(op, beta_1_t, beta_2_t, beta_1, beta_2);
  ASSERT_EQ(beta_1_t_copy, beta_1_t);
  ASSERT_EQ(beta_2_t_copy, beta_2_t);
}
TEST(optimizer, adamax)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(mt), prior_mean = dist(mt), prior_mean_copy = prior_mean, prior_variance = dist(mt), prior_variance_copy = prior_variance, current_gradient = dist(mt), 
	  						  			  theta_copy = theta,  beta_1_t = 0.9, eta = 0.003, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8;
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1.0) - beta_1) * current_gradient; 
  prior_variance_copy = (beta_2 * prior_variance_copy > fabs(current_gradient)) ? beta_2 * prior_variance : fabs(current_gradient);
  theta_copy -= (eta / (double(1.0) - beta_1_t) ) * (prior_mean_copy / (prior_variance_copy + epsilon));
  optimizer op;
  call_adamax(op, theta, prior_mean, prior_variance, current_gradient, beta_1_t, eta, beta_1, beta_2, epsilon);
  ASSERT_EQ(theta, theta_copy);

}
TEST(optimizer, adamax_moment_update)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double beta_1_t = dist(mt), beta_1_t_copy = beta_1_t, beta_1 = dist(mt);
  beta_1_t_copy *= beta_1;
  optimizer op;
  call_adamax_moment_update(op, beta_1_t, beta_1);
  ASSERT_EQ(beta_1_t_copy, beta_1_t);
}
TEST(optimizer, nadam)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double theta = dist(mt),  prior_mean = dist(mt), prior_mean_copy = prior_mean, prior_variance = dist(mt), prior_variance_copy = prior_variance, current_gradient = dist(mt), 
										theta_copy = theta, eta = 0.001, gamma = 0.9,  beta_1 = 0.9,
									   	beta_2 = 0.999, beta_1_t = 0.9, beta_2_t = 0.999,  epsilon = 1e-8;
  prior_mean_copy = beta_1 * prior_mean_copy + (double(1) - beta_1) * current_gradient;
  prior_variance_copy = beta_2 * prior_variance_copy + (double(1) - beta_2) * current_gradient * current_gradient;
  double prior_mean_hat = prior_mean_copy / (double(1) - beta_1_t );
  double prior_variance_hat = prior_variance_copy / (double(1) - beta_2_t); 
  theta -= eta / ( sqrt(prior_variance_hat) + epsilon ) * (beta_1 * prior_mean_hat + (double(1) - beta_1) / (double(1) - beta_1_t) * current_gradient  );
  optimizer op;
  call_nadam(op, theta, prior_mean, prior_variance, current_gradient, eta, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);
  ASSERT_EQ(theta, theta_copy);
}
TEST(optimizer, nadam_moment_update)
{
  std::random_device rd;
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::mt19937 mt(rd());
  double beta_1_t = dist(mt), beta_1_t_copy = beta_1_t, beta_2_t = dist(mt), beta_2_t_copy = beta_2_t, beta_1 = dist(mt), beta_2 = dist(mt);
  beta_1_t_copy *= beta_1;
  beta_2_t_copy *= beta_2;
  optimizer op;
  call_nadam_moment_update(op, beta_1_t, beta_2_t, beta_1, beta_2);
  ASSERT_EQ(beta_1_t_copy, beta_1_t);
  ASSERT_EQ(beta_2_t_copy, beta_2_t);
}
