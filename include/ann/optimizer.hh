#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "typedefs.cuh"
#include <memory>
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  class optimizer
  {
	public:
	  optimizer() = default;
  	  optimizer(const optimizer&) = default;
	  optimizer(optimizer&&) = default;
	  optimizer & operator = (const optimizer&) = default;
	  optimizer & operator = (optimizer&&) = default;
	  ~optimizer() = default;
	  CUDA_CALLABLE_MEMBER void sgd(double & theta, const double & gradient, const double & eta)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED SGD");
#else
		printf("CUDA_DISABLED SGD");
#endif
		theta -=  ( eta * gradient );
	  }
	  CUDA_CALLABLE_MEMBER void momentum(double & theta, double & prior_velocity, const double & current_gradient, const double & gamma, const double & eta)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED MOMENTUM ");
#else
		printf("CUDA_DISABLED MOMENTUM");
#endif
	    double current_velocity = gamma * prior_velocity + eta * current_gradient;
		theta -= current_velocity;
		prior_velocity = current_velocity;
	  }
	  CUDA_CALLABLE_MEMBER void nesterov_momentum(double & theta, double & prior_velocity, const double & current_gradient, const double & gamma, const double & eta)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED NESTEROV_MOMENTUM ");
#else
		printf("CUDA_DISABLED NESTEROV_MOMENTUM");
#endif
		double velocity = gamma * prior_velocity + eta * current_gradient * (  theta - gamma * prior_velocity) ;
		theta -= velocity;
		prior_velocity = velocity;
	  }
	  CUDA_CALLABLE_MEMBER void conjugate_gradient(double & theta, double & prior_gradient, double & hessian, const double & current_gradient, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED CONJUGATE_GRADIENT");
#else
		printf("CUDA_DISABLED CONJUGATE_GRADIENT");
#endif
		double gamma = ( current_gradient - prior_gradient ) * current_gradient / ( prior_gradient * prior_gradient + epsilon );
		double step = current_gradient + ( gamma * hessian );
		theta += step;
		prior_gradient = current_gradient;
		hessian = step;
	  }
	  CUDA_CALLABLE_MEMBER void adagrad(double & theta, double & prior_gradient, const double & current_gradient,  const double & eta, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAGRAD");
#else
		printf("CUDA_DISABLED ADAGRAD");
#endif
		prior_gradient += current_gradient * current_gradient;
		theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
	  }
  	  CUDA_CALLABLE_MEMBER void adadelta(double & theta,  double & prior_gradient, double & prior_delta,
		  const double & current_gradient, const double & gamma, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADADELTA");
#else
		printf("CUDA_DISABLED ADADELTA");
#endif
		prior_gradient = gamma * prior_gradient + (1 - gamma) * current_gradient * current_gradient;
		double delta = -(sqrt(prior_delta + epsilon) / sqrt(prior_gradient + epsilon)) * current_gradient;
		theta += delta;
		prior_delta = gamma * prior_delta + (1 - gamma) * delta * delta;
	  }
  	  CUDA_CALLABLE_MEMBER void rms_prop(double & theta, double & prior_gradient, const double & current_gradient,  const double & eta, const double & beta, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED RMS_PROP");
#else
		printf("CUDA_DISABLED RMS_PROP");
#endif
		prior_gradient = beta * prior_gradient + (1 - beta) * current_gradient * current_gradient;
		theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
	  }
	  CUDA_CALLABLE_MEMBER void adam(
									  double & theta, double & prior_mean, double & prior_variance, const double & current_gradient, 
									  const double & beta_1_t, const double & beta_2_t, const double & eta, const double & beta_1,
									  const double & beta_2, const double & epsilon
									)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAM");
#else
		printf("CUDA_DISABLED ADAM");
#endif
		prior_mean = beta_1 * prior_mean + (double(1) - beta_1_t) * current_gradient;
		prior_variance = beta_2 * prior_variance + (double(1) - beta_2_t) * current_gradient * current_gradient;
		//bias corrected mean = prior_mean/ (1 - beta_1_t), bias corrected uncentered variance = prior_variance / (1 - beta_2_t) 
		theta -= eta * ( prior_mean / (double(1) - beta_1_t) ) / (sqrt( prior_variance / (double(1) - beta_2_t) ) + epsilon ) ;
	  }
	  CUDA_CALLABLE_MEMBER void adam_moment_update(double & beta_1_t, double & beta_2_t, const double & beta_1, const double & beta_2)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAM_MOMENT_COEFF");
#else
		printf("CUDA_DISABLED ADAM_MOMENT_COEFF");
#endif
		beta_1_t *= beta_1;
		beta_2_t *= beta_2;
	  }
	  CUDA_CALLABLE_MEMBER void adamax(
										  double & theta, double & prior_mean, double & prior_variance, const double & current_gradient, 
	  						  			  const double & beta_1_t, const double & eta, const double & beta_1, const double & beta_2, const double & epsilon 
									  )
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAMAX");
#else
		printf("CUDA_DISABLED ADAMAX");
#endif
		prior_mean = beta_1 * prior_mean + (double(1.0) - beta_1) * current_gradient; 
		prior_variance = (beta_2 * prior_variance > fabs(current_gradient)) ? beta_2 * prior_variance : fabs(current_gradient);
		theta -= (eta / (double(1.0) - beta_1_t) ) * (prior_mean / (prior_variance + epsilon)); 
	  }
	  CUDA_CALLABLE_MEMBER void adamax_moment_update(double & beta_1_t, const double & beta_1)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAMAX_MOMENT_UPDATE");
#else
		printf("CUDA_DISABLED ADAMAX_MOMENT_UPDATE");
#endif
		beta_1_t *= beta_1;
	  }
	  CUDA_CALLABLE_MEMBER void nadam(
										double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
										const double & eta, const double & gamma, const double & beta_1, const double & beta_2, const double & beta_1_t, const double & beta_2_t, const double & epsilon
								    	)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED NADAM");
#else
		printf("CUDA_DISABLED NADAM");
#endif
		prior_mean = beta_1 * prior_mean + (double(1) - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (double(1) - beta_2) * current_gradient * current_gradient;
		double prior_mean_hat = prior_mean / (double(1) - beta_1_t );
		double prior_variance_hat = prior_variance / (double(1) - beta_2_t); 
		theta -= eta / ( sqrt(prior_variance_hat) + epsilon ) * (beta_1 * prior_mean_hat + (double(1) - beta_1) / (double(1) - beta_1_t) * current_gradient  );
	  }
	  CUDA_CALLABLE_MEMBER void nadam_moment_update(double & beta_1_t, double & beta_2_t, const double & beta_1, const double & beta_2)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED NADAM");
#else
		printf("CUDA_DISABLED NADAM");
#endif
		beta_1_t *= beta_1;
		beta_2_t *= beta_2;
	  }
  };

void call_sgd(optimizer & op, double & theta, const double & gradient, double eta = 0.9);
void call_momentum(optimizer & op, double & theta, double & prior_velocity, const double & gradient, double gamma = 0.9, double eta = 0.1);
void call_nesterov_momentum(optimizer & op, double & theta, double & prior_velocity, const double & gradient, double gamma = 0.9, double eta = 0.1);
void call_conjugate_gradient(optimizer & op, double & theta, double & prior_gradient, double & hessian, const double & current_gradient, double epsilon = 1.e-30);
void call_adagrad(optimizer & op, double & theta, double & prior_gradient, const double & current_gradient, double eta = 0.01, double epsilon = 1.e-8);
void call_adadelta(optimizer & op, double & theta,  double & prior_gradient, double & prior_delta,
		  const double & current_gradient, double gamma = 0.95, double epsilon = 1e-6);
void call_rms_prop(optimizer & op, double & theta, double & prior_gradient, const double & current_gradient, double eta = 0.0001, double beta = 0.99, double epsilon = 1.e8);
void call_adam(
									  optimizer & op, double & theta, double & prior_mean, double & prior_variance, const double & current_gradient, 
									  double beta_1_t = 0.9, double beta_2_t = 0.999, double eta = 0.001, double beta_1 = 0.9,
									  double beta_2 = 0.999, double epsilon = 1e-8
									);
void call_adam_moment_update(optimizer & op, double & beta_1_t, double & beta_2_t, double beta_1 = 0.9, double beta_2 = 0.999);
void call_adamax(
										  optimizer & op, double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
	  						  			  double beta_1_t = 0.9, double eta = 0.002, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8
									  );
void call_adamax_moment_update(optimizer & op, double & beta_1_t, double beta_1 = 0.9);
void call_nadam(
										optimizer & op, double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
										double eta = 0.001, double mu = 0.9, double beta_1 = 0.9, double beta_2 = 0.999, double beta_1_t = 0.9, double beta_2_t = 0.999, double epsilon = 1e-8
								    	);
void call_nadam_moment_update(optimizer & op, double & beta_1_t, double & beta_2_t, double beta_1 = 0.9, double beta_2 = 0.999);


}
#endif

