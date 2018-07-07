#include "ann/optimizer.hh"
namespace zinhart
{  
  HOST void call_sgd(optimizer & op, double & theta, const double & gradient, double eta)
  { op.sgd(theta, gradient, eta); }
  HOST void call_momentum(optimizer & op, double & theta, double & prior_velocity,
	  const double & gradient,  double gamma, double eta)
  { op.momentum(theta, prior_velocity, gradient, gamma, eta); }
  HOST void call_nesterov_momentum(optimizer & op, double & theta, double & prior_velocity, const double & gradient, double gamma, double eta)
  { op.nesterov_momentum(theta, prior_velocity, gradient, gamma, eta); }
  HOST void call_conjugate_gradient(optimizer & op, double & theta, double & prior_gradient, double & hessian, const double & current_gradient, double epsilon)
  { op.conjugate_gradient(theta, prior_gradient, hessian , current_gradient, epsilon); }
  HOST void call_adagrad(optimizer & op, double & theta, double & prior_gradient, const double & current_gradient, double eta, double epsilon)
  { op.adagrad(theta, prior_gradient, current_gradient, eta, epsilon); }
  HOST void call_adadelta(optimizer & op, double & theta,  double & prior_gradient, double & prior_delta,
		  const double & current_gradient, double gamma, double epsilon)
  {  op.adadelta(theta, prior_gradient, prior_delta, current_gradient, gamma, epsilon);  }
  HOST void call_rms_prop(optimizer & op, double & theta, double & prior_gradient, const double & current_gradient, double eta, double beta, double epsilon)
  {  op.rms_prop(theta, prior_gradient, current_gradient, eta, beta, epsilon); }
  HOST void call_adam(
									  optimizer & op, double & theta, double & prior_mean, double & prior_variance, const double & current_gradient, 
									  double beta_1_t, double beta_2_t, double eta, double beta_1,
									  double beta_2, double epsilon
									)
  {  op.adam(theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon); }
  HOST void call_adam_moment_update(optimizer & op, double & beta_1_t, double & beta_2_t, double beta_1, double beta_2)
  { op.adam_moment_update(beta_1_t, beta_2_t, beta_1, beta_2); }
  HOST void call_adamax(
										  optimizer & op, double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
	  						  			  double  beta_1_t, double eta, double beta_1, double beta_2, double epsilon
									  )
  {op.adamax(theta, prior_mean, prior_variance, current_gradient, beta_1_t, eta, beta_1, beta_2, epsilon); }
  HOST void call_adamax_moment_update(optimizer & op, double & beta_1_t,  double beta_1)
  { op.adamax_moment_update(beta_1_t, beta_1); }
  HOST void call_nadam(
										optimizer & op, double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
										double eta, double mu,  double beta_1, double beta_2, double beta_1_t, double beta_2_t, double epsilon 
								    	)
  { op.nadam(theta, prior_mean, prior_variance, current_gradient, eta, mu, beta_1, beta_2, beta_1_t, beta_2_t, epsilon); }
  HOST void call_nadam_moment_update(optimizer & op, double & beta_1_t, double & beta_2_t, double beta_1, double beta_2)
  { op.nadam_moment_update(beta_1_t, beta_2_t, beta_1, beta_2);}
}
