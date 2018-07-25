#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "typedefs.cuh"
#include "concurrent_routines/concurrent_routines.hh"
#include <memory>
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  namespace optimizers
  {
	// to do ADD RPROP
	enum class OPTIMIZER_NAME : std::uint32_t{SGD = std::uint32_t{0}, RPROP, MOMENTUM, NESTEROV_MOMENTUM, CONJUGATE_GRADIENT, ADAGRAD, ADADELTA, RMS_PROP, ADAMAX, AMSGRAD, ADAM, NADAM};
	class optimizer
	{
	  public:
		optimizer() = default;
		optimizer(const optimizer&) = default;
		optimizer(optimizer&&) = default;
		optimizer & operator = (const optimizer&) = default;
		optimizer & operator = (optimizer&&) = default;
		~optimizer() = default;

		// Return whether the optimizer_interface is first order, second order, etc
		std::uint32_t order();
		// This overload is for SGD
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, 
											   precision_type * theta, const precision_type * gradient, std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta = 0.9,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		// This overload is for rprop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, precision_type * prior_gradient, const precision_type * current_gradient, std::uint32_t theta_length,
											   const precision_type & eta_plus = 1.2, const precision_type & eta_neg = -0.5,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );
		/*
		 *  This overload is shared by momentum, nesterov momentum, and adagrad
		 *  for momentum and nesterove momentom free_2 = gamma, free_3 = eta
		 *  for adagrad free_2 = eta, free_3 = epsilon
		 *  */
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, precision_type * theta, precision_type & free_1, 
											   const precision_type * current_gradient, const precision_type & free_2, const precision_type & free_3
											  );
		// This overload is for conjugate gradient
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, 
											   precision_type * theta, precision_type * prior_gradient,  precision_type * hessian, 
											   const precision_type * current_gradient, const precision_type & epsilon
											  );
		// This overload is for adadelta
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, 
											   precision_type & theta, precision_type & prior_gradient, precision_type & prior_delta, 
											   const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon
											  );

		// This overload is for rms_prop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, 
											   precision_type * theta, precision_type * prior_gradient, 
											   const precision_type * current_gradient, const precision_type & eta, 
											   const precision_type & beta, const precision_type & epsilon
											  );


		// This overload is for adamax
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, 
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, 
											   const precision_type * current_gradient, const precision_type & beta_1_t, 
											   const precision_type & eta, const precision_type & beta_1, 
											   const precision_type & beta_2, const precision_type & epsilon
											  );

		// This overload is for amsgrad
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, 
											   precision_type * prior_mean, precision_type * prior_variance, precision_type * prior_bias_corrected_variance,
											   const precision_type * current_gradient, const precision_type & eta, 
											   const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
											  );
		// This overload is for adam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient, 
											   const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
											   const precision_type & beta_2, const precision_type & epsilon
											  );
		// This overload is for nadam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, precision_type * current_gradient, 
											   const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
											   const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
											  ); 
	};
	template <class OPTIMIZER>
	  class optimizer_interface : public optimizer
	  {
		public:
		  optimizer_interface() = default;
		  optimizer_interface(const optimizer_interface&) = default;
		  optimizer_interface(optimizer_interface&&) = default;
		  optimizer_interface & operator = (const optimizer_interface&) = default;
		  optimizer_interface & operator = (optimizer_interface&&) = default;
		  ~optimizer_interface() = default;
		  // Return whether the optimizer_interface is first order, second order, etc
		  std::uint32_t order();
		  // This overload is for SGD
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, const precision_type & gradient, const precision_type & eta);
/*
		  // This overload is for rprop
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
												 precision_type & theta, 
												 precision_type & prior_gradient,
												 const precision_type & current_gradient,
												 const precision_type & eta_plus,const precision_type & eta_neg
												);*/
		  /*
		   *  This overload is shared by momentum, nesterov momentum, and adagrad
		   *  for momentum and nesterove momentom free_2 = gamma, free_3 = eta
		   *  for adagrad free_2 = eta, free_3 = epsilon
		   *  */
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & free_1, 
												 const precision_type & current_gradient, const precision_type & free_2, const precision_type & free_3
												);
		  // This overload is for conjugate gradient
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_gradient,  precision_type & hessian, 
												 const precision_type & current_gradient, const precision_type & epsilon
												);
		  // This overload is for adadelta
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_gradient, precision_type & prior_delta, 
												 const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon
												);

		  // This overload is for rms_prop
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_gradient, 
												 const precision_type & current_gradient, const precision_type & eta, 
												 const precision_type & beta, const precision_type & epsilon
												);

		  // This overload is for adamax
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
												 const precision_type & current_gradient, const precision_type & beta_1_t, 
												 const precision_type & eta, const precision_type & beta_1, 
												 const precision_type & beta_2, const precision_type & epsilon
												);

		  // This overload is for amsgrad
		  template <class precision_type>
 			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, 
												 precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
												 const precision_type & current_gradient, const precision_type & eta, 
												 const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
												);
		  // This overload is for adam
		  template <class precision_type>
 			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
												 const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
												 const precision_type & beta_2, const precision_type & epsilon
												);
		  // This overload is for nadam
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, precision_type & current_gradient, 
												 const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
												 const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
				                                ); 
  /**/
	  };

	  class stochastic_gradient_descent : public optimizer_interface<stochastic_gradient_descent>
	  {
		public:
		  stochastic_gradient_descent() = default;
		  stochastic_gradient_descent(const stochastic_gradient_descent&) = default;
		  stochastic_gradient_descent(stochastic_gradient_descent&&) = default;
		  stochastic_gradient_descent & operator = (const stochastic_gradient_descent&) = default;
		  stochastic_gradient_descent & operator = (stochastic_gradient_descent&&) = default;
		  ~stochastic_gradient_descent() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER  void update(precision_type & theta, const precision_type & gradient, const precision_type & eta);
	  };

	  class momentum : public optimizer_interface<momentum>
	  {
		public:
		  momentum() = default;
		  momentum(const momentum&) = default;
		  momentum(momentum&&) = default;
		  momentum & operator = (const momentum&) = default;
		  momentum & operator = (momentum&&) = default;
		  ~momentum() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_velocity, 
											const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
										   );
	  };
	  class nesterov_momentum : public optimizer_interface<nesterov_momentum>
	  {
		public:
		  nesterov_momentum() = default;
		  nesterov_momentum(const nesterov_momentum&) = default;
		  nesterov_momentum(nesterov_momentum&&) = default;
		  nesterov_momentum & operator = (const nesterov_momentum&) = default;
		  nesterov_momentum & operator = (nesterov_momentum&&) = default;
		  ~nesterov_momentum() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_velocity, 
											const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
										   );
	  };
	  class conjugate_gradient_descent : public optimizer_interface<conjugate_gradient_descent>
	  {
		public:
		  conjugate_gradient_descent() = default;
		  conjugate_gradient_descent(const conjugate_gradient_descent&) = default;
		  conjugate_gradient_descent(conjugate_gradient_descent&&) = default;
		  conjugate_gradient_descent & operator = (const conjugate_gradient_descent&) = default;
		  conjugate_gradient_descent & operator = (conjugate_gradient_descent&&) = default;
		  ~conjugate_gradient_descent() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_gradient, precision_type & hessian, 
											const precision_type & current_gradient, const precision_type & epsilon
										   );
	  };

	  class adagrad : public optimizer_interface<adagrad>
	  {
		public:
		  adagrad() = default;
		  adagrad(const adagrad&) = default;
		  adagrad(adagrad&&) = default;
		  adagrad & operator = (const adagrad&) = default;
		  adagrad & operator = (adagrad&&) = default;
		  ~adagrad() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_gradient, 
											const precision_type & current_gradient, const precision_type & eta, const precision_type & epsilon
										   );
	  };
	  class adadelta : public optimizer_interface<adadelta>
	  {
		public:
		  adadelta() = default;
		  adadelta(const adadelta&) = default;
		  adadelta(adadelta&&) = default;
		  adadelta & operator = (const adadelta&) = default;
		  adadelta & operator = (adadelta&&) = default;
		  ~adadelta() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta,  precision_type & prior_gradient, precision_type & prior_delta,
											const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon
										   );
	  };
	  class rms_prop : public optimizer_interface<rms_prop>
	  {
		public:
		  rms_prop() = default;
		  rms_prop(const rms_prop&) = default;
		  rms_prop(rms_prop&&) = default;
		  rms_prop & operator = (const rms_prop&) = default;
		  rms_prop & operator = (rms_prop&&) = default;
		  ~rms_prop() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_gradient, 
											const precision_type & current_gradient,  const precision_type & eta, 
											const precision_type & beta, const precision_type & epsilon
										   );
	  };
	  class adamax : public optimizer_interface<adamax>
	  {
		public:
		  adamax() = default;
		  adamax(const adamax&) = default;
		  adamax(adamax&&) = default;
		  adamax & operator = (const adamax&) = default;
		  adamax & operator = (adamax&&) = default;
		  ~adamax() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
											 const precision_type & current_gradient, const precision_type & beta_1_t, const precision_type & eta, 
											 const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
										    );
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER static void moment_update(precision_type & beta_1_t, const precision_type & beta_1);
	  };

	  class amsgrad : public optimizer_interface<amsgrad>
	  {
		public:
		  amsgrad() = default;
		  amsgrad(const amsgrad&) = default;
		  amsgrad(amsgrad&&) = default;
		  amsgrad & operator = (const amsgrad&) = default;
		  amsgrad & operator = (amsgrad&&) = default;
		  ~amsgrad() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, 
											precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
											const precision_type & current_gradient, const precision_type & eta, 
											const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
			                  			   );
	  };

	  class adam : public optimizer_interface<adam>
	  {
		public:
		  adam() = default;
		  adam(const adam&) = default;
		  adam(adam&&) = default;
		  adam & operator = (const adam&) = default;
		  adam & operator = (adam&&) = default;
		  ~adam() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
											const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
											const precision_type & beta_2, const precision_type & epsilon
										   );
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER static void moment_update(precision_type & beta_1_t, precision_type & beta_2_t, const precision_type & beta_1, const precision_type & beta_2);

	  };
	  class nadam : public optimizer_interface<nadam>
	  {
		public:
		  nadam() = default;
		  nadam(const nadam&) = default;
		  nadam(nadam&&) = default;
		  nadam & operator = (const nadam&) = default;
		  nadam & operator = (nadam&&) = default;
		  ~nadam() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, precision_type & current_gradient, 
											const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
											const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
										   );
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER static void moment_update(precision_type & beta_1_t, precision_type & beta_2_t, const precision_type & beta_1, const precision_type & beta_2);
	  };
  }// END NAMESPACE OPTIMIZERS
/*  class optimizer
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
		printf("CUDA_ENABLED SGD\n");
#else
		printf("CUDA_DISABLED SGD\n");
#endif
		theta -=  ( eta * gradient );
	  }
	  CUDA_CALLABLE_MEMBER void momentum(double & theta, double & prior_velocity, const double & current_gradient, const double & gamma, const double & eta)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED MOMENTUM\n");
#else
		printf("CUDA_DISABLED MOMENTUM\n");
#endif
	    double current_velocity = gamma * prior_velocity + eta * current_gradient;
		theta -= current_velocity;
		prior_velocity = current_velocity;
	  }
	  CUDA_CALLABLE_MEMBER void nesterov_momentum(double & theta, double & prior_velocity, const double & current_gradient, const double & gamma, const double & eta)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED NESTEROV_MOMENTUM\n");
#else
		printf("CUDA_DISABLED NESTEROV_MOMENTUM\n");
#endif
		double velocity = gamma * prior_velocity + eta * current_gradient * (  theta - gamma * prior_velocity) ;
		theta -= velocity;
		prior_velocity = velocity;
	  }
	  CUDA_CALLABLE_MEMBER void conjugate_gradient(double & theta, double & prior_gradient, double & hessian, const double & current_gradient, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED CONJUGATE_GRADIENT\n");
#else
		printf("CUDA_DISABLED CONJUGATE_GRADIENT\n");
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
		printf("CUDA_ENABLED ADAGRAD\n");
#else
		printf("CUDA_DISABLED ADAGRAD\n");
#endif
		prior_gradient += current_gradient * current_gradient;
		theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
	  }
  	  CUDA_CALLABLE_MEMBER void adadelta(double & theta,  double & prior_gradient, double & prior_delta,
		  const double & current_gradient, const double & gamma, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADADELTA\n");
#else
		printf("CUDA_DISABLED ADADELTA\n");
#endif
		prior_gradient = gamma * prior_gradient + (1 - gamma) * current_gradient * current_gradient;
		double delta = -(sqrt(prior_delta + epsilon) / sqrt(prior_gradient + epsilon)) * current_gradient;
		theta += delta;
		prior_delta = gamma * prior_delta + (1 - gamma) * delta * delta;
	  }
  	  CUDA_CALLABLE_MEMBER void rms_prop(double & theta, double & prior_gradient, const double & current_gradient,  const double & eta, const double & beta, const double & epsilon)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED RMS_PROP\n");
#else
		printf("CUDA_DISABLED RMS_PROP\n");
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
		printf("CUDA_ENABLED ADAM\n");
#else
		printf("CUDA_DISABLED ADAM\n");
#endif
		prior_mean = beta_1 * prior_mean + (double(1) - beta_1_t) * current_gradient;
		prior_variance = beta_2 * prior_variance + (double(1) - beta_2_t) * current_gradient * current_gradient;
		//bias corrected mean = prior_mean/ (1 - beta_1_t), bias corrected uncentered variance = prior_variance / (1 - beta_2_t) 
		theta -= eta * ( prior_mean / (double(1) - beta_1_t) ) / (sqrt( prior_variance / (double(1) - beta_2_t) ) + epsilon ) ;
	  }
	  CUDA_CALLABLE_MEMBER void adam_moment_update(double & beta_1_t, double & beta_2_t, const double & beta_1, const double & beta_2)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAM_MOMENT_COEFF\n");
#else
		printf("CUDA_DISABLED ADAM_MOMENT_COEFF\n");
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
		printf("CUDA_ENABLED ADAMAX\n");
#else
		printf("CUDA_DISABLED ADAMAX\n");
#endif
		prior_mean = beta_1 * prior_mean + (double(1.0) - beta_1) * current_gradient; 
		prior_variance = (beta_2 * prior_variance > fabs(current_gradient)) ? beta_2 * prior_variance : fabs(current_gradient);
		theta -= (eta / (double(1.0) - beta_1_t) ) * (prior_mean / (prior_variance + epsilon)); 
	  }
	  CUDA_CALLABLE_MEMBER void adamax_moment_update(double & beta_1_t, const double & beta_1)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED ADAMAX_MOMENT_UPDATE\n");
#else
		printf("CUDA_DISABLED ADAMAX_MOMENT_UPDATE\n");
#endif
		beta_1_t *= beta_1;
	  }
	  CUDA_CALLABLE_MEMBER void nadam(
										double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
										const double & eta, const double & gamma, const double & beta_1, const double & beta_2, const double & beta_1_t, const double & beta_2_t, const double & epsilon
								    	)
	  {
#if CUDA_ENABLED == 1
		printf("CUDA_ENABLED NADAM\n");
#else
		printf("CUDA_DISABLED NADAM\n");
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
		printf("CUDA_ENABLED NADAM\n");
#else
		printf("CUDA_DISABLED NADAM\n");
#endif
		beta_1_t *= beta_1;
		beta_2_t *= beta_2;
	  }
  };*/
/*
HOST void call_sgd(optimizer & op, double & theta, const double & gradient, double eta = 0.9);
HOST void call_momentum(optimizer & op, double & theta, double & prior_velocity, const double & gradient, double gamma = 0.9, double eta = 0.1);
HOST void call_nesterov_momentum(optimizer & op, double & theta, double & prior_velocity, const double & gradient, double gamma = 0.9, double eta = 0.1);
HOST void call_conjugate_gradient(optimizer & op, double & theta, double & prior_gradient, double & hessian, const double & current_gradient, double epsilon = 1.e-30);
HOST void call_adagrad(optimizer & op, double & theta, double & prior_gradient, const double & current_gradient, double eta = 0.01, double epsilon = 1.e-8);
HOST void call_adadelta(optimizer & op, double & theta,  double & prior_gradient, double & prior_delta,
		  const double & current_gradient, double gamma = 0.95, double epsilon = 1e-6);
HOST void call_rms_prop(optimizer & op, double & theta, double & prior_gradient, const double & current_gradient, double eta = 0.0001, double beta = 0.99, double epsilon = 1.e8);
HOST void call_adam(
									  optimizer & op, double & theta, double & prior_mean, double & prior_variance, const double & current_gradient, 
									  double beta_1_t = 0.9, double beta_2_t = 0.999, double eta = 0.001, double beta_1 = 0.9,
									  double beta_2 = 0.999, double epsilon = 1e-8
									);
HOST void call_adam_moment_update(optimizer & op, double & beta_1_t, double & beta_2_t, double beta_1 = 0.9, double beta_2 = 0.999);
HOST void call_adamax(
										  optimizer & op, double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
	  						  			  double beta_1_t = 0.9, double eta = 0.002, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8
									  );
HOST void call_adamax_moment_update(optimizer & op, double & beta_1_t, double beta_1 = 0.9);
HOST void call_nadam(
										optimizer & op, double & theta, double & prior_mean, double & prior_variance, double & current_gradient, 
										double eta = 0.001, double mu = 0.9, double beta_1 = 0.9, double beta_2 = 0.999, double beta_1_t = 0.9, double beta_2_t = 0.999, double epsilon = 1e-8
								    	);
HOST void call_nadam_moment_update(optimizer & op, double & beta_1_t, double & beta_2_t, double beta_1 = 0.9, double beta_2 = 0.999);
*/
#if CUDA_ENABLED == 1
//Kernels for each optimizer will go here
//and cache prefereces as well
#endif

}
#include "ext/optimizer.tcc"
#endif

