#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "typedefs.cuh"
#include "concurrent_routines/concurrent_routines.hh"
#include <type_traits>
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
	// all the optimizers
	typedef std::integral_constant<std::uint32_t, 0> SGD;
	typedef std::integral_constant<std::uint32_t, 1> MOMENTUM;
	typedef std::integral_constant<std::uint32_t, 2> NESTEROV_MOMENTUM;
	typedef std::integral_constant<std::uint32_t, 3> ADAGRAD;
	typedef std::integral_constant<std::uint32_t, 4> CONJUGATE_GRADIENT;
	typedef std::integral_constant<std::uint32_t, 5> ADADELTA;
	typedef std::integral_constant<std::uint32_t, 6> RMS_PROP;
	typedef std::integral_constant<std::uint32_t, 7> RPROP;
	typedef std::integral_constant<std::uint32_t, 8> ADAMAX;
	typedef std::integral_constant<std::uint32_t, 9> AMSGRAD;
	typedef std::integral_constant<std::uint32_t, 10> ADAM;
	typedef std::integral_constant<std::uint32_t, 11> NADAM;

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
		  CUDA_CALLABLE_MEMBER void operator()(SGD &&,
											   precision_type * theta, const precision_type * gradient, std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta = 0.9,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		/*
		 *  This overload is shared by momentum, nesterov momentum, and adagrad
		 *  for momentum and nesterov-momentum free_1 = prior_velocity, free_2 = gamma, free_3 = eta
		 *  for adagrad free_1 = prior_gradient free_2 = eta, free_3 = epsilon
		 *  */
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(MOMENTUM &&, 
											   precision_type * theta, std::uint32_t theta_length, precision_type * prior_velocity, 
											   const precision_type * current_gradient, const precision_type & eta, const precision_type & epsilon,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(NESTEROV_MOMENTUM &&, 
											   precision_type * theta, std::uint32_t theta_length, precision_type * prior_velocity, 
											   const precision_type * current_gradient, const precision_type & eta, const precision_type & epsilon,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(ADAGRAD &&, 
											   precision_type * theta, std::uint32_t theta_length, precision_type * prior_gradient, 
											   const precision_type * current_gradient, const precision_type & eta, const precision_type & epsilon,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );
		// This overload is for conjugate gradient
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(CONJUGATE_GRADIENT&&, 
											   precision_type * theta, precision_type * prior_gradient,  precision_type * hessian, 
											   const precision_type * current_gradient, const precision_type & epsilon,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		// This overload is for adadelta
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(ADADELTA&&, 
											   precision_type * theta, precision_type * prior_gradient, precision_type * prior_delta, 
											   const precision_type * current_gradient, std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type gamma = 0.99, const precision_type epsilon= 1.e-8,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );
							

		// This overload is for rms_prop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(RMS_PROP&&, 
											   precision_type * theta, precision_type * prior_gradient, 
											   const precision_type * current_gradient, std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta = 0.001, const precision_type & gamma = 0.9, const precision_type & epsilon = 1.e-8,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		//This overload is for rprop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(RPROP &&,
											   precision_type * theta, precision_type * prior_gradient, precision_type * current_delta, const precision_type * current_gradient, 
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta_pos = 1.2, const precision_type & eta_neg = 0.5, const precision_type & delta_max = 50, const precision_type & delta_min = 1.e-6,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );
/*
		// This overload is for adamax
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name, 
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, 
											   const precision_type * current_gradient, const precision_type & beta_1_t, 
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool(),
											   const precision_type & eta = 0.002 , const precision_type & beta_1 = 0.9, 
											   const precision_type & beta_2 = 0.999, const precision_type & epsilon = 1.e-8
											  );

		// This overload is for amsgrad
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, 
											   precision_type * prior_mean, precision_type * prior_variance, precision_type * prior_bias_corrected_variance,
											   const precision_type * current_gradient,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool(),
											   const precision_type & eta= 0.001, const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, const precision_type & epsilon = 1.e-8 
											  );
		// This overload is for adam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient, 
											   const precision_type & beta_1_t, const precision_type & beta_2_t, 
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool(),
											   const precision_type & eta= 0.001, const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, const precision_type & epsilon = 1.e-8 
											  );
		// This overload is for nadam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(OPTIMIZER_NAME name,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, precision_type * current_gradient, 
											   const precision_type & beta_1_t, const precision_type & beta_2_t,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool(),
											   const precision_type & eta = 0.001, const precision_type & gamma = 0.9, const precision_type & beta_1 = 0.9, 
											   const precision_type & beta_2 = 0.9, const precision_type & epsilon= 1.e-8
											  ); 
		*/
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
		   *  This overload is shared by momentum, nesterov momentum, and adagrad 
		   *  for momentum and nesterov momentom free_2 = gamma, free_3 = eta
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

		//This overload is for rprop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_gradient, 
			                                   precision_type & current_delta, const precision_type & current_gradient, 
											   const precision_type & eta_pos, const precision_type & eta_neg, 
											   const precision_type & delta_max, const precision_type & delta_min
											  );

		  // This overload is for adamax
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
												 const precision_type & current_gradient, const precision_type & beta_1_t, 
												 const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
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

	  class resilient_propagation : public optimizer_interface<resilient_propagation>
	  {
		public:
		  resilient_propagation() = default;
		  resilient_propagation(const resilient_propagation&) = default;
		  resilient_propagation(resilient_propagation&&) = default;
		  resilient_propagation & operator = (const resilient_propagation&) = default;
		  resilient_propagation & operator = (resilient_propagation&&) = default;
		  ~resilient_propagation() = default;
		  CUDA_CALLABLE_MEMBER std::uint32_t get_order();// memory order is 2
		  template <class precision_type>
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_gradient, precision_type & current_delta,
											const precision_type & current_gradient, const precision_type & eta_pos, const precision_type & eta_neg,
											const precision_type & delta_max = 50, const precision_type & delta_min = 10.e-6
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

