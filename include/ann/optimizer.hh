#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <multi_core/multi_core.hh>
#include <type_traits>
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  namespace optimizers
  {
	namespace optimizer_attributes
	{
	  enum sgd_optimizer                         : std::uint32_t;
	  enum momentum_optimizer                    : std::uint32_t;
	  enum nesterov_momentum_optimizer           : std::uint32_t;
	  enum adagrad_optimizer                     : std::uint32_t;
	  enum conjugate_gradient_optimizer          : std::uint32_t;
	  enum adadelta_optimizer                    : std::uint32_t;
	  enum rms_prop_optimizer                    : std::uint32_t;
	  enum rprop_optimizer                       : std::uint32_t;
	  enum adamax_optimizer                      : std::uint32_t;
	  enum amsgrad_optimizer                     : std::uint32_t;
	  enum adam_optimizer                        : std::uint32_t;
	  enum nadam_optimizer                       : std::uint32_t;
	  union optimum_type
	  {
		sgd_optimizer sgd;
		momentum_optimizer momentum;
		nesterov_momentum_optimizer nesterov_momentum;
		adagrad_optimizer adagrad;
		conjugate_gradient_optimizer conjugrad;
		adadelta_optimizer adadelta;
		rms_prop_optimizer rms_prop;
		rprop_optimizer rprop;
		adamax_optimizer adamax;
		amsgrad_optimizer amsgrad;
		adam_optimizer adam;
		nadam_optimizer nadam;
	  };
	}

	template <class precision_type>
	  class optimum
	  {
		public:
		  optimum() = default;
		  optimum(const optimum&) = default;
		  optimum(optimum &&) = default;
		  optimum & operator = (const optimum&) = default;
		  optimum & operator = (optimum &&) = default;
		  ~optimum() = default;
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::sgd_optimizer sgd, precision_type & weight, const precision_type & gradient, const precision_type & eta);
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::momentum_optimizer momentum, precision_type & weight, precision_type & prior_velocity, const precision_type & current_gradient, const precision_type & eta, const precision_type & gamma);
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::nesterov_momentum_optimizer nesterov, precision_type & weight, precision_type & prior_velocity, 
											  const precision_type & current_gradient, const precision_type & eta, const precision_type & gamma
											 );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adagrad_optimizer adagrad, precision_type & weight, precision_type & prior_gradient, 
										   const precision_type & current_gradient, const precision_type & eta, const precision_type & epsilon
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::conjugate_gradient_optimizer conjugad, precision_type & weight, precision_type & prior_gradient, precision_type & hessian, 
											  const precision_type & current_gradient, const precision_type & epsilon
											 );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adadelta_optimizer adadelta, precision_type & weight, precision_type & prior_gradient, precision_type & prior_delta, 
												   const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon);
		  
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::rms_prop_optimizer rms_prop, precision_type & weight, precision_type & prior_gradient, 
											  const precision_type & current_gradient,  const precision_type & eta, 
											  const precision_type & beta, const precision_type & epsilon
											 );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::rprop_optimizer rprop, precision_type & weight, precision_type & prior_gradient, precision_type & current_delta,
											  const precision_type & current_gradient, const precision_type & eta_pos, const precision_type & eta_neg,
											  const precision_type & delta_max = 50, const precision_type & delta_min = 10.e-6
											 );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adamax_optimizer adamax ,precision_type & weight, precision_type & prior_mean, precision_type & prior_variance, 
											   const precision_type & current_gradient, const precision_type & beta_1_t, const precision_type & eta, 
											   const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
											  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::amsgrad_optimizer amsgrad, precision_type & weight, 
											  precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
											  const precision_type & current_gradient, const precision_type & eta, 
											  const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
											 );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adam_optimizer adam, precision_type & weight, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
											  const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
											  const precision_type & beta_2, const precision_type & epsilon
											 );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::nadam_optimizer nadam, precision_type & weight, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
											  const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
											  const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
											 );
	  
	  };

	template <class precision_type>
	  class optimizer
	  {
		protected:
		  optimum<precision_type> opt;
		  std::uint32_t size;
		  virtual void safe_deallocate(){};
		public:
  		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0) = 0;
		  HOST virtual void set_size(const std::uint32_t & size) = 0;
		  HOST virtual std::uint32_t get_size()const = 0;
		  HOST virtual ~optimizer()
		  {safe_deallocate();};
	  };

	template <class precision_type>
  	  class sgd : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type learning_rate;
		public:
		  HOST sgd(precision_type learning_rate = 0.9);
		  HOST sgd(const sgd&) = delete;
		  HOST sgd(sgd &&) = delete;
		  HOST sgd & operator = (const sgd&) = delete;
		  HOST sgd & operator = (sgd &&) = delete;
		  HOST ~sgd() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };


	template <class precision_type>
  	  class momentum : public optimizer<precision_type>
	  {

		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type learning_rate;
		  precision_type momentum_term;
		  precision_type * velocity{nullptr};
		  HOST void safe_deallocate()override;
		public:
		  HOST momentum(std::uint32_t size, precision_type learning_rate = 0.9, precision_type momentum_term = 0.1);
		  HOST momentum(const momentum & m) = delete;
		  HOST momentum(momentum && m) = delete;
		  HOST momentum & operator = (const momentum & m) = delete;
		  HOST momentum & operator = (momentum && m) = delete;
		  HOST ~momentum();
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class nesterov_momentum : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type learning_rate;
		  precision_type momentum_term;
		  precision_type * velocity{nullptr};
		  HOST void safe_deallocate()override;
		public:
		  HOST nesterov_momentum(std::uint32_t size, precision_type learning_rate = 0.9, precision_type momentum_term = 0.1);
		  HOST nesterov_momentum(const nesterov_momentum & nm) = delete;
		  HOST nesterov_momentum(nesterov_momentum && nm) = delete;
		  HOST nesterov_momentum & operator = (const nesterov_momentum & nm) = delete;
		  HOST nesterov_momentum & operator = (nesterov_momentum && nm) = delete;
		  HOST ~nesterov_momentum();
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
 		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class adagrad : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type learning_rate;
		  precision_type epsilon;
		  precision_type * prior_gradient{nullptr};
		  HOST void safe_deallocate()override;
		public:
		  HOST adagrad(std::uint32_t size, precision_type learning_rate = 0.01, precision_type epsilon = 1.e-8);
		  HOST adagrad(const adagrad & a) = delete;
		  HOST adagrad(adagrad && a) = delete;
		  HOST adagrad & operator = (const adagrad & a) = delete;
		  HOST adagrad & operator = (adagrad && a) = delete;
		  HOST ~adagrad();
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
 		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };


	template <class precision_type>
  	  class conjugate_gradient : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * prior_gradient{nullptr};
		  precision_type * hessian{nullptr};
		  precision_type epsilon;
		  HOST void safe_deallocate()override;
		public:
		  HOST conjugate_gradient(std::uint32_t size, precision_type epsilon = 1.e-30);
		  conjugate_gradient(const conjugate_gradient&) = delete;
		  conjugate_gradient(conjugate_gradient &&) = delete;
		  conjugate_gradient & operator = (const conjugate_gradient&) = delete;
		  conjugate_gradient & operator = (conjugate_gradient &&) = delete;
		  HOST ~conjugate_gradient();
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class adadelta : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type gamma;
		  precision_type epsilon;
		  precision_type * prior_gradient{nullptr};
		  precision_type * prior_delta{nullptr};
		  HOST void safe_deallocate()override;
		public:
		  HOST adadelta(std::uint32_t size, precision_type gamma = 0.99, precision_type epsilon = 1.e-8 );
		  adadelta(const adadelta&) = delete;
		  adadelta(adadelta &&) = delete;
		  adadelta & operator = (const adadelta&) = delete;
		  adadelta & operator = (adadelta &&) = delete;
		  HOST ~adadelta();
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class rms_prop : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		public:
		  rms_prop() = default;
		  rms_prop(const rms_prop&) = default;
		  rms_prop(rms_prop &&) = default;
		  rms_prop & operator = (const rms_prop&) = default;
		  rms_prop & operator = (rms_prop &&) = default;
		  ~rms_prop() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class rprop : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		public:
		  rprop() = default;
		  rprop(const rprop&) = default;
		  rprop(rprop &&) = default;
		  rprop & operator = (const rprop&) = default;
		  rprop & operator = (rprop &&) = default;
		  ~rprop() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class adamax : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		public:
		  adamax() = default;
		  adamax(const adamax&) = default;
		  adamax(adamax &&) = default;
		  adamax & operator = (const adamax&) = default;
		  adamax & operator = (adamax &&) = default;
		  ~adamax() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };


	template <class precision_type>
  	  class amsgrad : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		public:
		  amsgrad() = default;
		  amsgrad(const amsgrad&) = default;
		  amsgrad(amsgrad &&) = default;
		  amsgrad & operator = (const amsgrad&) = default;
		  amsgrad & operator = (amsgrad &&) = default;
		  ~amsgrad() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };

	template <class precision_type>
  	  class adam : public optimizer<precision_type>
	  {
		
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		public:
		  adam() = default;
		  adam(const adam&) = default;
		  adam(adam &&) = default;
		  adam & operator = (const adam&) = default;
		  adam & operator = (adam &&) = default;
		  ~adam() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };


	template <class precision_type>
  	  class nadam : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		public:
		  nadam() = default;
		  nadam(const nadam&) = default;
		  nadam(nadam &&) = default;
		  nadam & operator = (const nadam&) = default;
		  nadam & operator = (nadam &&) = default;
		  ~nadam() = default;
		  HOST virtual void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
		  HOST virtual void set_size(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size()const override;
	  };
#if CUDA_ENABLED == 1
	void optimize(optimizer<double> * const o, double * weights, const double * const gradient, const std::uint32_t & length);
#else
	void optimize(optimizer<double> * const o, double * weights, const double * const gradient, const std::uint32_t length, const std::uint32_t n_threads = 1, const std::uint32_t thread_id = 0);
	void optimize_m(const std::shared_ptr<optimizer<double>> & o, double * weights, const double * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads = 1, const std::uint32_t & thread_id = 0);
#endif

  }// END NAMESPACE OPTIMIZERS
 }// END NAMESPAC ZINHART
#include <ann/ext/optimizer.tcc>
#endif
   /*
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

		// This overload is for adamax
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(ADAMAX &&, 
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, 
											   const precision_type * current_gradient, const precision_type & beta_1_t, 
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta = 0.002 , const precision_type & beta_1 = 0.9, 
											   const precision_type & beta_2 = 0.999, const precision_type & epsilon = 1.e-8,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		// This overload is for amsgrad
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(AMSGRAD &&,
											   precision_type * theta, 
											   precision_type * prior_mean, precision_type * prior_variance, precision_type * prior_bias_corrected_variance,
											   const precision_type * current_gradient,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta= 0.001, const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, const precision_type & epsilon = 1.e-8, 
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );
	    
		// This overload is for adam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(ADAM &&,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient, 
											   const precision_type & beta_1_t, const precision_type & beta_2_t, 
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta= 0.001, const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, const precision_type & epsilon = 1.e-8, 
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											  );

		// This overload is for nadam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void operator()(NADAM &&,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient, 
											   const precision_type & beta_1_t, const precision_type & beta_2_t,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta = 0.001, const precision_type & gamma = 0.9, const precision_type & beta_1 = 0.9, 
											   const precision_type & beta_2 = 0.9, const precision_type & epsilon= 1.e-8,
											   zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
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
			CUDA_CALLABLE_MEMBER void operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
												 const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
												 const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
				                                ); 
  
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
		   CUDA_CALLABLE_MEMBER void update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
											const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
											const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
										   );
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER static void moment_update(precision_type & beta_1_t, precision_type & beta_2_t, const precision_type & beta_1, const precision_type & beta_2);
	  };
	  */
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

#if CUDA_ENABLED == 1
//Kernels for each optimizer will go here
//and cache prefereces as well
#endif
*/

