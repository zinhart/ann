#ifndef OPTIMIZER_H
#define OPTIMIZER_H
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
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::sgd_optimizer sgd, 
			                               precision_type & weight, const precision_type & gradient, 
										   const precision_type & learning_rate = 0.01
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::momentum_optimizer momentum, 
			                               precision_type & weight, precision_type & prior_velocity, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.01, const precision_type & gamma = 0.9
										  ); 
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::nesterov_momentum_optimizer nesterov, 
			                               precision_type & weight, precision_type & prior_velocity, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.01, const precision_type & gamma = 0.9
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adagrad_optimizer adagrad, 
			                               precision_type & weight, precision_type & prior_gradient, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.01, const precision_type & epsilon = 1.e-8
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::conjugate_gradient_optimizer conjugad, 
			                               precision_type & weight, precision_type & prior_gradient, precision_type & hessian, 
										   const precision_type & current_gradient, const precision_type & epsilon = 1.e-30
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adadelta_optimizer adadelta, 
			                               precision_type & weight, precision_type & prior_gradient, precision_type & prior_delta, 
										   const precision_type & current_gradient, const precision_type & gamma = 0.99, const precision_type & epsilon = 1.e-6
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::rms_prop_optimizer rms_prop, 
			                               precision_type & weight, precision_type & prior_gradient, const precision_type & current_gradient,  
										   const precision_type & learning_rate = 0.001, const precision_type & gamma = 0.90, const precision_type & epsilon = 1.e-8
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::rprop_optimizer rprop, 
			                               precision_type & weight, precision_type & prior_gradient, precision_type & delta, const precision_type & current_gradient, 
										   const precision_type & learning_rate_pos = 1.2, const precision_type & learning_rate_neg = 0.5,
										   const precision_type & delta_max = 50, const precision_type & delta_min = 1.e-6
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adamax_optimizer adamax, 
			                               precision_type & weight, precision_type & mean, precision_type & variance, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.002, 
										   const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, const precision_type & beta_1_t = 0.9, 
										   const precision_type & epsilon = 1.e-8
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::amsgrad_optimizer amsgrad, 
										   precision_type & weight, precision_type & mean, precision_type & variance, precision_type & bias_corrected_variance, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.01, 
										   const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.99, 
										   const precision_type & epsilo = 1.e-8
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::adam_optimizer adam, 
			                               precision_type & weight, precision_type & mean, precision_type & variance, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.001, 
										   const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, 
										   const precision_type & beta_1_t = 0.9, const precision_type & beta_2_t = 0.999, 
										   const precision_type & epsilon = 1.e-8
										  );
		  CUDA_CALLABLE_MEMBER void update(optimizer_attributes::nadam_optimizer nadam, 
										   precision_type & weight, precision_type & mean, precision_type & variance, const precision_type & current_gradient, 
										   const precision_type & learning_rate = 0.001, const precision_type & gamma = 0.9, 
										   const precision_type & beta_1 = 0.9, const precision_type & beta_2 = 0.999, 
										   const precision_type & beta_1_t = 0.9, const precision_type & beta_2_t = 0.999, 
										   const precision_type & epsilon = 1.e-8
										  );
	  
	  };

	// The template pattern + thread_safety
	template <class precision_type>
	  class optimizer
	  {
		protected:
		  optimum<precision_type> opt;
		  std::uint32_t size;
  		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) = 0;
		  HOST virtual void set_size_impl(const std::uint32_t & size) = 0;
		  HOST virtual std::uint32_t get_size_impl()const = 0;
		  HOST virtual void update_bias_correction_impl(){};
		  HOST virtual precision_type get_bias_corrected_first_moment_impl()const{ return 0; };
		  HOST virtual precision_type get_bias_corrected_second_moment_impl()const{ return 0; };
		  HOST virtual void safe_deallocate_impl(){};
		public:
  		  HOST void update(precision_type * weights, const precision_type * const gradient, const std::uint32_t length, const std::uint32_t n_threads = 1, const std::uint32_t thread_id = 0);
		  HOST void set_size(const std::uint32_t size);
		  HOST std::uint32_t get_size()const;
		  HOST void update_bias_correction();
		  HOST precision_type get_bias_corrected_first_moment()const;
		  HOST precision_type get_bias_corrected_second_moment()const;
		  HOST void safe_deallocate();
		  HOST virtual ~optimizer();
	  };

	template <class precision_type>
  	  class sgd : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type learning_rate;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		public:
		  HOST sgd(precision_type learning_rate = 0.01);
		  HOST sgd(const sgd&) = delete;
		  HOST sgd(sgd &&) = delete;
		  HOST sgd & operator = (const sgd&) = delete;
		  HOST sgd & operator = (sgd &&) = delete;
		  HOST ~sgd() = default;
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
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST momentum(std::uint32_t size, precision_type learning_rate = 0.01, precision_type momentum_term = 0.9);
		  HOST momentum(const momentum & m) = delete;
		  HOST momentum(momentum && m) = delete;
		  HOST momentum & operator = (const momentum & m) = delete;
		  HOST momentum & operator = (momentum && m) = delete;
		  HOST ~momentum();
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
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
 		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST nesterov_momentum(std::uint32_t size, precision_type learning_rate = 0.01, precision_type momentum_term = 0.9);
		  HOST nesterov_momentum(const nesterov_momentum & nm) = delete;
		  HOST nesterov_momentum(nesterov_momentum && nm) = delete;
		  HOST nesterov_momentum & operator = (const nesterov_momentum & nm) = delete;
		  HOST nesterov_momentum & operator = (nesterov_momentum && nm) = delete;
		  HOST ~nesterov_momentum();
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
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
 		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST adagrad(std::uint32_t size, precision_type learning_rate = 0.01, precision_type epsilon = 1.e-8);
		  HOST adagrad(const adagrad & a) = delete;
		  HOST adagrad(adagrad && a) = delete;
		  HOST adagrad & operator = (const adagrad & a) = delete;
		  HOST adagrad & operator = (adagrad && a) = delete;
		  HOST ~adagrad();
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
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST conjugate_gradient(std::uint32_t size, precision_type epsilon = 1.e-30);
		  conjugate_gradient(const conjugate_gradient&) = delete;
		  conjugate_gradient(conjugate_gradient &&) = delete;
		  conjugate_gradient & operator = (const conjugate_gradient&) = delete;
		  conjugate_gradient & operator = (conjugate_gradient &&) = delete;
		  HOST ~conjugate_gradient();
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
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST adadelta(std::uint32_t size, precision_type gamma = 0.99, precision_type epsilon = 1.e-6 );
		  adadelta(const adadelta&) = delete;
		  adadelta(adadelta &&) = delete;
		  adadelta & operator = (const adadelta&) = delete;
		  adadelta & operator = (adadelta &&) = delete;
		  HOST ~adadelta();
	  };

	template <class precision_type>
  	  class rms_prop : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * prior_gradient{nullptr};
		  precision_type learning_rate;
		  precision_type gamma;
		  precision_type epsilon;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST rms_prop(std::uint32_t size, precision_type learning_rate = 0.001, precision_type gamma = 0.90, precision_type epsilon = 1.e-8);
		  rms_prop(const rms_prop&) = delete;
		  rms_prop(rms_prop &&) = delete;
		  rms_prop & operator = (const rms_prop&) = delete;
		  rms_prop & operator = (rms_prop &&) = delete;
		  HOST ~rms_prop();
	  };

	template <class precision_type>
  	  class rprop : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * prior_gradient{nullptr};
		  precision_type * delta{nullptr};
		  precision_type learning_rate_pos;
		  precision_type learning_rate_neg;
		  precision_type delta_max;
		  precision_type delta_min;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id);
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST rprop(std::uint32_t size, precision_type learning_rate_pos = 1.2, precision_type learning_rate_neg = 0.5, precision_type delta_max = 50, precision_type delta_min = 1.e-6);
		  rprop(const rprop&) = delete;
		  rprop(rprop &&) = delete;
		  rprop & operator = (const rprop&) = delete;
		  rprop & operator = (rprop &&) = delete;
		  HOST ~rprop();
	  };

	template <class precision_type>
  	  class adamax : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * mean{nullptr};
		  precision_type * variance{nullptr};
		  precision_type learning_rate;
		  precision_type beta_1;
		  precision_type beta_2;
		  precision_type beta_1_t;
		  precision_type epsilon;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void update_bias_correction_impl() override;
		  HOST virtual precision_type get_bias_corrected_first_moment_impl()const override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST adamax(std::uint32_t size, precision_type learning_rate = 0.002, precision_type beta_1 = 0.9, precision_type beta_2 = 0.999, precision_type beta_1_t = 0.9, precision_type epsilon = 1e-8);
		  adamax(const adamax&) = delete;
		  adamax(adamax &&) = delete;
		  adamax & operator = (const adamax&) = delete;
		  adamax & operator = (adamax &&) = delete;
		  HOST ~adamax();
	  };


	template <class precision_type>
  	  class amsgrad : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * mean{nullptr};
		  precision_type * variance{nullptr};
		  precision_type * bias_corrected_variance{nullptr};
		  precision_type learning_rate;
		  precision_type beta_1;
		  precision_type beta_2;
		  precision_type epsilon;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id) override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST amsgrad(std::uint32_t size, precision_type learning_rate = 0.01, precision_type beta_1 = 0.9, precision_type beta_2 = 0.99, precision_type epsilon = 1.e-8);
		  amsgrad(const amsgrad&) = delete;
		  amsgrad(amsgrad &&) = delete;
		  amsgrad & operator = (const amsgrad&) = delete;
		  amsgrad & operator = (amsgrad &&) = delete;
		  HOST ~amsgrad();
	  };

	template <class precision_type>
  	  class adam : public optimizer<precision_type>
	  {
		
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * mean{nullptr};
		  precision_type * variance{nullptr};
		  precision_type learning_rate;
		  precision_type beta_1;
		  precision_type beta_2;
		  precision_type beta_1_t;
		  precision_type beta_2_t;
		  precision_type epsilon;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id);
		  HOST virtual void update_bias_correction_impl() override;
		  HOST virtual precision_type get_bias_corrected_first_moment_impl()const override;
		  HOST virtual precision_type get_bias_corrected_second_moment_impl()const override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST adam(std::uint32_t size, precision_type learning_rate = 0.001, precision_type beta_1 = 0.9, precision_type beta_2 = 0.999, precision_type beta_1_t = 0.9, precision_type beta_2_t = 0.999, precision_type epsilon = 1.e-8);
		  adam(const adam&) = delete;
		  adam(adam &&) = delete;
		  adam & operator = (const adam&) = delete;
		  adam & operator = (adam &&) = delete;
		  HOST ~adam();
	  };


	template <class precision_type>
  	  class nadam : public optimizer<precision_type>
	  {
		private:
		  using optimizer<precision_type>::opt; 
		  using optimizer<precision_type>::size;
		  precision_type * mean{nullptr};
		  precision_type * variance{nullptr};
		  precision_type learning_rate;
		  precision_type gamma;
		  precision_type beta_1;
		  precision_type beta_2;
		  precision_type beta_1_t;
		  precision_type beta_2_t;
		  precision_type epsilon;
		  HOST virtual void update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id);
		  HOST virtual void update_bias_correction_impl() override;
		  HOST virtual precision_type get_bias_corrected_first_moment_impl()const override;
		  HOST virtual precision_type get_bias_corrected_second_moment_impl()const override;
		  HOST virtual void set_size_impl(const std::uint32_t & size) override;
		  HOST virtual std::uint32_t get_size_impl()const override;
		  HOST void safe_deallocate_impl()override;
		public:
		  HOST nadam(std::uint32_t size, precision_type learning_rate = 0.001, precision_type gamma = 0.9, precision_type beta_1 = 0.9, precision_type beta_2 = 0.999, precision_type beta_1_t = 0.9, precision_type beta_2_t = 0.999, precision_type epsilon = 1.e-8);
		  nadam(const nadam&) = delete;
		  nadam(nadam &&) = delete;
		  nadam & operator = (const nadam&) = delete;
		  nadam & operator = (nadam &&) = delete;
		  HOST ~nadam();
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
