#include <multi_core/multi_core.hh>
#include <cassert>
namespace zinhart
{
  namespace optimizers
  {

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::sgd_optimizer sgd, 
																precision_type & weight, const precision_type & gradient, 
																const precision_type & learning_rate
															   )
	  { weight -=  ( learning_rate * gradient ); }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::momentum_optimizer momentum, 
																precision_type & weight, precision_type & prior_velocity, const precision_type & current_gradient, 
																const precision_type & learning_rate, const precision_type & gamma
															   )
	  {
 		precision_type current_velocity{ gamma * prior_velocity + learning_rate * current_gradient };
		weight -= current_velocity;
		prior_velocity = current_velocity;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::nesterov_momentum_optimizer nesterov, 
		                                                        precision_type & weight, precision_type & prior_velocity, const precision_type & current_gradient, 
		                                                        const precision_type & learning_rate, const precision_type & gamma
									                           )
	  {
		precision_type current_velocity{ gamma * prior_velocity + learning_rate * current_gradient };
		weight -= current_velocity;
		prior_velocity = current_velocity;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::adagrad_optimizer adagrad, 
																precision_type & weight, precision_type & prior_gradient, const precision_type & current_gradient, 
																const precision_type & learning_rate, const precision_type & epsilon
															   )
	  {
		prior_gradient += current_gradient * current_gradient;
		weight -= learning_rate * current_gradient / sqrt(prior_gradient + epsilon);
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::conjugate_gradient_optimizer conjugad, 
																precision_type & weight, precision_type & prior_gradient, precision_type & hessian, const precision_type & current_gradient, 
																const precision_type & epsilon
															   )
	  {
		precision_type gamma { ( current_gradient - prior_gradient ) * current_gradient / ( prior_gradient * prior_gradient + epsilon ) };
		precision_type step { current_gradient + ( gamma * hessian ) };
		weight += step;
		prior_gradient = current_gradient;
		hessian = step;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::adadelta_optimizer adadelta, precision_type & weight, precision_type & prior_gradient, precision_type & prior_delta, 
											 const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon)
	  {
		prior_gradient = gamma * prior_gradient + (precision_type{1.0} - gamma) * current_gradient * current_gradient;
		precision_type delta { -(sqrt(prior_delta * prior_delta + epsilon) / sqrt(prior_gradient * prior_gradient + epsilon)) * current_gradient };
		weight += delta;
		prior_delta = gamma * prior_delta + (precision_type{1.0} - gamma) * delta * delta;
	  }
	
	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::rms_prop_optimizer rms_prop, 
																precision_type & weight, precision_type & prior_gradient, const precision_type & current_gradient,  
																const precision_type & learning_rate, const precision_type & gamma, const precision_type & epsilon
															   )
	  {
  		prior_gradient = gamma * prior_gradient * prior_gradient + (precision_type{1} - gamma) * current_gradient * current_gradient;
		weight -= learning_rate * current_gradient / sqrt(prior_gradient * prior_gradient + epsilon);
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::rprop_optimizer rprop, 
																precision_type & weight, precision_type & prior_gradient, precision_type & delta, const precision_type & current_gradient, 
																const precision_type & learning_rate_pos, const precision_type & learning_rate_neg,
																const precision_type & delta_max, const precision_type & delta_min
															   )
	  {
		if(current_gradient * prior_gradient > 0) // if the sign of the gradient has stayed positive
		{
	   	  delta = (delta * learning_rate_pos < delta_max) ? delta * learning_rate_pos : delta_max;
   		  weight += -current_gradient * delta;
		  prior_gradient = current_gradient; 
		}
		else if(current_gradient * prior_gradient < 0)// if the sign of the gradient has stayed negative
		{
		  delta = ( delta * learning_rate_neg > delta_min) ? delta * learning_rate_neg : delta_min;
		  prior_gradient = 0;
		} 
		else// if either the prior or current gradient is 0, because of a negative gradient
		{
		  weight += -current_gradient * delta;
		  prior_gradient = current_gradient; 
		}
	  }
	// adamax, the max operation is w.r.t the infinity norm
	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::adamax_optimizer adamax, 
																precision_type & weight, precision_type & mean, precision_type & variance, const precision_type & current_gradient, 
																const precision_type & learning_rate, 
																const precision_type & beta_1, const precision_type & beta_2, const precision_type & beta_1_t, 
																const precision_type & epsilon
															   )
	  {
		mean = beta_1 * mean + (precision_type{1.0} - beta_1) * current_gradient; 
		variance = (beta_2 * variance > fabs(current_gradient)) ? beta_2 * variance : fabs(current_gradient);
		weight -= (learning_rate / (precision_type{1.0} - beta_1_t) ) * (mean / (variance + epsilon)); 
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::amsgrad_optimizer amsgrad, 
																precision_type & weight, precision_type & mean, precision_type & variance, precision_type & bias_corrected_variance, const precision_type & current_gradient, 
																const precision_type & learning_rate, 
																const precision_type & beta_1, const precision_type & beta_2, 
																const precision_type & epsilon
															   )
	  {
  		mean = beta_1 * mean + (precision_type{1} - beta_1) * current_gradient;
		variance = beta_2 * variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		// max(variance > bias_corrected_variance)
		bias_corrected_variance = (variance > bias_corrected_variance) ? variance : bias_corrected_variance;
		weight -= learning_rate * ( mean ) / ( sqrt( bias_corrected_variance) + epsilon ) ;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::adam_optimizer adam, 
																precision_type & weight, precision_type & mean, precision_type & variance, const precision_type & current_gradient, 
										                        const precision_type & learning_rate, 
																const precision_type & beta_1, const precision_type & beta_2, 
										                        const precision_type & beta_1_t, const precision_type & beta_2_t, 
																const precision_type & epsilon
									                           )
	  {
		mean = beta_1 * mean + (precision_type{1} - beta_1) * current_gradient;
		variance = beta_2 * variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		precision_type bias_corrected_mean{ mean / (precision_type{1} - beta_1_t) };
		precision_type bias_corrected_variace{ variance / (precision_type{1} - beta_2_t) };
		weight -= learning_rate * ( bias_corrected_mean ) / (sqrt( bias_corrected_variace ) + epsilon ) ;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimizer_attributes::nadam_optimizer nadam, 
		                                                        precision_type & weight, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
																const precision_type & learning_rate, const precision_type & gamma, 
																const precision_type & beta_1, const precision_type & beta_2, 
																const precision_type & beta_1_t, const precision_type & beta_2_t, 
																const precision_type & epsilon
															   )
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		precision_type prior_bias_corrected_mean{prior_mean / (precision_type{1} - beta_1_t )};
		precision_type prior_bias_corrected_variance{prior_variance / (precision_type{1} - beta_2_t)};
		weight -= learning_rate / ( sqrt(prior_bias_corrected_variance) + epsilon ) * (beta_1 * prior_bias_corrected_mean + (precision_type{1} - beta_1) / (precision_type{1} - beta_1_t) * current_gradient  );

	  }

    // require members for subclasses to override
	template <class precision_type>
	  HOST void optimizer<precision_type>::update(precision_type * weights, const precision_type * const gradient, const std::uint32_t length, const std::uint32_t n_threads, const std::uint32_t thread_id)
	  { update_impl(weights, gradient, length, n_threads, thread_id); }
	
	template <class precision_type>
	  HOST void optimizer<precision_type>::set_size(const std::uint32_t size)
	  { set_size_impl(size); }

	template <class precision_type>
	  HOST std::uint32_t optimizer<precision_type>::get_size()const
	  { return get_size_impl(size); }
	
	// optional members for subclasses to override
	template <class precision_type>
	  HOST void optimizer<precision_type>::update_bias_correction()
	  { update_bias_correction_impl(); }

	template <class precision_type>
	  HOST precision_type optimizer<precision_type>::get_bias_corrected_first_moment()const
	  { return get_bias_corrected_first_moment_impl(); }

	template <class precision_type>
	  HOST precision_type optimizer<precision_type>::get_bias_corrected_second_moment()const
	  { return get_bias_corrected_second_moment_impl(); }

	template <class precision_type>
	  HOST void optimizer<precision_type>::safe_deallocate()
	  { safe_deallocate_impl(); }


	template <class precision_type>
	  HOST optimizer<precision_type>::~optimizer()
	  { safe_deallocate(); }

	template <class precision_type>
	  HOST sgd<precision_type>::sgd(precision_type learning_rate)
	  { this->learning_rate = learning_rate; }

	template <class precision_type>
	  HOST void sgd<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::sgd_optimizer(), *(weights + op), *(gradient + op), learning_rate );
	  }

	template <class precision_type>
  	  HOST void sgd<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t sgd<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
	  HOST momentum<precision_type>::momentum(std::uint32_t size, precision_type learning_rate, precision_type momentum_term)
	  {
		this->learning_rate = learning_rate; 
	    this->momentum_term = momentum_term;
		set_size_impl(size);
		velocity = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		  velocity[i] = 0.0;
	  }

	template <class precision_type>
	  HOST momentum<precision_type>::~momentum()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void momentum<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		assert(velocity != nullptr);
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::momentum_optimizer(), *(weights + op), *(velocity + op),*(gradient + op), learning_rate, momentum_term);
	  }

	template <class precision_type>
  	  HOST void momentum<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t momentum<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void momentum<precision_type>::safe_deallocate_impl()
	  {
		if(velocity != nullptr) // might have a previous state
		  delete [] velocity;
	  }

	template <class precision_type>
	  HOST nesterov_momentum<precision_type>::nesterov_momentum(std::uint32_t size, precision_type learning_rate, precision_type momentum_term)
	  {
		this->learning_rate = learning_rate; 
	    this->momentum_term = momentum_term;
		set_size_impl(size);
		velocity = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		  velocity[i] = 0.0;
	  }

	template <class precision_type>
	  HOST nesterov_momentum<precision_type>::~nesterov_momentum()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void nesterov_momentum<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::nesterov_momentum_optimizer(), *(weights + op), *(velocity + op),*(gradient + op), learning_rate, momentum_term);
	  }

	template <class precision_type>
  	  HOST void nesterov_momentum<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t nesterov_momentum<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void nesterov_momentum<precision_type>::safe_deallocate_impl()
	  {
		if(velocity != nullptr) // might have a previous state
		  delete [] velocity;
	  }

	template <class precision_type>
	  HOST adagrad<precision_type>::adagrad(std::uint32_t size, precision_type learning_rate, precision_type epsilon)
	  {
		this->learning_rate = learning_rate; 
	    this->epsilon = epsilon;
		set_size_impl(size);
		prior_gradient = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		  prior_gradient[i] = 0.0;
	  }
	template <class precision_type>
	  HOST adagrad<precision_type>::~adagrad()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void adagrad<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::adagrad_optimizer(), *(weights + op), *(prior_gradient + op), *(gradient + op), learning_rate, epsilon);
	  }

	template <class precision_type>
  	  HOST void adagrad<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t adagrad<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void adagrad<precision_type>::safe_deallocate_impl()
	  {
		if(prior_gradient != nullptr) // might have a previous state
		  delete [] prior_gradient;
	  }

	template <class precision_type>
	  conjugate_gradient<precision_type>::conjugate_gradient(std::uint32_t size, precision_type epsilon)
	  {
		set_size_impl(size);
		this->epsilon = epsilon;
		prior_gradient = new precision_type[get_size_impl()];
		hessian = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  prior_gradient[i] = 0.0;
		  hessian[i] = 0.0;
		}
	  }
 
	template <class precision_type>
	  HOST conjugate_gradient<precision_type>::~conjugate_gradient()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void conjugate_gradient<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::conjugate_gradient_optimizer(), *(weights + op), *(prior_gradient + op), *(hessian + op), *(gradient + op), epsilon);
	  }

	template <class precision_type>
  	  HOST void conjugate_gradient<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t conjugate_gradient<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void conjugate_gradient<precision_type>::safe_deallocate_impl()
	  {
		if(prior_gradient != nullptr) // might have a previous state
		  delete [] prior_gradient;
		if(hessian != nullptr)
		  delete [] hessian;
	  }

	template <class precision_type>
	  HOST adadelta<precision_type>::adadelta(std::uint32_t size, precision_type gamma, precision_type epsilon)
	  {
		set_size_impl(size);
		this->gamma = gamma;
		this->epsilon = epsilon;
		prior_gradient = new precision_type[get_size_impl()];
		prior_delta = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  prior_gradient[i] = 0.0;
		  prior_delta[i] = 0.0;
		}
	  }

	template <class precision_type>
	  HOST adadelta<precision_type>::~adadelta()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void adadelta<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::adadelta_optimizer(), *(weights + op), *(prior_gradient + op), *(prior_delta + op), *(gradient + op), gamma, epsilon);
	  }

	template <class precision_type>
  	  HOST void adadelta<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t adadelta<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void adadelta<precision_type>::safe_deallocate_impl()
	  {
		if(prior_gradient != nullptr) // might have a previous state
		  delete [] prior_gradient;
		if(prior_delta != nullptr)
		  delete [] prior_delta;
	  }

	template <class precision_type>
	  HOST rms_prop<precision_type>::rms_prop(std::uint32_t size, precision_type learning_rate, precision_type beta, precision_type epsilon)
	  {
		set_size_impl(size);
		this->learning_rate = learning_rate;
		this->gamma = gamma;
		this->epsilon = epsilon;
		prior_gradient = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		  prior_gradient[i] = 0.0;
	  }

	template <class precision_type>
	  HOST rms_prop<precision_type>::~rms_prop()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void rms_prop<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::rms_prop_optimizer(), *(weights + op), *(prior_gradient + op), *(gradient + op), learning_rate, gamma, epsilon);
	  }

	template <class precision_type>
  	  HOST void rms_prop<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t rms_prop<precision_type>::get_size_impl()const
	  { return size; }


	template <class precision_type>
  	  HOST void rms_prop<precision_type>::safe_deallocate_impl()
	  {
		if(prior_gradient != nullptr)
		  delete [] prior_gradient;
	  }

	template <class precision_type>
	  HOST rprop<precision_type>::rprop(std::uint32_t size, precision_type learning_rate_pos, precision_type learning_rate_neg, precision_type delta_max, precision_type delta_min)
	  {
		set_size_impl(size);
		this->learning_rate_pos = learning_rate_pos;
		this->learning_rate_neg = this->learning_rate_neg;
		this->delta_max = delta_max;
		this->delta_min = delta_min;
		prior_gradient = new precision_type[get_size_impl()];
		delta = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  prior_gradient[i] = 0.0;
		  delta[i] = 0.0;
		}
	  }

	template <class precision_type>
	  HOST rprop<precision_type>::~rprop()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void rprop<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::rprop_optimizer(), *(weights + op), *(prior_gradient + op), *(delta + op), *(gradient + op), learning_rate_pos, learning_rate_neg, delta_max, delta_min);
	  }

	template <class precision_type>
  	  HOST void rprop<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t rprop<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void rprop<precision_type>::safe_deallocate_impl()
	  {
		if(prior_gradient != nullptr)
		  delete [] prior_gradient;
		if(delta != nullptr)
		  delete [] delta;
	  }

	template <class precision_type>
	  HOST adamax<precision_type>::adamax(std::uint32_t size, precision_type learning_rate, precision_type beta_1, precision_type beta_2, precision_type beta_1_t, precision_type epsilon)
	  {
		set_size_impl(size);
		this->learning_rate = learning_rate;
		this->beta_1 = beta_1;
		this->beta_2 = beta_2;
		this->beta_1_t = beta_1_t;
		this->epsilon = epsilon;
		mean = new precision_type[get_size_impl()];
		variance = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  mean[i] = 0.0;
		  variance[i] = 0.0;
		}
	  }

	template <class precision_type>
	  HOST adamax<precision_type>::~adamax()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void adamax<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::adamax_optimizer(), *(weights + op), *(mean + op), *(variance + op), *(gradient + op), learning_rate, beta_1, beta_2, beta_1_t, epsilon);
	  }

	template <class precision_type>
	  HOST void adamax<precision_type>::update_bias_correction_impl()
	  {	beta_1_t *= beta_1; }

	template <class precision_type>
	  HOST precision_type adamax<precision_type>::get_bias_corrected_first_moment_impl()const
	  {	return beta_1_t; }

	template <class precision_type>
  	  HOST void adamax<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t adamax<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void adamax<precision_type>::safe_deallocate_impl()
	  {
		if(mean != nullptr)
		  delete [] mean;
		if(variance != nullptr)
		  delete [] variance;
	  }

	template <class precision_type>
	  HOST amsgrad<precision_type>::amsgrad(std::uint32_t size, precision_type learning_rate, precision_type beta_1, precision_type beta_2, precision_type epsilon)
	  {
		set_size_impl(size);
		this->learning_rate = learning_rate;
		this->beta_1 = beta_1;
		this->beta_2 = beta_2;
		this->epsilon = epsilon;
		mean = new precision_type[get_size_impl()];
		variance = new precision_type[get_size_impl()];
		bias_corrected_variance = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  mean[i] = 0.0;
		  variance[i] = 0.0;
		  bias_corrected_variance[i] = 0.0;
		}
	  }

	template <class precision_type>
	  HOST amsgrad<precision_type>::~amsgrad()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void amsgrad<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::amsgrad_optimizer(), *(weights + op), *(mean + op), *(variance + op), *(bias_corrected_variance + op), *(gradient + op), learning_rate, beta_1, beta_2, epsilon);
	  }

	template <class precision_type>
  	  HOST void amsgrad<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t amsgrad<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void amsgrad<precision_type>::safe_deallocate_impl()
	  {
		if(mean != nullptr)
		  delete [] mean;
		if(variance != nullptr)
		  delete [] variance;
		if(bias_corrected_variance != nullptr)
		  delete [] bias_corrected_variance;
	  }

	template <class precision_type>
	  HOST adam<precision_type>::adam(std::uint32_t size, precision_type learning_rate, precision_type beta_1, precision_type beta_2, precision_type beta_1_t, precision_type beta_2_t, precision_type epsilon)
	  {
		set_size_impl(size);
		this->learning_rate = learning_rate;
		this->beta_1 = beta_1;
		this->beta_2 = beta_2;
		this->beta_1_t = beta_1_t;
		this->beta_2_t = beta_2_t;
		this->epsilon = epsilon;
		mean = new precision_type[get_size_impl()];
		variance = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  mean[i] = 0.0;
		  variance[i] = 0.0;
		}
	  }

	template <class precision_type>
	  HOST adam<precision_type>::~adam()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void adam<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::adam_optimizer(), *(weights + op), *(mean + op), *(variance + op), *(gradient + op), learning_rate, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);
	  }

	template <class precision_type>
	  HOST void adam<precision_type>::update_bias_correction_impl()
	  {	
		beta_1_t *= beta_1; 
		beta_2_t *= beta_2;
	  }

	template <class precision_type>
	  HOST precision_type adam<precision_type>::get_bias_corrected_first_moment_impl()const
	  {	return beta_1_t; }

	template <class precision_type>
	  HOST precision_type adam<precision_type>::get_bias_corrected_second_moment_impl()const
	  {	return beta_2_t; }

	template <class precision_type>
  	  HOST void adam<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t adam<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void adam<precision_type>::safe_deallocate_impl()
	  {
		if(mean != nullptr)
		  delete [] mean;
		if(variance != nullptr)
		  delete [] variance;
	  }

	template <class precision_type>
	  HOST nadam<precision_type>::nadam(std::uint32_t size, precision_type learning_rate, precision_type gamma, precision_type beta_1, precision_type beta_2, precision_type beta_1_t, precision_type beta_2_t, precision_type epsilon)
	  {
		set_size_impl(size);
		this->learning_rate = learning_rate;
		this->gamma = gamma;
		this->beta_1 = beta_1;
		this->beta_2 = beta_2;
		this->beta_1_t = beta_1_t;
		this->beta_2_t = beta_2_t;
		this->epsilon = epsilon;
		mean = new precision_type[get_size_impl()];
		variance = new precision_type[get_size_impl()];
		for(std::uint32_t i = 0; i < get_size_impl(); ++i)
		{
		  mean[i] = 0.0;
		  variance[i] = 0.0;
		}
	  }

	template <class precision_type>
	  HOST nadam<precision_type>::~nadam()
	  { safe_deallocate_impl(); }

	template <class precision_type>
	  HOST void nadam<precision_type>::update_impl(precision_type * weights, const precision_type * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
		std::uint32_t start{0}, stop{0};
		zinhart::multi_core::map(thread_id, n_threads, length, start, stop);
		for(std::uint32_t op{start}; op < stop; ++op)
		  opt.update(optimizer_attributes::nadam_optimizer(), *(weights + op), *(mean + op), *(variance + op), *(gradient + op), learning_rate, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);
	  }

	template <class precision_type>
	  HOST void nadam<precision_type>::update_bias_correction_impl()
	  {	
		beta_1_t *= beta_1; 
		beta_2_t *= beta_2;
	  }

	template <class precision_type>
	  HOST precision_type nadam<precision_type>::get_bias_corrected_first_moment_impl()const
	  {	return beta_1_t; }

	template <class precision_type>
	  HOST precision_type nadam<precision_type>::get_bias_corrected_second_moment_impl()const
	  {	return beta_2_t; }

	template <class precision_type>
  	  HOST void nadam<precision_type>::set_size_impl(const std::uint32_t & size)
	  { this->size = size; }
	
	template <class precision_type>
	  HOST std::uint32_t nadam<precision_type>::get_size_impl()const
	  { return size; }

	template <class precision_type>
  	  HOST void nadam<precision_type>::safe_deallocate_impl()
	  {
		if(mean != nullptr)
		  delete [] mean;
		if(variance != nullptr)
		  delete [] variance;
	  }
  }// END NAMESPACE OPTIMIZERS
}// END NAMESPACE ZINHART
