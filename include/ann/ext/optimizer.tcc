#include <cassert>
namespace zinhart
{
  namespace optimizers
  {

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::sgd_optimizer sgd, precision_type & theta, const precision_type & gradient, const precision_type & eta)
	  { theta -=  ( eta * gradient ); }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::momentum_optimizer momentum, precision_type & theta, precision_type & prior_velocity, const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta)
	  {
 		precision_type current_velocity{ gamma * prior_velocity + eta * current_gradient };
		theta -= current_velocity;
		prior_velocity = current_velocity;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::nesterov_momentum_optimizer nesterov, precision_type & theta, precision_type & prior_velocity, 
										const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
									   )
	  {
		precision_type current_velocity{ gamma * prior_velocity + eta * current_gradient };
		theta -= current_velocity;
		prior_velocity = current_velocity;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void update(optimum_attributes::adagrad_optimizer adagrad,precision_type & theta, precision_type & prior_gradient, 
										const precision_type & current_gradient, const precision_type & eta, const precision_type & epsilon
									   )
	  {
		prior_gradient += current_gradient * current_gradient;
		theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::conjugate_gradient_optimizer conjugad, precision_type & theta, precision_type & prior_gradient, precision_type & hessian, 
										const precision_type & current_gradient, const precision_type & epsilon
									   )
	  {
		precision_type gamma { ( current_gradient - prior_gradient ) * current_gradient / ( prior_gradient * prior_gradient + epsilon ) };
		precision_type step { current_gradient + ( gamma * hessian ) };
		theta += step;
		prior_gradient = current_gradient;
		hessian = step;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::adadelta_optimizer adadelta, precision_type & theta, precision_type & prior_gradient, precision_type & prior_delta, 
											 const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon)
	  {
		prior_gradient = gamma * prior_gradient + (precision_type{1.0} - gamma) * current_gradient * current_gradient;
		precision_type delta { -(sqrt(prior_delta * prior_delta + epsilon) / sqrt(prior_gradient * prior_gradient + epsilon)) * current_gradient };
		theta += delta;
		prior_delta = gamma * prior_delta + (precision_type{1.0} - gamma) * delta * delta;
	  }
	
	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::rms_prop_optimizer rms_prop, precision_type & theta, precision_type & prior_gradient, 
										const precision_type & current_gradient,  const precision_type & eta, 
										const precision_type & beta, const precision_type & epsilon
									   )
	  {
  		prior_gradient = gamma * prior_gradient * prior_gradient + (precision_type{1} - gamma) * current_gradient * current_gradient;
		theta -= eta * current_gradient / sqrt(prior_gradient * prior_gradient + epsilon);
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::rprop_optimizer rprop, precision_type & theta, precision_type & prior_gradient, precision_type & current_delta,
										const precision_type & current_gradient, const precision_type & eta_pos, const precision_type & eta_neg,
										const precision_type & delta_max, const precision_type & delta_min
									   )
	  {
		if(current_gradient * prior_gradient > 0) // if the sign of the gradient has stayed positive
		{
	   	  current_delta = ( current_delta * eta_pos < delta_max) ? current_delta * eta_pos : delta_max;
   		  theta += -current_gradient * current_delta;
		  prior_gradient = current_gradient; 
		}
		else if(current_gradient * prior_gradient < 0)// if the sign of the gradient has stayed negative
		{
		  current_delta = ( current_delta * eta_neg > delta_min) ? current_delta * eta_neg : delta_min;
		  prior_gradient = 0;
		} 
		else// if either the prior or current gradient is 0, because of a negative gradient
		{
		  theta += -current_gradient * current_delta;
		  prior_gradient = current_gradient; 
		}
	  }
	// adamax, the max operation is w.r.t the infinity norm
	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::adamax_optimizer adamax ,precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
										 const precision_type & current_gradient, const precision_type & beta_1_t, const precision_type & eta, 
										 const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
										)
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1.0} - beta_1) * current_gradient; 
		prior_variance = (beta_2 * prior_variance > fabs(current_gradient)) ? beta_2 * prior_variance : fabs(current_gradient);
		theta -= (eta / (precision_type{1.0} - beta_1_t) ) * (prior_mean / (prior_variance + epsilon)); 
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::amsgrad_optimizer amsgrad, precision_type & theta, 
										precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
										const precision_type & current_gradient, const precision_type & eta, 
										const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
									   )
	  {
  		prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		// max(prior_variance > prior_bias_corrected_variance)
		prior_bias_corrected_variance = (prior_variance > prior_bias_corrected_variance) ? prior_variance : prior_bias_corrected_variance;
		theta -= eta * ( prior_mean ) / ( sqrt( prior_bias_corrected_variance) + epsilon ) ;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::adam_optimizer adam, precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
										const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
										const precision_type & beta_2, const precision_type & epsilon
									   )
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		precision_type bias_corrected_mean{ prior_mean / (precision_type{1} - beta_1_t) };
		precision_type bias_corrected_variace{ prior_variance / (precision_type{1} - beta_2_t) };
		theta -= eta * ( bias_corrected_mean ) / (sqrt( bias_corrected_variace ) + epsilon ) ;
	  }

	template <class precision_type>
  	  CUDA_CALLABLE_MEMBER void optimum<precision_type>::update(optimum_attributes::nadam_optimizer nadam, precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
										const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
										const precision_type & beta_2, const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
									   )
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		precision_type prior_bias_corrected_mean{prior_mean / (precision_type{1} - beta_1_t )};
		precision_type prior_bias_corrected_variance{prior_variance / (precision_type{1} - beta_2_t)};
		theta -= eta / ( sqrt(prior_bias_corrected_variance) + epsilon ) * (beta_1 * prior_bias_corrected_mean + (precision_type{1} - beta_1) / (precision_type{1} - beta_1_t) * current_gradient  );

	  }


	template <class precision_type>
	  HOST void sgd<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void momentum<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void nesterov_momentum<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }


	template <class precision_type>
	  HOST void adagrad<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void conjugate_gradient<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void adadelta<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void rms_prop<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void rprop<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void adamax<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void amsgrad<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void adam<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }

	template <class precision_type>
	  HOST void nadam<precision_type>::update(precision_type * theta, const precision_type * const gradient, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	  {
	  }
	/*
	
	// Stochastic gradient descent
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void stochastic_gradient_descent::update(precision_type & theta, const precision_type & gradient, const precision_type & eta)
		{ theta -=  ( eta * gradient ); }
    
	// Momentum
	  template <class precision_type>
		 CUDA_CALLABLE_MEMBER void momentum::update(precision_type & theta, precision_type & prior_velocity,
												   	const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
												   )
		 {
   		   precision_type current_velocity{ gamma * prior_velocity + eta * current_gradient };
   		   theta -= current_velocity;
   		   prior_velocity = current_velocity;
   		 }

	// Nesterov momentum
  	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void nesterov_momentum::update(precision_type & theta, precision_type & prior_velocity, 
														    const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
														   )
		{
		  precision_type velocity { gamma * prior_velocity + eta * current_gradient * (  theta - gamma * prior_velocity) };
		  theta -= velocity;
		  prior_velocity = velocity;
		}

	
  	// adagrad	
  	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void adagrad::update(precision_type & theta, precision_type & prior_gradient, const precision_type & current_gradient, const precision_type & eta, const precision_type & epsilon)
		{
		  prior_gradient += current_gradient * current_gradient;
		  theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
		}
	  

	  // conjugate gradient descent
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void conjugate_gradient_descent::update(precision_type & theta, precision_type & prior_gradient, precision_type & hessian, 
																	 const precision_type & current_gradient, const precision_type & epsilon
																	)
	  {
		precision_type gamma { ( current_gradient - prior_gradient ) * current_gradient / ( prior_gradient * prior_gradient + epsilon ) };
		precision_type step { current_gradient + ( gamma * hessian ) };
		theta += step;
		prior_gradient = current_gradient;
		hessian = step;
	  }
	  // adadelta	
	  template <class precision_type>	 
		CUDA_CALLABLE_MEMBER void adadelta::update(precision_type & theta,  precision_type & prior_gradient, precision_type & prior_delta,
									               const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon
												  )
		{
		  prior_gradient = gamma * prior_gradient + (precision_type{1.0} - gamma) * current_gradient * current_gradient;
		  precision_type delta { -(sqrt(prior_delta * prior_delta + epsilon) / sqrt(prior_gradient * prior_gradient + epsilon)) * current_gradient };
		  theta += delta;
		  prior_delta = gamma * prior_delta + (precision_type{1.0} - gamma) * delta * delta;
		}
		
	 // rms_prop
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void rms_prop::update(precision_type & theta, precision_type & prior_gradient, const precision_type & current_gradient,  
												   const precision_type & eta, const precision_type & gamma, const precision_type & epsilon
												  )
		{
		  prior_gradient = gamma * prior_gradient * prior_gradient + (precision_type{1} - gamma) * current_gradient * current_gradient;
		  theta -= eta * current_gradient / sqrt(prior_gradient * prior_gradient + epsilon);
		}

	//rprop
	template <class precision_type>
	 CUDA_CALLABLE_MEMBER void resilient_propagation::update(precision_type & theta, precision_type & prior_gradient, precision_type & current_delta,
															 const precision_type & current_gradient, const precision_type & eta_pos, const precision_type & eta_neg,
															 const precision_type & delta_max, const precision_type & delta_min
															)
	 {
	   if(current_gradient * prior_gradient > 0) // if the sign of the gradient has stayed positive
	   {
		 current_delta = ( current_delta * eta_pos < delta_max) ? current_delta * eta_pos : delta_max;
		 theta += -current_gradient * current_delta;
		 prior_gradient = current_gradient; 
	   }
	   else if(current_gradient * prior_gradient < 0)// if the sign of the gradient has stayed negative
	   {
		 current_delta = ( current_delta * eta_neg > delta_min) ? current_delta * eta_neg : delta_min;
		 prior_gradient = 0;
	   } 
	   else// if either the prior or current gradient is 0, because of a negative gradient
	   {
		 theta += -current_gradient * current_delta;
		 prior_gradient = current_gradient; 
	   }
	 }

	// adamax, the max operation is w.r.t the infinity norm
  	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void adamax::update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
											   const precision_type & current_gradient, const precision_type & beta_1_t, const precision_type & eta, 
											   const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
									          )
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1.0} - beta_1) * current_gradient; 
		prior_variance = (beta_2 * prior_variance > fabs(current_gradient)) ? beta_2 * prior_variance : fabs(current_gradient);
		theta -= (eta / (precision_type{1.0} - beta_1_t) ) * (prior_mean / (prior_variance + epsilon)); 
	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void adamax::moment_update(precision_type & beta_1_t, const precision_type & beta_1)
	  {	beta_1_t *= beta_1; }

  	template <class precision_type>
 	  CUDA_CALLABLE_MEMBER void adam::update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
											 const precision_type & current_gradient, const precision_type & beta_1_t, const precision_type & beta_2_t, 
											 const precision_type & eta, const precision_type & beta_1,
										     const precision_type & beta_2, const precision_type & epsilon
                                            )
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		precision_type bias_corrected_mean{ prior_mean / (precision_type{1} - beta_1_t) };
		precision_type bias_corrected_variace{ prior_variance / (precision_type{1} - beta_2_t) };
		theta -= eta * ( bias_corrected_mean ) / (sqrt( bias_corrected_variace ) + epsilon ) ;
	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void adam::moment_update(precision_type & beta_1_t, precision_type & beta_2_t, const precision_type & beta_1, const precision_type & beta_2)
	  {
		beta_1_t *= beta_1;
		beta_2_t *= beta_2;
	  }

	  template <class precision_type>
	 	CUDA_CALLABLE_MEMBER void amsgrad::update(precision_type & theta, 
												  precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
												  const precision_type & current_gradient, const precision_type & eta, 
												  const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
										         )
		{
	  	  prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		  prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		  // max(prior_variance > prior_bias_corrected_variance)
		  prior_bias_corrected_variance = (prior_variance > prior_bias_corrected_variance) ? prior_variance : prior_bias_corrected_variance;
		  theta -= eta * ( prior_mean ) / ( sqrt( prior_bias_corrected_variance) + epsilon ) ;
		 }
	  
	// nadam 
	template <class precision_type>	 
	  CUDA_CALLABLE_MEMBER void nadam::update(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
											  const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, const precision_type & beta_2, 
											  const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & epsilon
											 )
	  {
		prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		precision_type prior_bias_corrected_mean{prior_mean / (precision_type{1} - beta_1_t )};
		precision_type prior_bias_corrected_variance{prior_variance / (precision_type{1} - beta_2_t)};
		theta -= eta / ( sqrt(prior_bias_corrected_variance) + epsilon ) * (beta_1 * prior_bias_corrected_mean + (precision_type{1} - beta_1) / (precision_type{1} - beta_1_t) * current_gradient  );
	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void nadam::moment_update(precision_type & beta_1_t, precision_type & beta_2_t, const precision_type & beta_1, const precision_type & beta_2)
	  {
		beta_1_t *= beta_1;
		beta_2_t *= beta_2;
	  }





	// This overload is for SGD
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer::operator()(SGD && s, 
													  precision_type * theta, const precision_type * gradient, std::uint32_t theta_length,
											          std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
													  const precision_type & eta,
													  zinhart::parallel::thread_pool & pool
													 )
	  {
		
		auto thread_launch = [](precision_type * thetas, const precision_type * gradients, const precision_type & eta_init, 
								std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
							   )
		{
		  std::uint32_t start{0}, stop{0};
		  optimizer_interface<stochastic_gradient_descent> opt;
		  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
		  for(std::uint32_t op{start}; op < stop; ++op)
		  {
			opt(thetas[op], gradients[op], eta_init);
		  }
		};
		for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
		  results.push_back(pool.add_task(thread_launch, theta, gradient, eta, thread, pool.size(), theta_length));

	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer::operator()(MOMENTUM && m,
													  precision_type * theta, std::uint32_t theta_length, precision_type * prior_velocity, 
													  const precision_type * current_gradient, const precision_type & gamma, const precision_type & eta,
													  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
													  zinhart::parallel::thread_pool & pool
													 )
	  {
		auto thread_launch = [](precision_type * thetas, precision_type * prior_velocity, const precision_type * gradients, precision_type gamma, precision_type eta,
								std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
							   )
		  {
			std::uint32_t start{0}, stop{0};
			optimizer_interface<momentum> opt;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op{start}; op < stop; ++op)
			{
			  opt(thetas[op], prior_velocity[op], gradients[op], gamma, eta);
			}
		  };
		  for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			results.push_back(pool.add_task(thread_launch, theta, prior_velocity, current_gradient, gamma, eta, thread, pool.size(), theta_length));

	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer::operator()(NESTEROV_MOMENTUM && nm, 
		                                              precision_type * theta, std::uint32_t theta_length, precision_type * prior_velocity, 
													  const precision_type * current_gradient, const precision_type & gamma, const precision_type & eta,
													  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
													  zinhart::parallel::thread_pool & pool
													 )
	  {
  		auto thread_launch = [](precision_type * thetas, precision_type * prior_velocity, const precision_type * gradients, precision_type gamma, precision_type eta,
							    std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
							   )
		{
		  std::uint32_t start{0}, stop{0};
		  optimizer_interface<nesterov_momentum> opt;
		  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
		  for(std::uint32_t op{start}; op < stop; ++op)
		  {
			opt(thetas[op], prior_velocity[op], gradients[op], gamma, eta);
		  }
		};
		for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
		  results.push_back(pool.add_task(thread_launch, theta, prior_velocity, current_gradient, gamma, eta, thread, pool.size(), theta_length));

	  }
	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer::operator()(ADAGRAD && a, 
													  precision_type * theta, std::uint32_t theta_length, precision_type * prior_gradient, 
													  const precision_type * current_gradient, const precision_type & eta, const precision_type & epsilon,
													  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
													  zinhart::parallel::thread_pool & pool
													 )
	  {
  		auto thread_launch = [](precision_type * thetas, precision_type * prior_gradient, const precision_type * gradients, precision_type eta, precision_type epsilon,
								std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
							   )
		{
		  std::uint32_t start{0}, stop{0};
		  optimizer_interface<adagrad> opt;
		  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
		  for(std::uint32_t op{start}; op < stop; ++op)
		  {
			opt(thetas[op], prior_gradient[op], gradients[op], eta, epsilon);
		  }
		};
		for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
		  results.push_back(pool.add_task(thread_launch, theta, prior_gradient, current_gradient, eta, epsilon, thread, pool.size(), theta_length));

	  }
	  // This overload is for conjugate gradient
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void optimizer::operator()(CONJUGATE_GRADIENT && cg, 
													    precision_type * theta, precision_type * prior_gradient,  precision_type * hessian, 
													    const precision_type * current_gradient, const precision_type & epsilon,
													    std::uint32_t theta_length,
													    std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
													    zinhart::parallel::thread_pool & pool
													   )
		{
		 auto thread_launch = [](precision_type * thetas, precision_type * prior_grad, precision_type * hess, const precision_type * gradient, const precision_type & epsilon,
							     std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
			                    )
		 {
			std::uint32_t start{0}, stop{0};
			optimizer_interface<conjugate_gradient_descent> opt;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op{start}; op < stop; ++op)
			{
			  opt(thetas[op], prior_grad[op], hess[op], gradient[op], epsilon);
			}
		 }; 
		 for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
	   	   results.push_back(pool.add_task(thread_launch, theta, prior_gradient, hessian, current_gradient, epsilon, thread, pool.size(), theta_length));
		}

		// This overload is for adadelta
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(ADADELTA && ad, 
														  precision_type * theta, precision_type * prior_gradient, precision_type * prior_delta, 
														  const precision_type * current_gradient, std::uint32_t theta_length,
														  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
														  const precision_type gamma, const precision_type epsilon,
														  zinhart::parallel::thread_pool & pool 
														 )
		  {
			auto thread_launch = [](precision_type * thetas, precision_type * prior_grad, precision_type * prior_delta, 
				                    const precision_type * gradient, const precision_type & gamma, const precision_type & epsilon,
								    std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
		  	  std::uint32_t start{0}, stop{0};
			  optimizer_interface<adadelta> opt;
		  	  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
	  		  for(std::uint32_t op{start}; op < stop; ++op)
				opt(thetas[op], prior_grad[op], prior_delta[op], gradient[op], gamma ,epsilon);
			 }; 
			 for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			   results.push_back(pool.add_task(thread_launch, theta, prior_gradient, prior_delta, current_gradient, gamma, epsilon, thread, pool.size(), theta_length));
		 }
							  

		// This overload is for rms_prop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(RMS_PROP && rms, 
														  precision_type * theta, precision_type * prior_gradient, 
														  const precision_type * current_gradient, std::uint32_t theta_length,
														  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
														  const precision_type & eta, const precision_type & gamma, const precision_type & epsilon,
														  zinhart::parallel::thread_pool & pool
														 )
		  {
			auto thread_launch = [](precision_type * thetas, precision_type * prior_grad, const precision_type * gradient, 
				                    const precision_type & eta, const precision_type & gamma, const precision_type & epsilon,
								    std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
		  	  std::uint32_t start{0}, stop{0};
			  optimizer_interface<rms_prop> opt;
		  	  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
	  		  for(std::uint32_t op{start}; op < stop; ++op)
				opt(thetas[op], prior_grad[op], gradient[op], eta, gamma ,epsilon);
			 }; 
			 for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			   results.push_back(pool.add_task(thread_launch, theta, prior_gradient, current_gradient, eta, gamma, epsilon, thread, pool.size(), theta_length));
		  }

		//This overload is for rprop
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(RPROP && rp,
														  precision_type * theta, precision_type * prior_gradient, precision_type * current_delta, const precision_type * current_gradient, 
														  std::uint32_t theta_length,
														  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
														  const precision_type & eta_pos, const precision_type & eta_neg, const precision_type & delta_max, const precision_type & delta_min,
														  zinhart::parallel::thread_pool & pool
														 )
		  {
  			auto thread_launch = [](precision_type * thetas, precision_type * prior_gradients, precision_type * current_delta, const precision_type * current_gradients,
								    const precision_type & eta_pos, const precision_type & eta_minus, const precision_type & dmax, const precision_type & dmin,
						            std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
			  std::uint32_t start{0}, stop{0};
			  optimizer_interface<resilient_propagation> opt;
			  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			  for(std::uint32_t op{start}; op < stop; ++op)
			  {
				opt(thetas[op], prior_gradients[op], current_delta[op], current_gradients[op], eta_pos, eta_minus, dmax, dmin);
			  }
			};
			for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			  results.push_back(pool.add_task(thread_launch, theta, prior_gradient, current_delta, current_gradient, eta_pos, eta_neg, delta_max, delta_min, thread, pool.size(), theta_length));
			}
  
		// This overload is for adamax
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(ADAMAX && amax, 
														  precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, 
														  const precision_type * current_gradient, const precision_type & beta_1_t, 
														  std::uint32_t theta_length,
														  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
														  const precision_type & eta, const precision_type & beta_1, 
														  const precision_type & beta_2, const precision_type & epsilon,
														  zinhart::parallel::thread_pool & pool
														 )
		  {
  			auto thread_launch = [](precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient,
								    const precision_type & beta_1_t, const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon,
						            std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
			  std::uint32_t start{0}, stop{0};
			  optimizer_interface<adamax> opt;
			  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			  for(std::uint32_t op{start}; op < stop; ++op)
			  {
				opt(theta[op], prior_mean[op], prior_variance[op], current_gradient[op], beta_1_t, eta, beta_1, beta_2, epsilon);
			  }
			};
			for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			  results.push_back(pool.add_task(thread_launch, theta, prior_mean, prior_variance, current_gradient, beta_1_t, eta, beta_1, beta_2, epsilon, thread, pool.size(), theta_length));
		}
		  
		// This overload is for amsgrad
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(AMSGRAD && amrad,
											   precision_type * theta, 
											   precision_type * prior_mean, precision_type * prior_variance, precision_type * prior_bias_corrected_variance,
											   const precision_type * current_gradient,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon, 
											   zinhart::parallel::thread_pool & pool
											  )
		  {
  			auto thread_launch = [](precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, precision_type * prior_bias_corrected_variance,
									const precision_type * current_gradient,
								    const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon,
						            std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
			  std::uint32_t start{0}, stop{0};
			  optimizer_interface<amsgrad> opt;
			  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			  for(std::uint32_t op{start}; op < stop; ++op)
			  {
				opt(theta[op], prior_mean[op], prior_variance[op], prior_bias_corrected_variance[op], current_gradient[op], eta, beta_1, beta_2, epsilon);
			  }
			};
			for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			  results.push_back(pool.add_task(thread_launch, theta, prior_mean, prior_variance, prior_bias_corrected_variance, current_gradient, eta, beta_1, beta_2, epsilon, thread, pool.size(), theta_length));
		  }
		// This overload is for adam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(ADAM && adm,
														  precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient, 
														  const precision_type & beta_1_t, const precision_type & beta_2_t, 
														  std::uint32_t theta_length,
														  std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
														  const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon,
														  zinhart::parallel::thread_pool & pool 
														 )
		  {
  			auto thread_launch = [](precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient,
								    const precision_type & beta_1_t, const precision_type & beta_2_t, 
									const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon,
						            std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
			  std::uint32_t start{0}, stop{0};
			  optimizer_interface<adam> opt;
			  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			  for(std::uint32_t op{start}; op < stop; ++op)
			  {
				opt(theta[op], prior_mean[op], prior_variance[op], current_gradient[op], beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon);
			  }
			};
			for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			  results.push_back(pool.add_task(thread_launch, theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon, thread, pool.size(), theta_length));
		  }
		
		// This overload is for nadam
		template <class precision_type>
		  CUDA_CALLABLE_MEMBER void optimizer::operator()(NADAM && ndm,
											   precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient, 
											   const precision_type & beta_1_t, const precision_type & beta_2_t,
											   std::uint32_t theta_length,
											   std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
											   const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, 
											   const precision_type & beta_2, const precision_type & epsilon,
											   zinhart::parallel::thread_pool & pool
												  )
		  {
  			auto thread_launch = [](precision_type * theta, precision_type * prior_mean, precision_type * prior_variance, const precision_type * current_gradient,
								    const precision_type & beta_1_t, const precision_type & beta_2_t, 
									const precision_type & eta, const precision_type & gamma, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon,
						            std::uint32_t thread_id, std::uint32_t n_threads, std::uint32_t n_elements
								   )
			{
			  std::uint32_t start{0}, stop{0};
			  optimizer_interface<nadam> opt;
			  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			  for(std::uint32_t op{start}; op < stop; ++op)
			  {
				opt(theta[op], prior_mean[op], prior_variance[op], current_gradient[op], eta, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon);
			  }
			};
			for(std::uint32_t thread = 0; thread < pool.size(); ++thread)
			  results.push_back(pool.add_task(thread_launch, theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, gamma ,beta_1, beta_2, epsilon, thread, pool.size(), theta_length));
		  }

	template <class OPTIMIZER>
  	  CUDA_CALLABLE_MEMBER std::uint32_t optimizer_interface<OPTIMIZER>::order()
	  {return static_cast<OPTIMIZER*>(this)->get_order();}
	// This overload is for SGD
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, const precision_type & gradient, const precision_type & eta)
	  { static_cast<OPTIMIZER*>(this)->update(theta, gradient, eta); }
	
	template <class OPTIMIZER>
  	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & free_1, const precision_type & current_gradient, const precision_type & free_2, const precision_type & free_3)
	  { static_cast<OPTIMIZER*>(this)->update(theta, free_1, current_gradient, free_2, free_3); }

	// This overload is for conjugate gradient
	template <class OPTIMIZER>
  	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_gradient,  precision_type & hessian, 
																           const precision_type & current_gradient, const precision_type & epsilon
																          )
	  { static_cast<OPTIMIZER*>(this)->update(theta, prior_gradient, hessian, current_gradient, epsilon); }


	// This overload is for adadelta
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_gradient, precision_type & prior_delta, 
																		   const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon
																		  )
	  { static_cast<OPTIMIZER*>(this)->update(theta, prior_gradient, prior_delta, current_gradient, gamma, epsilon); }

	// This overload is for rms_prop
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_gradient, 
																		   const precision_type & current_gradient, const precision_type & eta, 
																		   const precision_type & beta, const precision_type & epsilon
																		  )
	  { static_cast<OPTIMIZER*>(this)->update(theta, prior_gradient, current_gradient, eta, beta, epsilon); }

	  //This overload is for rprop
	  template <class OPTIMIZER>
		template <class precision_type>
		CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_gradient, 
																			 precision_type & current_delta, const precision_type & current_gradient, 
																			 const precision_type & eta_pos, const precision_type & eta_neg, 
																			 const precision_type & delta_max, const precision_type & delta_min
																			)
		{ static_cast<OPTIMIZER*>(this)->update(theta, prior_gradient, current_delta, current_gradient, eta_pos, eta_neg, delta_max, delta_min);}
	
	// This overload is for adamax
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
																		   const precision_type & current_gradient, const precision_type & beta_1_t, 
																		   const precision_type & eta, const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
																          )
	  
	  { static_cast<OPTIMIZER*>(this)->update(theta, prior_mean, prior_variance, current_gradient, beta_1_t, eta, beta_1, beta_2, epsilon); }

	// This overload is for amsgrad
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, 
																		   precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
																		   const precision_type & current_gradient, const precision_type & eta, 
																		   const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
																		  )
	  { static_cast<OPTIMIZER*>(this)->update(theta, prior_mean, prior_variance, prior_bias_corrected_variance, current_gradient, eta, beta_1, beta_2, epsilon); }

	// This overload is for adam
	template <class OPTIMIZER>
  	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
																		   const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
																		   const precision_type & beta_2, const precision_type & epsilon
																		  )	  
	  { static_cast<OPTIMIZER*>(this)->update(theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon); }

	  // This overload is for nadam
	  template <class OPTIMIZER>
		template <class precision_type>
		CUDA_CALLABLE_MEMBER void optimizer_interface<OPTIMIZER>::operator()(precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
																			 const precision_type & eta, const precision_type & gamma, 
																			 const precision_type & beta_1, const precision_type & beta_2,
																			 const precision_type & beta_1_t, const precision_type & beta_2_t, 
																			 const precision_type & epsilon
																			)
		{ static_cast<OPTIMIZER*>(this)->update(theta, prior_mean, prior_variance, current_gradient, eta, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon); };	
	*/
  }// END NAMESPACE OPTIMIZERS
}// END NAMESPACE ZINHART
