namespace zinhart
{
  namespace optimizers
  {
	template <class OPTIMIZER>
  	  CUDA_CALLABLE_MEMBER std::uint32_t optimizer<OPTIMIZER>::order()
	  {return static_cast<OPTIMIZER*>(this)->get_order();}
	// This overload is for SGD
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name, precision_type & theta, const precision_type & gradient, const precision_type & eta)
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, gradient, eta); }
	
	/*
	 *  This overload is shared by momentum, nesterov momentum, and adagrad
	 *  for momentum and nesterove momentom free_2 = gamma, free_3 = eta
	 *  for adagrad free_2 = eta, free_3 = epsilon
	 *  */
	template <class OPTIMIZER>
  	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name, precision_type & theta, precision_type & free_1, const precision_type & current_gradient, const precision_type & free_2, const precision_type & free_3)
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, free_1, current_gradient, free_2, free_3); }

	// This overload is for conjugate gradient
	template <class OPTIMIZER>
  	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name, 
																 precision_type & theta, precision_type & prior_gradient,  precision_type & hessian, 
																 const precision_type & current_gradient, const precision_type & epsilon
																)
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, prior_gradient, hessian, current_gradient, epsilon); }


	// This overload is for adadelta
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name, 
													 precision_type & theta, precision_type & prior_gradient, precision_type & prior_delta, 
													 const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon
													)
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, prior_gradient, prior_delta, current_gradient, gamma, epsilon); }

	// This overload is for rms_prop
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name, 
													 precision_type & theta, precision_type & prior_gradient, 
													 const precision_type & current_gradient, const precision_type & eta, 
													 const precision_type & beta, const precision_type & epsilon)
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, prior_gradient, current_gradient, eta, beta, epsilon); }
	
	// This overload is for adamax
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name,
																 precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
																 const precision_type & current_gradient, const precision_type & beta_1_t, const precision_type & eta, 
																 const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
																)
	  
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, prior_mean, prior_variance, current_gradient, beta_1_t, eta, beta_1, beta_2, epsilon); }

	// This overload is for amsgrad
	template <class OPTIMIZER>
	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name,
																 precision_type & theta, 
																 precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
																 const precision_type & current_gradient, const precision_type & eta, 
																 const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
															    )
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, prior_mean, prior_variance, prior_bias_corrected_variance, current_gradient, eta, beta_1, beta_2, epsilon); }




	// This overload is for adam
	template <class OPTIMIZER>
  	  template <class precision_type>
	  CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name,
																 precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, const precision_type & current_gradient, 
																 const precision_type & beta_1_t, const precision_type & beta_2_t, const precision_type & eta, const precision_type & beta_1,
																 const precision_type & beta_2, const precision_type & epsilon
															    )	  
	  { static_cast<OPTIMIZER*>(this)->update(name, theta, prior_mean, prior_variance, current_gradient, beta_1_t, beta_2_t, eta, beta_1, beta_2, epsilon); }

	  // This overload is for nadam
	  template <class OPTIMIZER>
		template <class precision_type>
		CUDA_CALLABLE_MEMBER void optimizer<OPTIMIZER>::operator()(OPTIMIZER_NAME name,
													   precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, precision_type & current_gradient, 
													   const precision_type & eta, const precision_type & gamma, 
													   const precision_type & beta_1, const precision_type & beta_2,
													   const precision_type & beta_1_t, const precision_type & beta_2_t, 
													   const precision_type & epsilon
													  )
		{ static_cast<OPTIMIZER*>(this)->update(name, theta, prior_mean, prior_variance, current_gradient, eta, gamma, beta_1, beta_2, beta_1_t, beta_2_t, epsilon); };	

	// Stochastic gradient descent
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void stochastic_gradient_descent::update(OPTIMIZER_NAME name, precision_type & theta, const precision_type & gradient, const precision_type & eta)
		{ theta -=  ( eta * gradient ); }
    
	// Momentum
	  template <class precision_type>
		 CUDA_CALLABLE_MEMBER void momentum::update(OPTIMIZER_NAME name,
			                                        precision_type & theta, precision_type & prior_velocity,
												   	const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
												   )
		 {
   		   precision_type current_velocity{ gamma * prior_velocity + eta * current_gradient };
   		   theta -= current_velocity;
   		   prior_velocity = current_velocity;
   		 }

	// Nesterov momentum
  	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void nesterov_momentum::update(OPTIMIZER_NAME name, 
														  precision_type & theta, precision_type & prior_velocity, 
														  const precision_type & current_gradient, const precision_type & gamma, const precision_type & eta
														 )
		{
		  precision_type velocity { gamma * prior_velocity + eta * current_gradient * (  theta - gamma * prior_velocity) };
		  theta -= velocity;
		  prior_velocity = velocity;
		}

	// conjugate gradient descent
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void conjugate_gradient_descent::update(OPTIMIZER_NAME name, 
																	 precision_type & theta, precision_type & prior_gradient, precision_type & hessian, 
																	 const precision_type & current_gradient, const precision_type & epsilon
																	)
	  {
		precision_type gamma { ( current_gradient - prior_gradient ) * current_gradient / ( prior_gradient * prior_gradient + epsilon ) };
		precision_type step { current_gradient + ( gamma * hessian ) };
		theta += step;
		prior_gradient = current_gradient;
		hessian = step;
	  }
	
  	// adagrad	
  	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void adagrad::update(OPTIMIZER_NAME name, precision_type & theta, precision_type & prior_gradient, const precision_type & current_gradient, const precision_type & eta, const precision_type & epsilon)
		{
		  prior_gradient += current_gradient * current_gradient;
		  theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
		}
	
	  // adadelta	
	  template <class precision_type>	 
		CUDA_CALLABLE_MEMBER void adadelta::update(OPTIMIZER_NAME name, 
									   precision_type & theta,  precision_type & prior_gradient, precision_type & prior_delta,
									   const precision_type & current_gradient, const precision_type & gamma, const precision_type & epsilon)
		{
		  prior_gradient = gamma * prior_gradient + (1 - gamma) * current_gradient * current_gradient;
		  precision_type delta { -(sqrt(prior_delta + epsilon) / sqrt(prior_gradient + epsilon)) * current_gradient };
		  theta += delta;
		  prior_delta = gamma * prior_delta + (1 - gamma) * delta * delta;
		}
		
	// rms_prop
	  template <class precision_type>
		CUDA_CALLABLE_MEMBER void rms_prop::update(OPTIMIZER_NAME name,
	                                             precision_type & theta, precision_type & prior_gradient, 
												 const precision_type & current_gradient,  const precision_type & eta, 
												 const precision_type & beta, const precision_type & epsilon)
		{
		  prior_gradient = beta * prior_gradient + (1 - beta) * current_gradient * current_gradient;
		  theta -= eta * current_gradient / sqrt(prior_gradient + epsilon);
		}

	// adamax, the max operation is w.r.t the infinity norm
  	template <class precision_type>
	  CUDA_CALLABLE_MEMBER void adamax::update(OPTIMIZER_NAME name,
											   precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
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
 	  CUDA_CALLABLE_MEMBER void adam::update(OPTIMIZER_NAME name,
											 precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, 
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
	 	CUDA_CALLABLE_MEMBER void amsgrad::update(OPTIMIZER_NAME name,
												  precision_type & theta, 
												  precision_type & prior_mean, precision_type & prior_variance, precision_type & prior_bias_corrected_variance,
												  const precision_type & current_gradient, const precision_type & eta, 
												  const precision_type & beta_1, const precision_type & beta_2, const precision_type & epsilon
										         )
		{
	  	  prior_mean = beta_1 * prior_mean + (precision_type{1} - beta_1) * current_gradient;
		  prior_variance = beta_2 * prior_variance + (precision_type{1} - beta_2) * current_gradient * current_gradient;
		  // max(prior_variance > prior_bias_corrected_variance)
		  prior_bias_corrected_variance = (prior_variance > prior_bias_corrected_variance) ? prior_variance : prior_bias_corrected_variance;
		  theta -= eta * ( prior_mean ) / (sqrt( prior_bias_corrected_variance ) + epsilon ) ;
		 }
	  
	// nadam 
	template <class precision_type>	 
	  CUDA_CALLABLE_MEMBER void nadam::update(OPTIMIZER_NAME name, 
		                               precision_type & theta, precision_type & prior_mean, precision_type & prior_variance, precision_type & current_gradient, 
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
  }// END NAMESPACE OPTIMIZERS
}// END NAMESPACE ZINHART
