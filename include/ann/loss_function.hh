#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H
#include "activation.hh"
#include "concurrent_routines/concurrent_routines.hh"
#include "function_space.hh"
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  namespace function_space
  {
	namespace error_metrics
	{
	  enum class LOSS_FUNCTION_NAME : std::uint32_t {MSE = std::uint32_t{0}, CROSS_ENTROPY_MULTI_CLASS};
	  enum class LOSS_FUNCTION_TYPE : std::uint32_t {OBJECTIVE = std::uint32_t{1}, DERIVATIVE};
	  class loss_function
	  {
		public:
		  CUDA_CALLABLE_MEMBER loss_function() = default;
		  CUDA_CALLABLE_MEMBER loss_function(const loss_function &) = default;
		  CUDA_CALLABLE_MEMBER loss_function(loss_function &&) = default;
		  CUDA_CALLABLE_MEMBER loss_function & operator = (const loss_function &) = default;
		  CUDA_CALLABLE_MEMBER loss_function & operator = (loss_function &&) = default;
		  CUDA_CALLABLE_MEMBER ~loss_function() = default;
		  // parallelizing the loop by splitting it into n sub loops
		  template <class precision_type, class container>
			HOST void operator()(LOSS_FUNCTION_NAME name, LOSS_FUNCTION_TYPE type, 
								 precision_type & error,
								 precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, std::uint32_t batch_size,
								 container  & results,
								 precision_type epsilon = 1.e-30, 
								 zinhart::parallel::thread_pool & pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
								);



		  // for parallelizing the entire loop
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator()(LOSS_FUNCTION_NAME name, OBJECTIVE label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, std::uint32_t batch_size);

		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator()(LOSS_FUNCTION_NAME name, DERIVATIVE label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, std::uint32_t batch_size);

		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator()(LOSS_FUNCTION_NAME name, OBJECTIVE label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths, precision_type epsilon = 1.e-30);

		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type operator()(LOSS_FUNCTION_NAME name, DERIVATIVE label, precision_type * outputs, precision_type * targets, std::uint32_t vector_lengths);

	  };
	  template <class LOSS_FUNCTION>
		class loss_function_interface : public loss_function
		{
		  public:
			CUDA_CALLABLE_MEMBER loss_function_interface() = default;
			CUDA_CALLABLE_MEMBER loss_function_interface(const loss_function_interface &) = default;
			CUDA_CALLABLE_MEMBER loss_function_interface(loss_function_interface &&) = default;
			CUDA_CALLABLE_MEMBER loss_function_interface & operator = (const loss_function_interface &) = default;
			CUDA_CALLABLE_MEMBER loss_function_interface & operator = (loss_function_interface &&) = default;
			CUDA_CALLABLE_MEMBER ~loss_function_interface() = default;
			
			template <class precision_type>
			  CUDA_CALLABLE_MEMBER precision_type operator()( OBJECTIVE label, precision_type kth_output, precision_type kth_target, std::uint32_t batch_size);
			template <class precision_type>
			  CUDA_CALLABLE_MEMBER precision_type operator()( DERIVATIVE label, precision_type kth_output, precision_type kth_target, std::uint32_t batch_size);

			template <class precision_type>
			  CUDA_CALLABLE_MEMBER precision_type operator()( OBJECTIVE label, precision_type kth_output, precision_type kth_target, precision_type epsilon);
			template <class precision_type>
			  CUDA_CALLABLE_MEMBER precision_type operator()( DERIVATIVE label, precision_type kth_output, precision_type kth_target);
		};

	  class mean_squared_error : public loss_function_interface<mean_squared_error>
	  {
		public:
		  mean_squared_error() = default;
		  mean_squared_error(const mean_squared_error &) = default;
		  mean_squared_error(mean_squared_error &&) = default;
		  mean_squared_error & operator = (const mean_squared_error &) = default;
		  mean_squared_error & operator = (mean_squared_error &&) = default;
		  ~mean_squared_error() = default;
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & kth_output, const precision_type & kth_target, std::uint32_t batch_size);
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & kth_output, const precision_type & kth_target, std::uint32_t batch_size);
	  };

	  class cross_entropy_multi_class : public loss_function_interface<cross_entropy_multi_class>
	  {
		public:
		  cross_entropy_multi_class() = default;
		  cross_entropy_multi_class(const cross_entropy_multi_class &) = default;
		  cross_entropy_multi_class(cross_entropy_multi_class &&) = default;
		  cross_entropy_multi_class & operator = (const cross_entropy_multi_class &) = default;
		  cross_entropy_multi_class & operator = (cross_entropy_multi_class &&) = default;
		  ~cross_entropy_multi_class() = default;
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type objective(const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon);
		  template <class precision_type>
			CUDA_CALLABLE_MEMBER precision_type derivative(const precision_type & kth_output, const precision_type & kth_target);
	  };


	  template <class precision_type>
		HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<zinhart::activation::LAYER_INFO> & total_layers, precision_type * device_total_outputs, std::uint32_t device_id = 0);

	  template <class precision_type>
		HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<zinhart::activation::LAYER_INFO> & total_layers, precision_type * device_total_outputs, const precision_type & epsilon = 1.e-30, std::uint32_t device_id = 0);

	}// END NAMESPACE ERROR_METRICS
  }// END NAMESPACE FUNCTION SPACE
}// END NAMESPACE ZINHART
#include "ext/loss_function.tcc"
#endif  

