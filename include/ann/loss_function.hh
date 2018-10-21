#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H
//#include <ann/activation.hh>
#include <multi_core/multi_core.hh>
#include <ann/function_space.hh>
#if CUDA_ENABLED == 1
#include <math.h>
#else
#include <cmath>
#endif
namespace zinhart
{
  namespace loss_functions
  {
	namespace loss_attributes
	{
	  enum mean_squared_error : std::uint32_t;
	  enum cross_entropy_multi_class : std::uint32_t;
      // grouped into one type for conveniece
	  union loss_function_type
	  {
		mean_squared_error mse;
		cross_entropy_multi_class ce;
	  };
	}

  template <class precision_type>
	class error_function
	{
	  public:
		HOST error_function() = default;
		HOST error_function(const error_function&) = default;
		HOST error_function(error_function&&) = default;
		HOST error_function & operator = (const error_function&) = default;
		HOST error_function & operator = (error_function&&) = default;
		HOST ~error_function() = default;

		HOST precision_type error(loss_attributes::mean_squared_error mse, zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length);
		HOST void error(loss_attributes::mean_squared_error mse, zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length, const std::uint32_t & output_size);

		HOST precision_type error(loss_attributes::cross_entropy_multi_class ce, zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length, const precision_type & epsilon = 1.e-30);
		HOST void error(loss_attributes::cross_entropy_multi_class ce, zinhart::function_space::derivative d,const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length);



		CUDA_CALLABLE_MEMBER precision_type objective(loss_attributes::mean_squared_error mse, const precision_type & kth_output, const precision_type & kth_target);
		CUDA_CALLABLE_MEMBER precision_type derivative(loss_attributes::mean_squared_error mse, const precision_type & kth_output, const precision_type & kth_target, const std::uint32_t & output_size);

		CUDA_CALLABLE_MEMBER precision_type objective(loss_attributes::cross_entropy_multi_class ce, const precision_type & kth_output, const precision_type & kth_target, const precision_type & epsilon = 1.e-30);
		CUDA_CALLABLE_MEMBER precision_type derivative(loss_attributes::cross_entropy_multi_class ce, const precision_type & kth_output, const precision_type & kth_target);
	};

  template <class precision_type>
	class loss_function
	{
	  public:
		HOST virtual ~loss_function() = default;
		error_function<precision_type> e;
		HOST virtual precision_type error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length) = 0;
		HOST virtual void error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length) = 0;
	};
  template <class precision_type>
	class mean_squared_error : public loss_function<precision_type>
	{
	  public:
		mean_squared_error() = default;
		mean_squared_error(const mean_squared_error &) = default;
		mean_squared_error(mean_squared_error &&) = default;
		mean_squared_error & operator = (const mean_squared_error &) = default;
		mean_squared_error & operator = (mean_squared_error &&) = default;
		~mean_squared_error() = default;
		std::uint32_t batch_size{2};
		using loss_function<precision_type>::e;
		HOST virtual precision_type error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length) override;
		HOST virtual void error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length) override;
	};

  template <class precision_type>
	class cross_entropy_multi_class : public loss_function<precision_type>
	{
	  public:
		cross_entropy_multi_class() = default;
		cross_entropy_multi_class(const cross_entropy_multi_class &) = default;
		cross_entropy_multi_class(cross_entropy_multi_class &&) = default;
		cross_entropy_multi_class & operator = (const cross_entropy_multi_class &) = default;
		cross_entropy_multi_class & operator = (cross_entropy_multi_class &&) = default;
		~cross_entropy_multi_class() = default;
		precision_type epsilon{1.e-30};
		using loss_function<precision_type>::e;
		HOST virtual precision_type error(zinhart::function_space::objective o, const precision_type * outputs, const precision_type * targets, const std::uint32_t & length) override;
		HOST virtual void error(zinhart::function_space::derivative d, const precision_type * outputs, const precision_type * targets, precision_type * results, const std::uint32_t & length) override;
	};
  }// END NAMESPACE LOSS_FUNCTIONS
/*
	  template <class precision_type>
		HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<zinhart::activation::LAYER_INFO> & total_layers, precision_type * device_total_outputs, std::uint32_t device_id = 0);

	  template <class precision_type>
		HOST std::int32_t call_loss_function(const LOSS_FUNCTION_NAME loss_function_name, const LOSS_FUNCTION_TYPE loss_function_type, const std::vector<zinhart::activation::LAYER_INFO> & total_layers, precision_type * device_total_outputs, const precision_type & epsilon = 1.e-30, std::uint32_t device_id = 0);*/

}// END NAMESPACE ZINHART
#include "ext/loss_function.tcc"
#endif  

