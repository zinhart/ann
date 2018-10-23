#include <ann/optimizer.hh>
namespace zinhart
{  
  namespace optimizers
  {
#if CUDA_ENABLED == 1
	void optimize(optimizer<double> * const o, double * weights, const double * const gradient, const std::uint32_t & length);
#else
	
	void optimize(optimizer<double> * const o, double * weights, const double * const gradient, const std::uint32_t length, const std::uint32_t n_threads, const std::uint32_t thread_id)
	{
	  o->update(weights, gradient, length, n_threads, thread_id);
	}

	void optimize_m(const std::shared_ptr<optimizer<double>> & o, double * weights, const double * const gradient, const std::uint32_t & length, const std::uint32_t & n_threads, const std::uint32_t & thread_id)
	{
	  o->update(weights, gradient, length, n_threads, thread_id);
	}/**/
	
	
#endif
  }// END NAMESPACE OPTIMIZERS
}// END NAMESPACE ZINHART
