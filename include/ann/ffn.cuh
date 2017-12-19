#ifndef FFN_CUH
#define FFN_CUH
#include "typedefs.cuh"
#include <cuda_runtime_api.h>
namespace zinhart
{
  float *reduc_fdata = nullptr;
  CONSTANT int d_n_cases;
  CONSTANT int d_n_trn_inputs;
  CONSTANT int d_n_targets;
  CONSTANT float * d_trn_data = nullptr;
  CONSTANT float * d_targets = nullptr;
  CONSTANT float ** d_whid = nullptr;
  CONSTANT float * d_wout = nullptr;
  CONSTANT double ** d_act = nullptr;
  CONSTANT double * d_output = nullptr;
  CONSTANT float * d_mse_out = nullptr;
  CONSTANT double * d_current_layer_delta = nullptr;
  CONSTANT double * d_prior_layer_delta = nullptr;
  CONSTANT int d_gradlen;
  CONSTANT float * d_gradient = nullptr;
  CONSTANT float **d_grad_ptr = nullptr;

  template <class Layer_Type>
	__global__ void device_hidden_activation ( 
		int istart, 
		int istop, 
		int ilayer,
	    int d_n_trn_inputs,	
		float * d_whid, 
		float * d_act, 
		float * d_nhid, 
		float * d_trn_data, 
		Layer<Layer_Type> activation_function )
	{
	  int icase, ihid, i_input, n_inputs, nhid;
	  float * f_inptr, * wptr;
	  double sum, * actptr, * d_inptr;

	  ihid = blockIdx.x * blockDim.x + threadIdx.x;
	  nhid = d_nhid[ilayer];
	  if(ihid >= nhid)
		return;

	  icase = blockIdx.y;

	  wptr = d_whid[ilayer];
	  actptr = d_act[ilayer];
	  sum = 0.0;

	  if(ilayer == 0)
	  {
		n_inputs = d_n_trn_inputs;
		f_inptr = d_trn_data + (icase +istart) * n_inputs;
		for(i_input = 0; i_input < n_inputs; ++i_input)
		  sum += wptr[i_input * nhid + ihid] * f_inptr[i_input];
		sum += w_ptr[n_inputs * nhid + ihid]; //bias and apparently assumes a bias of 1
	  }
	  else
	  {
		n_inputs = d_nhid[ilayer - 1];
		d_inptr = d_act[ilayer - 1] + icase * n_inputs;
		for(i_input = 0; i_input < n_inputs; ++i_input)
		  sum += wptr[i_input * nhid + ihid] * d_inptr[i_input];
		sum += wptr[n_inputs * nhid + ihid]; //bias and apparently assumes a bias of 1
	  }
	  actptr[icase * nhid + ihid] = activation_function(sum, Layer::OBJECTIVE);
	}
  //template <class Layer_Type>
	__global__ void device_output_activation ( 
		int istart, 
		int n_inputs, 
		int ilayer, 
		int d_ntarg,
		float * d_act, 
		float * d_wout,
		float * d_output
		)
	{
	  int icase, iout, i_input;
	  double sum, *inptr;

	  iout = blockIdx.x * blockDim.x + threadIdx.x;

	  if(iout >= d_n_targets)
		return;
	  icase = blockIdx.y; 
	  
	  //inptr = d_act[ilayer] + icase * n_inputs;
	  sum = 0.0;

	  for(i_input = 0; i_input < n_inputs; ++i_input)
		sum += d_wout[i_input * d_n_targets + iout] * inptr[i_input];
	  sum += d_wout[n_inputs * d_n_targets + iout];//bias

	  d_output[(icase + istart) * d_n_targets + iout] = sum;
	}
  template <class Layer_Type>
	__global__ void device_softmax ( int istart , int istop, int d_ntarg, float * d_output, Layer<Layer_Type> activation_function)
	{
	  int icase, iout;
	  double * outptr, sum;
	  
	}
  template <class error_metric>
	__global__ void device_output_delta( 
		int istart, 
		int istop,
	   	int ntarg,
	   	int d_ntarg,
	   	double * d_this_delta,
		double * d_targets,
		double * d_outputs,
	   	loss_function<error_metric> loss)
	{
	  int icase, iout;
	  iout = blockIdx.x * blockDim.x + threadIdx.x;
	  if(iout >= d_n_targets)
		return;
	  icase = blockIdx.y;
	  d_this_delta[icase * ntarg + iout] = loss(d_targets[(icase + start) * ntarg + iout ], d_outputs[(icase + start) * ntarg + iout ], loss_function::DELTA); 
	}
  __global__ void device_output_gradient (int nc,
	  int ilayer,
	  int d_ntarg,
	  int d_gradlen,
	  int * d_nhid,// for each hidden layer
	  float ** d_act,
	  float ** d_grad_ptr,
	  double * d_this_delta
	  )
  {
	int icase, iout, ihid, nhid;
	float *gptr;
	double input;
	ihid = blockIdx.x * blockDim.x + threadIdx.x;
	nhid = d_nhid[ilayer];
	icase = blockIdx.y;

	if(ihid > nhid)
	  return;
	else if(ihid < nhid)
	  input = d_act[ilayer][icase * nhid + ihid];
	else
	  input = 1.0;
	iout = blockIdx.z;

	gptr = d_grad_ptr[ilayer + 1] + icase * d_gradlen;
	gptr[iout * (nhid + 1) + ihid ] = d_this_delta[icase * d_n_targets + iout] * input;
  }
  //check from here
  
  __global__ void device_first_hidden_gradient (
	  int istart,       // First case in this batch
	  int istop,        // One past last case
	  int only_hidden    // Is this the only hidden layer?
   )
{
  int j, icase, iin, ihid, nhid, ninp1, n_next ;
  float *gptr, *next_weights, input ;
  double *delta_ptr, this_act, delta ;
  iin = blockIdx.x * blockDim.x + threadIdx.x ;
  icase = blockIdx.y ;
  if (iin > d_n_trn_inputs)
	return ;
  else if (iin < d_n_trn_inputs)
	input = d_trn_data[(icase+istart)*d_n_trn_inputs+iin] ;  // Feed coming into this layer
  else
	input = 1.0f ;             // Bias
  ihid = blockIdx.z ;
  //nhid = d_nhid[0] ;            // Neurons in this hidden layer
  ninp1 = d_n_trn_inputs + 1 ;  // We mustn't forget the bias
  if (only_hidden) 
  {
//	n_next = d_n_targets ;
	next_weights = d_wout + ihid * n_next ;
  }
  else 
  {
	//n_next = d_nhid[1] ;
	next_weights = d_whid[1] + ihid * n_next;
  }
 // delta_ptr = d_this_delta + icase * n_next ; // Delta for this case
  delta = 0.0 ;
  for (j=0 ; j<n_next ; j++)
	delta += delta_ptr[j] * next_weights[j] ;
  this_act = d_act[0][icase*nhid+ihid] ;
  delta *= this_act * (1.0 - this_act) ;//change this to the activation function of the current layer type
  gptr = d_grad_ptr[0] + icase * d_gradlen ;  // Gradient of first hidden layer
  gptr[ihid*ninp1+iin] = delta * input ;
}
  __global__ void device_subsequent_hidden_gradient ( int nc , int ilayer , int last_hidden )
  {
	int j, icase, iin, ihid, nhid, nin, ninp1, n_next ;
	float *gptr, *next_weights ;
 	double *delta_ptr, *prior_delta_ptr, this_act, delta, input ;
  	iin = blockIdx.x * blockDim.x + threadIdx.x ;
   	icase = blockIdx.y ;
  // 	nin = d_nhid[ilayer-1] ;      // Number of inputs to each neuron in this layer
   	if (iin > nin)
   	  return ;
   	else if (iin < nin)
   	  input = d_act[ilayer-1][icase*nin+iin] ;
   	else
   	  input = 1.0 ;              // Bias
 	ihid = blockIdx.z ;
  	//nhid = d_nhid[ilayer] ;       // Neurons in this hidden layer
   	ninp1 = nin + 1 ;             // We mustn't forget the bias, so nin+1
   	if (last_hidden) 
	{
	//  n_next = d_n_targets ;
	  next_weights = d_wout + ihid * n_next ;
  	}
   	else 
	{
	  //n_next = d_nhid[ilayer+1] ;
	  next_weights = d_whid[ilayer+1] + ihid * n_next ;
  	}
   	//delta_ptr = d_this_delta + icase * n_next ;      // Coming from the next layer, which was just done
   	//prior_delta_ptr = d_prior_delta + icase * nhid ; // Save for the next layer done, one layer back
   	delta = 0.0 ;
   	for (j=0 ; j<n_next ; j++)
   	  delta += delta_ptr[j] * next_weights[j] ;
   	this_act = d_act[ilayer][icase*nhid+ihid] ;
   	delta *= this_act * (1.0 - this_act) ;//change this to the activation function of the current layer type 
   	prior_delta_ptr[ihid] = delta ;            // Save it for the next layer back
 	gptr = d_grad_ptr[ilayer] + icase * d_gradlen ;  // Gradient of this hidden layer
  	gptr[ihid*ninp1+iin] = delta * input ;

  }
__global__ void device_move_delta ( int nhid )
{
  int icase, ihid ;
  ihid = blockIdx.x * blockDim.x + threadIdx.x ;
  if (ihid >= nhid)
	return ;
  icase = blockIdx.y ;
  //d_this_delta[icase*nhid+ihid] = d_prior_delta[icase*nhid+ihid] ;
}
__global__ void device_fetch_gradient ( int nc )
{
  int index, icase ;
  float *gptr ;
  double sum ;
  index = blockIdx.x * blockDim.x + threadIdx.x ;
  if (index >= d_gradlen)
	return ;
  sum = 0.0 ;
  gptr = d_gradient + index ;
  for (icase=0 ; icase<nc ; icase++)   // For all cases in this batch
	sum += gptr[icase*d_gradlen] ;
  *gptr = sum ;
}
  template <class error_metric>
__global__ void device_mse (loss_function<error_metric> loss)
{
  __shared__ double partial_mse[REDUC_THREADS] ;
  int i, index ;
  unsigned int n ;
  double diff, sum_mse ;
  index = threadIdx.x ;
  n = d_ncases * d_n_targets ;
  sum_mse = 0.0 ;   
  for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) 
  {
	diff = d_output[i] - d_targets[i] ;
	sum_mse += diff * diff ;
  }
  partial_mse[index] = sum_mse ;
  __syncthreads() ;
  for (i=blockDim.x>>1 ; i ; i>>=1) 
  {
	if (index < i)
	  partial_mse[index] += partial_mse[index+i] ;
	__syncthreads() ;
  }
  if (index == 0)
	d_mse_out[blockIdx.x] = partial_mse[0] ;
}
  __global__ void device_ll ();

  __global__ void device_softmax_delta ( int istart , int istop , int ntarg ) ;
}
#endif
