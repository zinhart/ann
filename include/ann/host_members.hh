#ifndef HOST_MEMBERS_HH
#define HOST_MEMBERS_HH
#include "typedefs.cuh"
#include "layer.hh"
//#include "ffn.cuh"
#include <memory>
namespace zinhart
{
  struct Host_Members
  {
	short num_layers;
	std::uint32_t total_hidden_weights;
	std::uint32_t total_output_weights;
	std::uint32_t total_cases;
	std::uint32_t case_size;
	std::uint32_t total_targets;
	Layer_Wrapper layer_wrapper;
	std::shared_ptr<float> data = nullptr;
	std::shared_ptr<float> targets = nullptr;
	std::shared_ptr<float**> hidden_weights = nullptr;
    std::shared_ptr<float*> h_wout = nullptr;
	std::shared_ptr<double**> h_act = nullptr;
	std::shared_ptr<double*> h_output = nullptr;
	std::shared_ptr<float*> h_loss_out = nullptr;
	std::shared_ptr<double*> current_layer_delta = nullptr;
	std::shared_ptr<double*> prior_layer_delta = nullptr;
	int grad_len;
	std::shared_ptr<float*> complete_gradient = nullptr;
	std::shared_ptr<float**> gradient_ptr = nullptr;
/*	float * data = nullptr;
	float * targets = nullptr;
	float ** h_whid = nullptr;
	float * h_wout = nullptr;
	double ** h_act = nullptr;
	double * h_output = nullptr;
	float * h_mse_out = nullptr;
	double * h_current_layer_delta = nullptr;
	double * h_prior_layer_delta = nullptr;
	int h_gradlen;
	float * h_gradient = nullptr;
	float ** h_grad_ptr = nullptr;*/
  	int allocate( int training_set_rows, int training_case_inputs, int training_set_cols, double * training_data, int num_targets,
			double * target_vector, int max_batch, /*Layer_Container & lc,*/ char * error_msg[256] )
	  {

		  int i, j, n, n_total, n_max, n_prior, memsize;
//		  float *gptr, *fptr[lc.size()];
//		  double * dptr[lc.size()];
		 // char msg[256];
#if CUDA_ENABLED == 1
		  cudaError_t error_id;
		  loss_function<mse> apl;
		  apl(1,2, loss_function<mse>::OBJECTIVE);
		  Layer<Identity_Layer> l(5);
		  l.set_neurons(5);
		  std::cout<<"CUDA IS ENABLED IN THIS CONTEXT\n";
		  error_id = cudaSetDevice(0);
		  if(error_id != cudaSuccess)
		  {
//		   	sprintf(error_msg, 255, "CUDA init set device failed %d: %s", error_id , cudaGetErrorString(error_id)); 
//			return ERROR_CUDA_ERROR;
		  }
		  /*
			 Constant
		  */
		  // cudaMemcpyToSymbol(d_ncases, &training_set_rows, sizeof(int),0, cudaMemcpyHostToDevice);
		 // cudaMemcpyToSymbol(d_n_trn_inputs, &training_case_inputs, sizeof(int),0, cudaMemcpyHostToDevice);
		 // cudaMemcpyToSymbol(d_n_targets, &num_targets, sizeof(int),0, cudaMemcpyHostToDevice);
//		  memsize = (lc.size() - 1) * sizeof(int);
//		  total_memory += memsize;
/*

   Data - We must extract only the first n_inputs columns from the ncols columns in data

*/

/*		 fdata = (float *) malloc( ncases * n_inputs * sizeof(float) ) ;
		  //cudaMallocPitch()
		  if (fdata == NULL)
			return ERROR_INSUFFICIENT_MEMORY ;
		  memsize = ncases * n_inputs * sizeof(float) ;
		  total_memory += memsize ;
		  error_id = cudaMalloc ( (void **) &h_trn_data , (size_t) memsize ) ;
		  sprintf_s ( msg, 255 , "CUDA MALLOC data = %llx  (%d bytes, total=%.2lf MB)", (unsigned long long) h_trn_data, memsize, total_memory / (1024 * 1024) ) ;
//		  cudalog ( msg ) ;
	   	  if (error_id  !=  cudaSuccess) 
		  {
	  		sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc data (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
	  		return ERROR_CUDA_MEMORY ;
		  }
	   	  for (i=0 ; i<ncases ; i++) 
		  {
			for (j=0 ; j<n_inputs ; j++)
			  fdata[i*n_inputs+j] = (float) training_data[i*ncols+j] ;
		  }



   error_id = cudaMemcpy ( h_trn_data , fdata , ncases * n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;

   FREE ( fdata ) ;

   fdata = NULL ;



   if (error_id == cudaSuccess)

      error_id = cudaMemcpyToSymbol ( d_trn_data , &h_trn_data , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;



   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad data copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_ERROR ;

      }*/

/*

   Targets

*/



/*   fdata = (float *) MALLOC ( ncases * ntarg * sizeof(float) ) ;

   if (fdata == NULL)

      return ERROR_INSUFFICIENT_MEMORY ;



   memsize = ncases * ntarg * sizeof(float) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_targets , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC targets = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_targets, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc targets (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   for (i=0 ; i<ncases ; i++) {

      for (j=0 ; j<ntarg ; j++)

         fdata[i*ntarg+j] = (float) targets[i*ntarg+j] ;

      }



   error_id = cudaMemcpy ( h_targets , fdata , ncases * ntarg * sizeof(float) , cudaMemcpyHostToDevice ) ;

   FREE ( fdata ) ;

   fdata = NULL ;



   if (error_id == cudaSuccess)

      error_id = cudaMemcpyToSymbol ( d_targets , &h_targets , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;



   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad targets copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_ERROR ;

      }*/


/*

   Classes if this is a classifier

*/


/*
   if (classifier) {

      memsize = ncases * sizeof(int) ;

      total_memory += memsize ;

      error_id = cudaMalloc ( (void **) &h_class , (size_t) memsize ) ;

      sprintf_s ( msg, 255 , "CUDA MALLOC class = %llx  (%d bytes, total=%.2lf MB)",

                  (unsigned long long) h_class, memsize, total_memory / (1024 * 1024) ) ;

      cudalog ( msg ) ;

      if (error_id  !=  cudaSuccess) {

         sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc class (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

         return ERROR_CUDA_MEMORY ;

         }



      error_id = cudaMemcpy ( h_class , class_ids , ncases * sizeof(int) , cudaMemcpyHostToDevice ) ;



      if (error_id == cudaSuccess)

         error_id = cudaMemcpyToSymbol ( d_class , &h_class , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;



      if (error_id  !=  cudaSuccess) {

         sprintf_s ( error_msg , 255 , "CUDA init bad class copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;

         return ERROR_CUDA_ERROR ;

         }

      }*/



/*

   Activations

*/



/*   n_total = 0 ;

   for (i=0 ; i<n_layers-1 ; i++)

      n_total += nhid[i] ;



   memsize = n_total * max_batch * sizeof(double) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &activations , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC activations = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) activations, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc activations (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   memsize = (n_layers-1) * sizeof(void *) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_act , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC act = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_act, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc act (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   cudaMemcpyToSymbol ( d_act , &h_act , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;



   n_total = 0 ;

   for (i=0 ; i<n_layers-1 ; i++) {

      dptr[i] = activations + n_total * max_batch ;

      n_total += nhid[i] ;

      }



   error_id = cudaMemcpy ( h_act , &dptr[0] , (n_layers-1) * sizeof(void *) , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad act ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_ERROR ;

      }*/

/*

   Output activations

*/


/*   memsize = ncases * ntarg * sizeof(double) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_output , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC output = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_output, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  ==  cudaSuccess)

      error_id = cudaMemcpyToSymbol ( d_output , &h_output , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc output (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }*/

/*

   Hidden layer weights

*/

/*   n_total = 0 ;

   n_prior = n_inputs ;

   for (i=0 ; i<n_layers-1 ; i++) {

      n_total += nhid[i] * (n_prior + 1) ;

      n_prior = nhid[i] ;

      }



   n_hid_weights = n_total ;



   memsize = n_total * sizeof(float) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &hidden_weights , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC hidden_weights = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) hidden_weights, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hidden_weights (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   memsize = (n_layers-1) * sizeof(float *) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_whid , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC whid = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_whid, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc whid (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   cudaMemcpyToSymbol ( d_whid , &h_whid , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;



   n_total = 0 ;

   n_prior = n_inputs ;

   for (i=0 ; i<n_layers-1 ; i++) {

      fptr[i] = hidden_weights + n_total ;

      n_total += nhid[i] * (n_prior + 1) ;

      n_prior = nhid[i] ;

      }



   error_id = cudaMemcpy ( h_whid , &fptr[0] , (n_layers-1) * sizeof(float *) , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad whid ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_ERROR ;

      }*/

/*

   Output weights

*/


/*   n_out_weights = ntarg * (nhid[n_layers-2]+1) ;

   memsize = n_out_weights * sizeof(float) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_wout , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC wout = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_wout, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  ==  cudaSuccess)

      error_id = cudaMemcpyToSymbol ( d_wout , &h_wout , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc wout (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }*/

/*

   This delta, next delta

*/


/*   n_max = ntarg ;

   for (i=1 ; i<n_layers-1 ; i++) {  // We do not store delta for first hidden layer, so skip 0

      if (nhid[i] > n_max)

         n_max = nhid[i] ;

      }



   memsize = n_max * max_batch * sizeof(double) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_this_delta , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC this_delta = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_this_delta, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  ==  cudaSuccess)

      error_id = cudaMemcpyToSymbol ( d_this_delta , &h_this_delta , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc this_delta (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   memsize = n_max * max_batch * sizeof(double) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_prior_delta , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC prior_delta = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_prior_delta, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  ==  cudaSuccess)

      error_id = cudaMemcpyToSymbol ( d_prior_delta , &h_prior_delta , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc prior_delta (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }*/

/*

   Gradient (all layers, including output); grad_ptr

*/

/*   h_gradlen = 0 ;

   n_prior = n_inputs ;

   for (i=0 ; i<n_layers-1 ; i++) {

      h_gradlen += nhid[i] * (n_prior + 1) ;

      n_prior = nhid[i] ;

      }

   h_gradlen += ntarg * (n_prior + 1) ;

   cudaMemcpyToSymbol ( d_gradlen , &h_gradlen , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;



   memsize = h_gradlen * max_batch * sizeof(float) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_gradient , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC h_gradient = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_gradient, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc h_gradient (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   cudaMemcpyToSymbol ( d_gradient , &h_gradient , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;



   memsize = n_layers * sizeof(float *) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_grad_ptr , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC grad_ptr = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_whid, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc grad_ptr (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }



   cudaMemcpyToSymbol ( d_grad_ptr , &h_grad_ptr , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;



   gptr = h_gradient ;

   for (i=0 ; i<n_layers ; i++) {

      fptr[i] = gptr ;



      if (i == 0) {                        // First hidden layer?

         n = nhid[i] * (n_inputs+1) ;

         gptr += n ;

         }



      else if (i < n_layers-1) {           // Subsequent hidden layer?

         n = nhid[i] * (nhid[i-1]+1) ;

         gptr += n ;

         }

      }



   error_id = cudaMemcpy ( h_grad_ptr , &fptr[0] , n_layers * sizeof(void *) , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad grad_ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_ERROR ;

      }*/
/*

   MSE reduction stuff

*/



/*   memsize = REDUC_BLOCKS * sizeof(float) ;

   total_memory += memsize ;

   error_id = cudaMalloc ( (void **) &h_mse_out , (size_t) memsize ) ;

   sprintf_s ( msg, 255 , "CUDA MALLOC mse_out = %llx  (%d bytes, total=%.2lf MB)",

               (unsigned long long) h_mse_out, memsize, total_memory / (1024 * 1024) ) ;

   cudalog ( msg ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc mse_out (%d): %s", error_id, cudaGetErrorString(error_id) ) ;

      return ERROR_CUDA_MEMORY ;

      }

   cudaMemcpyToSymbol ( d_mse_out , &h_mse_out , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;



   MEMTEXT ( "CUDA init reduc_fdata" ) ;

   reduc_fdata = (float *) MALLOC ( REDUC_BLOCKS * sizeof(float) ) ;

   if (reduc_fdata == NULL) {

      sprintf_s ( error_msg , 255 , "CUDA init bad MALLOC reduc_fdata" ) ;

      return ERROR_INSUFFICIENT_MEMORY ;  // New error return

      }*/

/*

   Allocate fdata large enough to handle all subsequent double <-> float transactions

*/



/*   fdata = (float *) MALLOC ( h_gradlen * sizeof(float) ) ;

   if (fdata == NULL)

      return ERROR_INSUFFICIENT_MEMORY ;*/





/*

   Set cache/shared memory preferences

*/



/*   error_id = cudaFuncSetCacheConfig ( device_hidden_activation , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_output_activation , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_output_delta , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_softmax_delta , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_output_gradient , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_first_hidden_gradient , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_subsequent_hidden_gradient , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_move_delta , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_softmax , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_fetch_gradient , cudaFuncCachePreferL1 ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_mse  , cudaFuncCachePreferNone ) ;

   if (error_id == cudaSuccess)

      error_id = cudaFuncSetCacheConfig ( device_ll  , cudaFuncCachePreferNone ) ;

   if (error_id  !=  cudaSuccess) {

      sprintf_s ( error_msg , 255 , "CUDA init bad cudaFuncSetCacheConfig" ) ;

      return ERROR_CUDA_ERROR ;

      }



   MEMTEXT ( "MLFN.cu: mlfn_cuda_init finished" ) ;

   return 0 ;*/
#else
   std::cout<<"CUDA IS NOT ENABLED IN THIS CONTEXT\n";
#endif
	  }
	int deallocate();
  };
}
#endif
