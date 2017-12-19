#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH
#include "typedefs.cuh"

#if CUDA_ENABLED == 1
#include <cublas_v2.h>
#else
#include <lapacke.h>
#endif


#if CUDA_ENABLED == 1
void dgemm_wrapper(cublasHandle_t & ref,std::uint32_t ORDERING, 
				   std::uint8_t & transA, std::uint8_t & transB, 
				   std::uint32_t & M, std::uint32_t & N, std::uint32_t & K, 
				   double & ALPHA, double * A, std::uint32_t & LDA,
				   double * B, std::uint32_t & LDB, double & BETA, 
				   double * C, std::uint32_t & LDC
				   );

void sgemm_wrapper(cublasHandle_t & ref,std::uint32_t ORDERING, 
				   std::uint8_t & transA, std::uint8_t & transB, 
				   std::uint32_t & M, std::uint32_t & N, std::uint32_t & K, 
				   float & ALPHA, float * A, std::uint32_t & LDA,
				   float * B, std::uint32_t & LDB, float & BETA, 
				   float * C, std::uint32_t & LDC
				   );
#else
void dgemm_wrapper(std::uint32_t ORDERING, 
				   std::uint8_t & transA, std::uint8_t & transB, 
				   std::uint32_t & M, std::uint32_t & N, std::uint32_t & K, 
				   double & ALPHA, double * A, std::uint32_t & LDA,
				   double * B, std::uint32_t & LDB, double & BETA, 
				   double * C, std::uint32_t & LDC
				   );

void sgemm_wrapper(std::uint32_t ORDERING, 
				   std::uint8_t & transA, std::uint8_t & transB, 
				   std::uint32_t & M, std::uint32_t & N, std::uint32_t & K, 
				   float & ALPHA, float * A, std::uint32_t & LDA,
				   float * B, std::uint32_t & LDB, float & BETA, 
				   float * C, std::uint32_t & LDC
				   );
#endif


#endif /* END MATRIX_OPS.H */
