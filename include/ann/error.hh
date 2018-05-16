#ifndef ERROR_HH
#define ERROR_HH
#include "typedefs.cuh"
#include <cublas_v2.h>
namespace zinhart
{
  HOST const char* cublasGetErrorString(cublasStatus_t status);
}
#endif
