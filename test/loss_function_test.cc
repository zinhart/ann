#include <gtest/gtest.h>
#include "ann/loss_function.hh"
TEST(loss_function_test, cross_entropy_multi_class)
{
}
TEST(loss_function_test, mean_square_error)
{
  zinhart::loss_function<zinhart::mse> a;
  a(1,2,zinhart::loss_function<zinhart::mse>::OBJECTIVE);
}
