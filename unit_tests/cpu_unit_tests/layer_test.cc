#include <ann/ann.hh>
#include <ann/layer.hh>
#include <vector>
#include <gmock/gmock.h>
#include <random>
using namespace zinhart::models::layers;
TEST(layer_test, init_test)
{
  // declarations for random numbers
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(2, 10);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  const std::uint32_t alignment = 64;
  const std::uint32_t n_threads{thread_dist(mt)};
  std::uint32_t total_activations_length{neuron_dist(mt) * n_threads}, thread_activation_length{total_activations_length / n_threads}, i{0}, start{0}, stop{0};
  double * total_activations_ptr{nullptr};
  double * total_activations_ptr_test{nullptr};
  double * total_deltas_ptr{nullptr};
  double * total_deltas_ptr_test{nullptr};
  std::vector<layer<double>> total_layers;


  // allocate vectors
  total_activations_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_activations_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  total_deltas_ptr_test = (double*) mkl_malloc( total_activations_length * sizeof( double ), alignment );
  for(i = 0; i < total_activations_length; ++i)
  {
	total_activations_ptr[i] = real_dist(mt);
	total_activations_ptr_test[i] = real_dist(mt);
	total_deltas_ptr[i] = real_dist(mt);
	total_deltas_ptr_test[i] = real_dist(mt);
  }
  
  for(i = 0, start = 0, stop = thread_activation_length; stop != total_activations_length; ++i, start += thread_activation_length, stop += thread_activation_length)
  {	total_layers.emplace_back(layer<double>(start, stop, total_activations_ptr, total_deltas_ptr)); }

  mkl_free(total_activations_ptr);
  mkl_free(total_activations_ptr_test);
  mkl_free(total_deltas_ptr);
  mkl_free(total_deltas_ptr_test);
}
TEST(layer_test, activation_test_identity)// for each activaiton function
{
}
