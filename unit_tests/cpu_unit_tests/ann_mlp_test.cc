#include <ann/ann.hh>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
std::shared_ptr<zinhart::models::layers::layer<double>> get_random_layer(std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > & total_layers, std::uint32_t layer_id, std::uint32_t layer_size)
{

  std::shared_ptr<zinhart::models::layers::layer<double>> layer;
  if(layer_id == 1)
  {
	layer = std::make_shared<zinhart::models::layers::identity_layer<double>>();
	layer->set_size(layer_size);
  } 
  else if(layer_id == 2)
  {
	layer = std::make_shared<zinhart::models::layers::sigmoid_layer<double>>();
	layer->set_size(layer_size);
  }
  else if(layer_id == 3)
  {	
	layer = std::make_shared<zinhart::models::layers::softplus_layer<double>>();
	layer->set_size(layer_size);
  }
  else if(layer_id == 4)
  {
	layer = std::make_shared<zinhart::models::layers::tanh_layer<double>>();
	layer->set_size(layer_size);
  }
  else if(layer_id == 5)
  {
	layer = std::make_shared<zinhart::models::layers::relu_layer<double>>();
	layer->set_size(layer_size);
  }
  else if(layer_id == 6)
  {
	layer = std::make_shared<zinhart::models::layers::leaky_relu_layer<double>>();
	layer->set_size(layer_size);
  }
  else if(layer_id == 7)
  {
	layer = std::make_shared<zinhart::models::layers::exp_leaky_relu_layer<double>>();
	layer->set_size(layer_size);

  }
  else if(layer_id == 8)
  {
	layer = std::make_shared<zinhart::models::layers::exp_leaky_relu_layer<double>>();
	layer->set_size(layer_size);

  }
 /* else if(layer_id == 9)
  {
  }*/
  return layer;
}
TEST(ann_mlp, add_layer)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, /*total_activation_types() - 1*/7);// does not include input layer
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);
}
TEST(ann_mlp, remove_layer)
{
}

