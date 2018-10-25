#include <ann/ann.hh>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
std::shared_ptr<zinhart::models::layers::layer<double>> get_random_layer(std::uint32_t layer_id, std::uint32_t layer_size)
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
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, 7);// does not include input layer
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  zinhart::models::ann_mlp<double> mlp;
  const std::uint32_t input_layer{0};
  std::uint32_t ith_layer{0}, n_layers{layer_dist(mt)};
  const std::uint32_t input_layer_size = neuron_dist(mt);

  mlp.add_layer( std::make_shared< zinhart::models::layers::input_layer<double> >() );
  mlp[0]->set_size(input_layer_size);
  ASSERT_EQ(input_layer_size, mlp[0]->get_size());

  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
	mlp.add_layer( get_random_layer(layer_dist(mt), neuron_dist(mt)) );
  ASSERT_EQ(mlp.size(), ith_layer + 1);// + 1 for input layer
}

TEST(ann_mlp, remove_layer)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, 10);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, 7);// does not include input layer
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);

  zinhart::models::ann_mlp<double> mlp;
  const std::uint32_t input_layer{0};
  std::uint32_t ith_layer{0}, n_layers{layer_dist(mt)};
  const std::uint32_t input_layer_size = neuron_dist(mt);

  mlp.add_layer( std::make_shared< zinhart::models::layers::input_layer<double> >() );
  mlp[0]->set_size(input_layer_size);
  ASSERT_EQ(input_layer_size, mlp[0]->get_size());

  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
	mlp.add_layer( get_random_layer(layer_dist(mt), neuron_dist(mt)) );
  ASSERT_EQ(mlp.size(), ith_layer + 1);// + 1 for input layer

  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	std::uint32_t current_size = mlp.size();
	mlp.remove_layer(ith_layer);
	ASSERT_EQ(current_size - 1, mlp.size());
  }
    
}

TEST(ann_mlp, init)
{
}

