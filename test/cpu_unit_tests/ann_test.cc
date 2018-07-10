#include "ann/ann.hh"
#include "ann/models/multi_layer_perceptron.hh"
#include "ann/activation.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
#include <algorithm>

using namespace zinhart::models;
using namespace zinhart::activation;
TEST(ann_test, get_layer_add_layer_clear_layers)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(1, std::numeric_limits<std::uint16_t>::max() );// random number of neurons
  std::uniform_int_distribution<std::uint32_t> layer_dist(0, total_activation_types());// random activation function, -1 is to not include softmax as it's not finished
  std::uint32_t i, n_layers{layer_dist(mt)};
  ann<multi_layer_perceptron<double>, double> model;
  std::vector<LAYER_INFO> total_layers, total_layers_copy;
  for(i = 0; i < n_layers; ++i)
  {
	LAYER_INFO a_layer;
	a_layer.first = ACTIVATION_NAME(layer_dist(mt));
	a_layer.second = neuron_dist(mt);  
	total_layers.push_back(a_layer);
	model.add_layer(a_layer);
  }
  total_layers_copy = model.get_total_layers();
  ASSERT_EQ(total_layers.size(), total_layers_copy.size());
  ASSERT_EQ(total_layers_copy.size(), n_layers);
  for(std::uint32_t i = 0; i < total_layers.size(); ++i)
  {
	ASSERT_EQ(total_layers[i].first, total_layers_copy[i].first);
	ASSERT_EQ(total_layers[i].second, total_layers_copy[i].second);
  }
  model.clear_layers();
  total_layers_copy = model.get_total_layers();
  ASSERT_EQ(total_layers_copy.size(), 0);
}

TEST(ann_test, initialize_model_cleanup_model)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> neuron_dist(0,5000);
  std::uniform_int_distribution<std::uint32_t> layer_dist(1, total_activation_types());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 20);
  std::uint32_t total_activations, total_deltas, total_hidden_weights, total_gradients, total_bias;
  std::uint32_t ith_layer, n_layers{layer_dist(mt)}, n_threads{thread_dist(mt)};
  ann<multi_layer_perceptron<double>, double> model;
  std::vector<LAYER_INFO> total_layers, total_layers_copy;
  for(ith_layer = 0; ith_layer < n_layers; ++ith_layer)
  {
	LAYER_INFO a_layer;
	a_layer.first = ACTIVATION_NAME(layer_dist(mt));
	a_layer.second = neuron_dist(mt);  
	total_layers.push_back(a_layer);
	model.add_layer(a_layer);
  }
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  total_deltas = total_activations;
  
  //calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights += total_layers[ith_layer + 1].second * total_layers[ith_layer].second; 
  total_gradients = total_hidden_weights;
  total_bias = total_layers.size() - 1;
  
  model.init();

  ASSERT_EQ(total_activations, model.get_total_activations());
  ASSERT_EQ(total_deltas, model.get_total_deltas());
  ASSERT_EQ(total_hidden_weights, model.get_total_hidden_weights());
  ASSERT_EQ(total_gradients, model.get_total_gradients());
  ASSERT_EQ(total_bias, model.get_total_bias());

  model.cleanup();


  ASSERT_EQ(0, model.get_total_activations());
  ASSERT_EQ(0, model.get_total_deltas());
  ASSERT_EQ(0, model.get_total_hidden_weights());
  ASSERT_EQ(0, model.get_total_gradients());
  ASSERT_EQ(0, model.get_total_bias());
} 

/*
TEST(ann_test, forward_propagate)
{
}

TEST(ann_test, ann_train)
{
}*/



