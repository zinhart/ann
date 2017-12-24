#include "ann/ann.hh"
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
using namespace zinhart;

TEST(ann_test,ann_test_constructor)
{
  ann< ffn > model;
}
TEST(ann_test, add_layer)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(1, std::numeric_limits<std::uint16_t>::max() );
  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = dist(mt);  
  add_layer(model, a_layer);
}
TEST(ann_test, initialize_model)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(
  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::INPUT;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = LAYER_NAME::SIGMOID;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = LAYER_NAME::RELU;
  a_layer.second = dist(mt); 
  add_layer(model,a_layer);
  
  std::uint16_t case_size = dist(mt);
  std::tuple<std::uint32_t, std::uint16_t, std::shared_ptr<double>> total_observations;
  std::pair<std::uint32_t, std::shared_ptr<float>> total_targets;
  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t total_hidden_weights, total_activations, ith_layer, prior_layer_neurons;
  
  std::get<0>(total_observations) = dist(mt);//number of observations
  std::get<1>(total_observations) = dist(mt);//size of an observation
  std::get<2>(total_observations) = std::shared_ptr<double> ( new double[std::get<0>(total_observations) * std::get<1>(total_observations)], std::default_delete<double[]>() );//observations themselves 
 
  total_targets.first = dist(mt); // number of targets
  total_targets.second = std::shared_ptr<float> ( new float[total_targets.first], std::default_delete<float[]>() );//targets themselves 
    
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  
  //calc number of hidden weights
  for(ith_layer = 0, prior_layer_neurons = total_layers[0].second; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights += total_layers[ith_layer].second * (prior_layer_neurons + 1);//+ 1 for bias input
	prior_layer_neurons = total_layers[ith_layer].second;
  }
  ASSERT_EQ(initialize_model(model, total_observations, total_targets), 0);
  ASSERT_EQ(total_activations, get_total_activations(model).first);
  ASSERT_EQ(total_hidden_weights, get_total_hidden_weights(model).first);
  ASSERT_EQ(cleanup(model), 0);
}

TEST(ann_test, ann_train)
{
  //Random numbers will serve as random model configurations
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0,5000);// causes a bad alloc when appro > when a 3 layer model has > 5000 neurons in each //layer machine limitations :(

  ann< ffn > model;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::INPUT;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = LAYER_NAME::RELU;
  a_layer.second = dist(mt);  
  add_layer(model,a_layer);
  a_layer.first = LAYER_NAME::SOFTMAX;
  a_layer.second = dist(mt); 
  add_layer(model,a_layer);
  
  std::uint16_t case_size = dist(mt);
  std::tuple<std::uint32_t, std::uint16_t, std::shared_ptr<double>> total_observations;
  std::pair<std::uint32_t, std::shared_ptr<float>> total_targets;
  std::vector<LAYER_INFO> total_layers(get_total_layers(model));
  std::uint32_t total_hidden_weights, total_activations, ith_layer, prior_layer_neurons;
  
  std::get<0>(total_observations) = dist(mt);//number of observations
  std::get<1>(total_observations) = dist(mt);//size of an observation
  std::get<2>(total_observations) = std::shared_ptr<double> ( new double[std::get<0>(total_observations) * std::get<1>(total_observations)], std::default_delete<double[]>() );//observations themselves 
  total_targets.first = dist(mt); // number of targets
  total_targets.second = std::shared_ptr<float> ( new float[total_targets.first], std::default_delete<float[]>() );//targets themselves 
    
  //calc number of activations
  for(ith_layer = 1, total_activations = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations += total_layers[ith_layer].second;//accumulate neurons in the hidden layers and output layer
  
  //calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights = 0,prior_layer_neurons = total_layers[0].second; ith_layer < total_layers.size() - 1; ++ith_layer)
  {
	total_hidden_weights += total_layers[ith_layer].second * (prior_layer_neurons + 1);//+ 1 for bias input
	prior_layer_neurons = total_layers[ith_layer].second;
  }
  ASSERT_EQ(initialize_model(model, total_observations, total_targets), 0);
  ASSERT_EQ(total_activations, get_total_activations(model).first);
  ASSERT_EQ(total_hidden_weights, get_total_hidden_weights(model).first);
  ASSERT_EQ(train(model,dist(mt),dist(mt), dist(mt)), 0);
  ASSERT_EQ(cleanup(model), 0);
}
