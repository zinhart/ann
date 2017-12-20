#include "ann/ann.hh"
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
using namespace zinhart;

TEST(ann_test,ann_test_constructor)
{
  ann< ffn > network;
}
TEST(ann_test, add_layer)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(1, std::numeric_limits<std::uint16_t>::max() );
  ann< ffn > network;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = dist(mt);  
  network.add_layer(a_layer);
}
TEST(ann_test, initialize_network)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(1, std::numeric_limits<std::uint16_t>::max());
  ann< ffn > network;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = dist(mt);  
  network.add_layer(a_layer);
  
  std::uint16_t case_size = dist(mt);
  std::pair<std::uint32_t, std::shared_ptr<float>>  total_observations, total_targets;
  std::pair<std::uint32_t, std::shared_ptr<double>>  total_hidden_weights;
  
  total_observations.first = dist(mt); //number of observations
  total_observations.second = std::shared_ptr<float> ( new float[total_observations.first], std::default_delete<float[]>() );//observations themselves 
  total_targets.first = dist(mt); // number of targets
  total_targets.second = std::shared_ptr<float> ( new float[total_targets.first], std::default_delete<float[]>() );//targets themselves 
  total_hidden_weights.first = dist(mt); // number of weights
  total_hidden_weights.second = std::shared_ptr<double> ( new double[total_hidden_weights.first], std::default_delete<double[]>() );// weights themselves
  ASSERT_EQ(initialize_network(network, case_size, total_observations, total_targets, total_hidden_weights), 0);
  ASSERT_EQ(cleanup(network), 0);
}

TEST(ann_test, ann_train)
{
  ann< ffn > network;
  train(network,1,2,3);
}
/*TEST(ann_test,ann_test_forward_prop)
{
  zinhart::ann< zinhart::ffn<zinhart::mse, int> > y;
//  y.forward_propagate();
}
TEST(ann_test,ann_test_back_prop)
{
  zinhart::ann< zinhart::ffn<zinhart::mse, int> > y;
  //y.backward_propagate();
}*/
