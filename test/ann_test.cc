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
  std::uniform_int_distribution<std::uint32_t> dist(1, std::numeric_limits<std::uint16_t>::max());
  ann< ffn > network;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = dist(mt);  
  network.add_layer(a_layer);
}
TEST(ann_test, set_case_info)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> dist(1, std::numeric_limits<std::uint16_t>::max());
  ann< ffn > network;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = dist(mt);  
  network.add_layer(a_layer);

  //case info declarations 
  std::uint32_t total_observations; // input layer size
  std::uint32_t total_targets; // output layer size
  std::uint32_t total_hidden_weights;
  
  std::uint16_t case_size = dist(mt);

  total_observations = dist(mt);//number of observations 
  total_targets = dist(mt);//number of targets
  total_hidden_weights = dist(mt);//number of hidden weights 

  set_case_info(network, total_observations, total_targets, total_hidden_weights, case_size);
		  
}
/*
TEST(ann_test, ann_test_train)
{
  ann< ffn<mse, optimizer<sgd, SGD > > > network;
  LAYER_INFO a_layer;
  a_layer.first = IDENTITY();
  a_layer.second = 4;  
  network.add_layer(a_layer);
  network.train(1,2,3);
}*/
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
