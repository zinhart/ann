#include "ann/ann.hh"
//#include "ann/random_input.hh"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <memory>
using namespace zinhart;

std::random_device rd_ann;
std::mt19937 mt_ann(rd_ann());
std::uniform_int_distribution<unsigned int> pos_int_ann(0, std::numeric_limits<unsigned int>::max() );
std::uniform_real_distribution<double> pos_real_ann(0, std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> reals_ann(std::numeric_limits<double>::min(), std::numeric_limits<double>::max() );
std::uniform_real_distribution<double> neg_real_ann(std::numeric_limits<double>::min(), -1 );

TEST(ann_test,ann_test_constructor)
{
  ann< ffn > network;
}
TEST(ann_test, add_layer)
{
  ann< ffn > network;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = pos_int_ann(mt_ann);  
  network.add_layer(a_layer);
}
TEST(ann_test, set_case_info)
{
  ann< ffn > network;
  LAYER_INFO a_layer;
  a_layer.first = LAYER_NAME::IDENTITY;
  a_layer.second = pos_int_ann(mt_ann);  
  network.add_layer(a_layer);

  //case info declarations 
  std::uint32_t total_observations; // input layer size
  std::uint32_t total_targets; // output layer size
  std::uint32_t total_hidden_weights;
  
  std::uint32_t case_size = pos_int_ann(mt_ann);

  total_observations = pos_int_ann(mt_ann); //number of observations 
  total_targets = pos_int_ann(mt_ann);//number of targets
  total_hidden_weights = pos_int_ann(mt_ann);//number of hidden weights 

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
