#include "ann/layer.hh"
namespace zinhart
{
  pt call_activation(Layer & L, double & input, LAYER_NAME ln, ACTIVATION f)
  { return L(input, ln, f); } 
  pt call_activation(Layer & L, double & input, double & coefficient, LAYER_NAME ln, ACTIVATION f)
  { return L(input, coefficient, ln, f); }
}
