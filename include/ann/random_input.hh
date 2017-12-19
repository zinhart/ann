#ifndef RANDOM_INPUT_HH
#define RANDOM_INPUT_HH
#include <random>
#include <limits>
namespace zinhart
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<unsigned int> pos_int(0, std::numeric_limits<unsigned int>::max() );
  std::uniform_real_distribution<double> pos_real(0, std::numeric_limits<double>::max() );
  std::uniform_real_distribution<double> reals(std::numeric_limits<double>::min(), std::numeric_limits<double>::max() );
  std::uniform_real_distribution<double> neg_real(std::numeric_limits<double>::min(), -1 );
}
#endif
