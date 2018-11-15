#include <ann/ann.hh>
#include <bmp/bmp.hh>



// allocate everything based on information in total_layers
template <class precision_type>
  void init(const std::vector< std::shared_ptr<zinhart::models::layers::layer<precision_type>> > & total_layers,
			std::uint32_t & total_activations_length,
			std::uint32_t & total_hidden_weights_length,
		    const std::uint32_t n_threads
		   )
{
  std::uint32_t ith_layer{0};
  // calc number of activations
  for(ith_layer = 1, total_activations_length = 0; ith_layer < total_layers.size(); ++ith_layer )
	total_activations_length += total_layers[ith_layer]->get_size();//accumulate neurons in the hidden layers and output layer
  total_activations_length *= n_threads;

  // calc number of hidden weights
  for(ith_layer = 0, total_hidden_weights_length = 0; ith_layer < total_layers.size() - 1; ++ith_layer)
	total_hidden_weights_length += total_layers[ith_layer + 1]->get_size() * total_layers[ith_layer]->get_size(); 

}
void and_gate(const std::uint32_t n_threads);
void or_gate();
void nand_gate();
void nor_gate();
void xor_gate();
int main(int argc, char *argv[])
{
  /* Sketch
   * have a vector of strings with all layer types as a cmd arg
   * have gate type as a cmd arg
   * have number of threads as a gate arg(optional)
   * have vector of strings with all optimizers as a cmd arg
   * have vector of strings with all loss_functions as a cmd arg
   * be able to print all layer types, optimizer types, loss function types
   * option to save model structure to file
   * */
}


void and_gate(const std::uint32_t n_threads)
{
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;
  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
}
void or_gate()
{
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;
  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
}
void nand_gate()
{
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;
  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
}
void nor_gate()
{
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;
  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
}
void xor_gate()
{
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;
  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
}
