#include <ann/ann.hh>
#include <bmp/bmp.hh>
#include <algorithm>
#include <cstdlib>

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
  if(argc < 2)
  {
	std::cout<<"USAGE ./logic_gates args\n";
	std::cout<<"Main args:\n";
	std::cout<<"--gate <and, or, nand, nor, xor>"<<"\n";
	std::cout<<"--threads <n_threads> (defaults to std::thread::hardware_concurrency)\n";
	std::cout<<"--optimizer <optimizer>\n";
	std::cout<<"--loss_function <loss_function>\n";
	std::cout<<"--layers <n_layers>\n";
	std::cout<<"--save <outputfile>\n";

	std::cout<<"Informational args\n";
	std::cout<<"-g: lists all gates\n";
	std::cout<<"-l: lists all layers\n";
	std::cout<<"-o: lists all optimizers\n";
	std::cout<<"-e: lists all loss_functions\n";

	std::cout<<"Main and informational args are mutually exclusive\n";
	std::exit(0);
  }
  std::for_each(argv, argv + argc, [](char * init){std::cout<<init<<"\n";});
  if(argv[1] == "-g")
	std::cout<<"here\n";
  // check for informational args
  if(std::any_of(argv, argv + argc, [](char * init){return init == "-g" || init == "-l" || init == "-o" || init == "-e";}))
  {
	std::cout<<"found informational args\n";
	std::exit(0);
  }
  // check for main args
  std::string gate;
  std::string optimizer;
  std::string loss_function;
  std::vector<std::string> layers;
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
