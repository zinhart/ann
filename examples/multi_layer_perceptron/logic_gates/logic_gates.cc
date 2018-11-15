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
  using namespace zinhart::models::layers;
  using namespace zinhart::loss_functions;
  using namespace zinhart::optimizers;
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
	std::cout<<"--threads <n_threads> (defaults to std::thread::hardware_concurrency when not specified)\n";
	std::cout<<"--optimizer <optimizer>\n";
	std::cout<<"--loss_function <loss_function>\n";
	std::cout<<"--layers <n_layers>\n";
	std::cout<<"--batch_size <n_cases>\n";
	std::cout<<"--learning_rate <learning rate>";
	std::cout<<"--save <outputfile>\n";

	std::cout<<"Informational args\n";
	std::cout<<"-g: lists all gates\n";
	std::cout<<"-l: lists all layers\n";
	std::cout<<"-o: lists all optimizers\n";
	std::cout<<"-e: lists all loss_functions\n";

	std::cout<<"Main and informational args are mutually exclusive\n";
	std::exit(0);
  }
  const std::vector<std::string> args(argv, argv + argc);
  const std::vector<std::string> all_gates{"and", "or", "nand", "nor", "xor"};
  const std::vector<std::shared_ptr<layer<double>>> all_layers
  {
   std::make_shared<identity_layer<double>>(), 
   std::make_shared<sigmoid_layer<double>>(), std::make_shared<softplus_layer<double>>(), std::make_shared<tanh_layer<double>>(), 
   std::make_shared<relu_layer<double>>(), std::make_shared<leaky_relu_layer<double>>(), std::make_shared<exp_leaky_relu_layer<double>>(), 
   std::make_shared<softmax_layer<double>>(0)
  };
  const std::vector<std::shared_ptr<optimizer<double>>> all_optimizers
  {
	std::make_shared<sgd<double>>(),
	std::make_shared<momentum<double>>(0),
	std::make_shared<nesterov_momentum<double>>(0),
	std::make_shared<adagrad<double>>(0),
	std::make_shared<conjugate_gradient<double>>(0),
	std::make_shared<adadelta<double>>(0),
	std::make_shared<rms_prop<double>>(0),
	std::make_shared<adamax<double>>(0),
	std::make_shared<amsgrad<double>>(0),
	std::make_shared<adam<double>>(0),
	std::make_shared<nadam<double>>(0),
  };
  const std::vector<std::shared_ptr<loss_function<double>>> all_loss_functions
  {
	std::make_shared<cross_entropy_multi_class<double>>(),
	std::make_shared<mean_squared_error<double>>()
  };

  // check for informational args
  if(std::any_of(args.begin(), args.end(), [](const std::string & init){return init == "-g" || init == "-l" || init == "-o" || init == "-e";}))
  {
	std::uint32_t i{0};
  	std::vector<std::string> informational_flags;
	for(i = 0; i < args.size(); ++i)
	{
	  if(args[i] == "-g")
	  {
		std::cout<<"supported gates: ";
	  	std::for_each(all_gates.begin(), all_gates.end(), [](const std::string & init){std::cout<<init<<"|";});
		std::cout<<"\n";
	  }
	  else if(args[i] == "-l")
	  {
		std::cout<<"supported layers: ";
		std::for_each(all_layers.begin(), all_layers.end(), [](const std::shared_ptr<layer<double>> & l){std::cout<<l->name()<<"|";});
		std::cout<<"\n";
	  }
	  else if(args[i] == "-o")
	  {
		std::cout<<"supported optimizers: ";
		std::for_each(all_optimizers.begin(), all_optimizers.end(), [](const std::shared_ptr<optimizer<double>> & o){std::cout<<o->name()<<"|";});
		std::cout<<"\n";
	  }
	  else if(args[i] == "-e")
	  {
		std::cout<<"supported loss_functions: ";
		std::for_each(all_loss_functions.begin(), all_loss_functions.end(), [](const std::shared_ptr<loss_function<double>> & l){std::cout<<l->name()<<"|";});
		std::cout<<"\n";
	  }
	}
	std::exit(0);
  }

  // check for main args
  std::string gate;
  std::uint32_t n_threads{std::thread::hardware_concurrency()};
  std::string optimizer;
  std::string loss_function;
  std::vector<std::string> layers;

  auto found_gate = std::find(args.begin(), args.end(), "--gate");
  if(found_gate == args.end())
  {
	std::cerr<<"no gate option was found, see usage\n";
	std::exit(0);
  }
  else if(found_gate == args.end() - 1)// gate option was given but no argument
  {
	std::cerr<<"gate option was found but no argument was specified, see usage\n";
	std::exit(0);
  }
  else
  {
	gate = *(found_gate + 1);
	if(!std::any_of(all_gates.begin(), all_gates.end(), [&gate](const std::string & init){return init == gate;}))
	{
	  std::cerr<<gate<<" is not a supported logic gate, see usage\n";
	  std::exit(0);
	}
	std::cout<<gate<<"\n";
  }

  auto found_threads = std::find(args.begin(), args.end(), "--threads");
  if(found_threads == args.end())
  {
	std::cerr<<"no thread option was found using: "<<n_threads<<" threads\n";
  }
  else if(found_threads == args.end() - 1)// thread option was given but no argument
  {
	std::cerr<<"thread option was found but no argument was specified, see usage\n";
	std::exit(0);
  }
  else
  {
	std::string threads{*(found_threads + 1)};
	bool has_only_positive_digits = (threads.find_first_not_of( "0123456789" ) == std::string::npos);
	if(!has_only_positive_digits)
	{
	  std::cerr<<threads<<" is not valid thread value, see usage\n";
	  std::exit(0);
	}
	n_threads = (std::stoi(threads) == 0) ? n_threads : std::stoi(threads);
	std::cout<<n_threads;
  }
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
