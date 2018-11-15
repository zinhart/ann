#include <ann/ann.hh>
#include <bmp/bmp.hh>
#include <algorithm>
#include <regex>
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


bool find_arg(const std::vector<std::string> & args, const std::string & arg_to_find);
template <class argval, class unary_predicate>
  void find_arg(const std::vector<std::string> & args, unary_predicate && p, const std::string & arg_to_find, argval & val, bool required = true);
template <class argval>
  void find_arg(const std::vector<std::string> & args, const std::vector<std::string> & valid_values, const std::string & arg_to_find, argval & val);

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
  const std::vector<std::string> all_layers
  {
   std::make_shared<identity_layer<double>>()->name(), 
   std::make_shared<sigmoid_layer<double>>()->name(), std::make_shared<softplus_layer<double>>()->name(), std::make_shared<tanh_layer<double>>()->name(), 
   std::make_shared<relu_layer<double>>()->name(), std::make_shared<leaky_relu_layer<double>>()->name(), std::make_shared<exp_leaky_relu_layer<double>>()->name(), 
   std::make_shared<softmax_layer<double>>(0)->name()
  };
  const std::vector<std::string> all_optimizers
  {
	std::make_shared<sgd<double>>()->name(),
	std::make_shared<momentum<double>>(0)->name(),
	std::make_shared<nesterov_momentum<double>>(0)->name(),
	std::make_shared<adagrad<double>>(0)->name(),
	std::make_shared<conjugate_gradient<double>>(0)->name(),
	std::make_shared<adadelta<double>>(0)->name(),
	std::make_shared<rms_prop<double>>(0)->name(),
	std::make_shared<adamax<double>>(0)->name(),
	std::make_shared<amsgrad<double>>(0)->name(),
	std::make_shared<adam<double>>(0)->name(),
	std::make_shared<nadam<double>>(0)->name(),
  };
  const std::vector<std::string> all_loss_functions
  {
	std::make_shared<cross_entropy_multi_class<double>>()->name(),
	std::make_shared<mean_squared_error<double>>()->name()
  };

  // check for informational args
  if(find_arg(args, "-g"))
  {
	std::cout<<"supported gates: ";
	std::for_each(all_gates.begin(), all_gates.end(), [](const std::string & init){std::cout<<init<<"|";});
	std::cout<<"\n";
	std::exit(0);
  }

  if(find_arg(args, "-l"))
  {
	std::cout<<"supported layers: ";
	std::for_each(all_layers.begin(), all_layers.end(), [](const std::string & init){std::cout<<init<<"|";});
	std::cout<<"\n";
	std::exit(0);
  }
  if(find_arg(args, "-o"))
  {
	std::cout<<"supported optimizers: ";
	std::for_each(all_optimizers.begin(), all_optimizers.end(),[](const std::string & init){std::cout<<init<<"|";});
	std::cout<<"\n";
	std::exit(0);
  }
  if(find_arg(args, "-e"))
  {
	std::cout<<"supported loss_functions: ";
	std::for_each(all_loss_functions.begin(), all_loss_functions.end(), [](const std::string & init){std::cout<<init<<"|";});
	std::cout<<"\n";
	std::exit(0);
  }

  // check for main args
  std::string gate;
  std::uint32_t n_threads{std::thread::hardware_concurrency()};
  std::uint32_t batch_size{1};
  double learning_rate{0.0};
  std::string optimizer_name;
  std::string loss_function_name;
  std::vector<std::string> layers;

  find_arg(args, all_gates, "--gate", gate);
  find_arg(args, all_optimizers, "--optimizer", optimizer_name);
  find_arg(args, all_loss_functions, "--loss_function", loss_function_name);

  auto check_thread_val = [&n_threads](std::string threads)
  {
	bool has_only_positive_digits = (threads.find_first_not_of( "0123456789" ) == std::string::npos);
	if(!has_only_positive_digits)
	{
	  std::cerr<<threads<<" is not valid thread value, see usage\n";
	  std::exit(0);
	}
	return (std::stoi(threads) == 0) ? n_threads : std::stoi(threads);
  };
  find_arg(args, check_thread_val, "--threads", n_threads);

  auto check_batch_val = [](std::string batch_size)
  {
	bool has_only_positive_digits = (batch_size.find_first_not_of( "0123456789" ) == std::string::npos);
	if(!has_only_positive_digits)
	{
	  std::cerr<<batch_size<<" is not valid batch value, see usage\n";
	  std::exit(0);
	}
	return (std::stoi(batch_size) == 0) ? 1 : std::stoi(batch_size);
  };
  find_arg(args, check_batch_val, "--batch_size", n_threads);
  auto check_learning_rate = [](std::string learning_rate)
  {
	std::regex valid_floating_point("^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$");
	if(!regex_match(learning_rate, valid_floating_point))
	{
	  std::cerr<<learning_rate<<" is not a valid learning_rate, see usage\n";
	  std::exit(0);
	}
	return std::stod(learning_rate);
  };
  find_arg(args, check_learning_rate, "--learning_rate", learning_rate, false);


  auto found_layers = std::find(args.begin(), args.end(), "--layers");
  if(found_layers == args.end())
  {
	std::cerr<<"no --layers option was found, see usage\n";
	std::exit(0);
  }
  else if(found_layers == args.end() - 1)// thread option was given but no argument
  {

	std::cerr<<"--layers option was found but no argument was specified, see usage\n";
	std::exit(0);
  }
  else
  {
	auto next_layer = found_layers + 1;
	while(next_layer != args.end() )
	{
	  if((*next_layer).at(0) == '-')// another cmd line arg
		break;
	  if(!std::any_of(all_layers.begin(), all_layers.end(), [&next_layer](const std::string & init){return init == *next_layer;}))
	  {
		std::cerr<<"unsupported layer: "<<*next_layer<<"\n";
		std::exit(0);
	  }
	  layers.push_back(*next_layer);
	  std::cout<<*next_layer;
	  ++next_layer;
	}
  }
}
bool find_arg(const std::vector<std::string> & args, const std::string & arg_to_find)
{
  return std::any_of(args.begin(), args.end(), [&arg_to_find](const std::string & init){return init == arg_to_find;});
}
template <class argval, class unary_predicate>
  void find_arg(const std::vector<std::string> & args, unary_predicate && p, const std::string & arg_to_find, argval & val, bool required)
  {
	auto found_arg = std::find(args.begin(), args.end(), arg_to_find);
	if(found_arg == args.end())
	{
	  std::cerr<<"no "<<arg_to_find <<" option was found, see usage\n";
	  if(required)
		std::exit(0);
	}
	else if(found_arg == args.end() - 1)// gate option was given but no argument
	{
	  std::cerr<<arg_to_find<<" option was found but no argument was specified, see usage\n";
	  if(required)
  		std::exit(0);
	}
	else
	{
	  val = p(*(found_arg +1));
	  std::cout<<val<<"\n";
	}

  }
template <class argval>
  void find_arg(const std::vector<std::string> & args, const std::vector<std::string> & valid_values, const std::string & arg_to_find, argval & val)
  {
	auto found_arg = std::find(args.begin(), args.end(), arg_to_find);
	if(found_arg == args.end())
	{
	  std::cerr<<"no "<<arg_to_find <<" option was found, see usage\n";
	  std::exit(0);
	}
	if(found_arg == args.end() - 1)// gate option was given but no argument
	{
	  std::cerr<<arg_to_find<<" option was found but no argument was specified, see usage\n";
	  std::exit(0);
	}
	else
	{
	  val = *(found_arg + 1);
	  if(!std::any_of(valid_values.begin(), valid_values.end(), [&val](const std::string & init){return init == val;}))
	  {
		std::cerr<<val<<" is not a valid value for arg: "<< arg_to_find<<"\nsee usage\n";
		std::exit(0);
	  }
	  std::cout<<val<<"\n";
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
