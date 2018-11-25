#include <ann/ann.hh>
#include <bmp/bmp.hh>
#include <algorithm>
#include <regex>
#include <cstdlib>
#include <set>

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

std::set<std::string>  match_unique_args(const std::vector<std::string>  & args,  const std::regex expr);
std::vector<std::string> match_args(const std::vector<std::string> & args, const std::regex expr);

bool find_args(std::vector<std::string> & args, std::regex expr);

bool find_arg(const std::vector<std::string> & args, const std::string & arg_to_find);
template <class argval, class unary_predicate>
  void find_arg(const std::vector<std::string> & args, unary_predicate && p, const std::string & arg_to_find, argval & val, bool required = true);
  
bool regex_match(std::regex expr, std::string s);

template <class argval>
  void find_arg(const std::vector<std::string> & args, const std::vector<std::string> & valid_values, const std::string & arg_to_find, argval & val);
// new
template <class argval>
  void find_arg(const std::vector<std::string> & args, const std::regex & valid_values, const std::string & arg_to_find, argval & val);


void and_gate(const std::uint32_t n_threads, const std::string optimizer_name, const std::string loss_function_name,
	          const std::vector<std::string> layers, const std::uint32_t batch_size, double learning_rate, const std::string file
			 );

int main(int argc, char *argv[])
{
  /*
  using namespace zinhart::models::layers;
  using namespace zinhart::loss_functions;
  using namespace zinhart::optimizers;
  const std::vector<std::string> args_vect(argv, argv + argc);
  std::string args;
  std::for_each(args_vect.begin(), args_vect.end(), [&args](const std::string & init){args+=init;});
  std::for_each(args_vect.begin(), args_vect.end(), [&args](const std::string & init){std::cout<<init<<"\n";});
  const std::string help_args{"(-g|-l|-o|-e)+"};
  const std::string operational_args{"(--gate|--threads|--optimizer|--loss_function|--layer|--batch_size|--learning_rate|--save)+"};
  const std::string gates{"and|or|nand|nor|xor"};
  const std::string pos_integer{"([1-9][0-9]*)"};
  const std::string floating_point{"[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?"};

  std::regex help_args_expr(help_args);
  std::regex operational_args_expr(operational_args);
  std::regex gates_regex(gates);
  std::regex pos_integer_regex(pos_integer);
  std::regex floating_point_regex(floating_point);

  std::vector<std::string> matched_operational_args{match_args(args_vect, operational_args_expr)};
  std::cout<<matched_operational_args.size();
  std::for_each(std::begin(matched_operational_args), std::end(matched_operational_args), [](const std::string & match){std::cout<<match<<" ";});

  std::set<std::string> matched_help_args{match_unique_args(args_vect, help_args_expr)};
  std::cout<<matched_help_args.size();
  std::for_each(std::end(matched_help_args), std::end(matched_help_args), [](const std::string & match){std::cout<<match<<" ";});

  if(matched_operational_args.size() > 0)
  {
	auto gate{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--gate")};
	auto thread{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--threads")};

	auto optimizer{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--optimizer")};
	auto loss_function{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--loss_function")};
	auto batch_size{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--batch_size")};
	auto learning_rate{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--learning_rate")};
	auto save{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--save")};
//	auto gate{std::find(std::begin(matched_operational_args), std::end(matched_operational_args),"--gate")};


	if(gate == std::end(matched_operational_args))
	{
	  std::cerr<<"no gate value specified, see args\n";
	  std::exit(0);
	}
	else
	{
	}
	if(thread == std::end(matched_operational_args))
	{
	  std::cerr<<"no thread value specified, see args\n";
	  std::exit(0);
	}
	else
	{
	}
	if(optimizer == std::end(matched_operational_args))
	{
	  std::cerr<<"no optimizer value specified, see args\n";
	  std::exit(0);
	}
	else
	{
	}
	if(loss_function == std::end(matched_operational_args))
	{
	  std::cerr<<"no loss_function value specified, see args\n";
	  std::exit(0);
	}
	else
	{
	}
	if(batch_size == std::end(matched_operational_args))
	{
	  std::cerr<<"no batch_size value specified, see args\n";
	  std::exit(0);
	}
	else
	{
	}
	if(learning_rate == std::end(matched_operational_args))
	{
	  std::cerr<<"no learning_rate value specified, see args\n";
	  std::exit(0);
	}
	else
	{
	}






  }
  else if(matched_help_args.size() > 0)
  {
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
	auto g{matched_help_args.find("-g")};
	auto l{matched_help_args.find("-l")};
	auto o{matched_help_args.find("-o")};
	auto e{matched_help_args.find("-e")};
	if(g != matched_help_args.end())
	{
	  std::cout<<"supported gates: ";
	  std::for_each(all_gates.begin(), all_gates.end(), [](const std::string & init){std::cout<<init<<"|";});
	  std::cout<<"\n";
	}

	if(l != matched_help_args.end())
	{
	  std::cout<<"supported layers: ";
	  std::for_each(all_layers.begin(), all_layers.end(), [](const std::string & init){std::cout<<init<<"|";});
	  std::cout<<"\n";
	}
	if(o != matched_help_args.end())
	{
	  std::cout<<"supported optimizers: ";
	  std::for_each(all_optimizers.begin(), all_optimizers.end(),[](const std::string & init){std::cout<<init<<"|";});
	  std::cout<<"\n";
	}
	if(e != matched_help_args.end())
	{
	  std::cout<<"supported loss_functions: ";
	  std::for_each(all_loss_functions.begin(), all_loss_functions.end(), [](const std::string & init){std::cout<<init<<"|";});
	  std::cout<<"\n";
	}
  	std::exit(0);

  }
  else
  {
	std::cout<<"USAGE ./logic_gates args\n";
	std::cout<<"Main args:\n";
	std::cout<<"--gate <and, or, nand, nor, xor>"<<"\n";
	std::cout<<"--threads <n_threads> (defaults to std::thread::hardware_concurrency when not specified)\n";
	std::cout<<"--optimizer <optimizer>\n";
	std::cout<<"--loss_function <loss_function>\n";
	std::cout<<"--layers <names> <size>\n";
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





  // check for informational args


  // check for main args
  std::string gate;
  std::uint32_t n_threads{std::thread::hardware_concurrency()};
  std::uint32_t batch_size{1};
  double learning_rate{0.0};
  std::string optimizer_name;
  std::string loss_function_name;
  std::vector<std::string> layers;
  std::string file;
  std::regex valid_gate("^and|or|nand|nor|xor$");
  std::regex pos_integer("^([1-9][0-9]*)$");
  std::regex floating_point("^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$");

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
  if(gate == "and")
  {
	and_gate(n_threads, optimizer_name, loss_function_name,	layers, batch_size, learning_rate, file);
  }
  /*
  else if(gate == "or")
  {
  }
  else if(gate == "nand")
  {
  }
  else if(gate == "nor")
  {
  }
  else if(gate == "xor")
  {
  }*/
}

std::set<std::string>  match_unique_args(const std::vector<std::string>  & args,  const std::regex expr)
{
  std::set<std::string> matches;
  std::for_each(args.begin(), args.end(), [&matches, &expr](const std::string & s){ if(std::regex_match(s, expr)) matches.insert(s);});
  return matches;
}
std::vector<std::string>  match_args(const std::vector<std::string>  & args,  const std::regex expr)
{
  std::vector<std::string> matches;
  std::for_each(args.begin(), args.end(), [&matches, &expr](const std::string & s){ if(std::regex_match(s, expr)) matches.push_back(s);});
  return matches;
}

bool find_arg(const std::vector<std::string> & args, const std::string & arg_to_find)
{
  return std::any_of(args.begin(), args.end(), [&arg_to_find](const std::string & init){return init == arg_to_find;});
}

// new
template <class argval>
  void find_arg(const std::vector<std::string> & args, const std::regex & valid_values, const std::string & arg_to_find, argval & val)
  {
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


void and_gate(const std::uint32_t n_threads, const std::string optimizer_name, const std::string loss_function_name,
	          const std::vector<std::string> layers, const std::uint32_t batch_size, double learning_rate, const std::string file
			 )
{
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;


  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
}
