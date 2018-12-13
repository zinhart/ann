#include <ann/ann.hh>
#include <bmp/bmp.hh>
#include <token_parser/token_parser.hh>
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

void and_gate(const std::uint32_t n_threads, const std::string optimizer_name, const std::string loss_function_name,
	          const std::vector<std::string> layers, const std::uint32_t batch_size, double learning_rate, const std::string file
			 );

int main(int argc, char *argv[])
{
  
  using namespace zinhart::models::layers;
  using namespace zinhart::loss_functions;
  using namespace zinhart::optimizers;
  std::string gate{"\0"};
  std::string threads{"\0"};
  std::string optimizer{"\0"};
  std::string loss_function{"\0"};
  std::string batch_size{"\0"};
  std::vector<std::vector<std::string>> layer_strings;

  zinhart::parsers::token_parser ap;
  ap.add_token("--gate", "and|or|nand|nor|xor", "valid logic_gates are <and, or, nand, nor, xor>", true);
  ap.add_token("--threads", zinhart::parsers::expressions::pos_integer, "number of threads to use 1 - N", true );
  ap.add_token("--optimizer", "sgd|momentum", "the optimizer to use");
  ap.add_token("--loss_function", "ce|mse", "the loss function to use");
  ap.add_token("--batch_size", zinhart::parsers::expressions::pos_integer, "the number of cases to process before a weight update", true );
  ap.add_token("--layer", "(input|relu|sigmoid)([1-9][0-9]*)", "layer name and size", true);
  ap.process(argc, argv);
  auto token_values = ap.get_parsed_tokens();

  for(auto it = token_values.begin(); it != token_values.end(); ++it)
  {
	std::cout<<it->first<<" ";
	if(it->first == "--gate")
	  gate = *it->second.begin();
	if(it->first == "--threads")
	  threads = *it->second.begin();
	if(it->first == "--optimizer")
	  optimizer =  *it->second.begin();
	if(it->first == "--loss_function")
	  loss_function =  *it->second.begin();
	if(it->first == "--batch_size")
	  batch_size =  *it->second.begin();
	if(it->first == "--layer")
	{
  	  std::vector<std::string> v;
  	  for(auto inner_it = it->second.begin(); inner_it != it->second.end(); ++inner_it)
	  {
		v.push_back(*inner_it);
		std::cout<<*inner_it<<" ";
	  }
	  layer_strings.push_back(v);
	}
	std::cout<<"\n";
  }
  std::cout<<"Gate: "<<gate<<"\n";
  
  if(gate == "\0")
  {
  	std::cout<<"--gate <and, or, nand, nor, xor>"<<"\n";
	std::exit(0);
  }
  else if(threads == "\0")
  {
   	std::cout<<"--threads <n_threads> (defaults to std::thread::hardware_concurrency when not specified)\n";
	std::exit(0);
  }
  else if(optimizer == "\0")
  {
	std::cout<<"--optimizer <optimizer>\n";
	std::exit(0);
  }
  else if(loss_function == "\0")
  {
	std::cout<<"--loss_function <loss_function>\n";
	std::exit(0);
  }
  else if(batch_size == "\0")
  {
	std::cout<<"--batch_size <n_cases>\n";
	std::exit(0);
  }
  else if(layer_strings.size() == 0)
  {
	std::cout<<"--layers <name> <size>\n";
	std::exit(0);
  }
  // init
  
  /*
  std::vector< std::shared_ptr<zinhart::models::layers::layer<double>> > total_layers;
  std::shared_ptr< zinhart::loss_functions::loss_function<double> > loss_function{std::make_shared<zinhart::loss_functions::mean_squared_error<double>>()};
  std::shared_ptr< zinhart::optimizers::optimizer<double> > optimizer{std::make_shared<zinhart::optimizers::sgd<double>>()};
  zinhart::models::multi_layer_perceptron<double> model;
*/

  std::uint32_t total_activations_length{0};
  std::uint32_t total_hidden_weights_length{0};
  /*
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
*/
}
void and_gate(const std::uint32_t n_threads, const std::string optimizer_name, const std::string loss_function_name,
	          const std::vector<std::string> layers, const std::uint32_t batch_size, double learning_rate, const std::string file
			 )
{

}
