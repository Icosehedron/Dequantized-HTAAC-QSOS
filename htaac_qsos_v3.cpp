//HTAAC-QSOS v3:
//This is the third version of the HTAAC-QSOS code. This code removes the fragments that accomodate a Y-rotation based
//variational circuit, making it faster and completely dropping the qubit-based variational circuit structure.

#include <iostream>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "sparse_tensor.hpp"
#include "gate_set.hpp"
#include "graph_results.hpp"

const std::string name = "./problem/"; //Path to the problem folder (./problem/)
const std::string diagram_name = "s3v110c700-1";

//Hyperparameters for simulation
const int number_of_epochs = 200; //number of epochs per simulation, you can play with this
const int epochs_between_reports = -1; //Set to -1 to turn off reports
const int number_of_repetitions = 1; //number of repetitions of experiment (full runs). At first, you probably just want 1, but crank it up to more reps to compare an ensemble of random initializations and get general understanding

//Circuit hyperparameters
const int gate_repetitions = 3; //how many time to repeat the [sequence of n(n-1)/2 Lie generators of SO(n)]

//Graph hyperparameters;
const float coeff_base = 0.1f; // size of coefficient. Bigger makes us enforce the constraints harder. You will want to tune this.

//Reads one line of the CNF input (stored in the problem folder)
int* checkCNFLine(const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  int* result = new int[3];
  for (int i = 0; i < 3; ++i) {
    iss >> token;
    result[i] = std::stoi(token);
  }
  return result;
}

//Loads in the problem parameters, along with the w_plus, w_minus, W_plus, and W_minus matrices
std::tuple<std::string, int, int, int, int**> parseParamFile(const std::string& fileName) { 
  std::ifstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << fileName << std::endl;
    return std::make_tuple("", 0, 0, 0, nullptr);
  }

  std::string line;
  std::vector<int*> clauses;

  std::getline(file, line);
  std::string path_to_cnf = line;
  std::getline(file, line);
  int num_clauses = std::stoi(line);
  std::getline(file, line);
  int w_plus = std::stoi(line);
  std::getline(file, line);
  int W_plus = std::stoi(line);

  while(std::getline(file, line)) {
    int* clause = checkCNFLine(line);
    if(clause == nullptr){continue;}
    clauses.push_back(clause);
  }

  file.close();
  int temp_num_clauses = clauses.size();
  int** arr = new int*[temp_num_clauses];
  for (size_t i = 0; i < temp_num_clauses; ++i) {
    arr[i] = clauses[i];
  }
  return std::make_tuple(path_to_cnf, temp_num_clauses, w_plus, W_plus, arr);
}

int get_max(int num_clauses, int** max3sat){
  int max = 0;
  for(int i = 0; i < num_clauses; i++){
    for(int j = 0; j < 3; j++){
      if(max < max3sat[i][j]){max = max3sat[i][j];}
      else if(max < -max3sat[i][j]){max = -max3sat[i][j];}
    }
  }
  return max;
}

void ensureNormalization(float* state, int num_dim){
  float norm = 0;
  for(int i = 0; i < num_dim; i++){
    norm += state[i] * state[i];
  }
  norm = std::sqrt(norm);
  for(int i = 0; i < num_dim; i++){
    state[i] /= norm;
  }
  return;
}

int main() {
  //Load in the problem parameters (determined by prepare_m3s.cpp)
  std::tuple<std::string, int, int, int, int**> params = parseParamFile(name + "parameters.txt");
  std::string path_to_cnf = std::get<0>(params); //Path to the original CNF file
  int num_clauses = std::get<1>(params); //Number of clauses in the CNF
  int w_plus = std::get<2>(params); //Constant 8*w_plus value (equal to num_clauses)
  int W_plus = std::get<3>(params); //Constant 8*W_plus value (equal to 6 * num_clauses)
  int** max3sat = std::get<4>(params); //Max3Sat instance
  if(max3sat == nullptr){return 1;}

  int num_var = get_max(num_clauses, max3sat); //Number of variables in the original cnf
  int num_Var = num_var + 1; //Number of variables, including v_0

  //Set up tensors in (sparse) COO format
  SparseTensor* w_minus = new SparseTensor(num_Var, 2);
  SparseTensor* W_minus = new SparseTensor(num_Var, 4);

  //Read in the tensors
  w_minus->readFromFile(name + "w_minus_2d.txt");
  W_minus->readFromFile(name + "W_minus_4d.txt");

  //Prepare tensors for ultrafast execution
  w_minus->prepare_for_execution();
  W_minus->prepare_for_execution();

  std::cout << "Original source path: " << path_to_cnf << std::endl;
  std::cout << "Number of variables: " << num_var << std::endl;
  std::cout << "Number of clauses: " << num_clauses << std::endl << std::endl;
  std::cout << std::endl;

  //Create storage for scores and loss
  float** rounded_scores = new float*[number_of_repetitions];
  for(int i = 0; i < number_of_repetitions; i++){rounded_scores[i] = new float[number_of_epochs];}
  float** unrounded_scores = new float*[number_of_repetitions];
  for(int i = 0; i < number_of_repetitions; i++){unrounded_scores[i] = new float[number_of_epochs];}
  float** loss_scores = new float*[number_of_repetitions];
  for(int i = 0; i < number_of_repetitions; i++){loss_scores[i] = new float[number_of_epochs];}

  float coeff = coeff_base * num_Var * num_Var;

  for(int rep = 0; rep < number_of_repetitions; rep++){
    if(epochs_between_reports > 0){
      std::cout << "Repetition " << rep << ": ____________________________________________________" << std::endl;
    }

    //Create the circuit in Lie decomposition form
    std::vector<RotationLayer*> circuit;
    for(int i = 0; i < gate_repetitions; i++){
      circuit.push_back(new RotationLayer(num_Var, gate_repetitions));
    }

    for(int epoch; epoch < number_of_epochs; epoch++){
      //Initialize state
      float* state = new float[num_Var];
      std::fill(state, state + num_Var, 1.0f / std::sqrt(num_Var));

      //Calculate the output of the variational circuit
      for(int i = 0; i<gate_repetitions; i++){
        circuit[i]->feed_forward(state);
        ensureNormalization(state, num_Var);
      }

      //Calculate the penalty for deviating from the equal superposition
      float contraint_loss = 0;
      for(int i = 0; i<num_Var; i++){
        float deviation = state[i] * state[i] - 1.0f/num_Var;
        contraint_loss += deviation * deviation;
      }

      //Calculate the loss from w_minus, W_minus, and the population balancing term
      float w_minus_loss = w_minus->contract_2D(state);
      float W_minus_loss = W_minus->contract_4D(state);

      float proper_loss = w_minus_loss + W_minus_loss;
      float loss = proper_loss + coeff * contraint_loss;
      loss_scores[rep][epoch] = loss;

      //Calculate unrounded score
      float unrounded_score = (w_plus - w_minus_loss * num_Var + W_plus - W_minus_loss * num_Var * num_Var)/8;
      unrounded_scores[rep][epoch] = unrounded_score;

      //Round the solution and calculate rounded score
      float* rounded_state = new float[num_Var];
      for(int i = 0; i<num_Var; i++){
        rounded_state[i] = state[i] >= 0.0f ? 1.0f : -1.0f;
      }
      float w_minus_rounded_loss = w_minus->contract_2D(rounded_state, false);
      float W_minus_rounded_loss = W_minus->contract_4D(rounded_state, false);
      float rounded_score = (w_plus - w_minus_rounded_loss + W_plus - W_minus_rounded_loss)/8;
      delete[] rounded_state;
      rounded_scores[rep][epoch] = rounded_score;

      //Set up the backpropogation
      float* gradients = new float[num_Var];
      std::fill(gradients, gradients + num_Var, 0.0f);

      //Account from the gradient on the output state from contraint-based penalties
      for(int i = 0; i<num_Var; i++){
        float deviation = state[i] * state[i] - 1.0f/num_Var;
        gradients[i] += 4 * coeff * deviation * state[i];
      }

      //Account from the gradient on the output state from w_minus and W_minus
      w_minus->back_contract_2D(gradients, num_Var/8);
      W_minus->back_contract_4D(gradients, num_Var * num_Var/8);

      //Backpropogate through the variational circuit
      for(int i = gate_repetitions - 1; i >= 0; i--){
        circuit[i]->back_propagate(gradients);
        circuit[i]->update_parameters();
      }

      if(epochs_between_reports > 0 && epoch % epochs_between_reports == 0){
        std::cout << "Epoch number: " << epoch << std::endl;
        std::cout << "State: " << state[0] << " " << state[1] << " " << state[2] << " " << state[3] << std::endl;
        std::cout << "HTAACQSOS (unrounded): " << unrounded_score << std::endl;
        std::cout << "HTAACQSOS (rounded): " << rounded_score << " of " << num_clauses << " clauses satisfied." << std::endl;
        std::cout << "Pure Loss: " << loss << " (= " << w_minus_loss << " + " << W_minus_loss << " + " << contraint_loss << ")" << std::endl;
        std::cout << std::endl;
      }

      delete[] gradients;
      delete[] state;
    }

    for(int i = 0; i<gate_repetitions; i++){
      delete circuit[i];
    }
    circuit.clear();
  }

  graph_results(diagram_name, number_of_repetitions, number_of_epochs, rounded_scores, unrounded_scores, loss_scores);

  return 0;
}