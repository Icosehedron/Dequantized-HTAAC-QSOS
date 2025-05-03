//HTAAC-QSOS v10.1:
//This is a for-testing version of the HTAAC-QSOS code. This code implements the old Pauli string + population-balancing unitary approach, with typical variational methods.

#include <iostream>
#include <cstdlib>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include "sparse_tensor.hpp"
#include "torus_model.hpp"
#include "graph_results.hpp"
#include "quantum_circuit.hpp"

const std::string name = "./problem/"; //Path to the problem folder (./problem/)
const std::string diagram_name = "s3v110c700-2-benchmark-11";

//Hyperparameters for simulation
const int number_of_epochs = 200; //number of epochs per simulation, you can play with this
int number_of_repetitions = 1; //number of repetitions of experiment (full runs). At first, you probably just want 1, but crank it up to more reps to compare an ensemble of random initializations and get general understanding

const int num_pauli_strings = 0;
const bool use_population_balancing = false;
const bool rotation_mode = 2;

//For tuning
const bool automatic_tuning = false;
const bool tune_population = false;
const float lambda_base = 10.0f; //Base strength of pauli loss
const float beta_base = 1.0f; //Base strength of population balancing
const float test_scale = 6.0f;

//Mode-specific hyperparameters
float lambda = lambda_base; //Strength of pauli loss
float beta = beta_base; //Strength of population balancing


const bool allow_degree_2 = false; //Allow degree 2 variables in the circuit when using DQSOS
const bool live_time_limit = true;
const int execution_mode = 0;

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

//Loads in the problem parameters, along with the w_plus, w_minus, W_plus, and W_minus matrices
int* parsePopulationFile(const int num_Var, const std::string& fileName) { 
  std::ifstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << fileName << std::endl;
    return nullptr;
  }

  int* populations = new int[num_Var];
  std::copy(std::istream_iterator<int>(file), std::istream_iterator<int>(), populations);

  file.close();
  return populations;
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

std::vector<int> generatePaulis(int num_qubits, int max_paulis) {
  std::vector<int> result;
  int max_value = 1 << num_qubits;
  for (int i = 1; i < max_value; i++) {
    if (__builtin_popcount(i) <= max_paulis) {result.push_back(i);}
  }
  return result;
}

std::pair<int*, int*> synthesizeCZs(int num_qubits, int num_dim) {
  int mask1 = 0;
  for(int i = 0; i < num_qubits; i++){
    mask1 = (mask1 << 1) ^ (i & 1) ^ 1;
  }
  int mask2 = mask1 >> 1;

  int* CZ0 = new int[num_dim];
  int* CZ1 = new int[num_dim];

  for(int i = 0; i < num_dim; i++){
    CZ0[i] = i ^ (i & mask1) >> 1;
    CZ1[i] = i ^ (i & mask2) >> 1;
  }
  return std::make_pair(CZ0, CZ1);
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
  int num_qubits = static_cast<int>(std::ceil(std::log2(num_Var)));
  int num_dim = 1 << num_qubits; //Number of dimensions in the circuit
  std::vector<int> pauliList;
  if(num_pauli_strings >= 1){pauliList = generatePaulis(num_qubits, num_pauli_strings);}
  int num_paulis = pauliList.size();
  std::pair<int*, int*> CZs = synthesizeCZs(num_qubits, num_dim);

  int* populations = parsePopulationFile(num_Var, name + "populations.txt"); //Constant 8 * population value
  int max_population = *std::max_element(populations, populations + num_Var);
  for(int i = 0; i < num_Var; i++){
    populations[i] -= max_population;
  } 

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

  if(live_time_limit){number_of_repetitions = 1;}

  //Create storage for scores and loss
  std::vector<float*> rounded_scores;
  std::vector<float*> unrounded_scores;
  std::vector<float*> loss_scores;
  std::vector<float*> constraint_scores;

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();

  for(int rep = 0; rep < number_of_repetitions; rep++){
    rounded_scores.push_back(new float[number_of_epochs]);
    unrounded_scores.push_back(new float[number_of_epochs]);
    loss_scores.push_back(new float[number_of_epochs]);
    constraint_scores.push_back(new float[number_of_epochs]);

    float rep_factor = (float) rep / number_of_repetitions;
    if(automatic_tuning){
      if(tune_population){
        beta = std::exp(test_scale * 2 * rep_factor - test_scale) * beta_base;
      }
      else{
        lambda = std::exp(test_scale * 2 * rep_factor - test_scale) * lambda_base;
      }
    }

    TorusModel* torus_circuit;
    QuantumCircuit* quantum_circuit;

    if(rotation_mode == 0){
      quantum_circuit = new QuantumCircuit(num_qubits, num_Var, CZs.first, CZs.second, 50);
    }
    else if(rotation_mode == 2){
      torus_circuit = new TorusModel(num_Var, allow_degree_2, true);
    }
    

    float* pauli_losses = new float[num_paulis];

    for(int epoch = 0; epoch < number_of_epochs; epoch++){
      //Initialize state
      float* state;
      if(rotation_mode == 0){
        state = new float[num_dim];
        std::fill(state, state + num_dim, 1 * std::sqrt(float(num_Var)/float(num_dim)));
        quantum_circuit->feed_forward(state);
      }
      else if(rotation_mode == 2){
        state = torus_circuit->feed_forward();
      }

      //Calculate the penalty for deviating from the equal superposition (we do not use this approach, but it is here as a sanity check)
      float contraint_loss = 0;
      for(int i = 0; i<num_Var; i++){
        float deviation = state[i] * state[i] - 1.0f;
        contraint_loss += deviation * deviation;
      }
      constraint_scores[rep][epoch] = contraint_loss;

      //Calculate the penalty for deviating from the Pauli string conditions
      float pauli_loss_total = 0;
      for(int i = 0; i<num_paulis; i++){
        pauli_losses[i] = 0;
        for(int j = 0; j<num_Var; j++){
          float pauli_sign = __builtin_popcount(pauliList[i] & j) % 2 == 0 ? 1.0f : -1.0f;
          pauli_losses[i] += pauli_sign * state[j] * state[j];
        }
        pauli_loss_total += pauli_losses[i] * pauli_losses[i];
      }

      //Calculate the premium due to the poulation balancing term 
      float population_loss = 0;
      if(use_population_balancing){
        for(int i = 0; i<num_Var; i++){
          population_loss += populations[i] * state[i] * state[i];
        }
      }

      //Calculate the loss from w_minus and W_minus
      float w_minus_loss = w_minus->contract_2D(state);
      float W_minus_loss = W_minus->contract_4D(state);

      float loss = (w_minus_loss + W_minus_loss + population_loss/num_Var * beta)/8 + pauli_loss_total * lambda / (num_Var * num_Var);
      loss_scores[rep][epoch] = loss;

      //Calculate unrounded score
      float unrounded_score = (w_plus - w_minus_loss + W_plus - W_minus_loss)/8;
      unrounded_scores[rep][epoch] = unrounded_score;

      //Round the solution and calculate rounded score
      float* rounded_state = new float[num_Var];
      float w_minus_rounded_loss;
      float W_minus_rounded_loss;

      for(int i = 0; i<num_Var; i++){
        rounded_state[i] = state[i] > 0.0f ? 1.0f : -1.0f;
      }
      w_minus_rounded_loss = w_minus->contract_2D(rounded_state, false);
      W_minus_rounded_loss = W_minus->contract_4D(rounded_state, false);
      float rounded_score = (w_plus - w_minus_rounded_loss + W_plus - W_minus_rounded_loss)/8;
      delete[] rounded_state;
      rounded_scores[rep][epoch] = rounded_score;

      //Set up the backpropogation
      float* gradients = new float[num_Var];
      std::fill(gradients, gradients + num_Var, 0.0f);

      //Account for 8 * gradient on the output state from w_minus and W_minus
      w_minus->back_contract_2D(gradients);
      W_minus->back_contract_4D(gradients);

      for(int i = 0; i<num_paulis; i++){
        for(int j = 0; j<num_Var; j++){
          float pauli_sign = __builtin_popcount(pauliList[i] & j) % 2 == 0 ? 1.0f : -1.0f;
          gradients[j] += 8 * lambda * 2 * pauli_losses[i] * 2 * pauli_sign * state[j] / (num_Var * num_Var);
        }
      }

      if(use_population_balancing){
        for(int i = 0; i<num_Var; i++){
          gradients[i] += beta * populations[i] * 2 * state[i] / num_Var;
        }
      }

      if(rotation_mode == 0){
        quantum_circuit->back_propagate(gradients);
        quantum_circuit->update_parameters();
      }
      else if(rotation_mode == 2){
        torus_circuit->back_propagate(gradients);
        torus_circuit->update_parameters();
      }

      delete[] gradients;
    }

    delete quantum_circuit;
    delete torus_circuit;
    delete[] pauli_losses;

    //If using live time limit, extend the number of repetitions until the time limit is reached
    if(rep == number_of_repetitions - 1 && live_time_limit){ 
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      if(elapsed_time.count() < 30.0){
        number_of_repetitions++;
      }
    }
  }

  // End timing
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  graph_results(diagram_name, number_of_repetitions, number_of_epochs, rounded_scores.data(), unrounded_scores.data(), loss_scores.data(), constraint_scores.data(), elapsed_time.count(), false, execution_mode, 40);

  return 0;
}