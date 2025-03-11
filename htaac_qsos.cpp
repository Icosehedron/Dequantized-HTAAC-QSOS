#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "sparse_tensor.hpp"
#include "gate_set.hpp"
#include "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense"

const std::string name = "./problem/"; //Path to the problem folder (./problem/)

//Hyperparameters for simulation
const int number_of_epochs = 100; //number of epochs per simulation, you can play with this
const int number_of_repetitions = 1; //number of repetitions of experiment (full runs). At first, you probably just want 1, but crank it up to more reps to compare an ensemble of random initializations and get general understanding

//Circuit hyperparameters
const int gate_repetitions = 100; //how many time to repeat the [sequence of n(n-1)/2 Lie generators of SO(n)]

/**********************************************
*                 REMINDER:                   *
*      In the dequantized case, I think       *
*    coeff_base and reg more or less serve    *
*            the same purpose.                *
**********************************************/

//Graph hyperparameters;
const int max_paulis = 3; //Maximum length of the Pauli string: the higher the stronger the constraint.
const float coeff_base = 300.0; // size of coefficient. Bigger makes us enforce the constraints harder. You will want to tune this.
const float reg = 10.0; // regularizes the strength of population balancing term. Bigger makes us regulate less. You will want to tune this.

//Reads one line of the CNF input (stored in the problem folder)
int* checkCNFLine(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

  int* result = new int[3];
  for (int i = 0; i < 3; ++i) {
    result[i] = std::stoi(tokens[i]);
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

//Loads in the population data
int* parsePopulationFile(const std::string& fileName, int size) { 
  std::ifstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << fileName << std::endl;
    return nullptr;
  }

  std::string line;
  std::getline(file, line);

  std::istringstream iss(line);
  std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

  int* result = new int[size];
  for (int i = 0; i < size; ++i) {
    result[i] = std::stoi(tokens[i]);
  }

  file.close();
  return result;
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

void ensureNormalization(double* state, int num_dim){
  double norm = 0;
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
  std::tuple<std::string, int, int, int, int**> params = parseParamFile(name + "parameters.txt");
  std::string path_to_cnf = std::get<0>(params);
  int num_clauses = std::get<1>(params);
  int w_plus = std::get<2>(params);
  int W_plus = std::get<3>(params);
  int** max3sat = std::get<4>(params);
  if(max3sat == nullptr){return 1;}

  int num_var = get_max(num_clauses, max3sat); //Number of variables in the original cnf
  int num_Var = num_var + 1; //Number of variables, including v_0
  int num_qubits = std::ceil(std::log2(num_Var)); //Number of qubits being used
  int num_dim = std::pow(2, num_qubits); //Number of dimensions in the Hilbert space

  SparseTensor* w_minus = new SparseTensor(num_Var, 2);
  SparseTensor* W_minus = new SparseTensor(num_Var, 4);

  w_minus->readFromFile(name + "w_minus_2d.txt");
  W_minus->readFromFile(name + "W_minus_4d.txt");

  std::cout << "Original source path: " << path_to_cnf << std::endl;
  std::cout << "Number of qubits: " << num_qubits << std::endl;
  std::cout << "Number of variables: " << num_var << std::endl;
  std::cout << "Number of clauses: " << num_clauses << std::endl << std::endl;

  std::vector<int> pauliList = generatePaulis(num_qubits, max_paulis);
  int num_paulis = pauliList.size();
  double coeff = coeff_base / num_paulis;

  int* populations = parsePopulationFile(name + "populations.txt", num_Var);
  if(populations == nullptr){return 1;}
  int max_population = *std::max_element(populations, populations + num_Var);
  double* penalties = new double[num_dim];
  for(int i = 0; i<num_dim; i++){
    penalties[i] = i < num_Var ? (populations[i] - max_population)/reg : -max_population/reg;
  }
  delete[] populations;

  double** rounded_scores = new double*[number_of_repetitions];
  for(int i = 0; i < number_of_repetitions; i++){rounded_scores[i] = new double[number_of_epochs];}
  double** unrounded_scores = new double*[number_of_repetitions];
  for(int i = 0; i < number_of_repetitions; i++){unrounded_scores[i] = new double[number_of_epochs];}
  double** loss_scores = new double*[number_of_repetitions];
  for(int i = 0; i < number_of_repetitions; i++){loss_scores[i] = new double[number_of_epochs];}

  std::pair<int*, int*> CZ = synthesizeCZs(num_qubits, num_dim);
  int* CZ0 = CZ.first;
  int* CZ1 = CZ.second;

  for(int rep = 0; rep < number_of_repetitions; rep++){
    std::cout << "Repetition " << rep << ": ____________________________________________________" << std::endl;
    std::vector<RotationLayer*> circuit;
    for(int i = 0; i < gate_repetitions; i++){
      circuit.push_back(new RotationLayer(num_qubits));
    }

    for(int epoch; epoch < number_of_epochs; epoch++){
      double* state = new double[num_dim];
      std::fill(state, state + num_dim, 1.0 / std::sqrt(num_dim));

      for(int i = 0; i<gate_repetitions; i++){
        circuit[i]->feed_forward(state);
        ensureNormalization(state, num_dim);
      }

      double pauli_loss = 0;
      for(int i = 0; i<num_paulis; i++){
        int pauli = pauliList[i];
        double expectation_of_pauli_string = 0;
        for(int j = 0; j<num_dim; j++){
          expectation_of_pauli_string += __builtin_popcount(pauli ^ j) % 2 == 0 ? state[j]*state[j] : -state[j]*state[j];
        }
        pauli_loss += expectation_of_pauli_string * expectation_of_pauli_string;
      }

      double proper_loss = 0;

      //Calculate proper_loss
      //Calculate loss
      //Calculate unrounded, rounded problem solutions and store performance
      //Backpropagate and update parameters
    }

    for(int i = 0; i<gate_repetitions; i++){
      delete circuit[i];
    }
    circuit.clear();
  }

  //Plot performance

  return 0;
}