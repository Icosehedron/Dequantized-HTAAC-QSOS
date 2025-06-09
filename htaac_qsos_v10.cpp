//HTAAC-QSOS v10:
//This is the tenth version of the HTAAC-QSOS code. This code incorporates new variants of the DQSOS algorithm, incorporating various levels of local search and simulated annealing. These greatly strength the performance of the algorithm.

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
#include "Xorshift32.hpp"

const std::string name = "./problem/"; //Path to the problem folder (./problem/)
std::string diagram_name = "HG-3SAT-V300-C1200-5";

//Hyperparameters for simulation
int number_of_epochs = 6000; //number of epochs per simulation, you can play with this
int number_of_repetitions = 1; //number of repetitions of experiment (full runs). At first, you probably just want 1, but crank it up to more reps to compare an ensemble of random initializations and get general understanding

//Mode-specific hyperparameters
//0 = DQSOS, 1 = DQSOS + LS, 2 = DQSOS + SA, 
//3 = DQSOS-guided LS, 4 = DQSOS-guided SA,
//5 = Local Search, 6 = Simulated Annealing
int execution_mode = 0;
float annealing_strength = 10.0f; //Strength of the annealing process
int annealing_girth = 15; //Girth of the annealing process
const bool allow_degree_2 = false; //Allow degree 2 variables in the circuit when using DQSOS
const bool live_time_limit = true;

Xorshift32 sa_rng(33333); //Seeds the simulated annealing process

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

int main(int argc, char* argv[]) {
  if (argc > 1) {
    execution_mode = std::stoi(argv[1]);
    diagram_name = diagram_name + "-" + argv[1];
  }

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

  switch (execution_mode) {
    case 0:
      std::cout << "Mode: DQSOS" << std::endl;
      number_of_epochs = 1000;
      annealing_strength = 0.0f;
      number_of_repetitions = 600;
      break;
    case 1:
      std::cout << "Mode: DQSOS + Local Search" << std::endl;
      number_of_epochs = 2000;
      annealing_strength = 0.0f;
      number_of_repetitions = 300;
      break;
    case 2:
      std::cout << "Mode: DQSOS + Simulated Annealing" << std::endl;
      break;
    case 3:
      std::cout << "Mode: DQSOS-guided Local Search" << std::endl;
      number_of_epochs = 500;
      annealing_strength = 0.0f;
      number_of_repetitions = 1200;
      break;
    case 4:
      std::cout << "Mode: DQSOS-guided Simulated Annealing" << std::endl;
      break;
    case 5:
      std::cout << "Mode: Local Search" << std::endl;
      number_of_epochs = 2000;
      annealing_strength = 0.0f;
      number_of_repetitions = 1200;
      break;
    case 6:
      std::cout << "Mode: Simulated Annealing" << std::endl;
      number_of_epochs = 24000;
      annealing_strength = 2.0f;
      break;
    default:
      std::cout << "Unknown mode" << std::endl;
      break;
  }

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

    float rep_factor = (1 - (float) rep / number_of_repetitions);

    TorusModel* circuit;

    if(execution_mode == 0 || execution_mode == 1 || execution_mode == 2){
      circuit = new TorusModel(num_Var, allow_degree_2, true);
    }
    else if(execution_mode == 3 || execution_mode == 4 || execution_mode == 5 || execution_mode == 6){
      circuit = new TorusModel(num_Var, false, false);
    }
    

    for(int epoch = 0; epoch < number_of_epochs; epoch++){
      float temperature = (1 - (float) epoch / number_of_epochs) * annealing_strength;

      //Initialize state
      float* state = circuit->feed_forward();

      //Calculate the penalty for deviating from the equal superposition
      float contraint_loss = 0;
      for(int i = 0; i<num_Var; i++){
        float deviation = state[i] * state[i] - 1.0f;
        contraint_loss += deviation * deviation;
      }
      constraint_scores[rep][epoch] = contraint_loss;

      //Calculate the loss from w_minus, W_minus, and the population balancing term
      float w_minus_loss = w_minus->contract_2D(state);
      float W_minus_loss = W_minus->contract_4D(state);

      float loss = (w_minus_loss + W_minus_loss)/8;
      loss_scores[rep][epoch] = loss;

      //Calculate unrounded score
      float unrounded_score = (w_plus - w_minus_loss + W_plus - W_minus_loss)/8;
      unrounded_scores[rep][epoch] = unrounded_score;

      //Round the solution and calculate rounded score
      float* rounded_state = new float[num_Var];
      float w_minus_rounded_loss;
      float W_minus_rounded_loss;

      float rounded_score;

      if(execution_mode == 0 || execution_mode == 1 || execution_mode == 2){
        for(int i = 0; i<num_Var; i++){
          rounded_state[i] = state[i] > 0.0f ? 1.0f : -1.0f;
        }
        w_minus_rounded_loss = w_minus->contract_2D(rounded_state, false);
        W_minus_rounded_loss = W_minus->contract_4D(rounded_state, false);
        rounded_score = (w_plus - w_minus_rounded_loss + W_plus - W_minus_rounded_loss)/8;
      }
      if(execution_mode == 3 || execution_mode == 4 || execution_mode == 5 || execution_mode == 6){
        std::copy(state, state + num_Var, rounded_state);
        w_minus_rounded_loss = w_minus_loss;
        W_minus_rounded_loss = W_minus_loss;
        rounded_score = (w_plus - w_minus_rounded_loss + W_plus - W_minus_rounded_loss)/8;
      }

      delete[] rounded_state;
      rounded_scores[rep][epoch] = rounded_score;

      //Set up the backpropogation
      float* gradients = new float[num_Var];
      std::fill(gradients, gradients + num_Var, 0.0f);

      //Account from the gradient on the output state from w_minus and W_minus
      if(execution_mode == 0 || execution_mode == 1 || execution_mode == 2 || execution_mode == 3 || execution_mode == 4){
        w_minus->back_contract_2D(gradients);
        W_minus->back_contract_4D(gradients);
      }

      if(execution_mode == 0 || execution_mode == 1 || execution_mode == 2){
        circuit->back_propagate(gradients);
        circuit->update_parameters();
      }

      if(execution_mode == 1 || execution_mode == 2 || execution_mode == 3 || execution_mode == 4){
        std::vector<std::pair<float, int>> certainty;
        for (int i = 0; i < num_Var; i++) {
          certainty.emplace_back(state[i] > 0.0f ? -gradients[i] : gradients[i], i);
        }
        std::sort(certainty.begin(), certainty.end());

        float* experimental_state = new float[num_Var];
        std::copy(state, state + num_Var, experimental_state);
        bool* flips = new bool[num_Var]();
        bool any_flips = false;

        while(!any_flips){
          for(int i = 0; i<annealing_girth; i++){
            if(sa_rng.next_double() < 1.0f/annealing_girth){
              flips[certainty[i].second] = true;
              any_flips = true;
              experimental_state[certainty[i].second] = -experimental_state[certainty[i].second];
            }
          }
        }

        float experimental_loss = (w_minus->contract_2D(experimental_state, false) + W_minus->contract_4D(experimental_state, false))/8;
        if(execution_mode == 1 || execution_mode == 3){
          if(experimental_loss < loss){
            circuit->flip_parameters(flips);
          }
        }
        else if(execution_mode == 2 || execution_mode == 4){
          if(sa_rng.next_double() < std::exp((loss - experimental_loss) / temperature)){
            circuit->flip_parameters(flips);
          }
        }

        delete[] experimental_state;
        delete[] flips;
      }
      if(execution_mode == 5 || execution_mode == 6){
        float* experimental_state = new float[num_Var];
        std::copy(state, state + num_Var, experimental_state);
        bool* flips = new bool[num_Var]();
        bool any_flips = false;

        while(!any_flips){
          for(int i = 0; i<num_Var; i++){
            if(sa_rng.next_double() < 1.0f / num_Var){
              flips[i] = true;
              any_flips = true;
              experimental_state[i] = -experimental_state[i];
            }
          }
        }

        float experimental_loss = (w_minus->contract_2D(experimental_state, false) + W_minus->contract_4D(experimental_state, false))/8;
        if(execution_mode == 5){
          if(experimental_loss < loss){
            circuit->flip_parameters(flips);
          }
        }
        else if(execution_mode == 6){
          if(sa_rng.next_double() < std::exp((loss - experimental_loss) / temperature)){
            circuit->flip_parameters(flips);
          }
        }

        delete[] experimental_state;
        delete[] flips;
      }

      delete[] gradients;
    }

    delete circuit;

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

  graph_results(diagram_name, number_of_repetitions, number_of_epochs, rounded_scores.data(), unrounded_scores.data(), loss_scores.data(), constraint_scores.data(), elapsed_time.count(), false, execution_mode);

  return 0;
}