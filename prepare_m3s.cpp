#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "sparse_tensor.hpp"

const std::string path_to_cnf = "./imported_cnfs/Max3Sat_2016/s3v110c700-1.cnf";
const std::string path_to_problem = "./problem/";

int* checkCNFLine(const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  std::vector<std::string> tokens;
  while(iss >> token) {
    tokens.push_back(token);
  }

  if (tokens.size() < 4 || tokens[3] != "0") {
    return nullptr;
  }

  int* result = new int[3];
  for (int i = 0; i < 3; ++i) {
    result[i] = std::stoi(tokens[i]);
  }

  return result;
}

std::pair<int, int**> parseExternalCNF(const std::string& fileName) {
  std::ifstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << fileName << std::endl;
    return std::make_pair(0, nullptr);
  }

  std::string line;
  std::vector<int*> clauses;

  while(std::getline(file, line)) {
    int* clause = checkCNFLine(line);
    if(clause == nullptr){continue;}
    clauses.push_back(clause);
  }

  file.close();
  int num_clauses = clauses.size();
  int** arr = new int*[num_clauses];
  for (size_t i = 0; i < num_clauses; ++i) {
    arr[i] = clauses[i];
  }
  return std::make_pair(num_clauses, arr);
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

std::tuple<int, SparseTensor*, int, SparseTensor*> generate_8W_mat(int n, int num_clauses, int** max3sat){
  int w_plus = 7*num_clauses; //The contribution is exactly 7/8 for every clause
  SparseTensor* w_minus = new SparseTensor(n+1, 2);
  int W_plus = num_clauses; //The contribution is exactly 1/8 for every clause
  SparseTensor* W_minus = new SparseTensor(n+1, 4);

  for(int r = 0; r<num_clauses; r++){
    int i = std::max(max3sat[r][0], -max3sat[r][0]);
    int j = std::max(max3sat[r][1], -max3sat[r][1]);
    int k = std::max(max3sat[r][2], -max3sat[r][2]);
    int i_sgn = max3sat[r][0] > 0 ? 1 : -1;
    int j_sgn = max3sat[r][1] > 0 ? 1 : -1;
    int k_sgn = max3sat[r][2] > 0 ? 1 : -1;

    //We will skip symmetrizing w_minus and W_minus because we don't need symmetry for backpropogation, and it also
    //makes it harder to store the matrix because it will become a matrix of floats as oppose to a matrix of ints.
    w_minus->insert(new int[2]{0, i}, -i_sgn);
    w_minus->insert(new int[2]{0, j}, -j_sgn);
    w_minus->insert(new int[2]{0, k}, -k_sgn);
    w_minus->insert(new int[2]{i, j}, i_sgn * j_sgn);
    w_minus->insert(new int[2]{i, k}, i_sgn * k_sgn);
    w_minus->insert(new int[2]{j, k}, j_sgn * k_sgn);
    W_minus->insert(new int[4]{0, i, j, k}, -i_sgn * j_sgn * k_sgn);

    //Even though it is true that we are using one extra dimension than necessary (the zero index has no real purpose),
    //since we are using a Compressed Sparse Row format, the space complexity is still O(num_clauses) regardless of the dimension
  }
  return std::make_tuple(w_plus, w_minus, W_plus, W_minus);
}

void writeParametersToFile(const std::string &filename, int num_clauses, int** max3sat, int w_plus, int W_plus) {
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Error opening file for writing\n";
    return;
  }
  // Write the path to the CNF file
  file << path_to_cnf << "\n";

  // Write the int value to the file
  file << num_clauses << "\n";

  // Write the values of w_plus and W_plus
  file << w_plus << "\n";
  file << W_plus << "\n";

  // Write the 2D array data
  for (int i = 0; i < num_clauses; ++i) {
    for (int j = 0; j < 3; ++j) {
      file << max3sat[i][j] << " ";
    }
    file << "\n"; // End of a row
  }

  file.close();
}

void writePopulationsToFile(const std::string &filename, int* populations, int size) {
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Error opening file for writing\n";
    return;
  }

  // Write the population data
  for (int i = 0; i < size; i++) {
    file << populations[i] << " ";
  }

  file.close();
}

int main() {
  std::pair<int, int**> max3sat = parseExternalCNF(path_to_cnf);
  if(max3sat.second == nullptr){return 1;}
  int n = get_max(max3sat.first, max3sat.second);

  std::tuple<int, SparseTensor*, int, SparseTensor*> W_matrices = generate_8W_mat(n, max3sat.first, max3sat.second);
  int w_plus = std::get<0>(W_matrices);
  SparseTensor* w_minus = std::get<1>(W_matrices);
  int W_plus = std::get<2>(W_matrices);
  SparseTensor* W_minus = std::get<3>(W_matrices);

  writeParametersToFile(path_to_problem + "parameters.txt", max3sat.first, max3sat.second, w_plus, W_plus);
  w_minus->writeToFile(path_to_problem + "w_minus_2d.txt");
  W_minus->writeToFile(path_to_problem + "W_minus_4d.txt");

  int* populations = new int[n+1];
  populations[0] = n+1;
  for(int i = 0; i<max3sat.first; i++){
    for(int j = 0; j<3; j++){
      populations[std::max(max3sat.second[i][j], -max3sat.second[i][j])] += 1;
    }
  }

  writePopulationsToFile(path_to_problem + "populations.txt", populations, n+1);

  return 0;
}