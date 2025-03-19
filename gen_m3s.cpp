#include <iostream>
#include <string>
#include <fstream>
#include "Xorshift32.hpp"

Xorshift32 rng(12345);
const int num_var = 3;
const int num_clauses = 1;

int* generate_clause(int num_var){
  int* clause = new int[3];
  for(int z = 0; z<3; z++){
    while(true){
      int tmp = static_cast<int>(rng.next_double() * num_var) + 1;
      int signed_tmp = rng.next() % 2 == 0 ? tmp : -tmp;
      bool repeated = false;
      for(int h = 0; h < z; h++){
        if(clause[h] == tmp || clause[h] == -tmp){
          repeated = true;
          break;
        }
      }
      if(!repeated){
        clause[z] = signed_tmp;
        break;
      }
    }
  }
  return clause;
}

std::string generate_max3sat(int num_var, int num_clauses){
  std::string max3sat = "";
  for(int i = 0; i<num_clauses; i++){
    int* clause = generate_clause(num_var);
    max3sat += std::to_string(clause[0]) + " " + std::to_string(clause[1]) + " " + std::to_string(clause[2]) + " 0\n";
    delete[] clause;
  }
  return max3sat;
}

int main() {
  std::string name = "v" + std::to_string(num_var) + "c" + std::to_string(num_clauses);
  std::string file_path = "./gen_max3sat/" + name + ".cnf";

  std::ofstream outFile(file_path);

  if (!outFile) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }

  std::string max3sat = generate_max3sat(num_var, num_clauses);
  outFile << max3sat;
  outFile.close();

  std::cout << "Max3Sat generated at " << file_path << std::endl;
  return 0;
}