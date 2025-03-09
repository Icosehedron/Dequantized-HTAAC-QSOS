#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int* checkCNFLine(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

  if (tokens.size() < 4 || tokens[3] != "0") {
    return nullptr;
  }

  int* result = new int[3];
  for (int i = 0; i < 3; ++i) {
    result[i] = std::stoi(tokens[i]);
  }

  return result;
}

std::vector<int*> parseExternalCNF(const std::string& fileName) {
  std::ifstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << fileName << std::endl;
    return std::vector<int*>();
  }

  std::string line;
  std::vector<int*> clauses;

  while(std::getline(file, line)) {
    int* clause = checkCNFLine(line);
    if(clause == nullptr){continue;}
    clauses.push_back(clause);
  }

  file.close();
  return clauses;
}


int main() {
  std::cout << parseExternalCNF("./imported_cnfs/s3v110c700-1.cnf").size() << std::endl;
  return 0;
}