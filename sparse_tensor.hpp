#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP

#include <iostream>
#include <vector>
#include <fstream>

struct SparseTensor {
  private:
    int num_Var;
    int rank;
    std::vector<int> values;
    std::vector<int*> indices;
    std::vector<int> sort_indices;
  public:
    // Constructor to initialize the CSR matrix
    SparseTensor(int num_Var, int rank)
      : num_Var(num_Var), rank(rank) // Initialize row_ptr with 0, numRows + 1 elements
    {}

    ~SparseTensor() {
      for (int* ind : indices) {
          delete[] ind; // Free each allocated array
          ind = nullptr;
      }
    }

    // Method to insert a non-zero element into the matrix
    void insert(int* ind, int val) {
      int sort_ind = 0;
      std::sort(ind, ind+rank); // Sort indices for consistent ascending order
      for(int i = 0; i < rank; i++){
        sort_ind = sort_ind * num_Var + ind[i];
      }
      auto it = std::lower_bound(sort_indices.begin(), sort_indices.end(), sort_ind);
      int index = std::distance(sort_indices.begin(), it);
      // If the column is already present in this row, update its value
      if (it != sort_indices.end() && *it == sort_ind) {
          values[index] += val;  // Add the new value to the existing one
          delete[] ind;
          ind = nullptr;
      } else {
          // Insert new column index and value in sorted order
          if(it == sort_indices.end()){
            sort_indices.push_back(sort_ind);
            values.push_back(val);
            indices.push_back(ind);
          }
          else{
            sort_indices.insert(sort_indices.begin() + index, sort_ind);
            values.insert(values.begin() + index, val);
            indices.insert(indices.begin() + index, ind);
          }
      }
    }

    // Method to display the matrix (for demonstration purposes)
    void display() {
      for (int i = 0; i < values.size(); i++) {
          std::cout << "Value: " << values[i] << " Indices: ";
          for (int j = 0; j < rank; j++) {
              std::cout << indices[i][j] << " ";
          }
          std::cout << std::endl;
      }
    }

    // Function to write file
    void writeToFile(const std::string &filename) const {
      std::ofstream file(filename);
      if (!file) {
        std::cerr << "Error opening file for writing\n";
        return;
      }

      file << values.size() << "\n";
      file << num_Var << "\n";
      file << rank << "\n";
    
      for (int i = 0; i < values.size(); i++) {
        file << values[i] << " ";
        for (int j = 0; j < rank; j++) {
            file << indices[i][j] << " ";
        }
        file << "\n";
      }

      file.close();
    }
  
    // Function to read file
    void readFromFile(const std::string &filename) {
      std::ifstream file(filename);
      if (!file) {
          std::cerr << "Error opening file for reading\n";
          return;
      }
  
      size_t valuesSize;
      
      file >> valuesSize;
      file >> num_Var;
      file >> rank;

      values.clear();
      indices.clear();
      sort_indices.clear();
  
      values.resize(valuesSize);
      indices.resize(valuesSize);
      for(size_t i = 0; i < valuesSize; i++) {
        indices[i] = new int[rank];
      }
  
      for (size_t i = 0; i < valuesSize; i++) {
        file >> values[i];
        for (int j = 0; j < rank; j++) {
          file >> indices[i][j];
        }
      }
  
      file.close();
    }
};

#endif