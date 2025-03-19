#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <Accelerate/Accelerate.h>

struct SparseTensor {
  private:
    int num_Var;
    int rank;
    std::vector<float> values;
    std::vector<int>* indices;
    std::vector<int> sort_indices;
    int num_entries;
    float* input_state;
    float* values_compiled;
    float* values_compiled_doe;
    float* indices_compiled;
    float* index_access;
  public:
    // Constructor to initialize the CSR matrix
    SparseTensor(const int num_Var, const int rank)
      : num_Var(num_Var), rank(rank) // Initialize row_ptr with 0, numRows + 1 elements
    {
      indices = new std::vector<int>[rank];
    }

    ~SparseTensor() {
      for(int i = 0; i<rank; i++) {
          indices[i].clear();
          indices[i].shrink_to_fit();
      }
      delete[] indices_compiled;
      delete[] values_compiled_doe;
      delete[] indices;
      delete[] index_access;
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
            for(int i = 0; i < rank; i++){
              indices[i].push_back(ind[i]);
            }
          }
          else{
            sort_indices.insert(sort_indices.begin() + index, sort_ind);
            values.insert(values.begin() + index, val);
            for(int i = 0; i < rank; i++){
              indices[i].insert(indices[i].begin() + index, ind[i]);
            }
          }
      }
    }

    // Method to display the matrix (for demonstration purposes)
    void display() {
      for (int i = 0; i < values.size(); i++) {
          std::cout << "Value: " << values[i] << " Indices: ";
          for (int j = 0; j < rank; j++) {
              std::cout << indices[j][i] << " ";
          }
          std::cout << std::endl;
      }
    }

    // Sparse Tensor Contraction (Accelerate framework)
    float contract_4D(float* v, bool storeInput = true) {
      if(storeInput){input_state = v;}
      vDSP_vindex(v, indices_compiled, 1, index_access, 1, 4 * num_entries);
      vDSP_vmul(index_access, 1, index_access + 2 * num_entries, 1, index_access, 1, 2 * num_entries);
      vDSP_vmul(index_access, 1, index_access + num_entries, 1, index_access, 1, num_entries);
      return cblas_sdot(num_entries, index_access, 1, values_compiled, 1);
    }
    
    // Sparse Tensor Contraction (plain)
    float contract_4D_plain(float* v,  bool storeInput = true) {
      if(storeInput){input_state = v;}
      float sum = 0.0;
      for(int i = 0; i<values.size(); i++){
        sum += values[i] * v[indices[0][i]] * v[indices[1][i]] * v[indices[2][i]] * v[indices[3][i]];
      }
      return sum;
    }

    //Sparse Tensor Backpropagation (Accelerate framework)
    void back_contract_4D(float* g, float focus = 1.0f) {
      vDSP_vindex(input_state, indices_compiled, 1, index_access, 1, 4 * num_entries);
      vDSP_vmul(index_access + num_entries, 1, index_access + 3 * num_entries, 1, index_access + 4 * num_entries, 1, num_entries); //bd, ac
      vDSP_vmul(index_access, 1, index_access + 2 * num_entries, 1, index_access + 5 * num_entries, 1, num_entries);

      vDSP_vmul(index_access + 4 * num_entries, 1, values_compiled_doe, 1, index_access + 4 * num_entries, 1, 2 * num_entries); //bd v, ac v

      vDSP_vmul(index_access, 1, index_access + 4 * num_entries, 1, index_access, 1, 2 * num_entries); //abd v at a, abc at b
      vDSP_vmul(index_access + 2 * num_entries, 1, index_access + 4 * num_entries, 1, index_access + 2 * num_entries, 1, 2 * num_entries); //bcd at c, acd at d
      vDSP_vsmul(index_access, 1, &focus, index_access, 1, 4 * num_entries);

      for(int i = 0; i<num_entries; i++){
        g[indices[0][i]] += index_access[i + 2 * num_entries];
        g[indices[1][i]] += index_access[i + 3 * num_entries];
        g[indices[2][i]] += index_access[i];
        g[indices[3][i]] += index_access[i + num_entries];
      }
    }

    //Sparse Tensor Backpropagation (plain)
    void back_contract_4D_plain(float* g) {
      for(int i = 0; i<num_entries; i++){
        g[indices[0][i]] += values[i] * input_state[indices[1][i]] * input_state[indices[2][i]] * input_state[indices[3][i]];
        g[indices[1][i]] += values[i] * input_state[indices[0][i]] * input_state[indices[2][i]] * input_state[indices[3][i]];
        g[indices[2][i]] += values[i] * input_state[indices[0][i]] * input_state[indices[1][i]] * input_state[indices[3][i]];
        g[indices[3][i]] += values[i] * input_state[indices[0][i]] * input_state[indices[1][i]] * input_state[indices[2][i]];
      }
    }

    // Sparse Tensor Contraction (Accelerate framework)
    float contract_2D(float* v, bool storeInput = true) {
      if(storeInput){input_state = v;}
      vDSP_vindex(v, indices_compiled, 1, index_access, 1, 2 * num_entries);
      vDSP_vmul(index_access, 1, index_access + num_entries, 1, index_access, 1, num_entries);
      return cblas_sdot(num_entries, index_access, 1, values_compiled, 1);
    }
    
    // Sparse Tensor Contraction (plain)
    float contract_2D_plain(float* v, bool storeInput = true) {
      if(storeInput){input_state = v;}
      float sum = 0.0f;
      for(int i = 0; i<values.size(); i++){
        sum += values[i] * v[indices[0][i]] * v[indices[1][i]];
      }
      return sum;
    }

    //Sparse Tensor Backpropagation (Accelerate framework)
    void back_contract_2D(float* g, float focus = 1.0f) {
      vDSP_vindex(input_state, indices_compiled, 1, index_access, 1, 2 * num_entries);
      vDSP_vmul(index_access, 1, values_compiled_doe, 1, index_access, 1, num_entries * 2);
      vDSP_vsmul(index_access, 1, &focus, index_access, 1, num_entries * 2);

      for(int i = 0; i<num_entries; i++){
        g[indices[0][i]] += index_access[i + num_entries];
        g[indices[1][i]] += index_access[i];
      }
    }

    //Sparse Tensor Backpropagation (plain)
    void back_contract_2D_plain(float* g) {
      for(int i = 0; i<num_entries; i++){
        g[indices[0][i]] += values[i] * input_state[indices[1][i]];
        g[indices[1][i]] += values[i] * input_state[indices[0][i]];
      }
    }

    void reportInput(){
      std::cout << "input_state: " << input_state[0] << " " << input_state[1] << " " << input_state[2] << " " << input_state[3] << std::endl;
    }

    // Function to write file
    void writeToFile(const std::string &filename) {
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
            file << indices[j][i] << " ";
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
      delete[] indices;
      sort_indices.clear();
  
      values.resize(valuesSize);
      indices = new std::vector<int>[rank];
      for(int i = 0; i < rank; i++) {
        indices[i] = std::vector<int>(valuesSize);
      }
  
      for (size_t i = 0; i < valuesSize; i++) {
        file >> values[i];
        for (int j = 0; j < rank; j++) {
          file >> indices[j][i];
        }
      }
  
      file.close();
    }

    void prepare_for_execution(){
      num_entries = values.size();
      values_compiled = values.data();
      indices_compiled = new float[rank * num_entries];
      index_access = new float[(rank + 2) * num_entries];
      values_compiled_doe = new float[2 * num_entries];
      for(int i = 0; i<num_entries; i++){
        values_compiled_doe[i] = values_compiled[i] / 8;
        values_compiled_doe[i + num_entries] = values_compiled[i] / 8;
      }

      for(int i = 0; i < rank; i++){
        std::copy(indices[i].begin(), indices[i].end(), indices_compiled + i * num_entries);
      }
    }
};

float dot_product(const float* a, const float* b, int size){
  return cblas_sdot(size, a, 1, b, 1);
}

#endif