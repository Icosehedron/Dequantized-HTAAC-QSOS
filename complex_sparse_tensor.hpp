#ifndef COMPLEX_SPARSE_TENSOR_HPP
#define COMPLEX_SPARSE_TENSOR_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <Accelerate/Accelerate.h>

struct ComplexSparseTensor {
  private:
    int num_Var;
    int rank;
    int num_entries;
    float alpha;
    std::vector<float> values;
    std::vector<int>* indices;
    float* input_angle;
    float* input_real;
    float* input_imaginary;

    float* compiled_values;
    float* compiled_indices;

    float* vindex_storage_real;
    float* vindex_storage_imaginary;
    float* output_storage_real;
    float* output_storage_imaginary;
    
  public:
    ComplexSparseTensor(float alpha)
      : alpha(alpha)
    {}

    ~ComplexSparseTensor() {
      for(int i = 0; i<rank; i++) {
          indices[i].clear();
          indices[i].shrink_to_fit();
      }
      delete[] compiled_values;
      delete[] compiled_indices;
      delete[] input_real;
      delete[] input_imaginary;
      delete[] vindex_storage_real;
      delete[] vindex_storage_imaginary;
      delete[] output_storage_real;
      delete[] output_storage_imaginary;
    }
  
    // Function to read file
    void readFromFile(const std::string &filename) {
      std::ifstream file(filename);
      if (!file) {
          std::cerr << "Error opening file for reading\n";
          return;
      }

      file >> num_entries;
      file >> num_Var;
      file >> rank;

      values.resize(num_entries);
      indices = new std::vector<int>[rank];
      for(int i = 0; i < rank; i++){
        indices[i] = std::vector<int>(num_entries);
      }
  
      for (int i = 0; i < num_entries; i++) {
        file >> values[i];
        for (int j = 0; j < rank; j++) {
          file >> indices[j][i];
        }
      }
  
      file.close();
    }

    void prepare_for_execution(){
      compiled_values = new float[num_entries];
      compiled_indices = new float[rank * num_entries];
      vindex_storage_real = new float[rank * num_entries];
      vindex_storage_imaginary = new float[rank * num_entries];
      output_storage_real = new float[num_entries];
      output_storage_imaginary = new float[num_entries];

      for(int i = 0; i < rank; i++){
        std::copy(indices[i].begin(), indices[i].end(), compiled_indices + i * num_entries);
      }
      for(int i = 0; i < num_entries; i++){
        compiled_values[i] = values[i] / 8;
      }
    }

    float feed_forward(float* angles){
      const float neg_alpha = -1 * alpha;
      constexpr float zero = 0.0f;
      input_angle = angles;
      vvsincosf(input_real, input_imaginary, input_angle, &num_Var);
      vDSP_vsmul(input_imaginary, 1, &neg_alpha, input_imaginary, 1, num_Var);

      vDSP_vindex(input_real, compiled_indices, 1, vindex_storage_real, 1, rank * num_entries);
      vDSP_vindex(input_imaginary, compiled_indices, 1, vindex_storage_imaginary, 1, rank * num_entries);

      std::copy(compiled_values, compiled_values + num_entries, output_storage_real);
      vDSP_vfill(&zero, output_storage_imaginary, 1, num_entries);

      DSPSplitComplex vindex_storage = {vindex_storage_real, vindex_storage_imaginary};
      DSPSplitComplex output_storage = {output_storage_real, output_storage_imaginary};

      for(int i = 0; i<rank; i++){
        vDSP_zvmul(&vindex_storage + i * num_entries, 1, &output_storage, 1, &output_storage, 1, num_entries, 1);
      }

      float result;
      vDSP_sve(output_storage_real, 1, &result, num_entries);
      return result;
    }
};

float dot_product(const float* a, const float* b, int size){
  return cblas_sdot(size, a, 1, b, 1);
}

#endif