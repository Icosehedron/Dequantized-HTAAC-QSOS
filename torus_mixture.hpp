#ifndef TORUS_MIXTURE_HPP
#define TORUS_MIXTURE_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <Accelerate/Accelerate.h>
#include "Xorshift32.hpp"

//Hyperparameters for initialization
float initialization_range = 0.06f;

//Hyperparameters for optimizer (Adam)
const float adam_lr_base = 0.01f; //Learning rate
const float adam_beta_1 = 0.9f; //Adam default decay rate for momentum
const float adam_beta_2 = 0.999f; //Adam default decay rate for variance

Xorshift32 rng(54321); //Will make initialization deterministic
constexpr float INV_SQRT_2 = 0.70710678119f; // Precomputed value of 1/sqrt(2)

const float one_minus_beta1 = 1.0f - adam_beta_1;
const float one_minus_beta2 = 1.0f - adam_beta_2;

struct TorusModel{
  private:
    int num_Var;
    int num_parameters;
    bool allow_degree_2;
    float proper_lr;

    float* parameters;
    float* gradients;
    float* output;
    float* sin_output;
    float* cos_output;
    float* output_gradients;

    float* adam_momentum;
    float* adam_variance;
    float* gradients_squared;
    float* adam_momentum_corrected;
    float* adam_variance_corrected;
    int num_updates = 0;

    std::vector<int> directory_first;
    std::vector<int> directory_second;
    std::vector<int> directory_sign;
    std::vector<float> values_fw;
    std::vector<float> values_bw;
    std::vector<int> row_indices_fw;
    std::vector<long> column_starts_fw;
    std::vector<int> row_indices_bw;
    std::vector<long> column_starts_bw;

    SparseMatrixStructure forward_structure;
    SparseMatrixStructure backward_structure;

    SparseMatrix_Float forward;
    SparseMatrix_Float backward;
    DenseVector_Float input_fw;
    DenseVector_Float output_fw;
    DenseVector_Float input_bw;
    DenseVector_Float output_bw;
  public:
    TorusModel(int num_Var, bool allow_degree_2 = true)
      : num_Var(num_Var), allow_degree_2(allow_degree_2) {
      num_parameters = allow_degree_2 ? num_Var * num_Var : num_Var;
      proper_lr = adam_lr_base * num_Var / num_parameters;

      parameters = new float[num_parameters];
      gradients = new float[num_parameters];
      output = new float[num_Var];
      sin_output = new float[num_Var];
      cos_output = new float[num_Var];
      output_gradients = new float[num_Var];

      adam_momentum = new float[num_parameters];
      adam_variance = new float[num_parameters];
      gradients_squared = new float[num_parameters];
      adam_momentum_corrected = new float[num_parameters];
      adam_variance_corrected = new float[num_parameters];
      std::fill(adam_momentum, adam_momentum + num_parameters, 0.0f);
      std::fill(adam_variance, adam_variance + num_parameters, 0.0f);

      float initialization_factor = initialization_range * std::sqrt((float) num_Var / num_parameters * 3.0f);
      for(int i = 0; i<num_parameters; i++){
        parameters[i] = (rng.next_double() * 2.0f - 1.0f) * initialization_factor;
      }

      for(int i = 0; i < num_Var; i++){
        directory_first.push_back(i);
        directory_second.push_back(-1);
        directory_sign.push_back(1);

        if(allow_degree_2){
          for(int j = i+1; j < num_Var; j++){
            directory_first.push_back(i);
            directory_second.push_back(j);
            directory_sign.push_back(1);

            directory_first.push_back(i);
            directory_second.push_back(j);
            directory_sign.push_back(-1);
          }
        }
      }

      long current_column_start = 0;

      for(int i = 0; i<directory_first.size(); i++){
        column_starts_fw.push_back(current_column_start);

        if(directory_second[i] == -1){
          row_indices_fw.push_back(directory_first[i]);
          values_fw.push_back(1.0f);
          current_column_start++;
          continue;
        }

        row_indices_fw.push_back(directory_first[i]);
        values_fw.push_back(INV_SQRT_2);
        current_column_start++;
        
        row_indices_fw.push_back(directory_second[i]);
        values_fw.push_back(INV_SQRT_2 * directory_sign[i]);
        current_column_start++;
      }
      column_starts_fw.push_back(current_column_start);

      forward_structure = SparseMatrixStructure({
        .rowCount = num_Var,
        .columnCount = num_parameters,
        .columnStarts = column_starts_fw.data(),
        .rowIndices = row_indices_fw.data(),
        .attributes = {
          .kind = SparseOrdinary,
        },
        .blockSize = 1
      });

      forward = SparseMatrix_Float({.structure = forward_structure, .data = values_fw.data()});
      input_fw = DenseVector_Float({.count = num_parameters, .data = parameters});
      output_fw = DenseVector_Float({.count = num_Var, .data = output});

      current_column_start = 0;

      for(int i = 0; i<num_Var; i++){
        column_starts_bw.push_back(current_column_start);
        for(int j = 0; j<directory_first.size(); j++){
          if(directory_first[j] == i){
            if(directory_second[j] == -1){
              row_indices_bw.push_back(j);
              values_bw.push_back(1.0f);
              current_column_start++;
            }
            else{
              row_indices_bw.push_back(j);
              values_bw.push_back(INV_SQRT_2);
              current_column_start++;
            }
          }
          else if(directory_second[j] == i){
            row_indices_bw.push_back(j);
            values_bw.push_back(INV_SQRT_2 * directory_sign[j]);
            current_column_start++;
          }
        }
      }
      column_starts_bw.push_back(current_column_start);

      backward_structure = SparseMatrixStructure({
        .rowCount = num_parameters,
        .columnCount = num_Var,
        .columnStarts = column_starts_bw.data(),
        .rowIndices = row_indices_bw.data(),
        .attributes = {
          .kind = SparseOrdinary,
        },
        .blockSize = 1
      });

      backward = SparseMatrix_Float({.structure = backward_structure, .data = values_bw.data()});
      input_bw = DenseVector_Float({.count = num_Var, .data = output_gradients});
      output_bw = DenseVector_Float({.count = num_parameters, .data = gradients});
    }
    ~TorusModel() {
      delete[] parameters;
      delete[] gradients;
      delete[] output;
      delete[] cos_output;
      delete[] sin_output;
      delete[] output_gradients;
      delete[] adam_momentum;
      delete[] adam_variance;
      delete[] gradients_squared;
      delete[] adam_momentum_corrected;
      delete[] adam_variance_corrected;
    }
    std::pair<float*, float*> feed_forward_pair(){
      SparseMultiply(forward, input_fw, output_fw);
      vvcosf(cos_output, output, &num_Var);
      vvsinf(sin_output, output, &num_Var);
      return std::make_pair(sin_output, cos_output);
    }
    void back_propagate(float* grad_of_sin_output, float* grad_of_cos_output){
      vDSP_vmul(grad_of_cos_output, 1, sin_output, 1, output_gradients, 1, num_Var);
      vDSP_vmsb(grad_of_sin_output, 1, cos_output, 1, output_gradients, 1, output_gradients, 1, num_Var);
      SparseMultiply(backward, input_bw, output_bw);
    }
    void update_parameters(){
      num_updates++;
      const float beta1_correction = 1.0f - std::pow(adam_beta_1, num_updates);
      const float beta2_correction = 1.0f - std::pow(adam_beta_2, num_updates); 

      vDSP_vsmsma(adam_momentum, 1, &adam_beta_1, gradients, 1, &one_minus_beta1, adam_momentum, 1, num_parameters);

      vDSP_vsq(gradients, 1, gradients_squared, 1, num_parameters);
      vDSP_vsmsma(adam_variance, 1, &adam_beta_2, gradients_squared, 1, &one_minus_beta2, adam_variance, 1, num_parameters);

      vDSP_vsdiv(adam_momentum, 1, &beta1_correction, adam_momentum_corrected, 1, num_parameters);
      vDSP_vsdiv(adam_variance, 1, &beta2_correction, adam_variance_corrected, 1, num_parameters);

      for(int i = 0; i<num_parameters; i++){
        parameters[i] -= proper_lr * adam_momentum_corrected[i] / (std::sqrt(adam_variance_corrected[i]) + 1e-8);
      }
    }
};

#endif