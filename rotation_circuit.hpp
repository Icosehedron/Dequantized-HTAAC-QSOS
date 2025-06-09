#ifndef ROTATION_CIRCUIT_HPP
#define ROTATION_CIRCUIT_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <Accelerate/Accelerate.h>
#include "Xorshift32.hpp"

//Hyperparameters for initialization
float initialization_range_r = 0.10f;

//Hyperparameters for optimizer (Adam)
const float adam_lr_base_r = 0.003f; //Learning rate
const float adam_beta_1_r = 0.9f; //Adam default decay rate for momentum
const float adam_beta_2_r = 0.999f; //Adam default decay rate for variance

Xorshift32 rng_r(54321); //Will make initialization deterministic

const float one_minus_beta1_r = 1.0f - adam_beta_1_r;
const float one_minus_beta2_r = 1.0f - adam_beta_2_r;

struct RotationGate{
  private:
    int num_Var;
    int source_index;
    int target_index;

    float angle;
    float* inputs;
    float* outputs;

    float angle_gradient;
    float adam_momentum = 0;
    float adam_variance = 0;
    int num_updates = 0;

  public:
    RotationGate(int num_Var, int source_index, int target_index)
      : num_Var(num_Var), source_index(source_index), target_index(target_index) {
      angle = (rng_r.next_double() * 2.0f - 1.0f) * initialization_range_r;

      inputs = new float[num_Var];
      outputs = new float[num_Var];
    }
    ~RotationGate() {
      delete[] inputs;
      delete[] outputs;
    }
    void feed_forward(float* state){
      std::copy(state, state + num_Var, inputs);
      float cos_angle = std::cos(angle);
      float sin_angle = std::sin(angle);

      state[source_index] = inputs[source_index] * cos_angle - inputs[target_index] * sin_angle;
      state[target_index] = inputs[target_index] * cos_angle + inputs[source_index] * sin_angle;
    }
    void back_propagate(float* g){
      float cos_angle = std::cos(angle);
      float sin_angle = std::sin(angle);
      angle_gradient = 0.0f;
      float g1 = g[source_index] * cos_angle + g[target_index] * sin_angle;
      float g2 = g[target_index] * cos_angle - g[source_index] * sin_angle;

      angle_gradient += (-sin_angle * inputs[source_index] - cos_angle * inputs[target_index]) * g[source_index];
      angle_gradient += (-sin_angle * inputs[target_index] + cos_angle * inputs[source_index]) * g[target_index];

      g[source_index] = g1;
      g[target_index] = g2;
    }
    void update_parameters(){
      num_updates++;
      const float beta1_correction = 1.0f - std::pow(adam_beta_1_r, num_updates);
      const float beta2_correction = 1.0f - std::pow(adam_beta_2_r, num_updates); 

      adam_momentum = adam_beta_1_r * adam_momentum + one_minus_beta1_r * angle_gradient;
      adam_variance = adam_beta_2_r * adam_variance + one_minus_beta2_r * angle_gradient * angle_gradient;

      float adam_momentum_corrected = adam_momentum / beta1_correction;
      float adam_variance_corrected = adam_variance / beta2_correction;

      angle -= adam_lr_base_r * adam_momentum_corrected / (std::sqrt(adam_variance_corrected) + 1e-8);
    }
};

struct RotationLayer{
  private:
    int num_Var;
    float* temporary_state;
    std::vector<RotationGate*> gates;
  public:
    RotationLayer(int num_Var)
      : num_Var(num_Var) {
      for(int i = 1; i<num_Var; i++){
        for(int j = 0; j + i<num_Var; j++){
          gates.push_back(new RotationGate(num_Var, j, j+i));
        }
      }
    }
    ~RotationLayer() {
      for (auto& gate : gates) {
        delete gate;
      }
      gates.clear();
    }
    void feed_forward(float* state){
      for (int i = 0; i<gates.size(); i++) {
        gates[i]->feed_forward(state);
      }
    }
    void back_propagate(float* g){
      for(int i = gates.size() - 1; i>=0; i--){
        gates[i]->back_propagate(g);
      }
    }
    void update_parameters(){
      for(int i = 0; i<gates.size(); i++){
        gates[i]->update_parameters();
      }
    }
};

struct RotationCircuit{
  private:
    int num_Var;
    int num_layers;
    float* g;
    std::vector<RotationLayer*> layers;
  public:
    RotationCircuit(int num_Var, int num_layers)
      : num_Var(num_Var), num_layers(num_layers) {
      for(int i = 0; i<num_layers; i++){
        layers.push_back(new RotationLayer(num_Var));
      }
      g = new float[num_Var];
    }
    ~RotationCircuit() {
      for(int i = 0; i<num_layers; i++){
        delete layers[i];
      }
      layers.clear();
      delete[] g;
    }
    void feed_forward(float* state){
      for(int i = 0; i<num_layers; i++){
        layers[i]->feed_forward(state);
      }
    }
    void back_propagate(float* grad){
      std::copy(grad, grad + num_Var, g);
      for(int i = num_layers - 1; i>=0; i--){
        layers[i]->back_propagate(g);
      }
    }
    void update_parameters(){
      for(int i = 0; i<num_layers; i++){
        layers[i]->update_parameters();
      }
    }
};

#endif