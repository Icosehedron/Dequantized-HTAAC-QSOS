#ifndef QUANTUM_CIRCUIT_HPP
#define QUANTUM_CIRCUIT_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <Accelerate/Accelerate.h>
#include "Xorshift32.hpp"

//Hyperparameters for initialization
float initialization_range_q = 0.10f;

//Hyperparameters for optimizer (Adam)
const float adam_lr_base_q = 0.003f; //Learning rate
const float adam_beta_1_q = 0.9f; //Adam default decay rate for momentum
const float adam_beta_2_q = 0.999f; //Adam default decay rate for variance

Xorshift32 rng_q(54321); //Will make initialization deterministic

const float one_minus_beta1_q = 1.0f - adam_beta_1_q;
const float one_minus_beta2_q = 1.0f - adam_beta_2_q;

struct QuantumGate{
  private:
    int num_qubits;
    int num_dim;

    int target_qubit;
    int target_index;

    float angle;
    float* inputs;
    float* outputs;

    float angle_gradient;
    float adam_momentum = 0;
    float adam_variance = 0;
    int num_updates = 0;

  public:
    QuantumGate(int num_qubits, int target_qubit)
      : num_qubits(num_qubits), target_qubit(target_qubit) {
      num_dim = 1 << num_qubits;
      target_index = 1 << target_qubit;
      angle = (rng_q.next_double() * 2.0f - 1.0f) * initialization_range_q;

      inputs = new float[num_dim];
      outputs = new float[num_dim];
    }
    ~QuantumGate() {
      delete[] inputs;
      delete[] outputs;
    }
    void feed_forward(float* state){
      std::copy(state, state + num_dim, inputs);
      float cos_angle = std::cos(angle);
      float sin_angle = std::sin(angle);

      for(int i = 0; i<num_dim; i++){
        if((i & target_index) == 0){
          state[i] = inputs[i] * cos_angle - inputs[i ^ target_index] * sin_angle;
          state[i ^ target_index] = inputs[i ^ target_index] * cos_angle + inputs[i] * sin_angle;
        }
      }
    }
    void back_propagate(float* g){
      float cos_angle = std::cos(angle);
      float sin_angle = std::sin(angle);
      angle_gradient = 0.0f;
      for(int i = 0; i<num_dim; i++){
        if((i & target_index) == 0){
          float g1 = g[i] * cos_angle + g[i ^ target_index] * sin_angle;
          float g2 = g[i ^ target_index] * cos_angle - g[i] * sin_angle;

          angle_gradient += (-sin_angle * inputs[i] - cos_angle * inputs[i ^ target_index]) * g[i];
          angle_gradient += (-sin_angle * inputs[i ^ target_index] + cos_angle * inputs[i]) * g[i ^ target_index];

          g[i] = g1;
          g[i ^ target_index] = g2;
        }
      }
    }
    void update_parameters(){
      num_updates++;
      const float beta1_correction = 1.0f - std::pow(adam_beta_1_q, num_updates);
      const float beta2_correction = 1.0f - std::pow(adam_beta_2_q, num_updates); 

      adam_momentum = adam_beta_1_q * adam_momentum + one_minus_beta1_q * angle_gradient;
      adam_variance = adam_beta_2_q * adam_variance + one_minus_beta2_q * angle_gradient * angle_gradient;

      float adam_momentum_corrected = adam_momentum / beta1_correction;
      float adam_variance_corrected = adam_variance / beta2_correction;

      angle -= adam_lr_base_q * adam_momentum_corrected / (std::sqrt(adam_variance_corrected) + 1e-8);
    }
};

struct QuantumLayer{
  private:
    int num_qubits;
    int num_dim;
    int* cz0;
    int* cz1;
    float* temporary_state;
    std::vector<QuantumGate*> gates0;
    std::vector<QuantumGate*> gates1;
  public:
    QuantumLayer(int num_qubits, int* cz0, int* cz1)
      : num_qubits(num_qubits), cz0(cz0), cz1(cz1) {
      num_dim = 1 << num_qubits;
      temporary_state = new float[num_dim];
      for(int i = 0; i<num_qubits; i++){
        gates0.push_back(new QuantumGate(num_qubits, i));
        gates1.push_back(new QuantumGate(num_qubits, i));
      }
    }
    ~QuantumLayer() {
      for(int i = 0; i<num_qubits; i++){
        delete gates0[i];
        delete gates1[i];
      }
    }
    void feed_forward(float* state){
      for(int i = 0; i<num_qubits; i++){
        gates0[i]->feed_forward(state);
      }
      for(int i = 0; i<num_dim; i++){
        temporary_state[i] = state[cz0[i]];
      }
      for(int i = 0; i<num_dim; i++){
        state[i] = temporary_state[i];
      }
      for(int i = 0; i<num_qubits; i++){
        gates1[i]->feed_forward(state);
      }
      for(int i = 0; i<num_dim; i++){
        temporary_state[i] = state[cz1[i]];
      }
      for(int i = 0; i<num_dim; i++){
        state[i] = temporary_state[i];
      }
    }
    void back_propagate(float* g){
      for(int i = 0; i<num_dim; i++){
        temporary_state[i] = g[cz1[i]];
      }
      for(int i = 0; i<num_dim; i++){
        g[i] = temporary_state[i];
      }
      for(int i = num_qubits - 1; i>=0; i--){
        gates1[i]->back_propagate(g);
      }
      for(int i = 0; i<num_dim; i++){
        temporary_state[i] = g[cz0[i]];
      }
      for(int i = 0; i<num_dim; i++){
        g[i] = temporary_state[i];
      }
      for(int i = num_qubits - 1; i>=0; i--){
        gates0[i]->back_propagate(g);
      }
    }
    void update_parameters(){
      for(int i = 0; i<num_qubits; i++){
        gates0[i]->update_parameters();
        gates1[i]->update_parameters();
      }
    }
};

struct QuantumCircuit{
  private:
    int num_qubits;
    int num_layers;
    int num_dim;
    int num_Var;
    float* g;
    std::vector<QuantumLayer*> layers;
  public:
    QuantumCircuit(int num_qubits, int num_Var, int* cz0, int* cz1, int num_layers)
      : num_qubits(num_qubits), num_Var(num_Var), num_layers(num_layers) {
      for(int i = 0; i<num_layers; i++){
        layers.push_back(new QuantumLayer(num_qubits, cz0, cz1));
      }
      num_dim = 1 << num_qubits;
      g = new float[num_dim];
    }
    ~QuantumCircuit() {
      for(int i = 0; i<num_layers; i++){
        delete layers[i];
      }
    }
    void feed_forward(float* state){
      for(int i = 0; i<num_layers; i++){
        layers[i]->feed_forward(state);
      }
    }
    void back_propagate(float* grad){
      std::fill(g, g + num_dim, 0.0f);
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