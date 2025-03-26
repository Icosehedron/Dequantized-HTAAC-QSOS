#ifndef GATE_SET_HPP
#define GATE_SET_HPP
#define PI 3.14159265358979323846

#include <iostream>
#include <algorithm>
#include "Xorshift32.hpp"

//Hyperparameters for optimizer (Adam)
const float adam_lr_base = 0.001; //Learning rate
const float adam_beta_1 = 0.9; //Adam default decay rate for momentum
const float adam_beta_2 = 0.999; //Adam default decay rate for variance

Xorshift32 rng(54321); //Will make initialization deterministic

struct RotationGate {
  private:
    float rotation_angle;
    float cos_val;
    float sin_val;
    int index1;
    int index2;
    float adam_lr;
    float output1 = 0.0f;
    float output2 = 0.0f;
    float grad = 0.0f;
    float adam_momentum = 0.0f;
    float adam_variance = 0.0f;
    int num_updates = 0;
  public:
  void update_angle(){
    cos_val = std::cos(rotation_angle);
    sin_val = std::sin(rotation_angle);
  }

  RotationGate(const int index1, const int index2, const int gate_repetitions) 
    : index1(index1), index2(index2) {
    // Initialize rotation_angle with random angle between 0 and 2*PI radians
    rotation_angle = rng.next_double() * 2 * PI;
    adam_lr = adam_lr_base / gate_repetitions;
    update_angle();
  }

  void feed_forward(float* state) {
    output1 = cos_val * state[index1] - sin_val * state[index2];
    output2 = sin_val * state[index1] + cos_val * state[index2];
    state[index1] = output1;
    state[index2] = output2;
  }

  void back_propagate(float* output_grad) {
    float grad1 = output_grad[index1] * cos_val + output_grad[index2] * sin_val;
    float grad2 = output_grad[index1] * -sin_val + output_grad[index2] * cos_val;
    grad = output1 * output_grad[index2] -output2 * output_grad[index1];
    output_grad[index1] = grad1;
    output_grad[index2] = grad2;
  }

  void update_parameters() {
    num_updates += 1;
    adam_momentum = adam_beta_1 * adam_momentum + (1 - adam_beta_1) * grad;
    adam_variance = adam_beta_2 * adam_variance + (1 - adam_beta_2) * grad * grad;
    float momentum = adam_momentum/(1 - std::pow(adam_beta_1, num_updates));
    float variance = adam_variance/(1 - std::pow(adam_beta_2, num_updates));
    rotation_angle -= adam_lr * momentum / (std::sqrt(variance) + 1e-8);
    update_angle();
  }
};

struct RotationLayer {
  private:
    int num_dim;
    int hilbert_dimension;
    int num_gates;
    std::vector<RotationGate*> gates;
  public:
  RotationLayer(const int num_dim, const int gate_repetitions) 
    : num_dim(num_dim) {
    hilbert_dimension = num_dim;
    num_gates = hilbert_dimension * (hilbert_dimension - 1) / 2;
    for (int i = 1; i < hilbert_dimension; i++) {
      for (int j = 0; j + i < hilbert_dimension; j++) {
        gates.push_back(new RotationGate(j, j+i, gate_repetitions));
      }
    }
  }
  ~RotationLayer() {
    for (int i = 0; i<num_gates; i++) {
      delete gates[i]; // Deallocate each RotationGate object
    }
    gates.clear();
  }

  void feed_forward(float* state) {
    for(int i = 0; i<num_gates; i++) {gates[i]->feed_forward(state);}
  }

  void back_propagate(float* output_grad) {
    for(int i = num_gates - 1; i>=0; i--){gates[i]->back_propagate(output_grad);}
  }

  void update_parameters() {
    for(int i = 0; i<num_gates; i++){gates[i]->update_parameters();}
  }
};



#endif