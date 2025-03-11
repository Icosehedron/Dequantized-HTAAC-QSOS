#ifndef GATE_SET_HPP
#define GATE_SET_HPP
#define PI 3.14159265358979323846

#include <iostream>
#include <algorithm>
#include "Xorshift32.hpp"
#include "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense"

//Hyperparameters for optimizer (Adam)
const double adam_lr = 0.001; //Learning rate
const double adam_beta_1 = 0.9; //Adam default decay rate for momentum
const double adam_beta_2 = 0.999; //Adam default decay rate for variance

Xorshift32 rng(54321); //Will make initialization deterministic

struct RotationGate {
  private:
    double rotation_angle;
    double cos_val;
    double sin_val;
    int index1;
    int index2;
    double output1 = 0;
    double output2 = 0;
    double grad = 0;
    double adam_momentum = 0;
    double adam_variance = 0;
    int num_updates = 0;
  public:
  void update_angle(){
    cos_val = cos(rotation_angle);
    sin_val = sin(rotation_angle);
  }

  RotationGate(int index1, int index2) 
    : index1(index1), index2(index2) {
    // Initialize rotation_angle with random angle between 0 and 2*PI radians
    rotation_angle = rng.next_double() * 2 * PI;
    update_angle();
  }

  void feed_forward(double* state) {
    output1 = cos_val * state[index1] - sin_val * state[index2];
    output2 = sin_val * state[index1] + cos_val * state[index2];
    state[index1] = output1;
    state[index2] = output2;
  }

  void back_propagate(double* output_grad) {
    double grad1 = output_grad[index1] * cos_val + output_grad[index2] * sin_val;
    double grad2 = output_grad[index1] * -sin_val + output_grad[index2] * cos_val;
    output_grad[index1] = grad1;
    output_grad[index2] = grad2;
    grad = output1 * grad2 - output2 * grad1;
  }

  void update_parameters() {
    num_updates += 1;
    adam_momentum = adam_beta_1 * adam_momentum + (1 - adam_beta_1) * grad;
    adam_variance = adam_beta_2 * adam_variance + (1 - adam_beta_2) * grad * grad;
    double momentum = adam_momentum/(1 - pow(adam_beta_1, num_updates));
    double variance = adam_variance/(1 - pow(adam_beta_2, num_updates));
    rotation_angle -= adam_lr * momentum / (sqrt(variance) + 1e-8);
    update_angle();
  }
};

struct RotationLayer {
  private:
    int num_qubits;
    int num_gates;
    std::vector<RotationGate*> gates;
  public:
  RotationLayer(int num_qubits) 
    : num_qubits(num_qubits) {
    num_gates = num_qubits * (num_qubits - 1) / 2;
    for (int i = 0; i < num_qubits; i++) {
      for (int j = i + 1; j < num_qubits; j++) {
        gates.push_back(new RotationGate(i, j));
      }
    }
  }
  ~RotationLayer() {
    for (int i = 0; i<num_gates; i++) {
      delete gates[i]; // Deallocate each RotationGate object
    }
    gates.clear();
  }

  void feed_forward(double* state) {
    for(int i = 0; i<num_gates; i++) {gates[i]->feed_forward(state);}
  }

  void back_propagate(double* output_grad) {
    for(int i = num_gates - 1; i>=0; i--){gates[i]->back_propagate(output_grad);}
  }

  void update_parameters() {
    for(int i = 0; i<num_gates; i++){gates[i]->update_parameters();}
  }
};



#endif