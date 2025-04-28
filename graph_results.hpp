#ifndef GRAPH_RESULTS_HPP
#define GRAPH_RESULTS_HPP

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "gnuplot-iostream.h"

void graph_results(std::string diagram_name, int number_of_reps, int number_of_epochs, float** rounded_scores, float** unrounded_scores, float** loss_scores, float** constraint_scores = nullptr, double elapsed_time = 0.0, bool compare = false, int execution_mode = 0){
  std::filesystem::create_directory("./saved_figures/");
  std::filesystem::create_directory("./saved_figures/" + diagram_name + "/");

  std::vector<float> best_by_reps;

  int best_index = -1;
  float best_score = -1.0f;
  float avg_score = 0.0f;
  for(int i = 0; i<number_of_reps; i++){
    float rep_best_score = -1.0f;
    for(int j = 0; j<number_of_epochs; j++){
      if(rounded_scores[i][j] > rep_best_score){
        rep_best_score = rounded_scores[i][j];
      }
    }
    if(rep_best_score > best_score){
      best_index = i;
      best_score = rep_best_score;
    }
    avg_score += rep_best_score;
    best_by_reps.push_back(rep_best_score);
  }
  avg_score /= number_of_reps;

  Gnuplot gp;
  gp << "set terminal pngcairo enhanced font 'Arial,12' size 800,600\n";

  std::vector<std::pair<int, float>> best_points;
  std::vector<std::pair<int, float>> avg_points;

  for(int i = 0; i < number_of_epochs; i++){
    best_points.push_back(std::make_pair(i, loss_scores[best_index][i]));
    float total = 0.0f;
    for(int j = 0; j < number_of_reps; j++){
      total += loss_scores[j][i];
    }
    avg_points.push_back(std::make_pair(i, total/number_of_reps));
  }
  gp << "set title 'Loss over Epochs'\n";
  gp << "set output './saved_figures/" << diagram_name << "/loss_over_epochs.png'\n";
  gp << "set xlabel 'Epochs'\n";
  gp << "set ylabel 'Loss'\n";
  gp << "plot '-' with linespoints lw 2 lc 'blue' title 'Best', "
        "'-' with lines lw 2 lc 'red' title 'Average'\n";
  gp.send1d(best_points);
  gp.send1d(avg_points);
  best_points.clear();
  best_points.shrink_to_fit();
  avg_points.clear();
  avg_points.shrink_to_fit();

  for(int i = 0; i < number_of_epochs; i++){
    best_points.push_back(std::make_pair(i, rounded_scores[best_index][i]));
    float total = 0.0f;
    for(int j = 0; j < number_of_reps; j++){
      total += rounded_scores[j][i];
    }
    avg_points.push_back(std::make_pair(i, total/number_of_reps));
  }
  gp << "set title 'Rounded HTAAC-QSOS Solution'\n";
  gp << "set output './saved_figures/" << diagram_name << "/rounded_sol_over_epochs.png'\n";
  gp << "set xlabel 'Epochs'\n";
  gp << "set ylabel 'Satisfied'\n";
  gp << "plot '-' with linespoints lw 2 lc 'blue' title 'Best', "
        "'-' with lines lw 2 lc 'red' title 'Average'\n";
  gp.send1d(best_points);
  gp.send1d(avg_points);
  best_points.clear();
  best_points.shrink_to_fit();
  avg_points.clear();
  avg_points.shrink_to_fit();

  for(int i = 0; i < number_of_reps; i++){
    best_points.push_back(std::make_pair(i, rounded_scores[i][number_of_epochs-1]));
    float nearby = 0.0f;
    int neighborhood = 0;
    for(int j = -20; j <= 20; j++){
      if(i+j >= 0 && i+j < number_of_reps){
        nearby += best_by_reps[i+j];
        neighborhood++;
      }
    }
    avg_points.push_back(std::make_pair(i, nearby/neighborhood));
  }
  gp << "set title 'Rounded HTAAC-QSOS Solution by Rep'\n";
  gp << "set output './saved_figures/" << diagram_name << "/rounded_sol_over_reps.png'\n";
  gp << "set xlabel 'Rep'\n";
  gp << "set ylabel 'Satisfied'\n";
  gp << "plot '-' with points pt 7 lc 'blue' title 'Final', "
        "'-' with lines lw 2 lc 'red' title 'Average'\n";
  gp.send1d(best_points);
  gp.send1d(avg_points);
  best_points.clear();
  best_points.shrink_to_fit();
  avg_points.clear();
  avg_points.shrink_to_fit();

  for(int i = 0; i < number_of_epochs; i++){
    best_points.push_back(std::make_pair(i, unrounded_scores[best_index][i]));
    float total = 0.0f;
    for(int j = 0; j < number_of_reps; j++){
      total += unrounded_scores[j][i];
    }
    avg_points.push_back(std::make_pair(i, total/number_of_reps));
  }
  gp << "set title 'Unrounded HTAAC-QSOS Solution'\n";
  gp << "set output './saved_figures/" << diagram_name << "/unrounded_sol_over_epochs.png'\n";
  gp << "set xlabel 'Epochs'\n";
  gp << "set ylabel 'Satisfied'\n";
  gp << "plot '-' with linespoints lw 2 lc 'blue' title 'Best', "
        "'-' with lines lw 2 lc 'red' title 'Average'\n";
  gp.send1d(best_points);
  gp.send1d(avg_points);
  best_points.clear();
  best_points.shrink_to_fit();
  avg_points.clear();
  avg_points.shrink_to_fit();

  if(compare){
    for(int i = 0; i < number_of_epochs; i++){
      float total1 = 0.0f;
      float total2 = 0.0f;
      for(int j = 0; j < number_of_reps; j++){
        if(j >= number_of_reps/2){total1 += rounded_scores[j][i];}
        else{total2 += rounded_scores[j][i];}
      }
      best_points.push_back(std::make_pair(i, 2*total1/number_of_reps));
      avg_points.push_back(std::make_pair(i, 2*total2/number_of_reps));
    }
    gp << "set title 'Rounded QSOS Solution by Choice'\n";
    gp << "set output './saved_figures/" << diagram_name << "/rounded_sol_by_factor.png'\n";
    gp << "set xlabel 'Epochs'\n";
    gp << "set ylabel 'Satisfied'\n";
    gp << "plot '-' with linespoints lw 2 lc 'blue' title 'Choice (second half)', "
          "'-' with lines lw 2 lc 'red' title 'Choice (first half)'\n";
    gp.send1d(best_points);
    gp.send1d(avg_points);
    best_points.clear();
    best_points.shrink_to_fit();
    avg_points.clear();
    avg_points.shrink_to_fit();
  }

  if(constraint_scores != nullptr){
    for(int i = 0; i < number_of_epochs; i++){
      best_points.push_back(std::make_pair(i, constraint_scores[best_index][i]));
      float total = 0.0f;
      for(int j = 0; j < number_of_reps; j++){
        total += constraint_scores[j][i];
      }
      avg_points.push_back(std::make_pair(i, total/number_of_reps));
    }
    gp << "set title 'HTAAC-QSOS Constraint'\n";
    gp << "set output './saved_figures/" << diagram_name << "/constraint_over_epochs.png'\n";
    gp << "set xlabel 'Epochs'\n";
    gp << "set ylabel 'Deviation'\n";
    gp << "plot '-' with linespoints lw 2 lc 'blue' title 'Best', "
          "'-' with lines lw 2 lc 'red' title 'Average'\n";
    gp.send1d(best_points);
    gp.send1d(avg_points);
    best_points.clear();
    best_points.shrink_to_fit();
    avg_points.clear();
    avg_points.shrink_to_fit();
  }

  std::ofstream file("./saved_figures/" + diagram_name + "/scores.txt");
  if (!file) {
    std::cerr << "Error opening file for writing\n";
    return;
  }

  switch (execution_mode) {
    case 0:
      file << "Execution mode: DQSOS\n";
      break;
    case 1:
      file << "Execution mode: DQSOS + Local Search\n";
      break;
    case 2:
      file << "Execution mode: DQSOS + Simulated Annealing\n";
      break;
    case 3:
      file << "Execution mode: DQSOS-guided Local Search\n";
      break;
    case 4:
      file << "Execution mode: DQSOS-guided Simulated Annealing\n";
      break;
    case 5:
      file << "Execution mode: Local Search\n";
      break;
    case 6:
      file << "Execution mode: Simulated Annealing\n";
      break;
    default:
      file << "Execution mode: Unknown\n";
      break;
  }
  
  file << "Best index: " << best_index << " of " << number_of_reps << "\n";
  file << "Best rounded score: " << best_score << "\n";
  file << "Average rounded score: " << avg_score << "\n";
  file << "Number of epochs: " << number_of_epochs << "\n";
  if(elapsed_time > 0.0){
    file << "Elapsed time: " << elapsed_time << " seconds\n";
    file << "Time taken to reach best score: " << (elapsed_time/number_of_reps * (best_index + 1)) << " seconds\n";
  }

  file.close();
}

#endif // GRAPH_RESULTS_HPP