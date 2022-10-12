#define LOG
#include "pso/pso.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

constexpr double PI = 3.1415926525;

int main(void) {

  auto cost_func = [&](const std::vector<double> &features) {
    double x = features[0];
    double y = features[1];
    double temp = sqrt(x * x + y * y);
    double cost = sin(temp) / temp +
                  exp(0.5 * cos(2 * PI * x) + 0.5 * cos(2 * PI * y)) - 2.71289;
    return -cost;
  };

  cc_opt::pso::PSOParam param;
  param.n_features = 2;
  param.size_pop = 500;
  param.max_iter = 300;
  param.person_learning_factor = 2;
  param.group_learning_factor = 2;
  param.inertia_weight = 0.9;
  param.time_step = 1;
  param.lower_bound = std::vector<double>(2, -1.0);
  param.upper_bound = std::vector<double>(2, 1.0);
  param.early_stop = 20;

  cc_opt::pso::PSO<decltype(cost_func)> pso(cost_func, param);
  pso.Run();

  auto opt_features = pso.GetBestFeature();
  for (auto feat_val : opt_features) {
    std::cout << feat_val << "\n";
  }
  std::cout << "\n";

  return 0;
}