#define LOG
#include "ga/ga.h"

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

  // cc_opt::ga::GAParam param;
  // param.n_features = 2;
  // param.size_pop = 50;
  // param.max_iter = 500;
  // param.prob_mut = 0.01;
  // param.remain_rate = 0.5;
  // param.lower_bound = std::vector<double>(param.n_features, -1.0);
  // param.upper_bound = std::vector<double>(param.n_features, 1.0);
  // param.early_stop = true;
  // param.seg_len = 10;
  // cc_opt::ga::GA<decltype(cost_func)> ga(cost_func, param);

  // ga.Run();

  // auto opt_features = ga.GetBestFeature();
  // for(auto feat_val: opt_features){
  //   std::cout << feat_val << "\n";
  // }
  // std::cout << "\n";

  cc_opt::ga::FloatGAParam param;
  param.n_features = 2;
  param.size_pop = 50;
  param.max_iter = 500;
  param.prob_mut = 0.01;
  param.remain_rate = 0.5;
  param.lower_bound = std::vector<double>(param.n_features, -1.0);
  param.upper_bound = std::vector<double>(param.n_features, 1.0);
  param.early_stop = 20;
  cc_opt::ga::FloatGA<decltype(cost_func)> float_ga(cost_func, param);

  float_ga.Run();
  auto opt_features = float_ga.GetBestFeature();
  for (auto feat_val : opt_features) {
    std::cout << feat_val << "\n";
  }
  std::cout << "\n";

  return 0;
}