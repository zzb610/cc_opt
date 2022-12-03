#define LOG
// #define PARALLEL

#include "ga/rcga.h"
#include <vector>
#include <iostream>

constexpr double PI = 3.1415926525;

int main(void) {

  // auto cost_func = [&](const std::vector<double> &features) {
  //   double x = features[0];
  //   double y = features[1];
  //   double temp = sqrt(x * x + y * y);
  //   double cost = sin(temp) / temp +
  //                 exp(0.5 * cos(2 * PI * x) + 0.5 * cos(2 * PI * y)) - 2.71289;
  //   return -cost;
  // };

    auto cost_func = [&](const std::vector<double> &features) {
    double x = features[0];
    double y = features[1];
 
    double cost = x * x + y * y;
    return cost;
  };

  cc_opt::ga::RCGAParam param;
  param.n_features = 2;
  param.size_pop = 50;
  param.max_iter = 500;
  param.prob_mut = 0.01;
  param.remain_rate = 0.5;
  param.lower_bound = std::vector<double>(param.n_features, 0.0);
  param.upper_bound = std::vector<double>(param.n_features, 1.0);
  param.early_stop = 20;
  cc_opt::ga::RCGA<decltype(cost_func)> rcga(cost_func, param);

  rcga.InitPopulation({{0, 0}});

  rcga.Run();
  auto opt_features = rcga.GetBestFeature();
  for (auto feat_val : opt_features) {
    std::cout << feat_val << "\n";
  }
  std::cout << "\n";

  auto best_feature = rcga.GetBestFeature();
  std::cout << "best feature: ";
  for(auto f: best_feature){
    std::cout << f << " ";
  }
  std::cout << "\n";

  return 0;
}