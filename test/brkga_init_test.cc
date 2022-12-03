#define LOG
// #define PARALLEL

#include "ga/brkga.h"
#include <vector>

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

  cc_opt::ga::BRKGAParam param;
  param.n_features = 2;
  param.size_pop = 50;
  param.elite_rate = 0.20;
  param.mutant_rate = 0.20;
  param.inherit_elite_prob = 0.60;

  param.max_iter = 500;
  param.lower_bound = std::vector<double>(param.n_features, 0);
  param.upper_bound = std::vector<double>(param.n_features, 1.0);

  cc_opt::ga::BRKGA<decltype(cost_func)> brkga(cost_func, param);

  std::vector<std::vector<double>> init_pop{{{0.0, 0.5}}};
  brkga.InitPopulation(init_pop);
  brkga.Run();

  auto best_feature = brkga.GetBestFeature();
  std::cout << "best feature: ";
  for(auto f: best_feature){
    std::cout << f << " ";
  }
  std::cout << "\n";

  return 0;
}