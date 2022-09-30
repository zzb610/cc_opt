#include "ga/ga.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

void SetAttrByFeature(std::vector<int> &ops,
                      const std::vector<double> &feature) {}

int main(void) {

  int64_t n_features = 2;
  int64_t size_pop = 50;
  int64_t max_iter = 500;
  double prob_mut = 0.01;
  double remain_rate = 0.5;
  std::vector<double> lower_bound(n_features, -1);
  std::vector<double> upper_bound(n_features, 1);
  bool early_stop = false;
  double seg_len = 10;

  std::vector<int> ops_;

  auto cost_func = [&ops_](std::vector<double> features) {
    SetAttrByFeature(ops_, features);

    double x1 = features[0], x2 = features[1];
    double cost = 0.5 + (std::sin(x1 * x1 + x2 * x2) - 0.5) /
                            std::sqrt(1 + 0.001 * std::sin(x1 * x1 + x2 * x2));

    return cost;
  };

  cc_opt::GA<decltype(cost_func)> ga(cost_func, n_features, size_pop, max_iter,
                                     prob_mut, remain_rate, lower_bound,
                                     upper_bound, early_stop, seg_len);

  ga.Run();

  auto opt_features = ga.GetBestFeature();
  for(auto feat_val: opt_features){
    std::cout << feat_val << "\n";
  }
  std::cout << "\n";

  // cc_opt::FloatGA<decltype(cost_func)> float_ga(cost_func, n_features,
  // size_pop, max_iter,
  //                                    prob_mut, remain_rate, lower_bound,
  //                                    upper_bound, early_stop);

  // float_ga.Run();

  return 0;
}