#pragma once

#include "chromo.h"
#include "crossover.h"
#include "operator.h"
#include "../utils/rand.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#define LOG

namespace cc_opt {
namespace ga {

struct BRKGAParam {

  int64_t max_iter;
  bool early_stop = true;

  int64_t size_pop;
  double elite_rate;
  double mutant_rate;
  double inherit_elite_prob;

  int64_t n_features;
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
};

template <typename T, typename LessTy>
std::vector<size_t> ArgSort(const std::vector<T> &array, LessTy less) {
  std::vector<size_t> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&array, &less](const auto &lhs, const auto &rhs) {
              return less(array[lhs], array[rhs]);
            });
  return indices;
}

template <typename T> T RandomChoice(const std::vector<T> &array) {
  auto n_element = array.size();
  auto rand_idx = GetRandomInt<size_t>(0, n_element - 1);
  return array.at(rand_idx);
}

template <typename CostFuncTy> class BRKGA {

public:
  BRKGA(CostFuncTy cost_func, const BRKGAParam &param)
      : cost_func_(cost_func), max_iter_(param.max_iter),
        early_stop_(param.early_stop), size_pop_(param.size_pop),
        elite_rate_(param.elite_rate), mutant_rate_(param.mutant_rate),
        inherit_elite_prob_(param.inherit_elite_prob),
        n_features_(param.n_features), lower_bound_(param.lower_bound),
        upper_bound_(param.upper_bound) {}

  double GetBestCost() const { return -best_chromo_.fitness; }

  std::vector<double> GetBestFeature() const {
    return Decode(best_chromo_, lower_bound_, upper_bound_);
  }

  void GetChromoFit(FloatChromo &chromo) {
    auto features = Decode(chromo, this->lower_bound_, this->upper_bound_);
    auto cost = this->cost_func_(features);
    // minimize cost functions
    chromo.fitness = -cost;
  }

  void InitPopulation() {
    for (auto i = 0; i < this->size_pop_; ++i) {
      auto genes = GenRandomFloatVec<double>(n_features_, 0, 1);
      FloatChromo chromo{genes};
      GetChromoFit(chromo);
      this->population_.push_back(std::move(chromo));
    }
    best_chromo_ = population_[0];
  }

  std::vector<std::vector<size_t>> DivPopulation() {
    auto gt = [&](const FloatChromo &lhs, const FloatChromo &rhs) {
      return lhs.fitness > rhs.fitness;
    };
    auto sorted_indices = ArgSort<FloatChromo, decltype(gt)>(population_, gt);

    int64_t elite_num = size_pop_ * elite_rate_;

    std::vector<size_t> elite_indices(sorted_indices.begin(),
                                      sorted_indices.begin() + elite_num);
    std::vector<size_t> non_elite_indices(sorted_indices.begin() + elite_num,
                                          sorted_indices.end());

    return {elite_indices, non_elite_indices};
  }

  void Run() {
    InitPopulation();

    int64_t first_best_iter = 0;
    for (auto iter = 0; iter < max_iter_; ++iter) {
      // divide group to elite subgroup and non-elite subgroup
      auto divided_indices = DivPopulation();
      auto elite_indices = divided_indices[0],
           non_elite_indices = divided_indices[1];

      std::vector<FloatChromo> new_population;
      // add all elites to next generation population
      for (auto idx : elite_indices) {
        new_population.push_back(population_[idx]);
      }

      // add mutant to next generation
      int64_t num_mutant = size_pop_ * mutant_rate_;
      for (auto i = 0; i < num_mutant; ++i) {
        auto mutant_genes = GenRandomFloatVec<double>(n_features_, 0, 1);
        FloatChromo mutant{mutant_genes};
        GetChromoFit(mutant);
        new_population.push_back(std::move(mutant));
      }

      // crossover elites and non-elites
      int64_t elite_num = size_pop_ * elite_rate_;
      int64_t num_children = size_pop_ - num_mutant - elite_num;
      for (auto i = 0; i < num_children; ++i) {
        auto elite_idx = RandomChoice(elite_indices);
        auto non_elite_idx = RandomChoice(non_elite_indices);

        auto child =
            CrossoverWithProb(population_[elite_idx],
                              population_[non_elite_idx], inherit_elite_prob_);
        GetChromoFit(child);
        new_population.push_back(std::move(child));
      }

      // evolve to next generation
      population_ = new_population;
      auto gen_best_chromo =
          *std::max_element(population_.begin(), population_.end(),
                            [&](const auto &lhs, const auto &rhs) {
                              return lhs.fitness < rhs.fitness;
                            });
      // update best
      if (gen_best_chromo.fitness > best_chromo_.fitness) {
        best_chromo_ = gen_best_chromo;
        first_best_iter = iter;
      }

      // early stop
      if (early_stop_ && iter - first_best_iter > kEarlyStopIter) {
#ifdef LOG
        std::cout << "early stop at: " << iter
                  << " get best at: " << first_best_iter << "\n";
#endif
        break;
      }

// log
#ifdef LOG
      std::cout << "iter: " << iter << " gen cost: " << -gen_best_chromo.fitness
                << " best cost: " << -best_chromo_.fitness << "\n";
#endif
    }
  }

private:
  CostFuncTy cost_func_;

  int64_t max_iter_;
  bool early_stop_;

  int64_t size_pop_;
  double elite_rate_;
  double mutant_rate_;
  double inherit_elite_prob_;

  int64_t n_features_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;

  std::vector<FloatChromo> population_;
  FloatChromo best_chromo_;
};

} // namespace ga
} // namespace cc_opt