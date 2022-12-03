#include "chromo.h"
#include "crossover.h"
#include "operator.h"

#include <iostream>

namespace cc_opt {
namespace ga {

struct RCGAParam {
  int64_t max_iter;
  int64_t early_stop = 50;

  int64_t size_pop;
  double remain_rate;
  double prob_mut;

  int64_t n_features;
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
};

template <typename CostFuncT> class RCGA {

public:
  RCGA(CostFuncT cost_func, const RCGAParam &param)
      : cost_func_(cost_func), n_features_(param.n_features),
        size_pop_(param.size_pop), max_iter_(param.max_iter),
        prob_mut_(param.prob_mut), lower_bound_(param.lower_bound),
        upper_bound_(param.upper_bound), early_stop_(param.early_stop),
        remain_rate_(param.remain_rate) {}

  std::vector<double> GetBestFeature() const {
    return Decode(best_chromo_, lower_bound_, upper_bound_);
  }

  FloatChromo GetBestChromo() const {
    return *std::max_element(this->population_.begin(), this->population_.end(),
                             [&](const auto &lhs, const auto &rhs) {
                               return lhs.fitness < rhs.fitness;
                             });
  }

  void GetChromoFit(FloatChromo &chromo) {
    auto features = Decode(chromo, lower_bound_, upper_bound_);
    auto cost = cost_func_(features);
    // minimize cost
    chromo.fitness = -cost;
  }

  void InitPopulation(const std::vector<std::vector<double>> &init_pop) {
    for (const auto &gene : init_pop) {

      FloatChromo chromo{gene};
      population_.push_back(std::move(chromo));
    }
    auto init_num = init_pop.size();
    auto random_num = size_pop_ - init_num;
    for (size_t i = 0; i < random_num; ++i) {
      auto genes = GenRandomFloatVec<double>(n_features_, 0, 1);
      FloatChromo chromo{genes};
      population_.push_back(std::move(chromo));
    }

    for (auto &chromo : population_) {
      GetChromoFit(chromo);
    }
    best_chromo_ = population_[0];
  }

  void InitPopulation() {
    int64_t n_genes = n_features_;
    for (auto i = 0; i < size_pop_; ++i) {
      auto gene = GenRandomFloatVec<double>(n_genes, 0, 1);
      FloatChromo chromo{gene};
      GetChromoFit(chromo);
      population_.push_back(std::move(chromo));
    }
  }

  void Selection() {
    SelectTournament<FloatChromo>(population_, remain_rate_, 3);
  }

  void CrossOver() {
    while (population_.size() < size_pop_) {
      auto parent_a_idx = GetRandomInt<int64_t>(0, population_.size() - 1);
      auto parent_b_idx = GetRandomInt<int64_t>(0, population_.size() - 1);

      auto childs =
          Crossover2Point(population_[parent_a_idx], population_[parent_b_idx]);
      for (auto &child : childs) {
        GetChromoFit(child);
        population_.push_back(std::move(child));
      }
    }
  }

  void Mutation() {
    for (auto &chromo : population_) {
      FloatMutate(chromo, prob_mut_);
      GetChromoFit(chromo);
    }
  }

  void Run() {
    if (population_.empty()) {
      InitPopulation();
    }
    best_chromo_ = GetBestChromo();
    int64_t first_best_iter = 0;
    for (auto iter = 0; iter < max_iter_; ++iter) {
      Selection();
      CrossOver();
      Mutation();

      // update best
      auto gen_best_chromo = GetBestChromo();
      if (gen_best_chromo.fitness > best_chromo_.fitness) {
        best_chromo_ = gen_best_chromo;
        first_best_iter = iter;
      }

      if (early_stop_ != -1 && iter - first_best_iter >= early_stop_) {
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
  CostFuncT cost_func_;

  int64_t max_iter_;
  int64_t early_stop_;

  int64_t size_pop_;
  double prob_mut_;

  int64_t n_features_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;

  std::vector<FloatChromo> population_;
  FloatChromo best_chromo_;
  double remain_rate_;
};
} // namespace ga
} // namespace cc_opt