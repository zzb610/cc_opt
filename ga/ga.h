#pragma once

#include "chromo.h"
#include "operator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#define LOG

namespace cc_opt {

namespace ga {

template <typename CostFuncTy, typename ChromoType> class GABase {
public:
  GABase(CostFuncTy cost_func, int64_t n_features, int64_t size_pop,
         int64_t max_iter, double prob_mut,
         const std::vector<double> &lower_bound,
         const std::vector<double> &upper_bound, bool early_stop)
      : cost_func_(cost_func), n_features_(n_features), size_pop_(size_pop),
        max_iter_(max_iter), prob_mut_(prob_mut), lower_bound_(lower_bound),
        upper_bound_(upper_bound), early_stop_(early_stop) {

    assert(size_pop_ % 2 == 0);
  }

  virtual void GetChromoFit(ChromoType &chromo) = 0;

  virtual void InitPopulation() = 0;
  virtual void Selection() = 0;
  virtual void CrossOver() = 0;
  virtual void Mutation() = 0;

  virtual std::vector<double> GetBestFeature() const = 0;

  double GetBestCost() const {
    return -best_chromo_.fitness;
  }

  ChromoType GetBestChromo() {
    return *std::max_element(this->population_.begin(), this->population_.end(),
                             [&](const auto &lhs, const auto &rhs) {
                               return lhs.fitness < rhs.fitness;
                             });
  }

  void Run() {
    this->InitPopulation();
    best_chromo_ = this->GetBestChromo();
    int64_t first_best_iter = 0;
    for (auto iter = 0; iter < this->max_iter_; ++iter) {
      this->Selection();
      this->CrossOver();
      this->Mutation();

      // update best
      auto gen_best_chromo = this->GetBestChromo();
      if (gen_best_chromo.fitness > best_chromo_.fitness) {
        best_chromo_ = gen_best_chromo;
        first_best_iter = iter;
      }

      if (this->early_stop_ && iter - first_best_iter >= kEarlyStopIter) {
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

protected:
  CostFuncTy cost_func_;

  int64_t max_iter_;
  bool early_stop_;

  int64_t size_pop_;
  double prob_mut_;

  int64_t n_features_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;

  std::vector<ChromoType> population_;
  ChromoType best_chromo_;
};

struct GAParam {
  int64_t max_iter;
  bool early_stop = true;

  int64_t size_pop;
  double remain_rate;
  double prob_mut;
  int64_t seg_len;

  int64_t n_features;
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
};

template <typename CostFuncTy>
class GA : public GABase<CostFuncTy, BinaryChromo> {
public:
  GA(CostFuncTy cost_func, const GAParam &param)
      : GABase<CostFuncTy, BinaryChromo>(cost_func, param.n_features,
                                         param.size_pop, param.max_iter,
                                         param.prob_mut, param.lower_bound,
                                         param.upper_bound, param.early_stop),
        remain_rate_(param.remain_rate), seg_len_(param.seg_len) {}

  void GetChromoFit(BinaryChromo &chromo) override final {

    auto features =
        Decode(chromo, seg_len_, this->lower_bound_, this->upper_bound_);
    auto cost = this->cost_func_(features);
    // minimize cost
    chromo.fitness = -cost;
  }

  std::vector<double> GetBestFeature() const override final {
    return Decode(this->best_chromo_, seg_len_, this->lower_bound_,
                  this->upper_bound_);
  };

  void InitPopulation() override final {
    int64_t n_genes = seg_len_ * this->n_features_;
    for (auto i = 0; i < this->size_pop_; ++i) {
      std::string gene;
      for (int j = 0; j < n_genes; ++j) {
        auto rand_01 = GetRandomInt<int64_t>(0, 1);
        gene.push_back(rand_01 == 0 ? '0' : '1');
      }
      BinaryChromo chromo{gene};
      this->GetChromoFit(chromo);
      this->population_.push_back(std::move(chromo));
    }
  }

  void Selection() override final {
    SelectTournament<BinaryChromo>(this->population_, remain_rate_, 3);
  }

  void CrossOver() override final {
    while (this->population_.size() < this->size_pop_) {
      auto parent_a_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);
      auto parent_b_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);

      auto childs = Crossover2Point(this->population_[parent_a_idx],
                                    this->population_[parent_b_idx]);
      for (auto &child : childs) {
        this->GetChromoFit(child);
        this->population_.push_back(std::move(child));
      }
    }
  }

  void Mutation() override final {
    for (auto &chromo : this->population_) {
      BinaryMutate(chromo, this->prob_mut_);
      GetChromoFit(chromo);
    }
  }

private:
  int64_t seg_len_; // binary gene number each feature
  double remain_rate_;
};

struct FloatGAParam {
  int64_t max_iter;
  bool early_stop = true;

  int64_t size_pop;
  double remain_rate;
  double prob_mut;

  int64_t n_features;
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
};

template <typename CostFuncTy>
class FloatGA : public GABase<CostFuncTy, FloatChromo> {
public:
  FloatGA(CostFuncTy cost_func, const FloatGAParam &param)
      : GABase<CostFuncTy, FloatChromo>(cost_func, param.n_features,
                                        param.size_pop, param.max_iter,
                                        param.prob_mut, param.lower_bound,
                                        param.upper_bound, param.early_stop),
        remain_rate_(param.remain_rate) {}

  std::vector<double> GetBestFeature() const override final {
    return Decode(this->best_chromo_, this->lower_bound_, this->upper_bound_);
  }

  void GetChromoFit(FloatChromo &chromo) override final {
    auto features = Decode(chromo, this->lower_bound_, this->upper_bound_);
    auto cost = this->cost_func_(features);
    // minimize cost
    chromo.fitness = -cost;
  }

  void InitPopulation() override final {
    int64_t n_genes = this->n_features_;
    for (auto i = 0; i < this->size_pop_; ++i) {
      auto gene = GenRandomFloatVec<double>(n_genes, 0, 1);
      FloatChromo chromo{gene};
      this->GetChromoFit(chromo);
      this->population_.push_back(std::move(chromo));
    }
  }

  void Selection() override final {
    SelectTournament<FloatChromo>(this->population_, remain_rate_, 3);
  }

  void CrossOver() override final {
    while (this->population_.size() < this->size_pop_) {
      auto parent_a_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);
      auto parent_b_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);

      auto childs = Crossover2Point(this->population_[parent_a_idx],
                                    this->population_[parent_b_idx]);
      for (auto &child : childs) {
        GetChromoFit(child);
        this->population_.push_back(std::move(child));
      }
    }
  }

  void Mutation() override final {
    for (auto &chromo : this->population_) {
      FloatMutate(chromo, this->prob_mut_);
      this->GetChromoFit(chromo);
    }
  }

private:
  double remain_rate_;
};
} // namespace ga

} // namespace cc_opt