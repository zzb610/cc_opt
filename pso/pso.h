#pragma once

#include "../utils/rand.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define LOG

namespace cc_opt {
namespace pso {
constexpr int64_t kEarlyStopIter = 100;

struct Particle {
  Particle() = default;
  Particle(const std::vector<double> &position, double fitness)
      : position(position), fitness(fitness) {}
  std::vector<double> position;
  double fitness;
};

struct PSOParam {
  int64_t n_features;
  int64_t size_pop;
  int64_t max_iter;
  double person_learning_factor;
  double group_learning_factor;
  double inertia_weight;
  int64_t time_step;
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
  bool early_stop;
};

template <typename CostFuncTy> class PSO {
public:
  PSO(CostFuncTy cost_func, const PSOParam &param)
      : cost_func_(cost_func), n_features_(param.n_features),
        size_pop_(param.size_pop), max_iter_(param.max_iter),
        person_learning_factor_(param.person_learning_factor),
        group_learning_factor_(param.group_learning_factor),
        time_step(param.time_step), inertia_weight_(param.inertia_weight),
        lower_bound_(param.lower_bound), upper_bound_(param.upper_bound),
        early_stop_(param.early_stop) {}

  double GetFitness(const std::vector<double> &position) const {
    auto cost = cost_func_(position);
    return -cost;
  }

  Particle GetBestParticle() const {
    assert(!population_.empty());
    auto best_particle = population_[0];
    for (auto i = 1; i < population_.size(); ++i) {
      if (population_[i].fitness > best_particle.fitness) {
        best_particle = population_[i];
      }
    }
    return best_particle;
  }

  std::vector<double> GetBestFeature() const { return best_feature_; }

  double GetBestCost() const { return best_cost_; }

  void InitPopulation() {
    for (auto i = 0; i < size_pop_; ++i) {
      auto position =
          GenRandomFloatVec<double>(n_features_, lower_bound_, upper_bound_);
      double fitness = GetFitness(position);
      Particle particle(position, fitness);
      population_.push_back(particle);
      person_best_.push_back(particle);
    }
    group_best_ = GetBestParticle();
  }

  void Run() {
    InitPopulation();

    int64_t best_iter = 0;
    best_cost_ = -GetFitness(group_best_.position);
    Particle best_particle = group_best_;

    std::vector<std::vector<double>> velocity;
    for (auto i = 0; i < size_pop_; ++i) {
      auto vel = GenRandomFloatVec<double>(n_features_, -1, 1);
      velocity.push_back(std::move(vel));
    }

    for (auto iter = 0; iter < max_iter_; ++iter) {
      for (auto i = 0; i < size_pop_; ++i) {
        for (auto j = 0; j < n_features_; ++j) {
          double r1 = GetRandomFloat<double>(0, 1);
          double r2 = GetRandomFloat<double>(0, 1);

          velocity[i][j] =
              inertia_weight_ * velocity[i][j] +
              (person_learning_factor_ * r1 *
               (person_best_[i].position[j] - population_[i].position[j])) +
              (group_learning_factor_ * r2 *
               (group_best_.position[j] - population_[i].position[j]));
          // update personal position and fitness
          population_[i].position[j] =
              population_[i].position[j] + velocity[i][j] * time_step;

          population_[i].position[j] =
              std::min(population_[i].position[j], upper_bound_[j]);
          population_[i].position[j] =
              std::max(population_[i].position[j], lower_bound_[j]);
        }
        population_[i].fitness = GetFitness(population_[i].position);

        // update personal best
        if (population_[i].fitness > person_best_[i].fitness) {
          person_best_[i] = population_[i];
        }
        // update group best
        if (population_[i].fitness > group_best_.fitness) {
          group_best_ = population_[i];

          // update global best
          if (group_best_.fitness > best_particle.fitness) {
            best_iter = iter;
            best_particle = group_best_;
            best_cost_ = -best_particle.fitness;
          }
        }
      }

#ifdef LOG
      std::cout << "iter: " << iter << " best cost: " << best_cost_ << "\n";
#endif
      // print features
      if (early_stop_ && (iter - best_iter) > kEarlyStopIter) {
#ifdef LOG
        std::cout << "early stop at: " << iter << " get best iter at " << best_iter << "\n";
#endif
        break;
      }
    }

    best_feature_ = best_particle.position;
  }

private:
  CostFuncTy cost_func_;

  int64_t max_iter_;
  bool early_stop_;

  int64_t size_pop_;
  int64_t n_features_;
  int64_t time_step;

  double person_learning_factor_;
  double group_learning_factor_;
  double inertia_weight_;

  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;

  std::vector<double> best_feature_;
  double best_cost_;

  std::vector<Particle> population_;
  std::vector<Particle> person_best_;
  Particle group_best_;
};

} // namespace pso
} // namespace cc_opt
