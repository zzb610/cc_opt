#pragma once

#include "chromo.h"
#include "../utils/rand.h"

namespace cc_opt {
namespace ga {
template <typename ChromoTy>
inline void SelectTournament(std::vector<ChromoTy> &population,
                             double remain_rate, int64_t candidate_num) {
  assert(candidate_num >= 2);

  int64_t pop_size = population.size();
  int64_t remain_num = pop_size * remain_rate;
  if (remain_num % 2 == 1) {
    remain_num -= 1;
  }
  assert(remain_num % 2 == 0);

  std::vector<ChromoTy> next_population;
  while (next_population.size() < remain_num) {
    auto candidate_idx =
        GenRandomIntVec<int64_t>(candidate_num, 0, pop_size - 1);
    ChromoTy *winner = &population[candidate_idx[0]];
    for (int64_t i = 1; i < candidate_idx.size(); ++i) {
      int64_t cand_idx = candidate_idx[i];
      if (population[cand_idx].fitness > winner->fitness) {
        winner = &population[cand_idx];
      }
    }
    next_population.push_back(*winner);
  }

  population = next_population;
}
} // namespace ga
} // namespace cc_opt