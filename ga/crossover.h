#pragma once

#include "rand.h"

#include <cstdint>
#include <vector>

namespace cc_opt {
namespace ga {

template <typename ChromoTy>
inline std::vector<ChromoTy> Crossover2Point(const ChromoTy &parent_a,
                                             const ChromoTy &parent_b) {
  auto n_genes = parent_a.gene.size();
  ChromoTy child_a{parent_a.gene}, child_b{parent_b.gene};
  int64_t begin_idx = GetRandomInt<int64_t>(0, n_genes - 1);
  int64_t end_idx = GetRandomInt<int64_t>(0, n_genes - 1);
  if (begin_idx > end_idx) {
    std::swap(begin_idx, end_idx);
  }
  for (int64_t i = begin_idx; i < end_idx; ++i) {
    child_a.gene[i] = parent_b.gene.at(i);
    child_b.gene[i] = parent_a.gene.at(i);
  }
  return {child_a, child_b};
}

template <typename ChromoTy>
inline ChromoTy CrossoverWithProb(const ChromoTy &parent_a,
                                  const ChromoTy &parent_b, double prob) {

  auto num_genes = parent_a.gene.size();
  auto mask = GenRandomFloatVec<double>(num_genes, 0, 1);

  decltype(parent_a.gene) child_gene(num_genes);

  for (auto i = 0; i < num_genes; ++i) {
    if (mask[i] < prob) {
      child_gene[i] = parent_a.gene[i];
    } else {
      child_gene[i] = parent_b.gene[i];
    }
  }

  return ChromoTy{child_gene};
}

} // namespace ga
} // namespace cc_opt