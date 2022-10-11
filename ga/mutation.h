#pragma once

#include "chromo.h"
#include "rand.h"

namespace cc_opt {
namespace ga {

inline void BinaryMutate(BinaryChromo &chromo, double mut_rate) {
  int64_t n_genes = chromo.gene.size();
  std::vector<double> mut_mask = GenRandomFloatVec<double>(n_genes, 0, 1);
  for (int64_t i = 0; i < n_genes; ++i) {
    if (mut_mask[i] < mut_rate) {
      chromo.gene[i] = (chromo.gene[i] == '0' ? '1' : '0');
    }
  }
}

inline void FloatMutate(FloatChromo &f_chromo, double mut_rate) {
  int64_t n_genes = f_chromo.gene.size();
  std::vector<double> mut_mask = GenRandomFloatVec<double>(n_genes, 0, 1);
  for (int64_t i = 0; i < n_genes; ++i) {
    if (mut_mask[i] < mut_rate) {
      f_chromo.gene[i] = GetRandomFloat<double>(0, 1);
    }
  }
}

} // namespace ga
} // namespace cc_opt