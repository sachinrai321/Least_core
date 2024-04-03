r"""
Permutation-based samplers.

TODO: explain the formulation and the different samplers.


## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
"""

from __future__ import annotations

import logging
import math
import warnings
from itertools import permutations
from typing import Callable

from pydvl.utils.types import Seed
from pydvl.valuation.samplers import EvaluationStrategy, IndexSampler
from pydvl.valuation.samplers.truncation import NoTruncation, TruncationPolicy
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility import Utility

__all__ = [
    "PermutationSampler",
    "AntitheticPermutationSampler",
    "DeterministicPermutationSampler",
    "PermutationEvaluationStrategy",
]


logger = logging.getLogger(__name__)


class PermutationSampler(StochasticSamplerMixin, IndexSampler):
    """Sample permutations of indices and iterate through each returning
    increasing subsets, as required for the permutation definition of
    semi-values.

    For a permutation `(3,1,4,2)`, this sampler returns in sequence the following
    [Samples][pydvl.valuation.samplers.Sample] (tuples of index and subset):

    `(3, {3})`, `(1, {3,1})`, `(4, {3,1,4})` and `(2, {3,1,4,2})`.

    !!! info "Batching"
        PermutationSamplers always batch their outputs to include a whole permutation
        of the index set, i.e. the batch size is always the number of indices.

    Args:
        indices: The set of items (indices) to sample from.
        truncation: A policy to stop the permutation early.
        seed: Seed for the random number generator.
    """

    def __init__(
        self,
        indices: IndexSetT,
        truncation: TruncationPolicy | None = None,
        seed: Seed | None = None,
    ):
        super().__init__(indices=indices, seed=seed)
        self.truncation = truncation or NoTruncation()
        self.batch_size = len(indices)

    def generate(self) -> SampleGenerator:
        """Generates the permutation samples.

        Samples are yielded one by one, not as whole permutations. These are batched
        together by calling iter() on the sampler.
        """
        while True:
            permutation = self._rng.permutation(self._indices)
            for i, idx in enumerate(permutation):
                yield Sample(idx, frozenset(permutation[: i + 1]))
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    def __getitem__(self, key: slice | list[int]) -> IndexSampler:
        """Permutation samplers cannot be split across indices, so we return
        a copy of the full sampler."""

        warnings.warn(
            "Permutation samplers cannot be split across indices, "
            "returning a copy of the full sampler.",
            RuntimeWarning,
        )
        return self.__class__(self._indices, seed=self._rng)

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        return n * math.comb(n - 1, subset_len) if n > 0 else 1.0

    def strategy(self) -> PermutationEvaluationStrategy:
        return PermutationEvaluationStrategy(self)


class AntitheticPermutationSampler(PermutationSampler):
    """Samples permutations like
    [PermutationSampler][pydvl.valuation.samplers.PermutationSampler], but after
    each permutation, it returns the same permutation in reverse order.

    This sampler was suggested in (Mitchell et al. 2022)<sup><a
    href="#mitchell_sampling_2022">1</a></sup>

    !!! tip "New in version 0.7.1"
    """

    def generate(self) -> SampleGenerator:
        while True:
            permutation = self._rng.permutation(self._indices)
            for perm in permutation, permutation[::-1]:
                for i, idx in enumerate(perm):
                    yield Sample(idx, frozenset(perm[: i + 1]))
                    self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break


class DeterministicPermutationSampler(PermutationSampler):
    """Samples all n! permutations of the indices deterministically, and
    iterates through them, returning sets as required for the permutation-based
    definition of semi-values.

    !!! Warning
        This sampler requires caching to be enabled or computation
        will be doubled wrt. a "direct" implementation of permutation MC

    !!! Warning
        This sampler is not parallelizable, as it always iterates over the whole
        set of permutations in the same order. Different processes would always
        return the same values for all indices.
    """

    def generate(self) -> SampleGenerator:
        for permutation in permutations(self._indices):
            for i, idx in enumerate(permutation):
                yield Sample(idx, frozenset(permutation[: i + 1]))
                self._n_samples += 1


class PermutationEvaluationStrategy(EvaluationStrategy[PermutationSampler]):
    """Computes marginal values for permutation sampling schemes.

    This strategy iterates over permutations from left to right, computing the marginal
    utility wrt. the previous one at each step to save computation.
    """

    def __init__(self, sampler: PermutationSampler):
        super().__init__(sampler)
        self.n = len(sampler.indices)
        self.truncation = sampler.truncation

    def setup(
        self,
        utility: Utility,
        coefficient: Callable[[int, int], float] | None = None,
    ):
        super().setup(utility, coefficient)
        self.truncation.reset(utility)  # Perform initial setup (e.g. total_utility)
        self._is_setup = True

    def process(
        self,
        utility: Utility,
        is_interrupted: NullaryPredicate,
        permutation: SampleBatch,
    ) -> list[ValueUpdate]:
        if not self._is_setup:
            raise ValueError("Evaluation strategy not set up")
        self.truncation.reset(utility)  # Reset before every batch (will be cached)
        r = []
        truncated = False
        curr = prev = utility.default_score
        for sample in permutation:
            assert sample.idx is not None
            if not truncated:
                # FIXME: If utility accepted Samples, we could subclass for CSShapley
                #   and use this strategy directly
                curr = utility(sample)
            marginal = curr - prev
            marginal *= self.coefficient(self.n, len(sample.subset))
            r.append(ValueUpdate(sample.idx, marginal))
            prev = curr
            if not truncated and self.truncation(sample.idx, curr, self.n):
                truncated = True
            if is_interrupted():
                break
        return r
