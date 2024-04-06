r"""
Powerset samplers.

TODO: explain the formulation and the different samplers.

## Inner and outer indices

...

## Stochastic samplers

...


## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
[^3]: <a name="wu_variance_2023"></a>Wu, Mengmeng, Ruoxi Jia, Changle Lin, Wei Huang,
      and Xiangyu Chang. [Variance Reduced Shapley Value Estimation for Trustworthy Data
      Valuation](https://doi.org/10.1016/j.cor.2023.106305). Computers & Operations
      Research 159 (1 November 2023): 106305.
[^4]: <a name="maleki_bounding_2014"></a>Maleki, Sasan, Long Tran-Thanh, Greg Hines,
      Talal Rahwan, and Alex Rogers. [Bounding the Estimation Error of Sampling-Based
      Shapley Value Approximation](https://arxiv.org/abs/1306.4265). arXiv:1306.4265
      [Cs], 12 February 2014.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Generator, Iterable

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.numeric import powerset, random_subset, random_subset_of_size
from pydvl.utils.types import Seed
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler, SamplerT
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    IndexT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = [
    "AntitheticSampler",
    "DeterministicUniformSampler",
    "LOOSampler",
    "IndexSampler",
    "PowersetIndexIteration",
    "PowersetSampler",
    "TruncatedUniformStratifiedSampler",
    "UniformSampler",
    "UniformStratifiedSampler",
]


logger = logging.getLogger(__name__)


class PowersetIndexIteration(Enum):
    Sequential = "sequential"
    Random = "random"


class PowersetSampler(IndexSampler, ABC):
    """
    An abstract class for samplers which iterate over the powerset of the
    complement of an index in the training set.

    This is done in two nested loops, where the outer loop iterates over the set
    of indices, and the inner loop iterates over subsets of the complement of
    the current index. The outer iteration can be either sequential or at random.

    ## Slicing of powerset samplers

    Powerset samplers can be sliced for parallel computation. For those which are
    embarrassingly parallel, this is done by slicing the set of "outer" indices and
    returning new samplers over those slices. This includes all truly powerset-based
    samplers, such as
    [DeterministicUniformSampler][pydvl.valuation.samplers.DeterministicUniformSampler]
    and [UniformSampler][pydvl.valuation.samplers.UniformSampler]. In contrast, slicing
    a [PermutationSampler][pydvl.valuation.samplers.PermutationSampler] creates a new
    sampler which iterates over the same indices.
    """

    def __init__(
        self,
        indices: IndexSetT,
        batch_size: int = 1,
        index_iteration: PowersetIndexIteration = PowersetIndexIteration.Sequential,
        outer_indices: IndexSetT | None = None,
    ):
        """
        Args:
            indices: The set of items (indices) to sample from.
            batch_size: The number of samples to generate per batch. Batches are
                processed together by
                [UtilityEvaluator][pydvl.valuation.utility.evaluator.UtilityEvaluator].
            index_iteration: the order in which indices are iterated over
            outer_indices: The set of items (indices) over which to iterate
                when sampling. Subsets are taken from the complement of each index
                in succession. For embarrassingly parallel computations, this set
                is sliced into new samplers.
        """
        super().__init__(indices, batch_size)
        self._index_iteration = index_iteration
        self._outer_indices = np.array(
            outer_indices if outer_indices is not None else self._indices,
            copy=False,
            dtype=self._indices.dtype,
        )

    def complement(self, exclude: IndexSetT) -> NDArray[IndexT]:
        return np.setxor1d(self._indices, exclude)  # type: ignore

    def index_iterator(self) -> Generator[IndexT, None, None]:
        """Iterates over indices in the order specified at construction.

        FIXME: this is probably not very useful, but I couldn't decide
          which method of iteration is better
        """
        if self._index_iteration is PowersetIndexIteration.Sequential:
            for idx in self._outer_indices:
                yield idx
        elif self._index_iteration is PowersetIndexIteration.Random:
            while True:
                yield np.random.choice(self._outer_indices, size=1).item()

    @abstractmethod
    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        """Iterates over subsets given an index (e.g. subsets of its complement)."""

    def __len__(self) -> int:
        """Returns the number of outer indices over which the sampler iterates."""
        return len(self._outer_indices)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._indices}, {self._outer_indices})"

    def __getitem__(self, key: slice | list[int]) -> PowersetSampler:
        if isinstance(key, slice) or isinstance(key, Iterable):
            return self.__class__(
                self._indices,
                index_iteration=self._index_iteration,
                outer_indices=self._outer_indices[key],
            )
        raise TypeError("Indices must be an iterable or a slice")

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> PowersetEvaluationStrategy:
        return PowersetEvaluationStrategy(self, utility, coefficient)

    def generate(self) -> SampleGenerator:
        """Generates samples iterating in sequence over the outer indices, then over
        subsets of the complement of the current index. Each PowersetSampler defines
        its own
        [subset_iterator][pydvl.valuation.samplers.PowersetSampler.subset_iterator] to
        generate the subsets."""
        while True:
            for idx in self.index_iterator():
                for subset in self.subset_iterator(idx):
                    yield Sample(idx, frozenset(subset))
                    self._n_samples += 1

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        """Correction coming from Monte Carlo integration so that the mean of
        the marginals converges to the value: the uniform distribution over the
        powerset of a set with n-1 elements has mass 2^{n-1} over each subset."""
        return float(2 ** (n - 1)) if n > 0 else 1.0


class PowersetEvaluationStrategy(EvaluationStrategy[PowersetSampler]):
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        r = []
        for sample in batch:
            u_i = self.utility(
                Sample(sample.idx, frozenset({sample.idx}.union(sample.subset)))
            )
            u = self.utility(sample)
            marginal = (u_i - u) * self.coefficient(self.n_indices, len(sample.subset))
            r.append(ValueUpdate(sample.idx, marginal))
            if is_interrupted():
                break
        return r


class LOOSampler(PowersetSampler):
    """Leave-One-Out sampler.

    As first item, it returns the full set of indices. Then, for each index in the set,
    it returns a tuple with an index and the complement set.

    !!! tip "New in version 0.9.0"
    """

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        yield self.complement([idx])

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        return 1.0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy:
        return LOOEvaluationStrategy(self, utility, coefficient)


class LOOEvaluationStrategy(EvaluationStrategy[LOOSampler]):
    """Computes marginal values for LOO."""

    def __init__(
        self,
        sampler: LOOSampler,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ):
        super().__init__(sampler, utility, coefficient)
        self.total_utility = utility(Sample(None, frozenset(sampler.indices)))

    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        r = []
        for sample in batch:
            assert sample.idx is not None
            u = self.utility(sample)
            marginal = self.total_utility - u
            marginal *= self.coefficient(self.n_indices, len(sample.subset))
            r.append(ValueUpdate(sample.idx, marginal))
            if is_interrupted():
                break
        return r


class DeterministicUniformSampler(PowersetSampler):
    """An iterator to perform uniform deterministic sampling of subsets.

    For every index $i$, each subset of the complement `indices - {i}` is
    returned.

    !!! Note
        Outer indices are iterated over sequentially

    ??? Example
        ``` pycon
        >>> for idx, s in DeterministicUniformSampler(np.arange(2)):
        >>>    print(f"{idx} - {s}", end=", ")
        1 - [], 1 - [2], 2 - [], 2 - [1],
        ```

    Args:
        indices: The set of items (indices) to sample from.
        outer_indices: The set of items ("outer indices") over which to iterate
    """

    def __init__(self, indices: IndexSetT, outer_indices: IndexSetT | None = None):
        super().__init__(
            indices,
            index_iteration=PowersetIndexIteration.Sequential,
            outer_indices=outer_indices,
        )

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        for subset in powerset(self.complement([idx])):
            yield subset


class UniformSampler(StochasticSamplerMixin, PowersetSampler):
    """An iterator to perform uniform random sampling of subsets.

    Iterating over every index $i$, either in sequence or at random depending on
    the value of ``index_iteration``, one subset of the complement
    ``indices - {i}`` is sampled with equal probability $2^{n-1}$. The
    iterator never ends.

    ??? Example
        The code
        ```python
        for idx, s in UniformSampler(np.arange(3)):
           print(f"{idx} - {s}", end=", ")
        ```
        Produces the output:
        ```
        0 - [1 4], 1 - [2 3], 2 - [0 1 3], 3 - [], 4 - [2], 0 - [1 3 4], 1 - [0 2]
        (...)
        ```
    """

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        yield random_subset(self.complement([idx]), seed=self._rng)


class AntitheticSampler(StochasticSamplerMixin, PowersetSampler):
    """An iterator to perform uniform random sampling of subsets, and their
    complements.

    Works as [UniformSampler][pydvl.valuation.samplers.UniformSampler], but for every
    tuple $(i,S)$, it subsequently returns $(i,S^c)$, where $S^c$ is the
    complement of the set $S$ in the set of indices, excluding $i$.
    """

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        _complement = self.complement([idx])
        subset = random_subset(_complement, seed=self._rng)
        yield subset
        yield np.setxor1d(_complement, subset)

    # FIXME: is a uniform 2^{1-n} weight correct here too?


class UniformStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    """For every index, sample a set size, then a set of that size."""

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        k = int(self._rng.choice(np.arange(len(self.indices)), size=1).item())
        yield random_subset_of_size(self.complement([idx]), size=k, seed=self._rng)

    # FIXME: is a uniform 2^{1-n} weight correct here too?


class TruncatedUniformStratifiedSampler(UniformStratifiedSampler):
    r"""A sampler which samples set sizes between two bounds.

    This sampler was suggested in (Watson et al. 2023)<sup><a
    href="#watson_accelerated_2023">1</a></sup> for $\delta$-Shapley

    !!! tip "New in version 0.9.0"
    """

    def __init__(
        self,
        indices: IndexSetT,
        *,
        lower_bound: int,
        upper_bound: int,
        index_iteration: PowersetIndexIteration = PowersetIndexIteration.Sequential,
        outer_indices: IndexSetT | None = None,
        seed: Seed | None = None,
    ):
        super().__init__(
            indices,
            index_iteration=index_iteration,
            outer_indices=outer_indices,
            seed=seed,
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        k = self._rng.integers(
            low=self.lower_bound, high=self.upper_bound + 1, size=1
        ).item()
        yield random_subset_of_size(self.complement([idx]), size=k, seed=self._rng)

    # FIXME: is a uniform 2^{1-n} weight correct here too?


class VarianceReducedStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    r"""VRDS sampler.

    This sampler was suggested in (Wu et al. 2023)<sup><a
    href="#wu_variance_2023">3</a></sup>, a generalization of the stratified
    sampler in (Maleki et al. 2014)<sup><a href="#maleki_bounding_2014">4</a></sup>

    Args:
        indices: The set of items (indices) to sample from.
        samples_per_setsize: A function which returns the number of samples to
            take for a given set size.
        max_samples: The maximum number of samples to take.
        index_iteration: the order in which indices are iterated over
        outer_indices: The set of items (indices) over which to iterate
            when sampling. Subsets are taken from the complement of each index
            in succession.

    !!! tip "New in version 0.9.0"
    """

    def __init__(
        self,
        indices: IndexSetT,
        *,
        samples_per_setsize: Callable[[int], int],
        index_iteration: PowersetIndexIteration = PowersetIndexIteration.Sequential,
        outer_indices: IndexSetT | None = None,
    ):
        super().__init__(
            indices, index_iteration=index_iteration, outer_indices=outer_indices
        )
        self.samples_per_setsize = samples_per_setsize
        # HACK: closure around the argument to avoid weight() being an instance method
        # FIXME: is this the correct weight anyway?
        self.weight = lambda n, subset_len: samples_per_setsize(subset_len)

    def subset_iterator(self, idx: IndexT) -> Generator[IndexSetT, None, None]:
        for k in range(1, len(self.complement([idx]))):
            for _ in range(self.samples_per_setsize(k)):
                yield random_subset_of_size(
                    self.complement([idx]), size=k, seed=self._rng
                )

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        raise NotImplementedError  # This should never happen
