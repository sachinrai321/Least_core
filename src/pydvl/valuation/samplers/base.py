"""
Base classes for samplers and evaluation strategies.

See [pydvl.valuation.samplers][pydvl.valuation.samplers] for details.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from pydvl.valuation.samplers.utils import take_n
from pydvl.valuation.types import (
    BatchGenerator,
    IndexSetT,
    IndexT,
    NullaryPredicate,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility import Utility

__all__ = ["EvaluationStrategy", "IndexSampler"]


# Sequence.register(np.ndarray)  # <- Doesn't seem to work

logger = logging.getLogger(__name__)


class IndexSampler(ABC, Iterable[SampleBatch]):
    r"""Samplers are custom iterables over batches of subsets of indices.

    Calling ``iter()`` on a sampler returns a generator over **batches** of `Samples`.
    A [Sample][pydvl.valuation.samplers.Sample] is a tuple of the form $(i, S)$, where $i$ is
    an index of interest, and $S \subset I \setminus \{i\}$ is a subset of the
    complement of $i$ in $I$.

    !!! Note
        Samplers are **not** iterators themselves, so that each call to ``iter()``
        e.g. in a new for loop creates a new iterator.

    Derived samplers must implement
    [weight()][pydvl.valuation.samplers.IndexSampler.weight] and
    [generate()][pydvl.valuation.samplers.IndexSampler.generate]. See the module's
    documentation for more on these.

    Args:
        indices: The set of items (indices) to sample from.
        batch_size: The number of samples to generate per batch. Batches are
            processed by
            [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
            so that individual valuations in batch are guaranteed to be received in the
            right sequence.

    ??? Example
        ``` pycon
        >>>from pydvl.valuation.samplers import DeterministicUniformSampler
        >>>for idx, s in DeterministicUniformSampler(np.arange(2)):
        >>>    print(s, end="")
        [][2,][][1,]
        ```
    """

    def __init__(self, indices: IndexSetT, batch_size: int = 1):
        """
        Args:
            indices: The set of items (indices) to sample from.
        """
        self._indices = np.array(indices, copy=False)
        self._n_samples = 0
        self._batch_size = batch_size

    @property
    def indices(self) -> NDArray[IndexT]:
        return self._indices

    @indices.setter
    def indices(self, indices: NDArray[IndexT]) -> None:
        raise AttributeError("Cannot set indices of sampler")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n: int) -> None:
        raise AttributeError("Cannot reset a sampler's number of samples")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if value < 1:
            raise ValueError("batch_size must be at least 1")
        self._batch_size = value

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._indices})"

    @overload
    def __getitem__(self, key: slice) -> IndexSampler:
        ...

    @overload
    def __getitem__(self, key: list[int]) -> IndexSampler:
        ...

    @abstractmethod
    def __getitem__(self, key: slice | list[int]) -> IndexSampler:
        ...

    def __iter__(self) -> BatchGenerator:
        """Batches the samples and yields them."""
        yield from take_n(self.generate(), self.batch_size)

    @abstractmethod
    def generate(self) -> SampleGenerator:
        """Generates single samples.

        `IndexSampler.__iter__()` will batch these samples according to the batch size
        set upon construction.

        Yields:
            A tuple (idx, subset) for each sample.

        Receives:
            Feedback required by the sampler.
        """
        ...

    @staticmethod
    @abstractmethod
    def weight(n: int, subset_len: int) -> float:
        r"""Factor by which to multiply Monte Carlo samples, so that the
        mean converges to the desired expression.

        By the Law of Large Numbers, the sample mean of $\delta_i(S_j)$
        converges to the expectation under the distribution from which $S_j$ is
        sampled.

        $$
        \frac{1}{m}  \sum_{j = 1}^m \delta_i (S_j) c (S_j) \longrightarrow
           \underset{S \sim \mathcal{D}_{- i}}{\mathbb{E}} [\delta_i (S) c ( S)]
        $$

        We add a factor $c(S_j)$ in order to have this expectation coincide with
        the desired expression.

        Args:
            n: The total number of indices in the training data.
            subset_len: The size of the subset $S_j$ for which the marginal is being
                computed
        """
        ...

    @abstractmethod
    def strategy(self) -> EvaluationStrategy[IndexSampler]:
        """Returns the strategy for this sampler."""
        ...  # return SomeEvaluationStrategy(self)


SamplerType = TypeVar("SamplerType", bound=IndexSampler)


class EvaluationStrategy(ABC, Generic[SamplerType]):
    """An evaluation strategy for samplers.

    Mediates between an [IndexSampler][pydvl.valuation.samplers.IndexSampler] and a
    [UtilityEvaluator][pydvl.valuation.utility.evaluator.UtilityEvaluator].

    Different sampling schemes require different strategies for the evaluation of the
    utilities. For instance permutations generated by
    [PermutationSampler][pydvl.valuation.samplers.PermutationSampler] must be evaluated
    in sequence to save computation, see
    [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy].

    This class defines the common interface.

    ??? Example "Usage pattern in valuation methods"
        ```python
            def fit(self):
                strategy = self.sampler.strategy()
                with self.evaluator as evaluator:
                    strategy.setup(evaluator.utility)
                    for batch in evaluator.map(strategy.process, self.sampler)
                        for evaluation in batch:
                            self.result.update(evaluation.idx, evaluation.update)
                        if self.is_done(self.result):
                            evaluator.interrupt()
        ```

    Args:
        sampler: Where to take samplers from.
    """

    def __init__(self, sampler: SamplerType):
        # FIXME: if EvaluationStrategy.process() is submitted as a future by the
        #  evaluator, this will be copied to every process!
        #  we only ever need: len(sampler|sampler.indices) and sampler.weight
        #  the latter is only an instance method because of the VRDS sampler requiring
        #  some configuration from the constructor. This might be the case in other
        #  instances. One could try to use some closure.
        #  More importantly: does this even matter? A sampler's biggest attribute is
        #  the index set, which will anyway be < 1MB, <10MB at worst (unlikely)
        self.sampler: SamplerType | None = None
        self.coefficient: Callable[[int, int], float] = lambda n, k: 1.0
        self._is_setup = False

    def setup(
        self,
        utility: Utility,
        coefficient: Callable[[int, int], float] | None = None,
    ):
        """Set up the strategy for the given evaluator.

        FIXME the coefficient does not belong here, but in the methods. Either we
          return more information from process() so that it can be used in the methods
          or we allow for some manipulation of the strategy after it has been created.
          The latter is rigid but a quick fix, which I need right now.
        Args:
            utility:
            coefficient: An additional coefficient to multiply marginals with. This
                depends on the valuation method, hence the delayed setup.
        """
        if self.sampler is not None:
            if coefficient is not None:

                def coefficient_fun(n: int, subset_len: int) -> float:
                    return self.sampler.weight(n, subset_len) * coefficient(
                        n, subset_len
                    )

                self.coefficient = coefficient_fun
            self.coefficient = self.sampler.weight
        self._is_setup = True

    @abstractmethod
    def process(
        self, utility: Utility, is_interrupted: NullaryPredicate, batch: SampleBatch
    ) -> list[ValueUpdate]:
        """Processes batches of samples using the evaluator, with the strategy
        required for the sampler.

        !!! Warning
            This method is intended to be used by the evaluator to process the samples
            in one batch, which means it might be sent to another process. Be careful
            with the objects you use here, as they will be pickled and sent over the
            wire.

        Args:
            is_interrupted:
            utility:
            batch:

        Yields:
            Updates to values as tuples (idx, update)
        """
        ...
