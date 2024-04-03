"""
A utility evaluator computes utility values for subsets of indices.

The utility evaluator is responsible for processing batches of samples and computing
the utility of each sample in the batch. There are serial and parallel implementations.

## Interrupting computation

There are two mechanisms to interrupt the computation of utilities:

* Via a [TruncationPolicy][pydvl.valuation.samplers.truncation.TruncationPolicy] set
  in the [Sampler][pydvl.valuation.samplers] that is being evaluated (only for certain
  samplers). This policy is evaluated after each sample in a batch, and can decide to
  stop the computation of the batch. When this happens, further samples in the batch are
  either discarded or evaluated with a default value, depending on the policy.

* Via a call to
  [interrupt()][pydvl.valuation.utility.evaluator.UtilityEvaluator.interrupt]. This
  cancels evaluation of any remaining samples in a batch, and stops the computation
  of further batches.
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Generator, Iterable, Literal, Protocol

from pydvl.parallel import (
    ParallelConfig,
    effective_n_jobs,
    init_executor,
    init_parallel_backend,
)
from pydvl.utils.types import Seed
from pydvl.valuation.types import NullaryPredicate, SampleBatch, ValueUpdate
from pydvl.valuation.utility import Utility
from pydvl.valuation.utility.utility import UtilityBase

logger = logging.getLogger(__name__)

__all__ = ["UtilityEvaluator", "SerialUtilityEvaluator", "ParallelUtilityEvaluator"]


class BatchProcessor(Protocol):
    def __call__(
        self, utility: UtilityBase, is_interrupted: NullaryPredicate, batch: SampleBatch
    ) -> list[ValueUpdate]:
        ...


class UtilityEvaluator(ABC):
    """Abstract base class for utility evaluators.

    Derived classes must implement the context manager protocol to ensure proper
    acquisition and cleanup of resources. This is done in `__enter__` and `__exit__`.

    FIXME: this is almost unrelated to the utility and could be abstracted
    """

    def __init__(self, utility: UtilityBase):
        self._utility = utility
        self._in_context = False

    @abstractmethod
    def interrupt(self):
        """Interrupt the computation of the utilities.
        Checked inside batch processors to stop the current batch.
        """
        ...

    @abstractmethod
    def is_interrupted(self) -> bool:
        ...

    @property
    def utility(self) -> UtilityBase:
        return self._utility

    @abstractmethod
    def map(
        self,
        process: BatchProcessor,
        batches: Iterable[SampleBatch],
    ) -> Generator[list[ValueUpdate], None, None]:
        ...

    def __enter__(self) -> UtilityEvaluator:
        if self._in_context:  # Careful with re-entrance
            return self
        logger.debug(f"Entering {self.__class__.__name__}")
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        """
        Return True: any exception that occurred in the with block will be suppressed.
        Return False or None: any exception will be re-raised after exit.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Trace
        """
        logger.debug(f"Exiting {self.__class__.__name__}")
        return False


class SerialUtilityEvaluator(UtilityEvaluator):
    def __init__(self, utility: UtilityBase):
        super().__init__(utility)
        self._interrupted = False

    def is_interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self):
        self._interrupted = True

    def __enter__(self) -> UtilityEvaluator:
        self._interrupted = False
        return super().__enter__()

    def map(
        self, process: BatchProcessor, batches: Iterable[SampleBatch]
    ) -> Generator[list[ValueUpdate], None, None]:
        for batch in batches:
            yield process(self.utility, self.is_interrupted, batch)
            if self.is_interrupted():
                break


class ParallelUtilityEvaluator(UtilityEvaluator):
    """An evaluator to compute utilities in parallel.

    Args:
        utility: Utility object to use. It will be sent to the parallel backend's store. Note
            that this can be an expensive step!
            FIXME: this should be delayed to the context manager. The strategy should
              not take an evaluator, or if it does take one, it should somehow only see
              the objects that have been put()
        parallel_config:
        n_jobs: number of jobs across which to distribute the computation.
        seed: Either an instance of a numpy random number generator or a seed for it.

    """

    def __init__(
        self,
        utility: Utility,
        *,
        parallel_config: ParallelConfig,
        n_jobs: int = -1,
        seed: Seed | None = None,
    ):
        super().__init__(utility)
        self.parallel_config = parallel_config
        self.parallel_backend = init_parallel_backend(parallel_config)
        self.n_jobs = n_jobs
        self.seed = seed
        self._pending_batches: set[Future] = set()
        self._utility_ref = None
        self._interrupted: ParallelFlag | None = None

    def interrupt(self):
        if self._interrupted is None:
            warnings.warn("The interrupt flag is not available outside a context")
            return
        self._interrupted.set()

    # FIXME: probably not how this works. In ray, we should use an actor with
    #  setter/getter
    def is_interrupted(self) -> bool:
        if self._interrupted is None:
            warnings.warn("The interrupt flag is not available outside a context")
            return False
        return self._interrupted.get()

    def __enter__(self) -> ParallelUtilityEvaluator:
        super().__enter__()
        self._utility_ref = self.parallel_backend.put(self._utility)
        self._interrupted = self.parallel_backend.make_flag()
        self._pending_batches = set()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pending_batches:
            for future in self._pending_batches:
                future.cancel()
        del self._pending_batches
        del self._utility_ref  # FIXME: will all backends do ref counting?
        return super().__exit__(exc_type, exc_val, exc_tb)

    def map(
        self, process: BatchProcessor, batches: Iterable[SampleBatch]
    ) -> Generator[list[ValueUpdate], None, None]:
        if not self._in_context:
            raise RuntimeError(
                "The context manager must be entered before calling the evaluator"
            )
        # For the type checker
        assert self._utility_ref is not None
        assert self._interrupted is not None

        max_workers = effective_n_jobs(self.n_jobs)
        n_submitted_jobs = 2 * self.n_jobs  # number of jobs in the queue
        # seed_sequence = ensure_seed_sequence(self.seed)
        batch_it = iter(batches)
        with init_executor(
            max_workers=max_workers, config=self.parallel_config, cancel_futures=True
        ) as executor:
            while True:
                completed_batches, self._pending_batches = wait(
                    self._pending_batches, timeout=0.1, return_when=FIRST_COMPLETED
                )
                for batch_future in completed_batches:
                    yield batch_future.result()
                    if self.is_interrupted():
                        return

                # Ensure that we always have n_submitted_jobs in the queue or running
                n_remaining_slots = n_submitted_jobs - len(self._pending_batches)
                # seeds = seed_sequence.spawn(n_remaining_slots)

                try:
                    for i in range(n_remaining_slots):
                        self._pending_batches.add(
                            executor.submit(
                                process,
                                utility=self._utility_ref,
                                is_interrupted=self._interrupted_ref,
                                batch=list(next(batch_it))
                                # seed=seeds[i]  # workers can't seed the utility yet
                            )
                        )
                except StopIteration as e:
                    if len(self._pending_batches) == 0:  # batch_it was exhausted
                        raise e
