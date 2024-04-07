from __future__ import annotations

import logging
from typing import Callable, Generator

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.valuation.base import (
    Valuation,
    ensure_backend_has_generator_return,
    make_parallel_flag,
)
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler, PowersetSampler
from pydvl.valuation.samplers.base import EvaluationStrategy
from pydvl.valuation.samplers.powerset import NoIndexIteration
from pydvl.valuation.scorers.classwise import ClasswiseScorer
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import BatchGenerator, IndexSetT
from pydvl.valuation.utility.base import UtilityBase
from pydvl.valuation.utility.classwise import CSSample

__all__ = ["ClasswiseShapley"]

logger = logging.getLogger(__name__)


def unique_labels(array: NDArray) -> NDArray:
    """Labels of the dataset."""
    # Object, String, Unicode, Unsigned integer, Signed integer, boolean
    if array.dtype.kind in "OSUiub":
        return np.unique(array)
    raise ValueError("Dataset must be categorical to have unique labels.")


class ClasswiseSampler(IndexSampler):
    def __init__(
        self,
        in_class: IndexSampler,
        out_of_class: PowersetSampler,
        label: int | None = None,
    ):
        super().__init__()
        self.in_class = in_class
        self.out_of_class = out_of_class
        self.label = label

    def for_label(self, label: int) -> ClasswiseSampler:
        return ClasswiseSampler(self.in_class, self.out_of_class, label)

    def from_data(self, data: Dataset) -> Generator[list[CSSample], None, None]:
        assert self.label is not None

        without_label = np.where(data.y != self.label)[0]
        with_label = np.where(data.y == self.label)[0]

        # HACK: the outer sampler is over full subsets of T_{-y_i}
        self.out_of_class._index_iteration = NoIndexIteration

        for ooc_batch in self.out_of_class.from_indices(without_label):
            # NOTE: The inner sampler can be a permutation sampler => we need to
            #  return batches of the same size as that sampler in order for the
            #  in_class strategy to work correctly.
            for ooc_sample in ooc_batch:
                for ic_batch in self.in_class.from_indices(with_label):
                    # FIXME? this sends the same out_of_class_subset for all samples
                    #   maybe a few 10s of KB... probably irrelevant
                    yield [
                        CSSample(
                            idx=ic_sample.idx,
                            label=self.label,
                            subset=ooc_sample.subset,
                            in_class_subset=ic_sample.subset,
                        )
                        for ic_sample in ic_batch
                    ]

    def from_indices(self, indices: IndexSetT) -> BatchGenerator:
        raise AttributeError("Cannot sample from indices directly.")

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy[IndexSampler]:
        return self.in_class.make_strategy(utility, coefficient)


class ClasswiseShapley(Valuation):
    def __init__(
        self,
        utility: UtilityBase,
        sampler: ClasswiseSampler,
        is_done: StoppingCriterion,
        progress: bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.labels: NDArray | None = None
        if not isinstance(utility.scorer, ClasswiseScorer):
            raise ValueError("Scorer must be a ClasswiseScorer.")
        self.scorer: ClasswiseScorer = utility.scorer
        self.is_done = is_done
        self.progress = progress

    def fit(self, data: Dataset):
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations
            algorithm=f"classwise-shapley",
            indices=data.indices,
            data_names=data.data_names,
        )
        ensure_backend_has_generator_return()
        flag = make_parallel_flag()
        parallel = Parallel(return_as="generator_unordered")

        self.utility.training_data = data
        self.labels = unique_labels(np.concatenate((data.y, self.utility.test_data.y)))

        # FIXME, DUH: this loop needs to be in the sampler or we will never converge
        for label in self.labels:
            sampler = self.sampler.for_label(label)
            strategy = sampler.make_strategy(self.utility)
            processor = delayed(strategy.process)
            delayed_evals = parallel(
                processor(batch=list(batch), is_interrupted=flag)
                for batch in sampler.from_data(data)
            )
            for evaluation in tqdm(iterable=delayed_evals, disable=not self.progress):
                self.result.update(evaluation.idx, evaluation.update)
                if self.is_done(self.result):
                    flag.set()
                    break

    def _normalize(self) -> ValuationResult:
        r"""
        Normalize a valuation result specific to classwise Shapley.

        Each value $v_i$ associated with the sample $(x_i, y_i)$ is normalized by
        multiplying it with $a_S(D_{y_i})$ and dividing by $\sum_{j \in D_{y_i}} v_j$.
        For more details, see (Schoch et al., 2022) <sup><a
        href="#schoch_csshapley_2022">1</a> </sup>.

        Returns:
            Normalized ValuationResult object.
        """
        assert self.result is not None
        assert self.utility.training_data is not None

        u = self.utility

        logger.info("Normalizing valuation result.")
        u.model.fit(u.training_data.x, u.training_data.y)

        for idx_label, label in enumerate(self.labels):
            self.scorer.label = label
            active_elements = u.training_data.y == label
            indices_label_set = np.where(active_elements)[0]
            indices_label_set = u.training_data.indices[indices_label_set]

            self.scorer.label = label
            in_class_acc, _ = self.scorer.estimate_in_class_and_out_of_class_score(
                u.model, u.test_data.x, u.test_data.y
            )

            sigma = np.sum(self.result.values[indices_label_set])
            if sigma != 0:
                self.result.scale(in_class_acc / sigma, indices=indices_label_set)

        return self.result
