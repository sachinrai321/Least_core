from __future__ import annotations

import logging
from typing import Callable, Generator

import numpy as np
from numpy.typing import NDArray

from pydvl.valuation.base import Valuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler, PowersetSampler
from pydvl.valuation.scorers.classwise import ClasswiseScorer
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import NullaryPredicate, ValueUpdate
from pydvl.valuation.utility import Utility
from pydvl.valuation.utility.classwise import CSSample
from pydvl.valuation.utility.evaluator import UtilityEvaluator

__all__ = ["ClasswiseShapley"]

logger = logging.getLogger(__name__)


def unique_labels(array: NDArray) -> NDArray:
    """Labels of the dataset."""
    # Object, String, Unicode, Unsigned integer, Signed integer, boolean
    if array.dtype.kind in "OSUiub":
        return np.unique(array)
    raise ValueError("Dataset must be categorical to have unique labels.")


class ClasswiseShapley(Valuation):
    def __init__(
        self,
        evaluator: UtilityEvaluator,
        in_class_sampler_factory: Callable[[...], IndexSampler],
        out_of_class_sampler_factory: Callable[[...], PowersetSampler],
        is_done: StoppingCriterion,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.data = evaluator.utility.data
        if not isinstance(evaluator.utility.scorer, ClasswiseScorer):
            raise ValueError("Scorer must be a ClasswiseScorer.")
        self.scorer: ClasswiseScorer = evaluator.utility.scorer
        self.in_class_sampler_factory = in_class_sampler_factory
        self.out_of_class_sampler_factory = out_of_class_sampler_factory
        self.inner_sampler: PowersetSampler | None = None
        self.outer_sampler: IndexSampler | None = None
        self.is_done = is_done

    def indices_without_label(self, label: int):
        return np.where(self.data.y_train != label)[0]

    def indices_with_label(self, label):
        return np.where(self.data.y_train == label)[0]

    def sampler(self, label: int) -> Generator[CSSample, None, None]:
        self.outer_sampler = self.out_of_class_sampler_factory(
            self.indices_without_label(label)
        )
        self.inner_sampler = self.in_class_sampler_factory(
            self.indices_with_label(label)
        )
        for out_of_class_sample in self.outer_sampler.generate():
            for in_class_sample in self.inner_sampler.generate():
                # FIXME? this sends the same out_of_class_subset for all samples
                #   maybe a few 10s of KB... probably irrelevant
                yield CSSample(
                    idx=in_class_sample.idx,
                    label=label,
                    subset=out_of_class_sample.subset,
                    in_class_subset=in_class_sample.subset,
                )

    # TODO: Do I need this? If the utility accepted Sample, we could subclass
    #  for CSShapley and use whatever EvaluationStrategy directly (typically
    #  PermuationEvaluationStrategy)
    def batch_processor(
        self, utility: Utility, is_interrupted: NullaryPredicate, batch: list[CSSample]
    ) -> list[ValueUpdate]:
        for sample in batch:
            self.scorer.label = sample.label
            # TODO do stuff
            raise NotImplementedError

    def fit(self):
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations
            algorithm=f"classwise-shapley",
            indices=self.data.indices,
            data_names=self.data.data_names,
        )

        labels = unique_labels(np.concatenate((self.data.y_train, self.data.y_test)))

        with self.evaluator as evaluator:
            for label in labels:
                sampler = self.sampler(label)
                strategy = self.inner_sampler.strategy()
                for evaluation in evaluator.map(strategy.process, sampler):
                    self.result.update(evaluation.idx, evaluation.update)
                    if self.is_done(self.result):
                        evaluator.interrupt()  # FIXME: does this work?

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
        u = self.evaluator.utility
        unique_labels = np.unique(np.concatenate((u.data.y_train, u.data.y_test)))

        logger.info("Normalizing valuation result.")
        u.model.fit(u.data.x_train, u.data.y_train)

        for idx_label, label in enumerate(unique_labels):
            self.scorer.label = label
            active_elements = u.data.y_train == label
            indices_label_set = np.where(active_elements)[0]
            indices_label_set = u.data.indices[indices_label_set]

            self.scorer.label = label
            in_class_acc, _ = self.scorer.estimate_in_class_and_out_of_class_score(
                u.model, u.data.x_test, u.data.y_test
            )

            sigma = np.sum(self.result.values[indices_label_set])
            if sigma != 0:
                self.result.scale(in_class_acc / sigma, indices=indices_label_set)

        return self.result
