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
from pydvl.valuation.scorers.classwise import ClasswiseScorer
from pydvl.valuation.stopping import StoppingCriterion
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


class ClasswiseShapley(Valuation):
    def __init__(
        self,
        utility: UtilityBase,
        # TODO: create the factories
        in_class_sampler_factory: Callable[..., IndexSampler],
        out_of_class_sampler_factory: Callable[..., PowersetSampler],
        is_done: StoppingCriterion,
        progress: bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.labels: NDArray | None = None
        if not isinstance(utility.scorer, ClasswiseScorer):
            raise ValueError("Scorer must be a ClasswiseScorer.")
        self.scorer: ClasswiseScorer = utility.scorer
        self.in_class_sampler_factory = in_class_sampler_factory
        self.out_of_class_sampler_factory = out_of_class_sampler_factory
        self.inner_sampler: IndexSampler | None = None
        self.outer_sampler: PowersetSampler | None = None
        self.is_done = is_done
        self.progress = progress

    def indices_without_label(self, data: Dataset, label: int):
        return np.where(data.y != label)[0]

    def indices_with_label(self, data: Dataset, label: int):
        return np.where(data.y == label)[0]

    def sampler(self, data: Dataset, label: int) -> Generator[CSSample, None, None]:
        self.outer_sampler = self.out_of_class_sampler_factory(
            self.indices_without_label(data, label)
        )
        self.inner_sampler = self.in_class_sampler_factory(
            self.indices_with_label(data, label)
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

        for label in self.labels:
            sampler = self.sampler(data, label)
            strategy = self.inner_sampler.make_strategy(self.utility)
            processor = delayed(strategy.process)
            delayed_evals = parallel(
                processor(batch=list(batch), is_interrupted=flag) for batch in sampler
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
