r"""
This module contains the base class for all semi-value valuation methods.

A **semi-value** is any valuation function with the form:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
                     \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],
$$

where $U$ is the utility, and the coefficients $w(k)$ satisfy the property:

$$
\sum_{k=1}^n w(k) = 1.
$$

This is the largest class of marginal-contribution-based valuation methods. These
compute the value of a data point by evaluating the change in utility when the data
point is removed from one or more subsets of the data.
"""
from __future__ import annotations

from abc import abstractmethod

from pydvl.valuation.base import Valuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.evaluator import UtilityEvaluator

__all__ = ["SemivalueValuation"]


class SemivalueValuation(Valuation):
    r"""Abstract class to define semi-values.

    Implementations must only provide the `coefficient()` method, corresponding
    to the semi-value coefficient.

    !!! Note
        For implementation consistency, we slightly depart from the common definition
        of semi-values, which includes a factor $1/n$ in the sum over subsets.
        Instead, we subsume this factor into the coefficient $w(k)$.
        TODO: see ...

    Args:
        evaluator: object to compute utilities.
        sampler: Sampling scheme to use.
        is_done: Stopping criterion to use.
    """

    algorithm_name = "Semi-Value"

    def __init__(
        self,
        evaluator: UtilityEvaluator,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.sampler = sampler
        self.data = evaluator.utility.data
        self.is_done = is_done

    @abstractmethod
    def coefficient(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed
        """
        ...

    def fit(self):
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations
            algorithm=f"{self.__class__.__name__}-{self.sampler.__class__.__name__}-{self.evaluator.utility.model}-{self.is_done}",
            indices=self.data.indices,
            data_names=self.data.data_names,
        )

        strategy = self.sampler.strategy()

        with self.evaluator as evaluator:
            strategy.setup(evaluator.utility, self.coefficient)
            for batch in evaluator.map(strategy.process, self.sampler):
                for evaluation in batch:
                    self.result.update(evaluation.idx, evaluation.update)
                if self.is_done(self.result):
                    evaluator.interrupt()

        # FIXME: remove NaN checking after fit()?
        import logging

        import numpy as np

        logger = logging.getLogger(__name__)
        nans = np.isnan(self.result.values).sum()
        if nans > 0:
            logger.warning(
                f"{nans} NaN values in current result. "
                "Consider setting a default value for the Scorer"
            )
