from __future__ import annotations

from pydvl.valuation.base import Valuation
from pydvl.valuation.samplers.powerset import PowersetSampler
from pydvl.valuation.utility.evaluator import UtilityEvaluator

__all__ = ["LeastCoreValuation"]


class LeastCoreValuation(Valuation):
    def __init__(
        self,
        evaluator: UtilityEvaluator,
        sampler: PowersetSampler,
        n_constraints: int,
        non_negative_subsidy: float,
        solver_options: dict | None = None,
    ):
        self.evaluator = evaluator
        self.sampler = sampler
        self.n_constraints = n_constraints
        self.non_negative_subsidy = non_negative_subsidy
        self.solver_options = solver_options

    def fit(self):
        pass
