from __future__ import annotations

import math

from pydvl.utils import Seed
from pydvl.valuation.samplers import TruncatedUniformStratifiedSampler
from pydvl.valuation.semivalue import SemivalueValuation
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.evaluator import UtilityEvaluator

__all__ = ["DeltaShapleyValuation"]


class DeltaShapleyValuation(SemivalueValuation):
    r"""Computes $\delta$-Shapley values.

    $\delta$-Shapley does not accept custom samplers. Instead it uses a truncated
    hierarchical powerset sampler with a lower and upper bound on the size of the sets
    to sample from.

    TODO See ...
    """

    algorithm_name = "Delta-Shapley"

    def __init__(
        self,
        evaluator: UtilityEvaluator,
        is_done: StoppingCriterion,
        lower_bound: int,
        upper_bound: int,
        seed: Seed | None = None,
    ):
        sampler = TruncatedUniformStratifiedSampler(
            evaluator.utility.data.indices,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=seed,
        )
        super().__init__(evaluator, sampler, is_done)

    def coefficient(self, n: int, k: int) -> float:
        return float(1 / math.comb(n, k))
