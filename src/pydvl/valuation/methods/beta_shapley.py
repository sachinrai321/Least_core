import scipy as sp

from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.semivalue import SemivalueValuation
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.evaluator import UtilityEvaluator

__all__ = ["BetaShapleyValuation"]


class BetaShapleyValuation(SemivalueValuation):
    """Computes Beta-Shapley values."""

    algorithm_name = "Beta-Shapley"

    def __init__(
        self,
        evaluator: UtilityEvaluator,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        alpha: float,
        beta: float,
    ):
        super().__init__(evaluator, sampler, is_done)

        self.alpha = alpha
        self.beta = beta
        self.const = sp.special.beta(alpha, beta)

    def coefficient(self, n: int, k: int) -> float:
        j = k + 1
        w = sp.special.beta(j + self.beta - 1, n - j + self.alpha) / self.const
        # return math.comb(n - 1, j - 1) * w * n
        return float(w)
