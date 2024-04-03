from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.semivalue import SemivalueValuation

__all__ = ["GroupTestingValuation"]

from pydvl.valuation.utility.evaluator import UtilityEvaluator


class GroupTestingValuation(SemivalueValuation):
    algorithm_name = "Group-Testing-Shapley"

    def __init__(self, evaluator: UtilityEvaluator, sampler: IndexSampler):
        raise NotImplementedError()
