from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.semivalue import SemivalueValuation
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["GroupTestingValuation"]


class GroupTestingValuation(SemivalueValuation):
    algorithm_name = "Group-Testing-Shapley"

    def __init__(self, utility: UtilityBase, sampler: IndexSampler):
        raise NotImplementedError()
