"""
This module implements 2D-Shapley, as introduced in (Liu et al., 2023)<sup><a
href="#liu_2dshapley_2023">1</a></sup>.


## References

[^1]: <a name="liu_2dshapley_2023"></a>Liu, Zhihong, Hoang Anh Just, Xiangyu Chang, Xi
      Chen, and Ruoxi Jia. [2D-Shapley: A Framework for Fragmented Data
      Valuation](https://proceedings.mlr.press/v202/liu23s.html). In Proceedings of the
      40th International Conference on Machine Learning, 21730–55. PMLR, 2023.

"""
from __future__ import annotations

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import Sample


class TwoDSample(Sample):
    """A sample for 2D-Shapley, consisting of a set of indices and a set of features."""

    features: frozenset[int]


class TwoDShapley(Valuation):
    def fit(self, data: Dataset):
        # With the right sampler and a subclassed utility, this should follow a very
        # similar pattern to the other methods.
        # Note that it should be trivial to generalize to other coefficients, sampling
        # strategies, etc.
        pass
