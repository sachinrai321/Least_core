from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from pydvl.valuation.scorers.classwise import ClasswiseScorer
from pydvl.valuation.types import IndexSetT, IndexT, Sample
from pydvl.valuation.utility import Utility

__all__ = ["CSSample", "ClasswiseUtility"]


@dataclass(frozen=True)
class CSSample(Sample):
    label: int | None
    in_class_subset: frozenset[IndexT]

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.label, self.in_class_subset))


class ClasswiseUtility(Utility[CSSample]):
    """
    FIXME: probably unnecessary, just a test
    """

    scorer: ClasswiseScorer

    def __call__(self, sample: CSSample) -> float:
        return cast(float, self._utility_wrapper(sample))

    def _utility(self, sample: CSSample) -> float:
        self.scorer.label = sample.label
        # TODO: do the thing
        raise NotImplementedError
