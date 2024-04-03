from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator, Iterable, Protocol, Sequence, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BatchGenerator",
    "IndexT",
    "IndexSetT",
    "LossFunction",
    "NameT",
    "NullaryPredicate",
    "Sample",
    "SampleBatch",
    "SampleGenerator",
    "UtilityEvaluation",
    "ValueUpdate",
]

IndexT = np.int_
IndexSetT = Union[Sequence[IndexT], NDArray[IndexT]]
NameT = Union[np.object_, np.int_]


@dataclass(frozen=True)
class ValueUpdate:
    idx: int
    update: float


# Wow, this escalated quickly...
@dataclass(frozen=True)
class Sample:
    idx: IndexT | None
    subset: frozenset[IndexT]

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset))


SampleBatch = Iterable[Sample]
SampleGenerator = Generator[Sample, None, None]
BatchGenerator = Generator[SampleBatch, None, None]


@dataclass(frozen=True)
class UtilityEvaluation:
    idx: IndexT
    subset: IndexSetT
    evaluation: float

    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.evaluation))


class LossFunction(Protocol):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        ...


NullaryPredicate = Callable[[], bool]
