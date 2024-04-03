from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from pydvl.utils import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import IndexSetT

__all__ = ["Valuation", "ModelFreeValuation"]


class Valuation(ABC):
    def __init__(self):
        self.result: ValuationResult | None = None

    @abstractmethod
    def fit(self):
        ...

    def values(self, indices: IndexSetT | None = None) -> ValuationResult:
        """Returns the valuation result, or a subset of it.

        The valuation must have been run with `fit()` before calling this method.

        Args:
            indices: indices of the subset to return. `None` to receive the full result.
        Returns:
            The result of the valuation.
        """
        if not self.is_fitted:
            raise RuntimeError("Valuation is not fitted")
        assert self.result is not None
        if indices is None:
            return self.result
        return self.result.subset(indices)

    @property
    def is_fitted(self) -> bool:
        return self.result is not None


class ModelFreeValuation(Valuation, ABC):
    """
    TODO: Just a stub
    """

    def __init__(self, data: Dataset, references: Iterable[Dataset]):
        super().__init__()
        self.data = data
        self.datasets = references
        self.result: ValuationResult | None = None

    @abstractmethod
    def fit(self):
        ...

    def values(self, indices: IndexSetT | None = None) -> ValuationResult:
        """Returns the valuation result, or a subset of it.

        The valuation must have been run with `fit()` before calling this method.

        Args:
            indices: indices of the subset to return. `None` to receive the full result.
        Returns:
            The result of the valuation.
        """
        if not self.is_fitted:
            raise RuntimeError("Valuation is not fitted")
        assert self.result is not None
        if indices is None:
            return self.result
        return self.result.subset(indices)

    @property
    def is_fitted(self) -> bool:
        return self.result is not None
