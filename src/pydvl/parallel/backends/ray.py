from __future__ import annotations

from concurrent.futures import Executor
from typing import Any, Callable, Iterable, Optional, TypeVar

import ray
from deprecate import deprecated
from ray import ObjectRef
from ray.util.joblib import register_ray

from pydvl.parallel.backend import BaseParallelBackend, CancellationPolicy
from pydvl.parallel.config import ParallelConfig

__all__ = ["RayParallelBackend"]


T = TypeVar("T")


class RayParallelBackend(BaseParallelBackend, backend_name="ray"):
    """Class used to wrap ray to make it transparent to algorithms.

    ??? Example
        ``` python
        import ray
        from pydvl.parallel import RayParallelBackend
        ray.init()
        parallel_backend = RayParallelBackend()
        ```
    """

    @deprecated(
        target=True,
        args_mapping={"config": None},
        deprecated_in="0.9.0",
        remove_in="0.10.0",
    )
    def __init__(self, config: Optional[ParallelConfig] = None) -> None:
        if not ray.is_initialized():
            raise RuntimeError(
                "Starting from v0.9.0, ray is no longer automatically initialized. "
                "Please use `ray.init()` with the desired configuration "
                "before using this class."
            )
        # Register ray joblib backend
        register_ray()

    @classmethod
    def executor(
        cls,
        max_workers: int | None = None,
        config: ParallelConfig = ParallelConfig(),
        cancel_futures: CancellationPolicy = CancellationPolicy.PENDING,
    ) -> Executor:
        from pydvl.parallel.futures.ray import RayExecutor

        return RayExecutor(max_workers, config=config, cancel_futures=cancel_futures)  # type: ignore

    def get(self, v: ObjectRef | Iterable[ObjectRef] | T, *args, **kwargs) -> T | Any:
        timeout: float | None = kwargs.get("timeout", None)
        if isinstance(v, ObjectRef):
            return ray.get(v, timeout=timeout)
        elif isinstance(v, Iterable):
            return [self.get(x, timeout=timeout) for x in v]
        else:
            return v

    def put(self, v: T, *args, **kwargs) -> ObjectRef[T] | T:
        try:
            return ray.put(v, **kwargs)  # type: ignore
        except TypeError:
            return v  # type: ignore

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a ray remote.

        Args:
            fun: the function to wrap
            kwargs: keyword arguments to pass to @ray.remote

        Returns:
            The `.remote` method of the ray `RemoteFunction`.
        """
        if len(kwargs) > 0:
            return ray.remote(**kwargs)(fun).remote  # type: ignore
        return ray.remote(fun).remote  # type: ignore

    def wait(
        self, v: list[ObjectRef], *args, **kwargs
    ) -> tuple[list[ObjectRef], list[ObjectRef]]:
        num_returns: int = kwargs.get("num_returns", 1)
        timeout: float | None = kwargs.get("timeout", None)
        return ray.wait(v, num_returns=num_returns, timeout=timeout)  # type: ignore

    def _effective_n_jobs(self, n_jobs: int) -> int:
        ray_cpus = int(ray._private.state.cluster_resources()["CPU"])  # type: ignore
        if n_jobs < 0:
            eff_n_jobs = ray_cpus
        else:
            eff_n_jobs = min(n_jobs, ray_cpus)
        return eff_n_jobs
