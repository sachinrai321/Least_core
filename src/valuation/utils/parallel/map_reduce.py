import weakref
from itertools import repeat
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import ray
from ray import ObjectRef

from ..config import ParallelConfig
from .backend import init_parallel_backend

T = TypeVar("T")
R = TypeVar("R")
Identity = lambda x, *args, **kwargs: x

MapFunction = Callable[..., R]
ReduceFunction = Callable[[Iterable[R]], R]

__all__ = ["MapReduceJob"]


def wrap_func_with_remote_args(func, *, timeout: int = 300):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, v in enumerate(args[:]):
            args[i] = get_value(v, timeout=timeout)
        for k, v in kwargs.items():
            kwargs[k] = get_value(v, timeout=timeout)
        return func(*args, **kwargs)

    return wrapper


def get_value(v: Union[ObjectRef, Iterable[ObjectRef], Any], *, timeout: int = 300):
    if isinstance(v, ObjectRef):
        return ray.get(v, timeout=timeout)
    elif isinstance(v, Iterable):
        return [get_value(x, timeout=timeout) for x in v]
    else:
        return v


class MapReduceJob(Generic[T, R]):
    """Takes an embarrassingly parallel fun and runs it in num_jobs parallel
    jobs, splitting the data into the same number of chunks, one for each job.

    It repeats the process num_runs times, allocating jobs across runs. E.g.
    if num_jobs = 90 and num_runs=10, each whole execution of fun uses 9 jobs,
    with the data split evenly among them. If num_jobs=2 and num_runs=10, two
    cores are used, five times in succession, and each job receives all data.

    Results are aggregated per run using reduce_func(), but not across runs.

    :param fun: Create with `MapReduceJob.from_fun(fun, reducer, ...)`
    :param data: values to split across jobs
    :param n_jobs: number of parallel jobs to run. Does not accept -1
    :param n_runs: number of times to run fun on the whole data.
    """

    def __init__(
        self,
        map_func: MapFunction,
        reduce_func: Optional[ReduceFunction] = None,
        map_kwargs: Optional[Dict] = None,
        reduce_kwargs: Optional[Dict] = None,
        config: ParallelConfig = ParallelConfig(),
        *,
        chunkify_inputs: bool = True,
        n_jobs: int = 1,
        n_runs: int = 1,
    ):
        self.config = config
        parallel_backend = init_parallel_backend(self.config)
        self._parallel_backend_ref = weakref.ref(parallel_backend)

        self._n_jobs = 1
        self.n_jobs = n_jobs
        self.n_runs = n_runs
        self.chunkify_inputs = chunkify_inputs

        if reduce_func is None:
            reduce_func = Identity

        self.map_kwargs = map_kwargs
        self.reduce_kwargs = reduce_kwargs

        if self.map_kwargs is None:
            self.map_kwargs = dict()

        if self.reduce_kwargs is None:
            self.reduce_kwargs = dict()

        self._map_func = map_func
        self._reduce_func = reduce_func

    def __call__(
        self,
        inputs: Union[Collection[T], Any],
        *,
        n_jobs: Optional[int] = None,
        n_runs: Optional[int] = None,
        chunkify_inputs: Optional[bool] = None,
    ) -> List[R]:
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if n_runs is not None:
            self.n_runs = n_runs
        if chunkify_inputs is not None:
            self.chunkify_inputs = chunkify_inputs

        map_results = self.map(inputs)
        reduce_results = self.reduce(map_results)
        return reduce_results

    def map(self, inputs: Union[Sequence[T], Any]) -> List[List[R]]:
        map_results = []

        map_func = self._wrap_function(self._map_func)

        for _ in range(self.n_runs):
            if self.chunkify_inputs and self.n_jobs > 1:
                chunks = self._chunkify(inputs, num_chunks=self.n_jobs)
            else:
                chunks = repeat(inputs, times=self.n_jobs)

            map_result = [
                map_func(next_chunk, **self.map_kwargs) for next_chunk in chunks
            ]

            map_results.append(map_result)

        return map_results

    def reduce(self, chunks: List[List[R]]) -> List[R]:
        futures = []

        reduce_func = self._wrap_function(self._reduce_func)

        for i in range(self.n_runs):
            future = reduce_func(chunks[i], **self.reduce_kwargs)
            futures.append(future)
        results = self.parallel_backend.get(futures, timeout=300)
        return results  # type: ignore

    def _wrap_function(self, func):
        remote_func = self.parallel_backend.wrap(wrap_func_with_remote_args(func))
        return remote_func.remote

    @staticmethod
    def _chunkify(data: Sequence[T], num_chunks: int) -> Iterator[Sequence[T]]:
        # Splits a list of values into chunks for each job
        n = len(data)
        chunk_size = n // num_chunks
        remainder = n % num_chunks
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, n)
            yield data[start_index:end_index]
        if remainder > 0:
            yield data[n - remainder :]

    @property
    def parallel_backend(self):
        parallel_backend = self._parallel_backend_ref()
        if parallel_backend is None:
            raise RuntimeError(f"Could not get reference to parallel backend instance")
        return parallel_backend

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = self.parallel_backend.effective_n_jobs(value)
