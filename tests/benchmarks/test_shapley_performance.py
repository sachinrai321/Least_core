import pytest

from tests.conftest import check_exact, check_total_value
from valuation.shapley import (
    combinatorial_exact_shapley,
    combinatorial_montecarlo_shapley,
    permutation_exact_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)

pytestmark = [
    pytest.mark.benchmark(
        group="shapley",
        min_rounds=5,
    ),
    pytest.mark.parametrize("num_samples", [3, 6]),
]


@pytest.mark.parametrize(
    "method",
    [
        combinatorial_exact_shapley,
        permutation_exact_shapley,
    ],
)
def test_exact_shapley_performance(exact_shapley, method, benchmark):
    u, exact_values = exact_shapley
    values = benchmark(method, u, progress=False)
    check_total_value(u, values)
    check_exact(values, exact_values)


@pytest.mark.parametrize("max_iterations", [100])
@pytest.mark.parametrize(
    "method",
    [
        pytest.param(combinatorial_montecarlo_shapley, marks=pytest.mark.xfail),
        permutation_montecarlo_shapley,
    ],
)
def test_montecarlo_shapley_performance(
    exact_shapley, method, max_iterations, benchmark
):
    u, exact_values = exact_shapley
    result = benchmark(
        method, u, max_iterations=max_iterations, progress=False, use_cache=False
    )
    values = result[0]
    # TODO: properly test these two checks because they fail for the combinatorial method
    check_total_value(u, values)
    check_exact(values, exact_values)


@pytest.mark.xfail
def test_truncated_montecarlo_shapley_performance(exact_shapley, benchmark):
    u, exact_values = exact_shapley
    result = benchmark(
        truncated_montecarlo_shapley,
        u,
        bootstrap_iterations=100,
        min_scores=10,
        score_tolerance=1e-6,
        min_values=10,
        value_tolerance=1e-6,
        max_iterations=100,
        num_workers=2,
        progress=False,
    )
    values = result[0]
    # TODO: properly test these two checks because they fail for the truncated method
    check_total_value(u, values)
    check_exact(values, exact_values)
