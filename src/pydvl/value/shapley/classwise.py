"""
Implementation of Class-wise Shapley, introduced in (Schoch, Haifeng and Ji,
2022)[^1].

# References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-Wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.

"""
import logging
import numbers
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import copy
from typing import Callable, Optional, Set, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.utils import (
    Dataset,
    ParallelConfig,
    Scorer,
    Seed,
    SupervisedModel,
    Utility,
    effective_n_jobs,
    ensure_seed_sequence,
    init_executor,
    init_parallel_backend,
    random_powerset_group_conditional,
)
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import TruncationPolicy
from pydvl.value.stopping import MaxChecks, StoppingCriterion

logger = logging.getLogger(__name__)

__all__ = ["compute_classwise_shapley_values", "ClasswiseScorer"]


def compute_classwise_shapley_values(
    u: Utility,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    done_sample_complements: Optional[StoppingCriterion] = None,
    normalize_values: bool = True,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """
    Computes the class-wise Shapley values as described in (Schoch, Haifeng and
    Ji, 2022)<sup><a href="#schoch_csshapley_2022">1</a></sup>.
    The values can be optionally normalized, depending on `normalize_values`.

    Args:
        u: Utility object containing model, data, and scoring function. The
            scorer must be of type
            [ClassWiseScorer][pydvl.value.shapley.classwise.ClasswiseScorer].
        done: Function that checks whether the computation needs to stop.
        truncation: Callable function that decides whether to interrupt processing a
            permutation and set subsequent marginals to zero.
        done_sample_complements: Function checking whether computation needs to stop.
            Otherwise, it will resample conditional sets until the stopping criterion is
            met.
        normalize_values: Indicates whether to normalize the values by the variation
            in each class times their in-class accuracy.
        done_sample_complements: Number of times to resample the complement set
            for each permutation.
        use_default_scorer_value: The first set of indices is the sampled complement
            set. Unless not otherwise specified, the default scorer value is used for
            this. If it is set to false, the base score is calculated from the utility.
        min_elements_per_label: The minimum number of elements for each opposite
            label.
        n_jobs: Number of parallel jobs to run.
        config: Parallel configuration.
        progress: Whether to display a progress bar.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        ValuationResult object containing computed data values.

    !!! tip "New in version 0.7.0"
    """

    _check_classwise_shapley_utility(u)

    parallel_backend = init_parallel_backend(config)
    u_ref = parallel_backend.put(u)
    n_jobs = effective_n_jobs(n_jobs, config)
    n_submitted_jobs = 2 * n_jobs

    pbar = tqdm(disable=not progress, position=0, total=100, unit="%")
    accumulated_result = ValuationResult.zeros(
        algorithm="classwise_shapley",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )
    terminate_exec = False
    seed_sequence = ensure_seed_sequence(seed)

    with init_executor(max_workers=n_jobs, config=config) as executor:
        pending: Set[Future] = set()
        while True:
            completed_futures, pending = wait(
                pending, timeout=60, return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                accumulated_result += future.result()
                if done(accumulated_result):
                    terminate_exec = True
                    break

            pbar.n = 100 * done.completion()
            pbar.refresh()
            if terminate_exec:
                break

            n_remaining_slots = n_submitted_jobs - len(pending)
            seeds = seed_sequence.spawn(n_remaining_slots)
            for i in range(n_remaining_slots):
                future = executor.submit(
                    _permutation_montecarlo_classwise_shapley,
                    u_ref,
                    truncation=truncation,
                    done_sample_complements=done_sample_complements,
                    use_default_scorer_value=use_default_scorer_value,
                    min_elements_per_label=min_elements_per_label,
                    seed=seeds[i],
                )
                pending.add(future)

    result = accumulated_result
    if normalize_values:
        result = _normalize_classwise_shapley_values(result, u)

    return result


def _permutation_montecarlo_classwise_shapley(
    u: Utility,
    *,
    done_sample_complements: StoppingCriterion = None,
    truncation: TruncationPolicy,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """Computes classwise Shapley value using truncated Monte Carlo permutation
    sampling for the subsets.

    Args:
        u: Utility object containing model, data, and scoring function. The scoring
            function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
        done_sample_complements: Function checking whether computation needs to stop.
            Otherwise, it will resample conditional sets until the stopping criterion is
            met.
        truncation: Callable function that decides whether to interrupt processing a
            permutation and set subsequent marginals to zero.
        use_default_scorer_value: The first set of indices is the sampled complement
            set. Unless not otherwise specified, the default scorer value is used for
            this. If it is set to false, the base score is calculated from the utility.
        min_elements_per_label: The minimum number of elements for each opposite
            label.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        ValuationResult object containing computed data values.
    """
    if done_sample_complements is None:
        done_sample_complements = MaxChecks(1)

    result = ValuationResult.zeros(
        algorithm="classwise_shapley",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )
    rng = np.random.default_rng(seed)
    x_train, y_train = u.data.get_training_data(u.data.indices)
    unique_labels = np.unique(y_train)
    scorer = cast(ClasswiseScorer, copy(u.scorer))
    u.scorer = scorer

    for label in unique_labels:
        u.scorer.label = label
        result += _permutation_montecarlo_classwise_shapley_for_label(
            u,
            label,
            done=done_sample_complements,
            truncation=truncation,
            use_default_scorer_value=use_default_scorer_value,
            min_elements_per_label=min_elements_per_label,
            seed=rng,
        )

    return result


def _check_classwise_shapley_utility(u: Utility):
    """
    Verifies if the provided utility object supports classwise Shapley values.

    Args:
        u: Utility object containing model, data, and scoring function. The scoring
            function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.

    Raises:
        ValueError: If ``u.data`` is not a classification problem.
        ValueError: If ``u.scorer`` is not an instance of
            :class:`~pydvl.utils.score.ClassWiseScorer`
    """

    dim_correct = u.data.y_train.ndim == 1 and u.data.y_test.ndim == 1
    is_integral = all(
        map(
            lambda v: isinstance(v, numbers.Integral), (*u.data.y_train, *u.data.y_test)
        )
    )
    if not dim_correct or not is_integral:
        raise ValueError(
            "The supplied dataset has to be a 1-dimensional classification dataset."
        )

    if not isinstance(u.scorer, ClasswiseScorer):
        raise ValueError(
            "Please set a subclass of ClassWiseScorer object as scorer object of the"
            " utility. See scoring argument of Utility."
        )


def _normalize_classwise_shapley_values(
    result: ValuationResult,
    u: Utility,
) -> ValuationResult:
    """
    Normalize a valuation result specific to classwise Shapley.

    Each value corresponds to a class c and gets normalized by multiplying
    `in-class-score / sigma`. In this context `sigma` is the magnitude of all values
    belonging to the currently viewed class. See footcite:t:`schoch_csshapley_2022` for
    more details.

    Args:
        result: ValuationResult object to be normalized.
        u: Utility object containing model, data, and scoring function. The scoring
            function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.

    Returns:
        Normalized ValuationResult object.
    """
    y_train = u.data.y_train
    unique_labels = np.unique(np.concatenate((y_train, u.data.y_test)))
    scorer = cast(ClasswiseScorer, u.scorer)

    for idx_label, label in enumerate(unique_labels):
        scorer.label = label
        active_elements = y_train == label
        indices_label_set = np.where(active_elements)[0]
        indices_label_set = u.data.indices[indices_label_set]

        u.model.fit(u.data.x_train, u.data.y_train)
        scorer.label = label
        in_cls_acc, _ = scorer.estimate_in_cls_and_out_of_cls_score(
            u.model, u.data.x_test, u.data.y_test
        )

        sigma = np.sum(result.values[indices_label_set])
        if sigma != 0:
            result.scale(in_cls_acc / sigma, indices=indices_label_set)

    return result


class ClasswiseScorer(Scorer):
    """A Scorer which is applicable for valuation in classification problems. Its value
    is based on in-cls and out-of-cls score :footcite:t:`schoch_csshapley_2022`. For
    each class ``label`` it separates the elements into two groups, namely in-cls
    instances and out-of-cls instances. The value function itself than estimates the
    in-cls metric discounted by the out-of-cls metric. In other words the value function
    for each element of one class is conditioned on the out-of-cls instances (or a
    subset of it). The form of the value function can be written as

    .. math::
        v_{y_i}(D) = f(a_S(D_{y_i}))) * g(a_S(D_{-y_i})))

    where f and g are continuous, monotonic functions and D is the test set.

    in order to produce meaningful results. For further reference see also section four
    of :footcite:t:`schoch_csshapley_2022`.

    Args:
        default: Score used when a model cannot be fit, e.g. when too little data is
            passed, or errors arise.
        range: Numerical range of the score function. Some Monte Carlo methods can
            use this to estimate the number of samples required for a certain quality of
            approximation. If not provided, it can be read from the ``scoring`` object
            if it provides it, for instance if it was constructed with
            :func:`~pydvl.utils.types.compose_score`.
        in_class_discount_fn: Continuous, monotonic increasing function used to
            discount the in-class score.
        out_of_class_discount_fn: Continuous, monotonic increasing function used to
            discount the out-of-class score.
        initial_label: Set initial label (Doesn't require to set parameter ``label``
            on ``ClassWiseDiscountedScorer`` in first iteration)
        name: Name of the scorer. If not provided, the name of the passed
            function will be prefixed by 'classwise '.

    !!! tip "New in version 0.7.0"
    """

    def __init__(
        self,
        scoring: str = "accuracy",
        default: float = 0.0,
        range: Tuple[float, float] = (-np.inf, np.inf),
        in_class_discount_fn: Callable[[float], float] = lambda x: x,
        out_of_class_discount_fn: Callable[[float], float] = np.exp,
        initial_label: Optional[int] = None,
        name: Optional[str] = None,
    ):
        disc_score_in_cls = in_class_discount_fn(range[1])
        disc_score_out_of_cls = out_of_class_discount_fn(range[1])
        transformed_range = (0, disc_score_in_cls * disc_score_out_of_cls)
        super().__init__(
            "accuracy",
            range=transformed_range,
            default=default,
            name=name or f"classwise {scoring}",
        )
        self._in_cls_discount_fn = in_class_discount_fn
        self._out_of_cls_discount_fn = out_of_class_discount_fn
        self.label = initial_label

    def __str__(self):
        return self._name

    def __call__(
        self: "ClasswiseScorer",
        model: SupervisedModel,
        x_test: NDArray[np.float_],
        y_test: NDArray[np.int_],
    ) -> float:
        """
        Args:
            model: Model used for computing the score on the validation set.
            x_test: Array containing the features of the classification problem.
            y_test: Array containing the labels of the classification problem.

        Returns:
            Calculated score.
        """
        in_cls_score, out_of_cls_score = self.estimate_in_cls_and_out_of_cls_score(
            model, x_test, y_test
        )
        disc_score_in_cls = self._in_cls_discount_fn(in_cls_score)
        disc_score_out_of_cls = self._out_of_cls_discount_fn(out_of_cls_score)
        return disc_score_in_cls * disc_score_out_of_cls

    def estimate_in_cls_and_out_of_cls_score(
        self,
        model: SupervisedModel,
        x_test: NDArray[np.float_],
        y_test: NDArray[np.int_],
        rescale_scores: bool = True,
    ) -> Tuple[float, float]:
        r"""
        Computes in-class and out-of-class scores using the provided scoring function,
        which can be expressed as:

        .. math::
            a_S(D=\{(\hat{x}_1, \hat{y}_1), \dots, (\hat{x}_K, \hat{y}_K)\}) &=
            \frac{1}{N} \sum_k s(y(\hat{x}_k), \hat{y}_k)

        In this context, the computation is performed twice: once on D_i and once on D_o
        to calculate the in-class and out-of-class scores. Here, D_i contains only
        samples with the specified 'label' from the validation set, while D_o contains
        all other samples. By default, the scores are scaled to have the same order of
        magnitude. In such cases, the raw scores are multiplied by:

        .. math::
            N_{y_i} = \frac{a_S(D_{y_i})}{a_S(D_{y_i})+a_S(D_{-y_i})} \quad \text{and}
            \quad N_{-y_i} = \frac{a_S(D_{-y_i})}{a_S(D_{y_i})+a_S(D_{-y_i})}

        :param model: Model used for computing the score on the validation set.
        :param x_test: Array containing the features of the classification problem.
        :param y_test: Array containing the labels of the classification problem.
        :param rescale_scores: If set to True, the scores will be denormalized. This is
            particularly useful when the inner score is calculated by an estimator of
            the form 1/N sum_i x_i.
        :return: Tuple containing the in-class and out-of-class scores.
        """
        scorer = self._scorer
        label_set_match = y_test == self.label
        label_set = np.where(label_set_match)[0]
        num_classes = len(np.unique(y_test))

        if len(label_set) == 0:
            return 0, 1 / (num_classes - 1)

        complement_label_set = np.where(~label_set_match)[0]
        in_cls_score = scorer(model, x_test[label_set], y_test[label_set])
        out_of_cls_score = scorer(
            model, x_test[complement_label_set], y_test[complement_label_set]
        )

        if rescale_scores:
            n_in_cls = np.count_nonzero(y_test == self.label)
            n_out_of_cls = len(y_test) - n_in_cls
            in_cls_score *= n_in_cls / (n_in_cls + n_out_of_cls)
            out_of_cls_score *= n_out_of_cls / (n_in_cls + n_out_of_cls)

        return in_cls_score, out_of_cls_score


def _permutation_montecarlo_classwise_shapley_for_label(
    u: Utility,
    label: int,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """
    Samples a random subset of the complement set and computes the truncated Monte Carlo
    estimator.

    Args:
    :param u: Utility object containing model, data, and scoring function. The scoring
        function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    :param done: Function checking whether computation needs to stop. Otherwise, it will
        resample conditional sets until the stopping criterion is met.
    :param label: The label for which to sample the complement (e.g. all other labels)
    :param truncation: Callable which decides whether to interrupt processing a
        permutation and set all subsequent marginals to zero.
    :param use_default_scorer_value: Use default scorer value even if additional_indices
        is not None.
    :param min_elements_per_label: The minimum number of elements for each opposite
        label.
    :param seed: Either an instance of a numpy random number generator or a seed for it.

    :return: ValuationResult object containing computed data values.
    """

    algorithm_name = "classwise_shapley"
    result = ValuationResult.zeros(
        algorithm="classwise_shapley",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    rng = np.random.default_rng(seed)
    _, y_train = u.data.get_training_data(u.data.indices)
    class_indices_set, class_complement_indices_set = split_indices_by_label(
        u.data.indices,
        y_train,
        label,
    )
    _, complement_y_train = u.data.get_training_data(class_complement_indices_set)
    indices_permutation = rng.permutation(class_indices_set)

    for subset_idx, subset_complement in enumerate(
        random_powerset_group_conditional(
            class_complement_indices_set,
            complement_y_train,
            min_elements_per_group=min_elements_per_label,
            seed=rng,
        )
    ):
        result += _permutation_montecarlo_shapley_rollout(
            u,
            indices_permutation,
            additional_indices=subset_complement,
            truncation=truncation,
            algorithm_name=algorithm_name,
            use_default_scorer_value=use_default_scorer_value,
        )
        if done(result):
            break

    return result


def _permutation_montecarlo_shapley_rollout(
    u: Utility,
    permutation: NDArray[np.int_],
    truncation: TruncationPolicy,
    algorithm_name: str,
    additional_indices: Optional[NDArray[np.int_]] = None,
    use_default_scorer_value: bool = True,
) -> ValuationResult:
    """
    A truncated version of a permutation-based MC estimator.
    values. It generates a permutation p[i] of the class label indices and iterates over
    all subsets starting from the empty set to the full set of indices.

    Args:
        u: Utility object containing model, data, and scoring function.
        permutation: Permutation of indices to be considered.
        truncation: Callable which decides whether to interrupt processing a
            permutation and set all subsequent marginals to zero.
        algorithm_name: For the results object. Used internally by different
            variants of Shapley using this subroutine
        additional_indices: Set of additional indices for data points which should be
            always considered.
        use_default_scorer_value: Use default scorer value even if additional_indices
            is not None.
    Returns:
         ValuationResult object containing computed data values.
    """
    if (
        additional_indices is not None
        and len(np.intersect1d(permutation, additional_indices)) > 0
    ):
        raise ValueError(
            "The class label set and the complement set have to be disjoint."
        )

    result = ValuationResult.zeros(
        algorithm=algorithm_name,
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    prev_score = (
        u.default_score
        if (
            use_default_scorer_value
            or additional_indices is None
            or additional_indices is not None
            and len(additional_indices) == 0
        )
        else u(additional_indices)
    )

    truncation_u = u
    if additional_indices is not None:
        # hack to calculate the correct value in reset.
        truncation_indices = np.sort(np.concatenate((permutation, additional_indices)))
        truncation_u = Utility(
            u.model,
            Dataset(
                u.data.x_train[truncation_indices],
                u.data.y_train[truncation_indices],
                u.data.x_test,
                u.data.y_test,
            ),
            u.scorer,
        )
    truncation.reset(truncation_u)

    is_terminated = False
    for i, idx in enumerate(permutation):
        if is_terminated or (is_terminated := truncation(i, prev_score)):
            score = prev_score
        else:
            score = u(
                np.concatenate((permutation[: i + 1], additional_indices))
                if additional_indices is not None and len(additional_indices) > 0
                else permutation[: i + 1]
            )

        marginal = score - prev_score
        result.update(idx, marginal)
        prev_score = score

    return result


def split_indices_by_label(
    indices: NDArray[np.int_], labels: NDArray[np.int_], label: int
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Splits the indices into two sets based on the value of  ``label``: those samples
    with and without that label.

    :param indices: The indices to be used for referring to the data.
    :param labels: Corresponding labels for the indices.
    :param label: Label to be used for splitting.
    :return: Tuple with two sets of indices.
    """
    active_elements = labels == label
    class_indices_set = np.where(active_elements)[0]
    class_complement_indices_set = np.where(~active_elements)[0]
    class_indices_set = indices[class_indices_set]
    class_complement_indices_set = indices[class_complement_indices_set]
    return class_indices_set, class_complement_indices_set
