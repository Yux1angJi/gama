from datetime import datetime
import logging
import os
import time
from typing import Callable, Tuple, Optional, Sequence, Any, List
import traceback
import numpy as np

import stopit
#from sklearn.base import TransformerMixin, is_classifier
from river.base import Classifier
from river import evaluate
from gama.utilities.river_metrics import get_metric
#from sklearn.model_selection import ShuffleSplit, cross_validate, check_cv
from river.compose.pipeline import Pipeline #River pipeline instead
from river import compose
# we dont have cross valudate checkcv and is classifier equivalent in river, we do have metrics.workswith though
# shuffle split is also just suffle streamer and then there isa splitter
from river import stream
from river import metrics


from gama.utilities.evaluation_library import Evaluation
from gama.utilities.generic.stopwatch import Stopwatch
import numpy as np
from gama.utilities.metrics import Metric
from gama.genetic_programming.components import Individual, PrimitiveNode, Fitness

log = logging.getLogger(__name__)

# Use progressive_val_score instead of cross validate

def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {
        terminal.output: terminal.value for terminal in primitive_node._terminals
    }
    return primitive_node._primitive.identifier(**hyperparameters)


def compile_individual(
    individual: Individual,
    parameter_checks=None,
    preprocessing_steps: Sequence[Tuple[str, Classifier
    ]] = None,
) -> Pipeline:
    steps = [
        (str(i), primitive_node_to_sklearn(primitive))
        for i, primitive in enumerate(individual.primitives)
    ]

    pipeline = Pipeline()
    for step in list(reversed(steps)):
        pipeline |= step

    return pipeline


def object_is_valid_pipeline(o):
    """ Determines if object behaves like a scikit-learn pipeline. """
    return (
        o is not None
        and hasattr(o, "learn_one")
        and hasattr(o, "predict_one")
        and hasattr(o, "steps")
    )


def pipeline_similarity(pipeline, exist_pipelines, dataset, method='cos', phi=0.08):
    """ Calculate the similarity between pipeline and existing pipelines.

    Returns
    -------
    float
        Similarity score which in [-1, 1] * phi.
        The higher score means the lower similarity.
    """
    if exist_pipelines is None:
        return 0
    num_exist_pipelines = len(exist_pipelines)

    if method == 'Q':
        n = [[[0 for _ in range(num_exist_pipelines)] for _ in range(2)] for _ in range(2)]
        for a, b in dataset:
            y_p1 = pipeline.predict_one(a)
            for idx, exist_pipeline in enumerate(exist_pipelines):
                y_p2 = exist_pipeline.predict_one(a)
                i = 1 if y_p1 == b else 0
                j = 1 if y_p2 == b else 0
                n[i][j][idx] += 1
        similarity = 0.0
        for i in range(num_exist_pipelines):
            similarity += 1.0 * \
                (n[1][1][i] * n[0][0][i] - n[1][0][i] * n[0][1][i]) / \
                (n[1][1][i] * n[0][0][i] + n[1][0][i] * n[0][1][i])
        similarity = - phi * similarity / num_exist_pipelines
        return similarity

    elif method == 'cos':
        similarity = 0.0
        num_sim = 0
        for a, b in dataset:
            y_p1 = pipeline.predict_proba_one(a)
            for idx, exist_pipeline in enumerate(exist_pipelines):
                y_p2 = exist_pipeline.predict_proba_one(a)
                l1 = []
                l2 = []
                for k, v in y_p1.items():
                    l1.append(v)
                    l2.append(y_p2[k])
                v1 = np.array(l1)
                v2 = np.array(l2)
                numer = np.sum(v1 * v2)
                denom = np.sqrt(np.sum(v1**2) * np.sum(v2**2))
                cos_dis = numer / denom
                if cos_dis < 0.95:
                    similarity += cos_dis
                    num_sim += 1
        if num_sim is not 0:
            similarity = - phi * similarity / num_sim
        else:
            similarity = - phi
        return similarity


def evaluate_pipeline(
    pipeline, x, y_train, timeout: float,metrics: str = 'accuracy', cv=5, subsample=None,
    diversity: bool = False, diversity_phi: float = 0.05, exist_pipelines: Optional[List[Any]] = None,
) -> Tuple:
    """ Score `pipeline` with online holdout evaluation according to `metrics` on (a subsample of) X, y

    Returns
    -------
    Tuple:
        prediction: np.ndarray if successful, None if not
        scores: tuple with one float per metric, each value is -inf on fail.
        estimators: list of fitted pipelines if successful, None if not
        error: None if successful, otherwise an Exception
    """
    if not object_is_valid_pipeline(pipeline):
        raise TypeError(f"Pipeline must not be None and requires learn_one, predict_one, steps.")
    if not timeout > 0:
        raise ValueError(f"`timeout` must be greater than 0, is {timeout}.")
    prediction, estimators = None, None
    # default score for e.g. timeout or failure
    scores = tuple([float("-inf")])
    river_metric = get_metric(metrics)

    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            dataset = []
            for a, b in stream.iter_pandas(x, y_train):
                dataset.append((a, int(b)))

            result = evaluate.progressive_val_score(
                dataset=dataset,
                model=pipeline,
                metric=river_metric,
            )

            if diversity:
                similarity_score = pipeline_similarity(pipeline, exist_pipelines, dataset[-len(dataset)//10:], phi=diversity_phi)
            else:
                similarity_score = 0.0
            scores = tuple([result.get() + similarity_score, similarity_score])
            estimators = pipeline

            prediction = np.empty(shape=(len(y_train),))
            y_pred = []
            for a, b in stream.iter_pandas(x, y_train):
                y_pred.append(pipeline.predict_one(a))
                pipeline = pipeline.learn_one(a, b)

            prediction = np.asarray(y_pred)


        except stopit.TimeoutException:
            # This exception is handled by the ThreadingTimeout context manager.
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()
            return prediction, scores, estimators, e

    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat an eval timeout the same way as
        # e.g. NaN exceptions and use the default score.
        return prediction, scores, estimators, stopit.TimeoutException()

    return prediction, tuple(scores), estimators, None


def evaluate_individual(
    individual: Individual,
    evaluate_pipeline: Callable,
    timeout: float = 1e6,
    deadline: Optional[float] = None,
    add_length_to_score: bool = True,
    **kwargs,
) -> Evaluation:
    """ Evaluate the pipeline specified by individual, and record

    Parameters
    ----------
    individual: Individual
        Blueprint for the pipeline to evaluate.
    evaluate_pipeline: Callable
        Function which takes the pipeline and produces validation predictions,
        scores, estimators and errors.
    timeout: float (default=1e6)
        Maximum time in seconds that the evaluation is allowed to take.
        Don't depend on high accuracy.
        A shorter timeout is imposed if `deadline` is in less than `timeout` seconds.
    deadline: float, optional
        A time in seconds since epoch.
        Cut off evaluation at `deadline` even if `timeout` seconds have not yet elapsed.
    add_length_to_score: bool (default=True)
        Add the length of the individual to the score result of the evaluation.
    **kwargs: Dict, optional (default=None)
        Passed to `evaluate_pipeline` function.

    Returns
    -------
    Evaluation

    """
    result = Evaluation(individual, pid=os.getpid())
    result.start_time = datetime.now()
    if deadline is not None:
        time_to_deadline = deadline - time.time()
        timeout = min(timeout, time_to_deadline)
    with Stopwatch() as wall_time, Stopwatch(time.process_time) as process_time:
        evaluation = evaluate_pipeline(individual.pipeline, timeout=timeout, **kwargs)
        result._predictions, result.score, result._estimators, error = evaluation

        if error is not None:
            result.error = f"{type(error)} {str(error)}"
    result.duration = wall_time.elapsed_time

    if add_length_to_score:
        result.score = result.score + (-len(individual.primitives),)
    individual.fitness = Fitness(
        result.score,
        result.start_time,
        wall_time.elapsed_time,
        process_time.elapsed_time,
    )

    return result