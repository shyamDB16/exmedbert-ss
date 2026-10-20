import math

from experiments.ai_runtime_exmedbert.training.endpoint_metrics import (
    binary_metrics,
    select_youden_threshold,
)


def test_binary_metrics_match_known_ranking_example():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5)
    assert math.isclose(metrics["auroc"], 0.75)
    assert math.isclose(metrics["average_precision"], 5 / 6)
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0


def test_validation_threshold_uses_youden_j():
    threshold = select_youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert threshold == 0.8
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=threshold)
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_binary_metrics_reject_single_class_input():
    try:
        binary_metrics([1, 1], [0.7, 0.8])
    except ValueError as error:
        assert "both classes" in str(error)
    else:
        raise AssertionError("single-class metrics must be rejected")
