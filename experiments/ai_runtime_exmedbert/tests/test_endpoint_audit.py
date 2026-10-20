import pytest

from experiments.ai_runtime_exmedbert.spark.profile_endpoints import (
    plos_label,
    readmission_label,
)


def test_plos_threshold_is_strictly_greater_than_seven_days():
    assert plos_label(7 * 24, 7) == 0
    assert plos_label(7 * 24 + 1, 7) == 1


def test_readmission_horizon_is_inclusive_and_requires_positive_gap():
    assert readmission_label(None, 30) == 0
    assert readmission_label(0, 30) == 0
    assert readmission_label(30, 30) == 1
    assert readmission_label(30.01, 30) == 0


def test_endpoint_contract_rejects_invalid_horizons():
    with pytest.raises(ValueError):
        plos_label(-1, 7)
    with pytest.raises(ValueError):
        readmission_label(1, 0)
