import numpy as np
import pytest

from src.scoring import binary_auroc, mape, relative_error


def test_relative_error_hand_computed():
    assert relative_error(0.09, 0.08) == pytest.approx(0.125)
    assert relative_error(0.08, 0.08) == 0.0


def test_mape_hand_computed():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 160.0])
    assert mape(actual, predicted) == pytest.approx(0.15)


def test_mape_rejects_zero_actual():
    with pytest.raises(ValueError):
        mape(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_auroc_hand_computed():
    # Pairs: 0.35 beats 0.1 only; 0.8 beats both. 3 of 4 pairs are concordant.
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert binary_auroc(y_true, scores) == pytest.approx(0.75)
