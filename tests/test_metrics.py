import numpy as np
import pandas as pd
import pytest

from xynapse.metrics import pearson, per_protein_spearman, regression_metrics, rmse, spearman


def test_correlations_and_rmse_ignore_nonfinite_pairs():
    assert spearman([1, 2, 3, np.nan], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, np.inf], [1, 2, 3, 0]) == pytest.approx(1.0)
    assert rmse([1, 2, np.nan], [2, 2, 100]) == pytest.approx(np.sqrt(0.5))


def test_constant_and_empty_inputs_return_nan_correlations():
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert np.isnan(pearson([1, 2, 3], [0, 0, 0]))
    assert np.isnan(spearman([np.nan], [0]))
    assert np.isnan(rmse([np.nan], [0]))


def test_invalid_shapes_or_methods_raise():
    with pytest.raises(ValueError, match="equal shape"):
        spearman([1, 2], [1])
    from xynapse.metrics import correlation
    with pytest.raises(ValueError, match="method"):
        correlation([1, 2], [1, 2], "kendall")


def test_per_protein_spearman_qualifies_only_enough_finite_variants():
    frame = pd.DataFrame({
        "protein_id": ["A"] * 35 + ["B"] * 40,
        "y_true": list(range(35)) + list(range(40)),
        "y_pred": list(range(35)) + [1] * 40,
    })
    result, count = per_protein_spearman(frame, min_variants=30)
    assert count == 2
    assert result.set_index("protein_id").loc["A", "spearman"] == pytest.approx(1.0)
    assert np.isnan(result.set_index("protein_id").loc["B", "spearman"])


def test_regression_metrics_reports_finite_sample_count():
    metrics = regression_metrics([1, 2, np.nan], [1, 4, 2])
    assert metrics["n"] == 2
    assert metrics["rmse"] == pytest.approx(np.sqrt(2))
