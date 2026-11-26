import numpy as np
import pandas as pd
import pytest

from xynapse.classification_metrics import (best_mcc_threshold, binary_metrics,
                                            calibration_curve_data, per_assay_auroc)


def test_best_mcc_threshold_handles_unbounded_oriented_scores():
    y = [0, 0, 1, 1]
    score = [-3.0, -1.0, 2.0, 5.0]
    threshold, mcc = best_mcc_threshold(y, score)
    assert mcc == pytest.approx(1.0)
    assert threshold > -1.0 and threshold <= 2.0
    result = binary_metrics(y, score, threshold)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["mcc"] == pytest.approx(1.0)
    assert np.isnan(result["brier"])


def test_binary_metrics_filter_nonfinite_and_report_prevalence():
    result = binary_metrics([0, 1, 1, np.nan], [0.1, 0.8, 0.9, 0.2], threshold=0.5)
    assert result["n_rows"] == 3
    assert result["prevalence"] == pytest.approx(2 / 3)
    assert result["n_positive"] == 2
    assert result["n_negative"] == 1
    assert result["brier"] == pytest.approx((0.1**2 + 0.2**2 + 0.1**2) / 3)


def test_binary_metrics_handles_single_class_without_auroc():
    result = binary_metrics([1, 1, 1], [0.2, 0.3, 0.4])
    assert np.isnan(result["auroc"])
    assert np.isnan(result["mcc"])
    assert result["auprc"] == pytest.approx(1.0)


def test_calibration_curve_counts_all_finite_rows():
    frame = calibration_curve_data([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=4)
    assert frame.n_rows.sum() == 4
    assert frame.fraction_of_positives.tolist() == [0.0, 1.0]


def test_per_assay_auroc_marks_missing_class_as_nan():
    frame = pd.DataFrame({
        "assay_id": ["a", "a", "a", "b", "b"],
        "label": [0, 1, 0, 1, 1],
        "score": [0.1, 0.8, 0.3, 0.2, 0.9],
    })
    result = per_assay_auroc(frame, "score").set_index("assay_id")
    assert result.loc["a", "auroc"] == pytest.approx(1.0)
    assert np.isnan(result.loc["b", "auroc"])


def test_fast_mcc_threshold_matches_brute_force():
    from sklearn.metrics import matthews_corrcoef

    rng = np.random.RandomState(42)
    for _ in range(10):
        y = rng.randint(0, 2, size=40)
        if len(np.unique(y)) < 2:
            continue
        score = np.round(rng.normal(size=40), 1)
        threshold, mcc = best_mcc_threshold(y, score)
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0, np.nextafter(score.max(), np.inf)], score)))
        expected = max(matthews_corrcoef(y, score >= value) for value in candidates)
        assert mcc == pytest.approx(expected)
        assert matthews_corrcoef(y, score >= threshold) == pytest.approx(mcc)
