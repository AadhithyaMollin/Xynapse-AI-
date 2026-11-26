import numpy as np
import pandas as pd
import pytest

from xynapse.labels import function_project_label, stability_project_ddg, within_assay_percentile


def test_project_ddg_sign():
    assert stability_project_ddg(3.0, 1.0) == 2.0
    assert stability_project_ddg(1.0, 3.0) == -2.0


def test_function_label_direction_both_ways():
    assert function_project_label(1, label_1_is_tolerated=True) == 0
    assert function_project_label(0, label_1_is_tolerated=True) == 1
    assert function_project_label(1, label_1_is_tolerated=False) == 1
    assert function_project_label(0, label_1_is_tolerated=False) == 0


def test_label_conversion_rejects_invalid_values():
    with pytest.raises(ValueError):
        function_project_label(2, label_1_is_tolerated=True)
    with pytest.raises(ValueError):
        stability_project_ddg(np.nan, 1.0)


def test_percentiles_are_within_assay_and_average_ties():
    actual = within_assay_percentile(pd.Series([1.0, 1.0, 3.0]))
    assert actual.tolist() == [0.5, 0.5, 1.0]
