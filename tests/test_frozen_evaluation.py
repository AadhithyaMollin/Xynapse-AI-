import numpy as np
import pandas as pd

from scripts.evaluate_frozen_tests import function_predictions, stability_prediction


class FakeClassifier:
    def predict_proba(self, frame):
        p = 1.0 / (1.0 + np.exp(-frame.esm_llr.to_numpy(float)))
        return np.column_stack([1 - p, p])


class FakeCalibrator:
    def predict(self, probability):
        return probability


class FakeRegressor:
    def predict(self, frame):
        return frame.esm_llr.to_numpy(float)


class FakeLogistic:
    def predict_proba(self, frame):
        p = np.full(len(frame), 0.4)
        return np.column_stack([1 - p, p])


def sample_features():
    return pd.DataFrame({
        "protein_id": ["P1", "P1"], "position": [1, 2], "wt": ["A", "C"], "mut": ["D", "E"],
        "esm_llr": [-1.0, 1.0], "esm_wt_logp": [-0.1, -0.2], "esm_mut_logp": [-1.0, -2.0],
        "blosum62": [-1.0, 1.0], "d_hydropathy": [0.2, -0.1], "d_volume": [1.0, -1.0],
        "d_charge": [0, 1], "esm_pos_mean_llr": [-0.3, 0.2], "esm_pos_min_llr": [-1.2, -0.5],
        "rel_position": [0.2, 0.3],
    })


def test_function_predictions_produce_finite_calibrated_and_logistic_outputs():
    frame = sample_features()
    feature_list = ["esm_llr", "esm_wt_logp", "esm_mut_logp", "blosum62", "d_hydropathy",
                    "d_volume", "d_charge", "esm_pos_mean_llr", "esm_pos_min_llr", "rel_position", "wt", "mut"]
    context = {
        "function_meta": {"feature_list": feature_list,
                          "category_lists": {"wt": ["A", "C"], "mut": ["D", "E"]}},
        "function_bundle": {"estimator": FakeClassifier(), "calibrator": FakeCalibrator(),
                            "calibration_method": "isotonic"},
        "function_logistic": FakeLogistic(),
    }
    calibrated, logistic = function_predictions(context, frame)
    assert np.allclose(calibrated, [1 / (1 + np.exp(1)), 1 / (1 + np.exp(-1))])
    assert np.allclose(logistic, [0.4, 0.4])


def test_stability_prediction_uses_saved_feature_order_and_rejects_nonfinite():
    frame = sample_features()
    feature_list = ["esm_llr", "esm_wt_logp", "esm_mut_logp", "blosum62", "d_hydropathy",
                    "d_volume", "d_charge", "esm_pos_mean_llr", "esm_pos_min_llr", "rel_position", "wt", "mut"]
    context = {"stability_meta": {"feature_list": feature_list,
                                   "category_lists": {"wt": ["A", "C"], "mut": ["D", "E"]}},
               "stability_model": FakeRegressor()}
    assert np.array_equal(stability_prediction(context, frame), [-1.0, 1.0])
