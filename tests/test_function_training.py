import numpy as np
import pandas as pd

from scripts.train_function import (ABLATIONS, CATEGORICAL, make_logistic,
                                    choose_calibrator)


def test_logistic_preprocessing_is_inside_pipeline_and_handles_categories():
    rows = 40
    frame = pd.DataFrame({
        "esm_llr": np.linspace(-2, 2, rows),
        "esm_wt_logp": np.linspace(-1, 0, rows),
        "esm_mut_logp": np.linspace(-2, 1, rows),
        "blosum62": np.tile([-1.0, 1.0], rows // 2),
        "d_hydropathy": np.linspace(-3, 3, rows),
        "d_volume": np.linspace(-2, 2, rows),
        "d_charge": np.tile([-1, 0, 1, 0], rows // 4),
        "esm_pos_mean_llr": np.linspace(-1, 1, rows),
        "esm_pos_min_llr": np.linspace(-2, 0, rows),
        "rel_position": np.linspace(0, 1, rows),
        "wt": np.tile(["A", "C"], rows // 2),
        "mut": np.tile(["D", "E"], rows // 2),
    })
    labels = np.tile([0, 1], rows // 2)
    categories = {"wt": ["A", "C"], "mut": ["D", "E"]}
    model = make_logistic(categories, [column for column in ABLATIONS["tier1_full"] if column not in CATEGORICAL], 42)
    model.fit(frame[ABLATIONS["tier1_full"]], labels)
    probabilities = model.predict_proba(frame[ABLATIONS["tier1_full"]])[:, 1]
    assert len(probabilities) == rows
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_calibration_selection_uses_cross_fitted_group_folds():
    labels = np.tile([0, 1], 40)
    scores = np.clip(0.1 + 0.8 * labels + np.linspace(-0.15, 0.15, len(labels)), 0, 1)
    frame = pd.DataFrame({"label": labels, "cv_fold": np.repeat(np.arange(4), 20)})
    method, final_calibrator, crossfit, evidence = choose_calibrator(frame, scores)
    assert method in {"isotonic", "platt"}
    assert len(crossfit) == len(labels)
    assert np.isfinite(crossfit).all()
    assert set(evidence.method) == {"isotonic", "platt"}
    assert evidence.cv_fold.nunique() == 4
    assert np.isfinite(evidence.brier).all()
    assert final_calibrator is not None
