from scripts.train_stability import CATEGORICAL, prepare_x, random_configs

import pandas as pd


def test_random_search_has_20_deterministic_unique_configurations():
    one = random_configs(42)
    two = random_configs(42)
    assert one == two
    assert len(one) == 20
    assert len({tuple(sorted(row.items())) for row in one}) == 20


def test_categorical_preparation_maps_unseen_values_to_missing():
    frame = pd.DataFrame({"wt": ["A", "Z"], "mut": ["C", "D"], "esm_llr": [0.1, 0.2]})
    categories = {"wt": ["A", "C"], "mut": ["C", "D"]}
    result = prepare_x(frame, ["wt", "mut", "esm_llr"], categories)
    assert result.wt.cat.categories.tolist() == ["A", "C"]
    assert pd.isna(result.loc[1, "wt"])
    assert result.mut.cat.categories.tolist() == ["C", "D"]
    assert CATEGORICAL == ["wt", "mut"]
