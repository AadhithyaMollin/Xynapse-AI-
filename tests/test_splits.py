import pandas as pd
import pytest

from xynapse.splits import assign_splits, make_training_table, parse_cluster_tsv


def synthetic_proteins(n=20):
    return pd.DataFrame({"protein_id": [f"P{i:05d}" for i in range(n)]})


def test_parse_cluster_tsv_validates_and_assigns_each_protein(tmp_path):
    path = tmp_path / "clusters.tsv"
    path.write_text("P00001\tP00001\nP00001\tP00002\nP00003\tP00003\n")
    result = parse_cluster_tsv(path, {"P00001", "P00002", "P00003"})
    assert result == {"P00001": "P00001", "P00002": "P00001", "P00003": "P00003"}


@pytest.mark.parametrize("contents", ["rep\tP00001\n", "rep\tP00001\nrep\tP00001\n", "rep\tUNKNOWN\n"])
def test_parse_cluster_tsv_rejects_bad_or_incomplete_rows(tmp_path, contents):
    path = tmp_path / "clusters.tsv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        parse_cluster_tsv(path, {"P00001", "P00002"})


def test_seeded_split_is_cluster_disjoint_and_reproducible():
    proteins = synthetic_proteins()
    cluster_map = {pid: f"C{int(pid[1:]) // 2:04d}" for pid in proteins.protein_id}
    one = assign_splits(proteins, cluster_map, case_study_ids={"P00000"},
                        external_test_ids={"P00001"}, seed=42)
    two = assign_splits(proteins, cluster_map, case_study_ids={"P00000"},
                        external_test_ids={"P00001"}, seed=42)
    pd.testing.assert_frame_equal(one, two)
    train_clusters = set(one.loc[one.split_id.eq("train"), "cluster_id"])
    test_clusters = set(one.loc[one.split_id.eq("test"), "cluster_id"])
    assert train_clusters.isdisjoint(test_clusters)
    assert one.loc[one.protein_id.isin(["P00000", "P00001"]), "split_id"].eq("test").all()
    assert one.loc[one.split_id.eq("test"), "cv_fold"].isna().all()
    assert one.loc[one.split_id.eq("train"), "cv_fold"].notna().all()
    assert one.groupby("cluster_id").cv_fold.nunique(dropna=True).max() <= 1


def test_training_table_joins_labels_and_preserves_fold_metadata():
    proteins = synthetic_proteins(20)
    cluster_map = {pid: pid for pid in proteins.protein_id}
    assignments = assign_splits(proteins, cluster_map, case_study_ids={"P00000"},
                                test_fraction=0.15, seed=42)
    features = pd.DataFrame({
        "protein_id": ["P00002", "P00003"], "position": [1, 1], "wt": ["A", "A"],
        "mut": ["C", "D"], "esm_llr": [-1.0, -2.0],
    })
    labels = pd.DataFrame({
        "protein_id": ["P00002", "P00003"], "position": [1, 1], "wt": ["A", "A"],
        "mut": ["C", "D"], "ddg": [0.5, 1.2], "source": ["x", "x"],
        "split_id": [None, None], "cluster_id": [None, None],
    })
    result = make_training_table(features, labels, assignments)
    assert list(result.protein_id) == ["P00002", "P00003"]
    assert result.ddg.tolist() == [0.5, 1.2]
    assert set(result.columns) >= {"cluster_id", "split_id", "cv_fold"}


def test_assignment_requires_exact_cluster_map_coverage():
    proteins = synthetic_proteins(10)
    cluster_map = {pid: pid for pid in proteins.protein_id[:-1]}
    with pytest.raises(ValueError, match="cover exactly"):
        assign_splits(proteins, cluster_map, case_study_ids=set())
