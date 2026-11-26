from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_demo_bundle import sha256
from scripts.claims_audit import DISCLAIMER, _managed_readme, audit
from scripts.make_reports import regenerate_rank_counts


def test_readme_managed_block_requires_single_ordered_pair():
    assert _managed_readme(f"outside\n<!-- XYNAPSE-ML:BEGIN -->\n{DISCLAIMER}\n<!-- XYNAPSE-ML:END -->") == f"\n{DISCLAIMER}\n"
    with pytest.raises(ValueError, match="exactly one"):
        _managed_readme("missing boundaries")
    with pytest.raises(ValueError, match="out of order"):
        _managed_readme("<!-- XYNAPSE-ML:END --><!-- XYNAPSE-ML:BEGIN -->")


def test_claims_audit_accepts_compliant_synthetic_project(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "README.md").write_text(
        f"<!-- XYNAPSE-ML:BEGIN -->\n{DISCLAIMER}\n<!-- XYNAPSE-ML:END -->\n", encoding="utf-8"
    )
    (tmp_path / "docs/page.md").write_text(f"# Page\n\n{DISCLAIMER}\n", encoding="utf-8")
    (tmp_path / "app/app.py").write_text(f'DISCLAIMER = "{DISCLAIMER}"\n', encoding="utf-8")
    assert all(passed for _, passed, _ in audit(tmp_path))


def test_claims_audit_detects_restricted_word_and_missing_disclaimer(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "README.md").write_text(
        f"<!-- XYNAPSE-ML:BEGIN -->\n{DISCLAIMER}\n<!-- XYNAPSE-ML:END -->\n", encoding="utf-8"
    )
    (tmp_path / "docs/page.md").write_text("This is accurate.\n", encoding="utf-8")
    (tmp_path / "app/app.py").write_text("pass\n", encoding="utf-8")
    failed = [(name, detail) for name, passed, detail in audit(tmp_path) if not passed]
    assert any("restricted claims wording" in name and "accurate" in detail for name, detail in failed)
    assert any("docs/page.md contains the required disclaimer" == name for name, _ in failed)
    assert any("app/app.py contains the required disclaimer" == name for name, _ in failed)


def test_rank_count_regeneration_is_deterministic():
    predictions = pd.DataFrame({
        "protein_id": ["P1", "P1", "P2"],
        "rank_class": ["uncertain", "uncertain", "potentially damaging"],
    })
    first = regenerate_rank_counts(predictions)
    second = regenerate_rank_counts(predictions)
    pd.testing.assert_frame_equal(first, second)
    assert first.to_dict("records") == [
        {"protein_id": "P1", "rank_class": "uncertain", "n_mutations": 2},
        {"protein_id": "P2", "rank_class": "potentially damaging", "n_mutations": 1},
    ]


def test_demo_sha256_helper(tmp_path: Path):
    item = tmp_path / "payload"
    item.write_bytes(b"demo bytes")
    assert sha256(item) == hashlib.sha256(b"demo bytes").hexdigest()


def test_tracked_demo_bundle_is_integrity_checked_when_available():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "demo_data/manifest.json"
    if not manifest_path.is_file():
        pytest.skip("tracked demo bundle is absent in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["disclaimer"] == DISCLAIMER
    assert set(manifest["proteins"]) == {"P60484", "P04637"}
    for relative_path, details in manifest["files"].items():
        path = root / relative_path
        assert path.is_file()
        assert path.stat().st_size == details["bytes"]
        assert sha256(path) == details["sha256"]
