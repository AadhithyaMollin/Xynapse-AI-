from __future__ import annotations

import http.server
import io
import json
import socketserver
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from scripts import pre_push_check, sync_to_shared, zenodo_fetch


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _tree(path: Path) -> None:
    (path / "scripts").mkdir(parents=True)
    (path / "reports").mkdir()
    (path / "README.md").write_text(
        "# Other content\n\n<!-- XYNAPSE-ML:BEGIN -->\nPredictions are computational hypotheses, not experimental findings.\n<!-- XYNAPSE-ML:END -->\n",
        encoding="utf-8",
    )
    (path / "scripts" / "tool.py").write_text("print('ok')\n", encoding="utf-8")
    (path / "data").mkdir()
    (path / "data" / "secret.parquet").write_bytes(b"not for sync")


def test_sync_copies_allowed_files_preserves_readme_and_never_deletes(tmp_path: Path):
    source, shared = tmp_path / "source", tmp_path / "shared"
    source.mkdir()
    shared.mkdir()
    _tree(source)
    (shared / "README.md").write_text("# Teammate content\n", encoding="utf-8")
    preserved = shared / "teammate.txt"
    preserved.write_text("keep", encoding="utf-8")

    planned = sync_to_shared.sync(shared, dry_run=True, root=source)
    assert "scripts/tool.py" in planned
    assert not (source / "reports" / ".sync_manifest.json").exists()
    sync_to_shared.sync(shared, root=source)

    readme = (shared / "README.md").read_text(encoding="utf-8")
    assert "# Teammate content" in readme
    assert "old" not in readme
    assert "Predictions are computational hypotheses" in readme
    assert preserved.read_text(encoding="utf-8") == "keep"
    assert not (shared / "data" / "secret.parquet").exists()
    manifest = json.loads((source / "reports" / ".sync_manifest.json").read_text())
    assert "scripts/tool.py" in manifest["files"]


def test_sync_refuses_conflicting_clone_change(tmp_path: Path):
    source, shared = tmp_path / "source", tmp_path / "shared"
    source.mkdir()
    shared.mkdir()
    _tree(source)
    sync_to_shared.sync(shared, root=source)
    (shared / "scripts" / "tool.py").write_text("someone else's change\n", encoding="utf-8")
    (source / "scripts" / "tool.py").write_text("new source change\n", encoding="utf-8")
    with pytest.raises(sync_to_shared.SyncConflict, match="scripts/tool.py"):
        sync_to_shared.sync(shared, root=source)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


def test_pre_push_check_accepts_allowlisted_staged_file(tmp_path: Path, capsys):
    repo = _init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "src/module.py")
    _, failures = pre_push_check.check(repo)
    assert failures == []
    assert "PASS: allowlist" in capsys.readouterr().out


def test_pre_push_check_rejects_data_and_secret(tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / "data" / "raw").mkdir(parents=True)
    (repo / "data" / "raw" / "unsafe.csv").write_text("not allowed\n", encoding="utf-8")
    (repo / "docs").mkdir()
    secret_assignment = "api" + "_key" + " = " + "abc123"
    (repo / "docs" / "unsafe.md").write_text(secret_assignment, encoding="utf-8")
    _git(repo, "add", "data/raw/unsafe.csv", "docs/unsafe.md")
    _, failures = pre_push_check.check(repo)
    assert "allowlist and forbidden locations/extensions" in failures
    assert "secret scan" in failures


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self):
        header = self.headers.get("Range")
        if not header:
            self.send_response(200)
            body = self.payload
        else:
            start_end = header.removeprefix("bytes=")
            start_raw, end_raw = start_end.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            body = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture

def range_zip_server():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("folder/member.txt", b"original member bytes\n")
    _RangeHandler.payload = payload.getvalue()
    server = socketserver.TCPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/archive.zip"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_generic_zip_range_fetch_extracts_member(tmp_path: Path, range_zip_server: str):
    dest = tmp_path / "member.txt"
    manifest = zenodo_fetch.fetch_url_member(range_zip_server, "folder/member.txt", dest)
    assert dest.read_bytes() == b"original member bytes\n"
    assert manifest["member_size"] == len(dest.read_bytes())


def test_generic_zip_range_fetch_rejects_member_size_mismatch(tmp_path: Path, range_zip_server: str, monkeypatch):
    original = zipfile.ZipFile.getinfo

    def mismatched(self, name):
        info = original(self, name)
        info.file_size += 1
        return info

    monkeypatch.setattr(zipfile.ZipFile, "getinfo", mismatched)
    dest = tmp_path / "bad.txt"
    with pytest.raises(ValueError, match="wrote .* expected"):
        zenodo_fetch.fetch_url_member(range_zip_server, "folder/member.txt", dest)
    assert not dest.exists()
