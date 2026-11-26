#!/usr/bin/env python
"""Fetch one member of a Zenodo ZIP archive without downloading the whole archive.

The Mega-scale stability record (7992926) ships a 1.01 GB ZIP, above the 500 MB
limit in the brief. Its members are each below 500 MB, so this script reads the
ZIP central directory over HTTP Range requests and range-downloads only the
requested member, then decompresses it locally.

Usage:
    python scripts/zenodo_fetch.py --record 7992926 \
        --archive Processed_K50_dG_datasets.zip \
        --member Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv \
        --dest data/raw/tsuboyama/Tsuboyama2023_Dataset1_20230416.csv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HOSTS = {
    "zenodo.org", "www.biorxiv.org", "marks.hms.harvard.edu", "proteingym.org",
    "ftp.ncbi.nlm.nih.gov", "rest.uniprot.org", "alphafold.ebi.ac.uk",
    "huggingface.co", "github.com",
}
INSECURE_HOST = "marks.hms.harvard.edu"
LOCAL_TEST_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_url_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials are forbidden in download URLs")
    if parsed.hostname not in ALLOWED_HOSTS | LOCAL_TEST_HOSTS:
        raise ValueError(f"Host is not on the approved download allowlist: {parsed.hostname}")
    if parsed.scheme != "https" and parsed.hostname not in LOCAL_TEST_HOSTS:
        raise ValueError("Downloads require HTTPS outside localhost tests")


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests."""

    def __init__(self, url: str, timeout: int = 120, verify: bool = True):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        self.verify = verify
        self.session = requests.Session()
        validate_url_host(url)
        head = self.session.head(url, timeout=timeout, allow_redirects=False, verify=verify)
        if not 200 <= head.status_code < 300:
            raise IOError(f"HEAD request failed or redirected with status {head.status_code}")
        if "Content-Length" not in head.headers:
            raise ValueError("Server did not provide Content-Length")
        self.size = int(head.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            timeout=self.timeout,
            verify=self.verify,
            allow_redirects=False,
        )
        response.raise_for_status()
        expected = end - self.pos + 1
        if response.status_code != 206 or len(response.content) != expected:
            raise IOError(f"Server did not honor byte range {self.pos}-{end}")
        data = response.content
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_member(record: int, archive: str, member: str, dest: Path) -> dict:
    api = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    api.raise_for_status()
    record_json = api.json()
    file_info = next((f for f in record_json["files"] if f["key"] == archive), None)
    if file_info is None:
        raise KeyError(f"{archive} not found in Zenodo record {record}")
    url = file_info["links"]["self"]
    validate_url_host(url)

    import zipfile

    remote = HTTPRangeFile(url)
    with zipfile.ZipFile(remote) as zf:
        info = zf.getinfo(member)
        if info.compress_size > 500_000_000:
            raise ValueError(f"Compressed member is {info.compress_size} bytes, over the 500 MB cap")

    # Parse the local file header to find where the compressed data starts.
    header_reader = HTTPRangeFile(url)
    header_reader.seek(info.header_offset)
    header = header_reader.read(30)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    name_len, extra_len = fields[-2], fields[-1]
    data_start = info.header_offset + 30 + name_len + extra_len

    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decompressor = zlib.decompressobj(-15)
    written = 0
    with requests.get(
        url,
        headers={"Range": f"bytes={data_start}-{data_start + info.compress_size - 1}"},
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in response.iter_content(1 << 20):
                data = decompressor.decompress(chunk)
                out.write(data)
                written += len(data)
            out.write(decompressor.flush())
    elapsed = time.perf_counter() - started

    if written != info.file_size:
        raise ValueError(f"{member}: wrote {written} bytes, expected {info.file_size}")

    manifest = {
        "record": record,
        "record_title": record_json.get("metadata", {}).get("title"),
        "archive": archive,
        "archive_size": file_info.get("size"),
        "member": member,
        "member_size": info.file_size,
        "member_compressed_size": info.compress_size,
        "dest": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
        "sha256": sha256(dest),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "download_seconds": round(elapsed, 2),
        "source_url": url,
    }
    manifest_path = dest.parent / f"{dest.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"member:   {member}")
    print(f"size:     {info.file_size} bytes (compressed {info.compress_size})")
    print(f"dest:     {dest}")
    print(f"sha256:   {manifest['sha256']}")
    print(f"download: {elapsed:.2f} s measured")
    print(f"manifest: {manifest_path}")
    return manifest


def fetch_url_member(url: str, member: str, dest: Path, verify: bool = True) -> dict:
    import zipfile

    validate_url_host(url)
    remote = HTTPRangeFile(url, verify=verify)
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(member)
        if info.compress_size > 500_000_000:
            raise ValueError(f"Compressed member is {info.compress_size} bytes, over the 500 MB cap")
        header_reader = HTTPRangeFile(url, verify=verify)
        header_reader.seek(info.header_offset)
        header = header_reader.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        name_len, extra_len = fields[-2], fields[-1]
        data_start = info.header_offset + 30 + name_len + extra_len
        dest.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        decompressor = zlib.decompressobj(-15)
        written = 0
        with requests.get(
            url,
            headers={"Range": f"bytes={data_start}-{data_start + info.compress_size - 1}"},
            stream=True,
            timeout=300,
            verify=verify,
            allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise IOError("Server did not honor the requested ZIP-member byte range")
            with dest.open("wb") as out:
                for chunk in response.iter_content(1 << 20):
                    data = decompressor.decompress(chunk)
                    out.write(data)
                    written += len(data)
                out.write(decompressor.flush())
        elapsed = time.perf_counter() - started
    if written != info.file_size:
        dest.unlink(missing_ok=True)
        raise ValueError(f"{member}: wrote {written} bytes, expected {info.file_size}")
    manifest = {
        "source_url": url,
        "member": member,
        "archive_size": remote.size,
        "member_size": info.file_size,
        "member_compressed_size": info.compress_size,
        "dest": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
        "sha256": sha256(dest),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "download_seconds": round(elapsed, 2),
    }
    manifest_path = dest.parent / f"{dest.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"member: {member}")
    print(f"size: {info.file_size} bytes (compressed {info.compress_size})")
    print(f"sha256: {manifest['sha256']}")
    print(f"download: {elapsed:.2f} s measured")
    return manifest


def fetch_url_file(url: str, dest: Path, verify: bool = True) -> dict:
    validate_url_host(url)
    head = requests.head(url, timeout=120, allow_redirects=False, verify=verify)
    if not 200 <= head.status_code < 300:
        raise IOError(f"HEAD request failed or redirected with status {head.status_code}")
    if "Content-Length" not in head.headers:
        raise ValueError("Server did not provide Content-Length")
    size = int(head.headers["Content-Length"])
    if size > 500_000_000:
        raise ValueError(f"Download is {size} bytes, over the 500 MB cap")
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0
    with requests.get(url, stream=True, timeout=300, verify=verify, allow_redirects=False) as response:
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise IOError(f"GET request failed or redirected with status {response.status_code}")
        with dest.open("wb") as out:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    out.write(chunk)
                    written += len(chunk)
    if written != size:
        dest.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {written} bytes, expected Content-Length {size}")
    elapsed = time.perf_counter() - started
    manifest = {
        "source_url": url,
        "size": written,
        "sha256": sha256(dest),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "download_seconds": round(elapsed, 2),
        "dest": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
    }
    manifest_path = dest.parent / f"{dest.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"size: {written} bytes")
    print(f"sha256: {manifest['sha256']}")
    print(f"download: {elapsed:.2f} s measured")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=int)
    mode.add_argument("--url")
    parser.add_argument("--archive")
    parser.add_argument("--member")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--insecure-host", help="Disable TLS verification only for this exact URL host")
    args = parser.parse_args()
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = ROOT / dest
    if args.record is not None:
        if not args.archive or not args.member:
            parser.error("--record requires --archive and --member")
        if args.insecure_host:
            parser.error("--insecure-host is available only with --url")
        fetch_member(args.record, args.archive, args.member, dest)
        return
    parsed = urlparse(args.url)
    verify = True
    if args.insecure_host:
        if args.insecure_host != INSECURE_HOST or parsed.hostname != args.insecure_host:
            parser.error(f"--insecure-host is permitted only for the exact host {INSECURE_HOST}")
        print(f"WARNING: TLS certificate verification disabled for GET/HEAD to {parsed.hostname}", file=sys.stderr)
        verify = False
    if args.member:
        fetch_url_member(args.url, args.member, dest, verify=verify)
    else:
        fetch_url_file(args.url, dest, verify=verify)


if __name__ == "__main__":
    main()
