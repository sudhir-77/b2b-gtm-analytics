"""Download SmolLM2 weights with requests and verify the checksum."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
WEIGHTS_NAME = "model.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_oid(oid: str) -> str:
    if oid.startswith("sha256:"):
        return oid.split(":", 1)[1]
    return oid


def fetch_file_index(session: requests.Session, repo: str) -> dict[str, dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    index: dict[str, dict] = {}
    for item in response.json():
        path = item.get("path")
        if not path:
            continue
        lfs = item.get("lfs") or {}
        oid = _normalize_oid(str(lfs.get("oid") or ""))
        size = int(lfs.get("size") or item.get("size") or 0)
        index[path] = {"sha256": oid, "size": size, "lfs": bool(lfs)}
    return index


def _stream_to(session: requests.Session, url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    digest = hashlib.sha256()
    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
    temporary.replace(dest)
    return digest.hexdigest()


def _matches(path: Path, meta: dict, pinned_sha256: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.name == WEIGHTS_NAME:
        return sha256_file(path) == pinned_sha256
    if meta.get("lfs") and meta.get("sha256"):
        return sha256_file(path) == meta["sha256"]
    if meta.get("size"):
        return path.stat().st_size == meta["size"]
    return True


def download_model(
    dest: Path,
    repo: str,
    pinned_sha256: str,
    session: requests.Session | None = None,
) -> dict[str, str]:
    """Cache model files under ``dest`` and verify checksums or sizes.

    ``model.safetensors`` must match ``pinned_sha256``. Other LFS files must
    match the Hub sha256. Small git files must match the published size.
    """
    dest.mkdir(parents=True, exist_ok=True)
    owns_session = session is None
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "b2b-gtm-analytics/1.0")
    try:
        index = fetch_file_index(session, repo)
        verified: dict[str, str] = {}
        for name in MODEL_FILES:
            if name not in index:
                raise RuntimeError(f"{repo} is missing {name}")
            meta = index[name]
            target = dest / name
            if name == WEIGHTS_NAME and meta.get("sha256") and meta["sha256"] != pinned_sha256:
                raise RuntimeError("pinned model sha256 does not match the Hub LFS oid")
            if not _matches(target, meta, pinned_sha256):
                url = f"https://huggingface.co/{repo}/resolve/main/{name}?download=true"
                digest = _stream_to(session, url, target)
                if name == WEIGHTS_NAME and digest != pinned_sha256:
                    raise RuntimeError(
                        f"checksum mismatch for {name}: {digest} != {pinned_sha256}"
                    )
                if meta.get("lfs") and meta.get("sha256") and digest != meta["sha256"]:
                    raise RuntimeError(f"checksum mismatch for {name}")
                if not meta.get("lfs") and meta.get("size") and target.stat().st_size != meta["size"]:
                    raise RuntimeError(f"size mismatch for {name}")
            verified[name] = sha256_file(target) if name == WEIGHTS_NAME else "verified"
        return verified
    finally:
        if owns_session:
            session.close()
