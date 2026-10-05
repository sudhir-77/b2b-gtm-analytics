import hashlib

import requests

from src.download import MODEL_FILES, WEIGHTS_NAME, download_model, sha256_file


class _Response:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self._content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Session(requests.Session):
    def __init__(self, files: dict[str, bytes]):
        super().__init__()
        self.files = files

    def get(self, url, timeout=None, stream=False, headers=None):
        if url.endswith("/tree/main"):
            items = []
            for name, data in self.files.items():
                digest = hashlib.sha256(data).hexdigest()
                items.append(
                    {
                        "path": name,
                        "size": len(data),
                        "lfs": {"oid": f"sha256:{digest}", "size": len(data)},
                    }
                )
            return _Response(items)
        for name, data in self.files.items():
            if f"/{name}?" in url or url.endswith(f"/{name}"):
                return _Response(content=data)
        raise AssertionError(url)

    def close(self):
        return None


def test_sha256_file(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_download_verifies_checksum_and_reuses_cache(tmp_path):
    files = {name: f"payload-{name}".encode() for name in MODEL_FILES}
    pinned = hashlib.sha256(files[WEIGHTS_NAME]).hexdigest()
    session = _Session(files)
    first = download_model(tmp_path / "model", "HuggingFaceTB/SmolLM2-360M-Instruct", pinned, session)
    assert first[WEIGHTS_NAME] == pinned
    second = download_model(tmp_path / "model", "HuggingFaceTB/SmolLM2-360M-Instruct", pinned, session)
    assert second[WEIGHTS_NAME] == pinned


def test_download_rejects_a_bad_pin(tmp_path):
    files = {name: f"payload-{name}".encode() for name in MODEL_FILES}
    session = _Session(files)
    try:
        download_model(tmp_path / "model", "org/model", "0" * 64, session)
    except RuntimeError as exc:
        assert "sha256" in str(exc)
        return
    raise AssertionError("bad checksum was accepted")
