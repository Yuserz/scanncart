"""`scripts/fetch_models.py`: the weights a fresh clone downloads from the repo's `models` release.

Run against an injected opener, never the network: a fake GitHub that answers the release route and
the asset downloads, so what is pinned is the tool's own rules - which assets land, that a weight
already on disk is never replaced, that a short download never becomes a weight, and that a missing
release is a warning rather than a failed setup.
"""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fetch_models", REPO_ROOT / "scripts" / "fetch_models.py")
fetch_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_models)

REPO = "someone/scanncart"
RELEASE_URL = f"https://api.github.com/repos/{REPO}/releases/tags/models"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _github(files: dict[str, bytes], *, sizes: dict[str, int] | None = None, release: bool = True):
    """A fake GitHub: the release lists `files` (sizes overridable to fake a short download)."""
    assets = [
        {"name": name, "size": (sizes or {}).get(name, len(body)), "browser_download_url": f"https://dl/{name}"}
        for name, body in files.items()
    ]
    asked: list[str] = []

    def opener(request):
        asked.append(request.full_url)
        if request.full_url == RELEASE_URL:
            if not release:
                raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
            return _Response(json.dumps({"assets": assets}).encode())
        name = request.full_url.rsplit("/", 1)[1]
        return _Response(files[name])

    opener.asked = asked
    return opener


def test_every_weight_and_record_lands_and_nothing_else(tmp_path):
    gh = _github({
        "scanncart-grocery-v2-stretch.pt": b"weights",
        "scanncart-grocery-v2-stretch.json": b'{"resize_mode": "stretch"}',
        "notes.txt": b"not a model",
    })
    code = fetch_models.main(["--repo", REPO, "--dest", str(tmp_path)], opener=gh)

    assert code == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "scanncart-grocery-v2-stretch.json",
        "scanncart-grocery-v2-stretch.pt",
    ]
    assert (tmp_path / "scanncart-grocery-v2-stretch.pt").read_bytes() == b"weights"
    assert "https://dl/notes.txt" not in gh.asked


def test_a_weight_already_on_disk_is_kept_and_not_downloaded(tmp_path, capsys):
    (tmp_path / "scanncart-grocery-v2-stretch.pt").write_bytes(b"this machine's own run")
    gh = _github({"scanncart-grocery-v2-stretch.pt": b"the release's"})

    assert fetch_models.main(["--repo", REPO, "--dest", str(tmp_path)], opener=gh) == 0
    assert (tmp_path / "scanncart-grocery-v2-stretch.pt").read_bytes() == b"this machine's own run"
    assert gh.asked == [RELEASE_URL]
    assert "not replaced" in capsys.readouterr().out


def test_a_short_download_never_becomes_a_weight(tmp_path, capsys):
    gh = _github({"v2.pt": b"half"}, sizes={"v2.pt": 1000})

    assert fetch_models.main(["--repo", REPO, "--dest", str(tmp_path)], opener=gh) == 0
    assert list(tmp_path.iterdir()) == [], "neither the weight nor its temporary file is left"
    assert "could not fetch" in capsys.readouterr().out


def test_no_release_is_a_warning_and_strict_makes_it_a_failure(tmp_path, capsys):
    gh = _github({}, release=False)

    assert fetch_models.main(["--repo", REPO, "--dest", str(tmp_path)], opener=gh) == 0
    assert "no release tagged 'models'" in capsys.readouterr().out
    assert fetch_models.main(["--repo", REPO, "--dest", str(tmp_path), "--strict"], opener=gh) == 1


def test_check_lists_what_is_missing_and_downloads_nothing(tmp_path, capsys):
    gh = _github({"v2.pt": b"weights"})

    assert fetch_models.main(["--repo", REPO, "--dest", str(tmp_path), "--check"], opener=gh) == 0
    assert list(tmp_path.iterdir()) == []
    assert "missing  v2.pt" in capsys.readouterr().out


@pytest.mark.parametrize(
    "remote, repo",
    [
        ("https://github.com/Yuserz/scanncart.git", "Yuserz/scanncart"),
        ("git@github.com:someone/scanncart.git\n", "someone/scanncart"),
        ("https://github.com/someone/scanncart", "someone/scanncart"),
        ("http://127.0.0.1:9/git/Yuserz/scanncart", None),
    ],
)
def test_the_repo_is_read_off_a_github_remote(remote, repo):
    assert fetch_models.repo_from_remote(remote) == repo
