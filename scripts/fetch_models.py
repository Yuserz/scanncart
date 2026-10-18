"""Download the model weights from this repo's GitHub Release into `sidecar/models/`.

Weights are not in git (`*.pt` is ignored: they are tens of megabytes each and change with every
retrain), so a fresh clone has the code and no detector. They travel as the assets of one GitHub
Release instead - tag `models` by default - and this fetches every weight file (`.pt`, `.onnx`) and
record (`.json`) attached to it that is not already on disk. The record matters as much as the
weight: it is what tells `resize_mode: auto` the geometry the weight was trained at.

Run by both setup scripts and by `make models`, through the sidecar's venv interpreter, which is why
it uses the standard library only. A run is safe to repeat:

* a file already in `sidecar/models/` is **kept, never replaced** - a same-named file may be a
  different training run, and replacing a weight is a decision rather than a side effect of setup;
* a download lands under a temporary name and is moved into place only once its size matches the
  release's, so an interrupted fetch never leaves a truncated weight that the picker would list;
* no release, no network, or a rate limit is a **warning and exit 0**, because the app installs and
  runs without weights (a stock model downloads on first use) - `--strict` makes it a failure, for
  a check that must see the weights arrive.

The repository is read from `--repo`, else `SCANNCART_MODELS_REPO`, else the `origin` remote, so a
fork fetches its own release. `GITHUB_TOKEN`, when set, is sent with the requests: it lifts the
anonymous API limit (60 requests an hour per address) and reaches a private fork's release.

    sidecar/.venv/bin/python scripts/fetch_models.py            # Windows: sidecar\\.venv\\Scripts\\python.exe
    sidecar/.venv/bin/python scripts/fetch_models.py --check    # list what is missing, download nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "sidecar" / "models"
DEFAULT_TAG = "models"
#: The upstream repository, used only when neither a flag, the environment nor a git remote names one.
FALLBACK_REPO = "Yuserz/scanncart"
#: What a release asset has to be to land in `sidecar/models/`: the two weight formats the picker
#: lists, and the record beside a weight. Anything else attached to the release is left alone.
MODEL_SUFFIXES = (".pt", ".onnx", ".json")
API = "https://api.github.com"
TIMEOUT_S = 60

Opener = Callable[[urllib.request.Request], "urllib.response.addinfourl"]


def _say(message: str) -> None:
    print(message, flush=True)


def repo_from_remote(url: str) -> str | None:
    """`owner/name` out of a GitHub remote URL (https or ssh), or None for anything else."""
    match = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return f"{match.group(1)}/{match.group(2)}" if match else None


def resolve_repo(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("SCANNCART_MODELS_REPO"):
        return os.environ["SCANNCART_MODELS_REPO"]
    try:
        url = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return repo_from_remote(url) or FALLBACK_REPO
    except (OSError, subprocess.SubprocessError):
        return FALLBACK_REPO


def _request(url: str, accept: str) -> urllib.request.Request:
    headers = {"Accept": accept, "User-Agent": "scanncart-fetch-models"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def release_assets(repo: str, tag: str, opener: Opener) -> list[dict] | None:
    """The release's model assets as `{name, size, url}`, or None when there is no such release."""
    url = f"{API}/repos/{repo}/releases/tags/{tag}"
    try:
        with opener(_request(url, "application/vnd.github+json")) as response:
            release = json.load(response)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    return [
        {"name": a["name"], "size": int(a["size"]), "url": a["browser_download_url"]}
        for a in release.get("assets", [])
        if a.get("name", "").lower().endswith(MODEL_SUFFIXES)
    ]


def download(asset: dict, dest: Path, opener: Opener) -> None:
    """One asset into `dest`, through a temporary file that is moved into place only when whole."""
    fd, tmp = tempfile.mkstemp(prefix=f".{asset['name']}.", suffix=".part", dir=dest)
    try:
        with os.fdopen(fd, "wb") as out, opener(
            _request(asset["url"], "application/octet-stream")
        ) as response:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        got = Path(tmp).stat().st_size
        if got != asset["size"]:
            raise OSError(f"{asset['name']}: got {got} bytes, the release says {asset['size']}")
        os.replace(tmp, dest / asset["name"])
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fetch(
    repo: str,
    tag: str,
    dest: Path,
    *,
    check: bool = False,
    opener: Opener | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """`(fetched, kept, missing)` asset names. Raises on a release that cannot be read."""
    opener = opener or (lambda req: urllib.request.urlopen(req, timeout=TIMEOUT_S))
    assets = release_assets(repo, tag, opener)
    if assets is None:
        raise LookupError(f"{repo} has no release tagged {tag!r}")
    dest.mkdir(parents=True, exist_ok=True)
    fetched, kept, missing = [], [], []
    for asset in assets:
        if (dest / asset["name"]).exists():
            kept.append(asset["name"])
        elif check:
            missing.append(asset["name"])
        else:
            _say(f"  downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB)")
            download(asset, dest, opener)
            fetched.append(asset["name"])
    return fetched, kept, missing


def main(argv: list[str] | None = None, opener: Opener | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", help="owner/name on GitHub (default: the origin remote)")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"release tag (default: {DEFAULT_TAG})")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="where weights go")
    parser.add_argument("--check", action="store_true", help="report what is missing, download nothing")
    parser.add_argument("--strict", action="store_true", help="exit nonzero when nothing could be fetched")
    args = parser.parse_args(argv)

    repo = resolve_repo(args.repo)
    _say(f"model weights: release {args.tag!r} of {repo} -> {args.dest}")
    try:
        fetched, kept, missing = fetch(repo, args.tag, args.dest, check=args.check, opener=opener)
    except LookupError as err:
        _say(f"  warning: {err} - no weights downloaded. Attach the .pt and .json files from the "
             f"training PC's sidecar/models/ to a release tagged {args.tag!r}, or copy them in by hand.")
        return 1 if args.strict else 0
    except (urllib.error.URLError, OSError, ValueError) as err:
        _say(f"  warning: could not fetch the weights ({err}) - re-run this when online.")
        return 1 if args.strict else 0

    for name in kept:
        _say(f"  kept     {name} (already in {args.dest.name}/, not replaced)")
    for name in fetched:
        _say(f"  ok       {name}")
    for name in missing:
        _say(f"  missing  {name}")
    if not (fetched or kept or missing):
        _say(f"  warning: release {args.tag!r} has no .pt/.onnx/.json assets attached")
        return 1 if args.strict else 0
    return 1 if (args.check and missing and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
