"""Credential loading for the sidecar.

`Settings` is persisted to data/settings.json and returned wholesale by
GET /api/settings to the renderer, so a secret must never live on it. The
Roboflow API key is read from the process environment instead, falling back to
sidecar/.env (gitignored; see .env.example).

Hand-rolled rather than depending on python-dotenv: the format needed is
KEY=value, and settings.py deliberately stays loader-free.

Paths are relative to the sidecar working directory, matching
settings_store.DEFAULT_SETTINGS_PATH.
"""

import os
from pathlib import Path

ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"
DEFAULT_ENV_PATH = ".env"

_QUOTES = ("'", '"')


def parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=value lines. Blank lines, `#` comments, a leading `export `,
    and matched surrounding quotes are all handled; anything else is skipped
    rather than raising, so a hand-edited file never crashes startup."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        out[key] = value
    return out


def load_env_file(path: str = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Read and parse an env file. A missing or unreadable file yields {}."""
    try:
        return parse_env_file(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}


def load_api_key(path: str = DEFAULT_ENV_PATH) -> str | None:
    """The Roboflow API key, or None if unset.

    The real environment wins over the file so a packaged deploy can inject the
    key without shipping a .env. Blank values are treated as unset.
    """
    value = os.environ.get(ROBOFLOW_API_KEY_ENV)
    if value and value.strip():
        return value.strip()
    value = load_env_file(path).get(ROBOFLOW_API_KEY_ENV)
    return value.strip() if value and value.strip() else None


def has_api_key(path: str = DEFAULT_ENV_PATH) -> bool:
    """Whether a key is available. This is the only thing the API ever exposes
    about the key — the value itself must not leave the sidecar process."""
    return load_api_key(path) is not None
