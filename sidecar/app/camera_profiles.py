"""Persist CameraProfiles to data/camera_profiles.json.

Same shape as settings_store: atomic write via a temp file plus os.replace, and
a missing or corrupt file yields no profiles rather than crashing startup.
"""

import json
import os
from dataclasses import asdict

from app.camera_caps import CameraProfile, ControlSupport

DEFAULT_PROFILES_PATH = "data/camera_profiles.json"


def load_profiles(path: str = DEFAULT_PROFILES_PATH) -> dict[str, CameraProfile]:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for key, value in (raw or {}).items():
        try:
            value["controls"] = ControlSupport(**value.get("controls", {}))
            out[key] = CameraProfile(**value)
        except (TypeError, ValueError):
            continue
    return out


def save_profile(profile: CameraProfile, path: str = DEFAULT_PROFILES_PATH) -> None:
    profiles = load_profiles(path)
    profiles[profile.device_key] = profile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({k: asdict(v) for k, v in profiles.items()}, fh, indent=2)
    os.replace(tmp, path)
