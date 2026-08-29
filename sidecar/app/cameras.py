"""Camera enumeration — turns OpenCV's bare device indices into named devices.

`camera_index` is an integer because that is all OpenCV accepts, but "0" tells
an operator nothing when a machine has a built-in webcam and a StreamCam
plugged in. This module pairs each workable index with a human-readable name so
the Admin Panel can offer "1 - Logitech StreamCam (1920x1080)".

The pairing is the awkward part, and it is a heuristic: OpenCV exposes an index
and no name, Windows exposes a name and no index. Neither offers a join key
without a DirectShow/Media Foundation binding (`pygrabber` et al.), which is a
dependency this project does not want for a label. So names are taken in
Windows enumeration order and zipped positionally onto the indices that
actually open, which is the order OpenCV's MSMF backend also walks.

Because the pairing can be wrong, every device also carries the resolution it
actually probed at. That is the operator's check: a StreamCam reports
1920x1080 where a typical built-in webcam reports 1280x720, so a mislabeled row
is visible rather than silent.

Probing opens each device, which is slow (~0.5-2 s each) and cannot be done
while capture holds the camera — callers must only probe when idle.
"""

import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import cv2

# Windows groups webcams under either PNPClass; a UVC device can land in
# 'Camera' while a vendor-driver device (the StreamCam here) lands in 'Image'.
_PNP_CLASSES = ("Camera", "Image")
_PS_COMMAND = (
    "Get-CimInstance Win32_PnPEntity | "
    "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
    "Select-Object -ExpandProperty Name"
)

# Past this, assume there are no more devices. Probing is slow and every miss
# costs a full open attempt, so the ceiling stays low.
DEFAULT_MAX_INDEX = 5


@dataclass
class CameraDevice:
    index: int
    name: str
    width: int
    height: int


def list_device_names() -> list[str]:
    """Best-effort camera names in Windows enumeration order. Returns [] on any
    failure (non-Windows, no PowerShell, timeout) — never raises, matching
    hardware._list_display_adapters."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_COMMAND],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _default_cap_factory(index: int):
    return cv2.VideoCapture(index)


def probe_index(index: int, cap_factory: Callable[[int], object] = _default_cap_factory):
    """The (width, height) a device opens at, or None if the index is unusable.

    Asks for 1080p first: OpenCV reports whatever the device clamps that to, so
    a 720p webcam and a 1080p StreamCam come back distinguishable instead of
    both reporting whatever default they happened to start in.
    """
    cap = None
    try:
        cap = cap_factory(index)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None
    finally:
        release = getattr(cap, "release", None)
        if callable(release):
            release()


def list_cameras(
    max_index: int = DEFAULT_MAX_INDEX,
    name_lister: Callable[[], list[str]] = list_device_names,
    cap_factory: Callable[[int], object] = _default_cap_factory,
) -> list[CameraDevice]:
    """Every camera index that opens, paired with a name where one is known.

    Stops at the first index that fails to open: OpenCV indices are dense, so a
    gap means the end of the list, and probing past it only wastes seconds.
    """
    try:
        names = name_lister()
    except Exception:
        names = []

    devices: list[CameraDevice] = []
    for index in range(max_index):
        probed = probe_index(index, cap_factory)
        if probed is None:
            break
        width, height = probed
        # Fall back to the index when Windows named fewer devices than opened —
        # a labeled device is a nicety, an unusable dropdown row is not.
        name = names[index] if index < len(names) else f"Camera {index}"
        devices.append(CameraDevice(index=index, name=name, width=width, height=height))
    return devices
