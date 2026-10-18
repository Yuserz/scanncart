"""Camera enumeration — turns OpenCV's bare device indices into named devices.

`camera_index` is an integer because that is all OpenCV accepts, but "0" tells
an operator nothing when a machine has a built-in webcam and a StreamCam
plugged in. This module pairs each workable index with a human-readable name so
the Admin Panel can offer "1 - Logitech StreamCam (1920x1080)".

The pairing is the awkward part: OpenCV exposes an index and no name. Capture
opens `cv2.VideoCapture(index, cv2.CAP_MSMF)`, and OpenCV's Media Foundation
backend resolves that index as the index-th device of `MFEnumDeviceSources`
(video-capture sources). So the names are read from that same enumeration -
`list_msmf_device_names`, through ctypes, with no new dependency - and zipped
positionally onto the indices, which makes the position a real join key rather
than a coincidence of two orders.

It used to be one. The names came from Windows' PnP list (Win32_PnPEntity,
classes Camera + Image), which WMI returns in its own order and which also holds
devices Media Foundation never opens (a Windows Hello IR camera, a scanner). On a
laptop with a built-in camera beside the StreamCam, the row labelled "Logitech
StreamCam" opened the built-in camera. The PnP list is kept only as the fallback
for when Media Foundation cannot be asked at all.

Every device also carries the resolution it actually probed at, through the same
open path capture uses (`camera._default_capture`), so a row can still be
checked against what the device is: a StreamCam reports 1920x1080 where a
typical built-in webcam reports 1280x720.

Probing opens each device, which is slow (~0.5-2 s each) and cannot be done
while capture holds the camera — callers must only probe when idle.
"""

import ctypes
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Callable

import cv2

from app.camera import _default_capture

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


# Media Foundation, as OpenCV's MSMF backend calls it (cap_msmf: MFCreateAttributes, SetGUID
# SOURCE_TYPE = VIDCAP, MFEnumDeviceSources, then the index-th IMFActivate). These three GUIDs
# appear byte-for-byte in this venv's cv2.pyd.
_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE = "c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3"
_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID = "8ac3587a-4ae7-42d8-99e0-0a6013eef90f"
_MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME = "60d0e559-52f8-4fa2-bbce-acdb34a8ec01"
_MF_VERSION = 0x00020070  # MF_SDK_VERSION 0x0002 << 16 | MF_API_VERSION 0x0070
_MFSTARTUP_FULL = 0
_COINIT_MULTITHREADED = 0
# IMFAttributes vtable slots (IUnknown's three first); IMFActivate inherits them.
_RELEASE = 2
_GET_ALLOCATED_STRING = 13
_SET_GUID = 24


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _GUID:
    g = _GUID()
    ctypes.memmove(ctypes.byref(g), uuid.UUID(text).bytes_le, 16)
    return g


def _com_method(obj: int, slot: int, *argtypes):
    """The `slot`-th vtable entry of COM object `obj`, bound to it, returning the HRESULT."""
    vtable = ctypes.cast(ctypes.c_void_p(obj), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return lambda *args: fn(obj, *args)


def _enumerate_msmf(ole32, mfplat, mf) -> list[str]:
    """The friendly name of every Media Foundation video-capture source, in enumeration order;
    "" where a device would not give one, so every later name keeps its index."""
    names: list[str] = []
    attrs = ctypes.c_void_p()
    if mfplat.MFCreateAttributes(ctypes.byref(attrs), 1) < 0 or not attrs.value:
        return []
    guid_arg = ctypes.POINTER(_GUID)
    try:
        key = _guid(_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE)
        vidcap = _guid(_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
        if _com_method(attrs.value, _SET_GUID, guid_arg, guid_arg)(
            ctypes.byref(key), ctypes.byref(vidcap)
        ) < 0:
            return []
        devices = ctypes.POINTER(ctypes.c_void_p)()
        count = ctypes.c_uint32()
        if mf.MFEnumDeviceSources(attrs, ctypes.byref(devices), ctypes.byref(count)) < 0:
            return []
        friendly = _guid(_MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)
        try:
            for i in range(count.value):
                activate = devices[i]
                if not activate:
                    names.append("")
                    continue
                try:
                    text = ctypes.c_void_p()
                    length = ctypes.c_uint32()
                    get_string = _com_method(
                        activate, _GET_ALLOCATED_STRING, guid_arg,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32),
                    )
                    hr = get_string(ctypes.byref(friendly), ctypes.byref(text), ctypes.byref(length))
                    if hr >= 0 and text.value:
                        names.append(ctypes.wstring_at(text.value, length.value))
                        ole32.CoTaskMemFree(text)
                    else:
                        names.append("")
                finally:
                    _com_method(activate, _RELEASE)()
        finally:
            if devices:
                ole32.CoTaskMemFree(ctypes.cast(devices, ctypes.c_void_p))
    finally:
        _com_method(attrs.value, _RELEASE)()
    return names


def list_msmf_device_names() -> list[str]:
    """Camera names in Media Foundation's enumeration order - the order capture's
    `cv2.VideoCapture(index, cv2.CAP_MSMF)` indexes into. Enumerates only; no device is
    opened. Returns [] on any failure (non-Windows, no Media Foundation - an N edition of
    Windows ships without it) and never raises: a label must not break the panel."""
    if sys.platform != "win32":
        return []
    try:
        ole32 = ctypes.WinDLL("ole32")
        mfplat = ctypes.WinDLL("mfplat")
        mf = ctypes.WinDLL("mf")
    except OSError:
        return []
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    mfplat.MFStartup.restype = ctypes.c_long
    mfplat.MFStartup.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    mfplat.MFShutdown.restype = ctypes.c_long
    mfplat.MFCreateAttributes.restype = ctypes.c_long
    mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    mf.MFEnumDeviceSources.restype = ctypes.c_long
    mf.MFEnumDeviceSources.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    # S_OK / S_FALSE mean this call initialised COM and owes the uninitialise; RPC_E_CHANGED_MODE
    # means the thread already runs COM in another mode, which Media Foundation works with too.
    co = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    try:
        if mfplat.MFStartup(_MF_VERSION, _MFSTARTUP_FULL) < 0:
            return []
        try:
            names = _enumerate_msmf(ole32, mfplat, mf)
        finally:
            mfplat.MFShutdown()
    except Exception:  # noqa: BLE001 - a label must never break the panel
        return []
    finally:
        if co in (0, 1):
            ole32.CoUninitialize()
    return [n or f"Camera {i}" for i, n in enumerate(names)]


def list_pnp_device_names() -> list[str]:
    """Best-effort camera names in Windows PnP order - **not** the order capture indexes
    into (see the module docstring), so only `list_device_names`' fallback. Returns [] on
    any failure (non-Windows, no PowerShell, timeout) — never raises, matching
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


def list_device_names(
    msmf_lister: Callable[[], list[str]] = list_msmf_device_names,
    pnp_lister: Callable[[], list[str]] = list_pnp_device_names,
) -> list[str]:
    """Camera names in the order capture's indices select from.

    Media Foundation's own list when it answers; the PnP list only when it does not, since a
    positional pairing off the wrong list is how a StreamCam row came to open a laptop camera.
    An empty answer is a non-answer rather than "no cameras": `list_cameras` stops at the first
    index that does not open, so the fallback can only label rows, never add one.
    """
    try:
        names = msmf_lister()
    except Exception:  # noqa: BLE001 - fall through to the PnP names
        names = []
    return names or pnp_lister()


def name_for_index(index: int, names: list[str]) -> str:
    """The Windows name at a positional index, or a numbered fallback.

    Names come back from Windows with no index attached and indices come back
    from OpenCV with no name attached (see the module docstring), so this is
    the one place that convention lives. `list_cameras` uses it when pairing
    the dropdown; anything else that needs a device's name (e.g. calibration's
    device_key) should call this rather than re-deriving the fallback.
    """
    return names[index] if index < len(names) else f"Camera {index}"


# The probe opens a device exactly the way capture will (MSMF, then OpenCV's auto backend only if
# MSMF cannot open it), so the size shown beside a row is the size of the device that row starts.
_default_cap_factory = _default_capture


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
        name = name_for_index(index, names)
        devices.append(CameraDevice(index=index, name=name, width=width, height=height))
    return devices
