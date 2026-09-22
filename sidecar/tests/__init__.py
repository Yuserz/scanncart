"""Shared helpers for the sidecar's tests.

Deliberately tiny - most test support lives beside the tests that use it - but this one is used
by three modules, and it exists because of a *protocol* fact rather than a local convenience.
"""

from __future__ import annotations


def next_frame(ws) -> dict:
    """The next frame message, skipping any status messages ahead of it.

    `/ws/stream` sends the current capture state immediately on connect (see `main.py`'s `stream`
    route), so the first message a client receives is a status, not a frame. Tests that want a
    frame - whether to assert on it or just to pull the pipeline forward so a detection is logged -
    have to say so, because `receive_json()` now answers a different message than it used to. The
    alternative, dropping the handshake status, would take the fix with it.
    """
    while True:
        msg = ws.receive_json()
        if msg.get("type") == "frame":
            return msg
