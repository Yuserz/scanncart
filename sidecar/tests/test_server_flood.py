"""Two floods, against a real server on the loop this repo serves on.

`tests/test_server_loop.py` checks the choices and `app/loops.py` carries the mechanisms; this is the
part that has to be *run*, because both failures being answered here are only visible with sockets in
front of a live server.

The first is the selector's count ceiling. Windows' `select()` takes 512 descriptors in one call and
raises `ValueError: too many file descriptors in select()` on 513 - measured, and measured again as a
server that died: 520 connections opened and held, with no request ever sent on them, left the sidecar
`refused` with that traceback in its log. `BatchedSelectSelector` polls in batches, so the same flood
is survived. The connections are silent on purpose: a connection that sends nothing is invisible to
every limit uvicorn has (the 503 below is decided at headers-complete, and there are no headers),
which is why the ceiling needed the selector rather than the bound.

The second is the bound. `SERVER_CONCURRENCY_LIMIT` is handed to uvicorn as `limit_concurrency`, and
past it uvicorn answers `503 Service Unavailable` with `connection: close` instead of working on the
request - shed rather than queued. The test walks up to the shipped number one connection at a time,
so the 503 is asserted at the bound itself rather than at some number this file picked.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator

import uvicorn

from app.loops import SERVER_CONCURRENCY_LIMIT
from app.main import build_app
from app_run import bind_port, server_config

HEALTH = "/api/health"

#: More descriptors than the one `select()` call that raised on this machine would take, and more
#: than either server was expected to hold open. Not read from the platform: `FD_SETSIZE` is not
#: exported, and a test that discovered it would be a test of CPython rather than of this module.
FLOOD = 520


def _get(sock: socket.socket, path: str = HEALTH) -> tuple[int, bytes]:
    """One request on an already-open connection: the status line, and everything that follows."""
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
    sock.settimeout(5)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            return 0, data
        data += chunk
    status = int(data.split(b" ", 2)[1])
    if status == 503:
        # `connection: close` on the shed response, so the body is already on its way when the
        # headers are; reading to the end is what lets the caller see whose 503 it is.
        while True:
            try:
                chunk = sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            data += chunk
    return status, data


@contextlib.contextmanager
def _serving() -> Iterator[int]:
    """The app's own server - real config, real loop, real bound - on an OS-assigned port.

    The config is `run.py`'s own, so what is exercised is what ships: the loop choice and the
    concurrency bound come from `app/loops.py` through the same function the entrypoint calls. It
    runs in a thread rather than a subprocess because the assertions are about the server's own
    behaviour and a second interpreter would add seconds to every run; uvicorn's signal handling is
    already thread-aware (`Server.capture_signals` yields off the main thread), so nothing needs
    patching. Its own logging is turned down here and not in the shipped config: sixty-odd access
    lines per test is noise in a suite, not evidence.
    """
    sock = bind_port(0)
    port = sock.getsockname()[1]
    config = server_config(build_app(), port)
    config.log_level = "warning"
    config.access_log = False
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1) as probe:
                    if _get(probe)[0] == 200:
                        break
            except OSError:
                pass
            assert time.monotonic() < deadline, "the server never came up"
            time.sleep(0.05)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def test_a_held_open_flood_does_not_end_the_server():
    """`FLOOD` silent connections, which is the shape that ended a server on the plain selector.

    Nothing is sent on them, so nothing about them can be shed: the only thing standing between this
    server and `ValueError: too many file descriptors in select()` is the batching selector. The
    assertion is that it *answers* rather than that it answers 200, and deliberately - with the
    concurrency bound in place, a probe arriving behind five hundred open connections is itself past
    the bound and gets a 503, which is this server working. A status either way is a live server; 0
    is a dead one.
    """
    with _serving() as port:
        held = []
        try:
            for _ in range(FLOOD):
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5)
                conn.connect(("127.0.0.1", port))
                held.append(conn)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as probe:
                status, _ = _get(probe)
            assert status in (200, 503), (
                f"the server answered {status} with {len(held)} silent connections held open - the "
                "selector's descriptor count is what this flood is aimed at"
            )
        finally:
            for conn in held:
                conn.close()


def test_the_connection_past_the_bound_is_answered_503():
    """The shipped bound, walked up to one connection at a time.

    Sequential rather than simultaneous on purpose: each response proves the server registered that
    connection (a response cannot come from a protocol that never ran `connection_made`), so the count
    this reaches is known rather than raced - and kept alive meanwhile, since uvicorn counts *open*
    connections, which is what a client holding a socket pool looks like from in here.
    """
    limit = SERVER_CONCURRENCY_LIMIT
    with _serving() as port:
        held = []
        try:
            for index in range(limit - 1):
                conn = socket.create_connection(("127.0.0.1", port), timeout=5)
                held.append(conn)
                status, _ = _get(conn)
                assert status == 200, (
                    f"connection {index + 1} of {limit} was answered {status} - the bound is set "
                    "below this app's own traffic"
                )

            past = socket.create_connection(("127.0.0.1", port), timeout=5)
            held.append(past)
            status, body = _get(past)
            assert status == 503, (
                f"connection {limit} with the other {limit - 1} still open was answered {status}, "
                "not 503: the flood is being queued instead of shed"
            )
            assert b"Service Unavailable" in body, (
                "the 503 did not come from uvicorn's own limit - the body is what tells a shed "
                "connection apart from an application error"
            )
        finally:
            for conn in held:
                conn.close()

        # ...and the bound is not sticky: the flood is gone, so the next client is served.
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5) as fresh:
                    if _get(fresh)[0] == 200:
                        break
            except OSError:
                pass
            assert time.monotonic() < deadline, (
                "the server never went back to serving once the flood was closed - the bound is "
                "being held by connections that are gone"
            )
            time.sleep(0.1)
