"""The frame stream under load, against a real server on the config this repo ships.

`tests/test_server_loop.py` asserts the *choice* (`ws_per_message_deflate=False` on the config
object uvicorn reads) and `app/run.py`'s `server_config` comment carries the measurements; this
file is the part that has to be *run*, because the failure it guards is only visible with preview
frames actually crossing the wire. The shape that starved the serving loop before the fix: three
websocket clients receiving a live-sized preview stream while `/api/health` is polled on fresh
connections — measured that morning as a health read outliving a 10 s timeout and a new handshake
timing out, with the process alive and py-spy parked in `permessage_deflate.encode`.

The load is incident-shaped, not incidental. The frames are incompressible (random noise, the same
content class as a live camera's JPEG), sized the way the rig's previews measured (~100 KB on the
wire), paced the way the camera paces inference (~30 fps), and fanned out to three clients — the
renderer, the transfer client and a probe, as the machine stood that morning. With the flag
regressed to uvicorn's default, that shape re-creates the starvation: deflating on the serving
loop costs tens of milliseconds per message per client, the drain queue backs up, and health
crawls past the second this test allows. With the fix, the same window answers health in
single-digit milliseconds — margin wide enough that a slow CI machine does not flake it.

Everything fake here is the *hardware* only. The server, the loop, the pipeline threads, the
preview encoder, the websocket protocol and the sockets are the shipped ones; the same pattern as
`tests/test_pipeline_e2e.py`, served over real sockets the way `tests/test_server_flood.py` does.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import http.client
import json
import os
import socket
import threading
import time
from collections.abc import Iterator

import numpy as np
import uvicorn
import websockets

from app.hardware import HardwareInfo
from app.main import AppState, build_app
from app.schemas import Detection
from app.settings import Settings
from app_run import bind_port, server_config

#: How many websocket clients the load fans out to: the count the rig actually held when the
#: starvation was measured, and the count that makes the regressed case saturate the serving
#: loop rather than merely slow it.
CLIENTS = 3

#: The window the load runs for. Long enough for the health prober to collect a meaningful
#: number of samples and for a regressed deflate queue to back up; short enough to keep the
#: suite's real-server time where the flood tests already put it.
WINDOW_S = 4.0

#: One health read is allowed one second on the wire — the assertion is structural (a read that
#: overruns raises out of the probe and cancels the window) as well as explicit below.
HEALTH_BUDGET_S = 1.0

#: A new websocket handshake must complete inside this too. The incident's second symptom was a
#: handshake that timed out at the library default while the process was alive.
HANDSHAKE_BUDGET_S = 5.0

#: A client that received fewer frames than this did not witness a stream, and the health
#: numbers beside it would prove nothing about one.
MIN_FRAMES_PER_CLIENT = 10


class _NoiseSource:
    """A camera that delivers incompressible frames at camera pace.

    Random noise is the worst case for the cost being guarded: a JPEG of it is as large as a
    preview gets and deflate on its base64 buys almost nothing, which is exactly the message
    profile a live counter produces. `latest()` paces itself at ~30 fps — the rate the capture
    thread feeds the real pipeline — by blocking like a frame wait would.
    """

    width = 640
    height = 360
    fps = 30.0

    def __init__(self) -> None:
        self._rng = np.random.default_rng(20261007)
        self._seq = 0

    def open(self) -> bool:
        return True

    def latest(self) -> tuple[int, np.ndarray]:
        time.sleep(1.0 / self.fps)
        self._seq += 1
        frame = self._rng.integers(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        return (self._seq, frame)

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _EmptyDetector:
    """Between customers: a model that runs every frame and boxes nothing.

    The preview still encodes and still streams — the frame message carries the JPEG whether or
    not there are boxes, which is precisely the traffic the starvation lived in.
    """

    names = {0: "banana", 1: "apple", 2: "milk"}

    def infer(self, frame: np.ndarray) -> list[Detection]:
        return []


def _fake_hardware() -> HardwareInfo:
    return HardwareInfo(cpu_count=8, ram_gb=16.0, cuda_available=False, accelerator="cpu")


def _request(port: int, method: str, path: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path)
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


@contextlib.contextmanager
def _streaming_server(*, start_capture: bool = True) -> Iterator[int]:
    """The app's own server, real config and loop, with a fake camera behind it.

    The same shape `tests/test_server_flood.py` serves — `run.py`'s own `server_config` on an
    OS-assigned port, in a thread, uvicorn's access log down because a streamed preview writes
    hundreds of lines the suite does not need — plus the one thing the flood tests never do:
    a capture actually running, so the websocket route has the traffic the loop guards exist for.
    """
    source = _NoiseSource()
    state = AppState(
        settings=Settings(preview_height=360, preview_max_fps=65),
        source_factory=lambda s: source,
        detector_factory=lambda s, d: _EmptyDetector(),
        db_path=":memory:",
        hardware_prober=_fake_hardware,
    )
    sock = bind_port(0)
    port = sock.getsockname()[1]
    config = server_config(build_app(lambda: state), port)
    config.log_level = "warning"
    config.access_log = False
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                if _request(port, "GET", "/api/health") == 200:
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, "the server never came up"
            time.sleep(0.05)
        if start_capture:
            status = _request(port, "POST", "/api/capture/start")
            assert status == 200, f"capture failed to start ({status})"
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def _health_latency(port: int) -> float:
    """One health read on a *fresh* connection, timed — the way the renderer polls it.

    Fresh, because the incident's second face was new work starving behind the stream: a pooled
    keep-alive connection can hide exactly the stall this measures.
    """
    start = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=HEALTH_BUDGET_S)
    try:
        conn.request("GET", "/api/health")
        response = conn.getresponse()
        response.read()
        assert response.status == 200
    finally:
        conn.close()
    return time.perf_counter() - start


async def _load_window(port: int, seconds: float) -> tuple[list[int], list[float]]:
    """`CLIENTS` websocket readers and one health prober, run to a deadline.

    The readers count frame messages (the handshake's status snapshot does not count — it is the
    first thing the server sends and carries no preview). The prober samples health on a fresh
    connection every fifth of a second; `to_thread` keeps its blocking socket off the client
    loop, and its one-second socket timeout is what makes the budget structural: a stalled read
    raises here and cancels the window rather than being averaged away.
    """
    loop = asyncio.get_running_loop()
    stop_at = loop.time() + seconds
    received = [0] * CLIENTS
    latencies: list[float] = []

    async def reader(index: int, ws: websockets.asyncio.client.ClientConnection) -> None:
        while loop.time() < stop_at:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if json.loads(raw).get("type") == "frame":
                received[index] += 1

    async def prober() -> None:
        while loop.time() < stop_at:
            latencies.append(await asyncio.to_thread(_health_latency, port))
            await asyncio.sleep(0.2)

    connections = [
        await websockets.connect(
            f"ws://127.0.0.1:{port}/ws/stream",
            max_size=10_000_000,
            open_timeout=HANDSHAKE_BUDGET_S,
        )
        for _ in range(CLIENTS)
    ]
    try:
        async with asyncio.TaskGroup() as group:
            for index, ws in enumerate(connections):
                group.create_task(reader(index, ws))
            group.create_task(prober())
    finally:
        for ws in connections:
            await ws.close()
    return received, latencies


def test_health_stays_under_a_second_while_three_clients_stream_preview_frames():
    """The incident's shape, replayed with the fix in place and bounded in seconds.

    Two things are asserted so the numbers mean something: the stream was real (every client fed
    frames for the whole window — a health that is fast because nothing is streaming proves
    nothing), and health answered every fresh-connection read inside the budget while that stream
    ran. A regressed permessage-deflate fails the second long before the window closes.
    """
    with _streaming_server() as port:
        received, latencies = asyncio.run(_load_window(port, WINDOW_S))

    for index, count in enumerate(received):
        assert count >= MIN_FRAMES_PER_CLIENT, (
            f"client {index} received only {count} frames in {WINDOW_S}s - the load this test "
            "claims to measure never happened, so its health numbers prove nothing"
        )
    assert len(latencies) >= 8, (
        f"only {len(latencies)} health reads landed in {WINDOW_S}s - the prober itself was "
        "starved, which is the failure under test rather than a pass"
    )
    worst = max(latencies)
    assert worst < 1.0, (
        f"the slowest /api/health under a {CLIENTS}-client preview stream took {worst:.2f}s - "
        "the serving loop is starving. That is the permessage-deflate shape: check that "
        "server_config still passes ws_per_message_deflate=False (app/run.py carries the "
        "measurement), or the load has grown past what one loop can answer."
    )


def _upgrade_response(port: int) -> list[str]:
    """A raw websocket upgrade that offers permessage-deflate; the extension lines back.

    Raw rather than through the `websockets` client on purpose: the library's negotiation is
    exactly what would hide a server-side regression behind a working client, while the header
    the server answers with is the fact itself.
    """
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET /ws/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n"
        "\r\n"
    )
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request.encode())
        sock.settimeout(5)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    lines = data.decode("latin-1").split("\r\n")
    assert lines[0].startswith("HTTP/1.1 101"), (
        f"the websocket upgrade failed instead of negotiating: {lines[0]!r}"
    )
    return [
        line for line in lines[1:] if line.lower().startswith("sec-websocket-extensions")
    ]


def test_the_wire_declines_permessage_deflate_even_when_offered():
    """What the socket answers when a client offers the extension.

    The config assertion in `tests/test_server_loop.py` pins the flag on the object uvicorn
    reads; this pins what a client actually receives, so a future uvicorn upgrade that drops or
    renames the keyword cannot quietly reintroduce the cost while the old assertion keeps
    passing on an argument nothing reads anymore.
    """
    with _streaming_server(start_capture=False) as port:
        extensions = _upgrade_response(port)
    assert extensions == [], (
        f"the server negotiated {extensions} - uvicorn's permessage-deflate default is back on "
        "the frame stream. It costs ~60 ms of serving-loop CPU per ~900 KB preview frame per "
        "client and starved /api/health past ten seconds when it shipped by default "
        "(app/run.py's server_config comment carries the measurement)."
    )
